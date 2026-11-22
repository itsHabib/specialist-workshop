from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from decision_workshop.core import digest, train, write_new
from experiments.decision_recovery import benchmark as bench
from experiments.decision_recovery.demo import startup


class RecoveryBenchmarkTests(unittest.TestCase):
    def test_demo_checks_cutoff_before_dispatch_and_can_replay_an_old_checkpoint(self):
        state = {name: "unknown" if kind == "category" else 1 for name, kind in bench.TASK["features"].items()}
        rows = [dict(id=str(i), group=str(i), split="train", state=state, target=label, candidates=bench.ACTIONS)
                for i, label in enumerate(["lookup", "lookup", "retry"])]
        model = train(bench.TASK, rows, min_score=0.8)
        with TemporaryDirectory() as folder:
            path = Path(folder) / "new.json"
            write_new(path, model)
            policy, identity = startup(path)
            self.assertEqual(identity["min_score"], 0.8)
            self.assertEqual(policy(state, bench.ACTIONS)["reasons"], ["below_min_score"])

            old = {k: v for k, v in model.items() if k not in ("min_score", "sha256", "tree_diagnostics")}
            legacy = Path(folder) / "legacy.json"
            write_new(legacy, {**old, "sha256": digest(old)})
            policy, identity = startup(legacy)
            self.assertEqual(identity["reason"], "model_cutoff_error")
            self.assertIs(policy, bench.status_first)
            policy, identity = startup(legacy, min_score=0)
            self.assertEqual(identity["min_score"], 0)
            self.assertEqual(policy(state, bench.ACTIONS)["action"], "lookup")

    def test_free_shift_and_seeded_families_preserve_the_work(self):
        ordinary = list(bench.worlds("test"))
        for cohort in ("shift", "free"):
            cases = list(bench.worlds("test", cohort))
            self.assertEqual(len(ordinary), len(cases))
            for (group, case, baseline), (other_group, other_case, variant) in zip(
                ordinary, cases, strict=True
            ):
                self.assertEqual(
                    (group, case, baseline.batches, baseline.fault, baseline.deadline),
                    (other_group, other_case, variant.batches, variant.fault, variant.deadline),
                )
                if cohort == "shift":
                    self.assertEqual(variant.outage_until, baseline.outage_until + 4)
                else:
                    self.assertEqual((variant.outage_until, variant.lookup_cost), (0, 0))
        old_groups = {g for g, _, _ in ordinary}
        new_groups = {g for g, _, _ in bench.worlds("test", seed=735269)}
        self.assertFalse(old_groups & new_groups)

    def test_reward_preserves_distinct_terminal_outcomes(self):
        for status, expected in (("verified", 1), ("incorrect", -1), ("review", 0), ("failed", 0)):
            self.assertEqual(bench.value({"outcome": {"status": status, "duplicate_effects": 0}}), expected)
            self.assertEqual(bench.value({"outcome": {"status": status, "duplicate_effects": 1}}), -1)

    def test_paired_estimate_rejects_different_job_sets(self):
        left = {("family", "1"): "verified", ("family", "2"): "failed"}
        right = {("family", "1"): "review", ("family", "2"): "failed"}
        result = bench.paired(left, right)
        self.assertEqual(
            (result["difference"], result["tree_only_verified"], result["baseline_only_verified"]),
            (0.5, 1, 0),
        )
        with self.assertRaises(ValueError):
            bench.paired(left, {("family", "1"): "verified"})
        with self.assertRaises(ValueError):
            bench.paired(left, {**right, ("other", "3"): "failed"})

    def test_lost_reply_is_eligible_for_lookup_under_same_shared_continuation(self):
        sc = bench.Scenario(
            "synthetic-test", ((1,), (2,), (4,)), "lost_reply", deadline=4, lookup_cost=1, batch_cost=2
        )
        branches = bench.alternatives(sc, ())
        self.assertEqual(branches["values"]["lookup"], 1)
        self.assertEqual(branches["values"]["retry"], 0)
        self.assertEqual(branches["replays"]["lookup"]["outcome"]["event_count"], 1)
