import unittest

from experiments.decision_recovery import benchmark as bench


class RecoveryBenchmarkTests(unittest.TestCase):
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
