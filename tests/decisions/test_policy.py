import json
from pathlib import Path
import shutil
import subprocess
import sys
import textwrap
import tempfile
import unittest

from decision_workshop.core import digest, train, write_new
from decision_workshop.policy import load_policy, save_policy
from decision_workshop.steering import LABELS, rank_candidates


HOST = {"adapter_revision": "input-cost/1", "fallback_revision": "native/1"}


class PolicyTests(unittest.TestCase):
    def setUp(self):
        scratch = tempfile.TemporaryDirectory()
        self.addCleanup(scratch.cleanup)
        self.temp = Path(scratch.name)
        task = {"name": "dispatch", "version": "1", "features": {"x": "number"}, "labels": LABELS}
        examples = [(0, "defer"), (0, "defer"), (0, "prefer"), (2, "prefer"), (2, "prefer"), (2, "prefer")]
        rows = [{"id": str(i), "group": str(i), "split": "train", "state": {"x": x},
                 "target": label, "candidates": LABELS} for i, (x, label) in enumerate(examples)]
        model = train(task, rows)
        path = self.temp / "training" / "tree.json"
        write_new(path, model)
        self.path, self.model = path, model

    def test_host_reload_dispatches_fixed_and_ranked_policies_from_its_own_directory(self):
        model_path = self.path
        training = model_path.parent
        fixed_digest = save_policy(training / "fixed.json", model_path, kind="fixed", min_score=0.8, **HOST)
        ranking_digest = save_policy(training / "ranking.json", model_path, kind="ranking", **HOST)
        host = self.temp / "caller"
        host.mkdir()
        training.rename(host / "deployment")
        source = Path(__file__).resolve().parents[2] / "decision_workshop"
        shutil.copytree(source, host / "decision_workshop", ignore=shutil.ignore_patterns("__pycache__"))
        script = textwrap.dedent("""
            import json, sys
            from pathlib import Path
            sys.path.insert(0, str(Path.cwd()))
            from decision_workshop.policy import load_policy
            directory = Path("deployment")
            settings = json.loads(sys.argv[1])
            def policy(name, sha):
                return load_policy(directory / (name + ".json"), directory / "tree.json",
                                   expected_sha256=sha, **settings)
            fixed = policy("fixed", sys.argv[2])
            ranking = policy("ranking", sys.argv[3])
            executed = []
            def dispatch(identifier):
                executed.append(identifier)
            decisions = []
            for value, candidates in [(0, ["defer", "prefer"]), (5, ["defer", "prefer"]), (5, None)]:
                choice = fixed.decide({"x": value}, candidates)
                decisions.append(choice["reasons"])
                dispatch(choice["action"] if choice["status"] == "suggest" else "default")
            roster = [("slow", {"x": 0}), ("z-new", {"x": 10}), ("a-new", {"x": 3})]
            for identifier, score in ranking.rank(iter(roster)):
                dispatch(identifier)
            print(json.dumps({"executed": executed, "reasons": decisions}))
        """)
        run = subprocess.run([sys.executable, "-I", "-S", "-c", script, json.dumps(HOST),
                              fixed_digest, ranking_digest], cwd=host, check=True,
                             text=True, capture_output=True)
        self.assertEqual(json.loads(run.stdout), {
            "executed": ["default", "prefer", "default", "z-new", "a-new", "slow"],
            "reasons": [["below_min_score"], [], ["candidates_unrecorded"]],
        })

    def test_ranking_ties_are_opt_in_and_missing_rosters_are_visible(self):
        model_path, model = self.path, self.model
        path = model_path.parent / "policy.json"
        save_policy(path, model_path, kind="ranking", **HOST)
        deployed = load_policy(path, model_path, **HOST)
        roster = [("z", {"x": 10}), ("a", {"x": 3})]
        self.assertEqual(rank_candidates(model, roster), [("a", 1.0), ("z", 1.0)])
        self.assertEqual(deployed.rank(roster), [("z", 1.0), ("a", 1.0)])
        self.assertEqual(deployed.rank([]), [])
        with self.assertRaisesRegex(ValueError, "roster is unrecorded"):
            deployed.rank(None)
        with self.assertRaisesRegex(ValueError, "distinct"):
            deployed.rank([roster[0], roster[0]])
        with self.assertRaisesRegex(ValueError, "requires a fixed policy"):
            deployed.decide({"x": 3}, LABELS)

    def test_integrity_host_revisions_and_checkpoint_substitution(self):
        model_path, model = self.path, self.model
        path = model_path.parent / "policy.json"
        sha = save_policy(path, model_path, kind="fixed", min_score=0.8, **HOST)
        for field in HOST:
            with self.assertRaisesRegex(ValueError, "revision mismatch"):
                load_policy(path, model_path, **{**HOST, field: "changed"})
        with self.assertRaisesRegex(ValueError, "pinned digest"):
            load_policy(path, model_path, expected_sha256="0" * 64, **HOST)
        with self.assertRaisesRegex(ValueError, "requires a ranking policy"):
            load_policy(path, model_path, expected_sha256=sha, **HOST).rank([])
        with self.assertRaises(FileExistsError):
            save_policy(path, model_path, kind="fixed", min_score=0.2, **HOST)

        alternate = {k: v for k, v in model.items() if k != "sha256"}
        alternate["settings"] = {**alternate["settings"], "max_depth": 4}
        other = model_path.parent / "another.json"
        write_new(other, {**alternate, "sha256": digest(alternate)})
        with self.assertRaisesRegex(ValueError, "checkpoint digest mismatch"):
            load_policy(path, other, **HOST)
        with self.assertRaises(ValueError):
            load_policy(path, model_path.parent / "gone.json", **HOST)
        manifest = json.loads(path.read_text())
        path.write_text(json.dumps({**manifest, "min_score": 0.1}))
        with self.assertRaisesRegex(ValueError, "policy digest mismatch"):
            load_policy(path, model_path, **HOST)

    def test_reject_invalid_settings_at_creation(self):
        path = self.path.parent / "policy.json"
        cases = [("fixed", None), ("fixed", True), ("fixed", 1.01), ("fixed", float("nan")),
                 ("ranking", 0.0), ("other", None)]
        for kind, cutoff in cases:
            with self.subTest(kind=kind, cutoff=cutoff), self.assertRaises(ValueError):
                save_policy(path, self.path, kind=kind, min_score=cutoff, **HOST)
            self.assertFalse(path.exists())

    def test_reject_unsupported_manifest_even_with_valid_digest(self):
        path = self.path.parent / "policy.json"
        save_policy(path, self.path, kind="ranking", **HOST)
        original = json.loads(path.read_text())
        del original["sha256"]
        for change in [{"tie_order": "alphabetical"}, {"new_setting": "ignored"},
                       {"format": "decision-workshop/policy-v2"}, {"min_score": None}]:
            body = {**original, **change}
            path.write_text(json.dumps({**body, "sha256": digest(body)}))
            with self.subTest(change=change), self.assertRaisesRegex(ValueError, "unsupported policy"):
                load_policy(path, self.path, **HOST)
