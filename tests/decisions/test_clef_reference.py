"""Receipt bookkeeping tests with synthetic scores. Numerical inference is exercised separately."""

import copy
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from decision_workshop import clef_reference as ref
from decision_workshop import clef_mlx_replay
from decision_workshop.clef_mlx_replay import file_hash, request_for
from decision_workshop.core import digest, train


class ClefReferenceTests(unittest.TestCase):
    def setUp(self):
        scratch = TemporaryDirectory()
        self.addCleanup(scratch.cleanup)
        self.root = Path(scratch.name)
        self.task = {"name": "ci", "version": "1", "features": {"stage": "category"},
                     "labels": ["inspect", "retry"], "question": "How should the failure be routed?",
                     "descriptions": {"inspect": "Inspect the failure.", "retry": "Run the check again."}}
        self.rows = [
            {"id": record_id, "group": split + "-group", "split": split, "state": {"stage": stage},
             "target": target, "candidates": candidates}
            for record_id, split, stage, target, candidates in (
                ("train-a", "train", "tests", "inspect", ["inspect", "retry"]),
                ("train-b", "train", "setup", "retry", ["inspect", "retry"]),
                ("dev-1", "dev", "tests", "inspect", ["inspect"]),
                ("dev-2", "dev", "tests", "retry", ["inspect"]),
                ("test-1", "test", "new-stage", "retry", None),
            )
        ]
        self.task_file = self.root / "task.json"
        self.rows_file = self.root / "records.jsonl"
        self.task_file.write_text(json.dumps(self.task), encoding="utf-8")
        self.write_rows()
        self.tree = train(self.task, self.rows, min_leaf=1, min_score=0.95)
        self.bert = {**self.tree, "algorithm": ref.BERT_ALGORITHM, "sha256": "test-bert-manifest",
                     "_score": lambda state: {"inspect": 0.3, "retry": 0.7}}
        fake_load = patch.object(ref, "load_model", side_effect=lambda path:
                                 self.bert if path.name == "model" else self.tree)
        fake_load.start()
        self.addCleanup(fake_load.stop)
        dev_hash = digest(self.rows[2:4])
        self.maker = {"task": "ci", "version": "1", "question": self.task["question"], "split": "dev",
                      "training_rows": 2, "development_rows": 2,
                      "bert": {"checkpoint": "model", "model_sha256": self.bert["sha256"],
                               "evaluation_records_sha256": dev_hash},
                      "tree": {"checkpoint": "tree.json", "model_sha256": self.tree["sha256"],
                               "evaluation_records_sha256": dev_hash}}
        (self.root / "report.json").write_text(json.dumps(self.maker), encoding="utf-8")
        inputs = {"task": "ci", "task_sha256": file_hash(self.task_file), "records_sha256": file_hash(self.rows_file),
                  "choice_field": "action", "trained_on_evaluation_rows": False,
                  "options": self.task["labels"], "split_counts": {"dev": 2, "test": 1},
                  "replay_script_sha256": file_hash(Path(clef_mlx_replay.__file__)),
                  "port_loader_sha256": ref.FILE_HASHES["clef_mlx.py"]}
        model = {"id": ref.MODEL_ID, "revision": ref.REVISION, "upstream_id": ref.UPSTREAM_ID,
                 "upstream_revision": ref.UPSTREAM_REVISION,
                 "files": {k: {"sha256": v} for k, v in ref.FILE_HASHES.items()}}
        cases = []
        for row in reversed(self.rows[2:]):
            probs = {"inspect": 0.9, "retry": 0.1} if row["id"] == "dev-1" else {"inspect": 0.2, "retry": 0.8}
            cases.append({"id": row["id"], "split": row["split"], "input": request_for(self.task, row["state"]),
                          "probabilities": probs.copy(), "raw_output": {"action": probs.copy()},
                          "error": None, "elapsed_seconds": 0.1})
        environment = {"os": "test-os", "machine": "test-machine", "python": "3.test",
                       "packages": {p: "1.test" for p in ("mlx", "mlx-lm", "mlx-vlm", "transformers", "huggingface_hub")}}
        self.receipt = {"format": "decision-workshop/clef-replay-v1", "status": "complete", "errors": [],
                        "model": model, "inputs": inputs, "cases": cases, "environment": environment}

    def write_rows(self):
        self.rows_file.write_text("".join(json.dumps(r) + "\n" for r in self.rows), encoding="utf-8")

    def compare(self, receipt=None, **kwargs):
        receipt = self.receipt if receipt is None else receipt
        path = self.root / "reference.json"
        path.write_text(json.dumps({**receipt, "sha256": digest(receipt)}), encoding="utf-8")
        return ref.compare(self.root, path, self.root, **kwargs)

    def test_same_state_scored_by_id_with_common_candidate_policy(self):
        report = self.compare(min_score=0.5)
        dev = report["splits"]["dev"]
        self.assertEqual([c["id"] for c in dev["cases"]], ["dev-1", "dev-2"])
        self.assertEqual(dev["models"]["clef"]["final"], {"correct": 1, "wrong": 0, "review": 1})
        self.assertEqual(dev["models"]["clef"]["raw"]["correct"], 2)
        self.assertEqual(dev["reference_availability"], {"present": 1, "missing": 1, "unknown": 0})
        case = dev["cases"][1]
        self.assertEqual(case["models"]["clef"]["preferred"], "retry")
        self.assertEqual(case["models"]["clef"]["reasons"], ["preferred_action_missing"])
        test = report["splits"]["test"]
        self.assertEqual(test["reference_availability"]["unknown"], 1)
        for name in ("clef", "bert", "tree"):
            reasons = test["cases"][0]["models"][name]["reasons"]
            self.assertIn("unseen_category", reasons)
            self.assertIn("candidates_unrecorded", reasons)
            self.assertEqual(test["models"][name]["final"]["review"], 1)
            self.assertEqual(sum(test["models"][name]["known_candidates"].values()), 0)

    def test_default_comparison_applies_frozen_cutoff_to_all_models(self):
        report = self.compare()
        self.assertEqual(report["min_score"], 0.95)
        first = report["splits"]["dev"]["cases"][0]["models"]
        self.assertEqual(first["clef"]["reasons"], ["below_min_score"])
        self.assertIn("below_min_score", first["bert"]["reasons"])
        self.assertEqual(first["tree"]["status"], "suggest")

    def test_rejects_resealed_incomplete_or_misbound_receipts(self):
        for change, error in [
            (lambda r: r.update(status="failed"), "incomplete"),
            (lambda r: r["model"].update(revision="other"), "revision"),
            (lambda r: r["inputs"].update(records_sha256="other"), "hash mismatch"),
            (lambda r: r["inputs"].update(trained_on_evaluation_rows=True), "inference policy"),
            (lambda r: r["cases"].pop(), "missing reference rows"),
            (lambda r: r["cases"].append(r["cases"][0]), "duplicate reference row"),
            (lambda r: r["cases"][0].update(id="train-a"), "unexpected reference row"),
            (lambda r: r["cases"][0].update(split="dev"), "split mismatch"),
            (lambda r: r["cases"][0].update(error="failed"), "incomplete reference inference"),
            (lambda r: r["cases"][0]["input"].update(state={"stage": "other"}), "input mismatch"),
            (lambda r: r["cases"][0]["probabilities"].update(inspect=0.8), "invalid probability"),
            (lambda r: r["cases"][0].update(probabilities={"inspect": False, "retry": True}), "invalid probability"),
            (lambda r: r["cases"][0]["raw_output"].update(action={}), "raw output mismatch"),
        ]:
            with self.subTest(error=error):
                receipt = copy.deepcopy(self.receipt)
                change(receipt)
                with self.assertRaisesRegex(ValueError, error):
                    self.compare(receipt)

    def test_checkpoint_and_maker_development_bindings_are_checked(self):
        self.bert["training"] = {**self.bert["training"], "records_sha256": "different training set"}
        with self.assertRaisesRegex(ValueError, "training records mismatch"):
            self.compare()
        self.bert["training"] = self.tree["training"]
        self.maker["tree"]["evaluation_records_sha256"] = "different development set"
        (self.root / "report.json").write_text(json.dumps(self.maker), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "development records disagree"):
            self.compare()

    def test_requires_shipped_driver_and_recorded_runtime_identity(self):
        for mutate, message in (
            (lambda r: r["inputs"].pop("replay_script_sha256"), "expected .*recorded None"),
            (lambda r: r["inputs"].update(replay_script_sha256="other"), "expected .*recorded 'other'"),
            (lambda r: r.pop("environment"), "receipt environment"),
            (lambda r: r["environment"].update(machine=" "), "environment.machine"),
            (lambda r: r["environment"].pop("python"), "environment.python"),
            (lambda r: r["environment"].pop("packages"), "environment.packages"),
            (lambda r: r["environment"]["packages"].pop("mlx-vlm"), "environment.packages.mlx-vlm"),
            (lambda r: r["environment"]["packages"].update(transformers=4), "environment.packages.transformers"),
        ):
            with self.subTest(message=message):
                receipt = copy.deepcopy(self.receipt)
                mutate(receipt)
                with self.assertRaisesRegex(ValueError, message):
                    self.compare(receipt)

    def test_cross_split_group_leakage_rejected_before_inference(self):
        self.rows[-1]["group"] = self.rows[0]["group"]
        self.write_rows()
        with self.assertRaisesRegex(ValueError, "occurs in splits"):
            self.compare()
