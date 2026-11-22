import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from decision_workshop.contracts import read_records
from decision_workshop.core import load_model, predict, train, write_new
from decision_workshop.evaluation import evaluate


TASK = {
    "name": "test routing",
    "version": "1",
    "features": {"retry": "boolean"},
    "labels": ["inspect", "retry"],
}


def record(row_id, split, retry, **overrides):
    return {
        "id": row_id,
        "group": f"group-{row_id}",
        "split": split,
        "state": {"retry": retry},
        "target": "retry" if retry else "inspect",
        "candidates": ["inspect", "retry"],
        **overrides,
    }


class ReplayTests(unittest.TestCase):
    def setUp(self):
        self.training = [
            record("a", "train", False), record("b", "train", False),
            record("c", "train", True), record("d", "train", True),
        ]
        self.model = train(TASK, self.training)

    def test_missing_preferred_label_yields_review_in_prediction_and_replay(self):
        row = record("e", "test", True, candidates=["inspect"])
        decision = predict(self.model, row["state"], row["candidates"])
        self.assertEqual(decision["preferred"], "retry")
        self.assertIsNone(decision["action"])
        self.assertEqual(decision["status"], "review")
        self.assertIn("preferred_action_missing", decision["reasons"])
        report = evaluate(self.model, [row], "test")
        self.assertEqual(report["reference_missing"], 1)
        self.assertEqual(report["preference_matches_before_candidate_check"], 1)
        self.assertEqual(report["model"]["review"], 1)
        self.assertEqual(report["model"]["matched"], 0)

    def test_paired_comparison_distinguishes_missing_null_and_outside_candidates(self):
        rows = [
            record("null", "dev", True, baselines={"planner": None}),
            record("missing", "dev", False),
            record("correct", "dev", False, baselines={"planner": "inspect"}),
            record("outside", "dev", True, candidates=["retry"], baselines={"planner": "inspect"}),
        ]
        report = evaluate(self.model, rows, "dev")
        comparison = report["recorded_baselines"]["planner"]
        self.assertEqual(report["rows"], 4)
        self.assertEqual(comparison["missing_records"], 1)
        self.assertEqual(comparison["outside_candidates"], 1)
        incumbent = comparison["incumbent"]
        self.assertEqual(incumbent["rows"], 3)
        self.assertEqual(incumbent["review"], 2)
        self.assertEqual(incumbent["matched"], 1)
        self.assertAlmostEqual(incumbent["match_per_record"], 1 / 3)
        self.assertEqual(incumbent["match_when_suggested"], 1.0)
        paired = comparison["model_on_same_records"]
        self.assertEqual((paired["matched"], paired["rows"]), (3, 3))

    def test_unknown_candidates_count_as_reviews_and_keep_raw_comparisons(self):
        rows = [
            record("unknown", "test", True, candidates=None, baselines={"planner": "retry"}),
            record("empty", "test", False, candidates=[], baselines={"planner": "inspect"}),
            record("wrong-preference", "test", False, target="retry", baselines={"planner": "retry"}),
            record("correct-preference", "test", False, baselines={"planner": None}),
        ]
        report = evaluate(self.model, rows, "test")
        self.assertEqual((report["candidate_sets_recorded"], report["candidate_sets_unrecorded"]), (3, 1))
        self.assertEqual((report["reference_in_candidates"], report["reference_missing"]), (2, 1))
        self.assertAlmostEqual(report["reference_in_candidates_rate"], 2 / 3)
        self.assertEqual(report["raw_model_preference"]["match_per_record"], 3 / 4)
        self.assertEqual(report["model"]["rows"], 4)
        self.assertEqual(report["model_with_recorded_candidates"]["rows"], 3)
        self.assertEqual(report["train_majority"]["rows"], 4)
        self.assertEqual(report["train_majority"]["coverage"], 2 / 4)
        self.assertEqual(report["model_when_reference_present"]["match_per_record"], 1 / 2)
        comparison = report["recorded_baselines"]["planner"]
        self.assertEqual(comparison["raw_incumbent"]["match_per_record"], 3 / 4)
        self.assertEqual(comparison["raw_model_preference"]["match_per_record"], 3 / 4)
        self.assertEqual(comparison["outside_candidates"], 1)
        for policy in ("incumbent", "model_on_same_records"):
            self.assertEqual(comparison[policy]["rows"], 3)
            self.assertEqual(comparison["when_reference_present"][policy]["rows"], 2)
            self.assertEqual(comparison["when_reference_present"][policy]["match_per_record"], 1 / 2)

        # A dataset containing only historical final decisions has no candidate recall estimate.
        only_unknown = evaluate(self.model, rows[:1], "test")
        self.assertIsNone(only_unknown["reference_in_candidates_rate"])
        self.assertEqual(only_unknown["model"]["match_per_record"], 0)
        self.assertEqual(only_unknown["model"]["review"], 1)
        self.assertIsNone(only_unknown["model_with_recorded_candidates"]["coverage"])
        self.assertIn("candidates_unrecorded", only_unknown["cases"][0]["decision"]["reasons"])
        historical = only_unknown["recorded_baselines"]["planner"]
        self.assertEqual(historical["raw_incumbent"]["matched"], 1)
        self.assertEqual(historical["raw_model_preference"]["matched"], 1)
        self.assertEqual(historical["incumbent"]["rows"], 0)

    def test_new_dataset_file_cannot_reuse_training_id_or_group(self):
        with TemporaryDirectory() as directory:
            model_file = Path(directory) / "trained.json"
            write_new(model_file, self.model)
            restored = load_model(model_file)
            # Each file is valid by itself; leakage must be checked against the checkpoint.
            for field in ("id", "group"):
                for split in ("dev", "test"):
                    with self.subTest(field=field, split=split):
                        row = record("new", split, True)
                        row[field] = self.training[0][field]
                        path = Path(directory) / f"{field}-{split}.jsonl"
                        path.write_text(json.dumps(row) + "\n", encoding="utf-8")
                        rows = read_records(path, TASK)
                        with self.assertRaisesRegex(ValueError, "used in checkpoint training"):
                            evaluate(restored, rows, split)

    def test_changing_held_out_labels_preserves_entire_checkpoint(self):
        held_out = [record("new-dev", "dev", False), record("new-test", "test", True)]
        flipped = [{**r, "target": "inspect" if r["target"] == "retry" else "retry"}
                   for r in held_out]
        self.assertEqual(train(TASK, self.training + held_out), self.model)
        self.assertEqual(train(TASK, self.training + flipped), self.model)

    def test_tampered_checkpoint_is_rejected(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / "model.json"
            write_new(path, self.model)
            self.assertEqual(load_model(path), self.model)
            with self.assertRaises(FileExistsError):
                write_new(path, {})
            tampered = json.loads(path.read_text(encoding="utf-8"))
            tampered["task"]["version"] = "2"
            path.write_text(json.dumps(tampered), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "digest mismatch"):
                load_model(path)
