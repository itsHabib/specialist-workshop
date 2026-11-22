"""Exercise the public review policy and the model identity check independently of GPU libraries."""

from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from decision_workshop import load_model, predict
from decision_workshop.base import MODEL_ID, REVISION, FILE_HASHES
from decision_workshop.checkpoint import FORMAT
from decision_workshop.core import digest, write_new


class NeuralPolicyTests(unittest.TestCase):
    task = {"name": "routing", "version": "1", "labels": ["one", "two"],
            "features": {"note": "text"}}

    def test_threshold_and_omitted_action_apply_to_full_vocabulary(self):
        model = {"algorithm": FORMAT, "task": self.task, "training": {"categories": {}},
                 "min_score": 0.6,
                 "_score": lambda state: {"one": 0.6, "two": 0.4}}
        request = {"note": "a previously unseen note"}
        self.assertEqual(predict(model, request, ["one", "two"])["action"], "one")
        decision = predict(model, request, ["two"], min_score=0.7)
        self.assertEqual(decision["preferred"], "one")
        self.assertIsNone(decision["action"])
        self.assertEqual(set(decision["reasons"]), {"preferred_action_missing", "below_min_score"})

    def test_modified_neural_file_rejected_before_model_loading(self):
        with TemporaryDirectory() as tmp:
            folder = Path(tmp)
            (folder / "decision_head.safetensors").write_bytes(b"changed")
            model = {"format": FORMAT, "algorithm": FORMAT, "task": self.task,
                     "base": {"model": MODEL_ID, "revision": REVISION, "files": FILE_HASHES},
                     "artifacts": {"decision_head.safetensors": "0" * 64}}
            model["sha256"] = digest(model)
            write_new(folder / "manifest.json", model)
            with self.assertRaisesRegex(ValueError, "files differ"):
                load_model(folder)
