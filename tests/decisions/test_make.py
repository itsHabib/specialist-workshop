import hashlib
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from decision_workshop import base
from decision_workshop.maker import make


class MakeTests(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)
        self.output = self.root / "run"
        self.source = self.root / "task"
        self.source.mkdir()
        self.base_dir = self.root / "public-base"
        task = {
            "name": "document_route", "version": "1",
            "features": {"text": "text"}, "labels": ["keep", "refer"],
        }
        (self.source / "task.json").write_text(json.dumps(task), encoding="utf-8")
        self.write_records(("train", "dev", "test"))

    def write_records(self, splits):
        with (self.source / "records.jsonl").open("w", encoding="utf-8") as handle:
            for split in splits:
                row = {
                    "id": split + "-id", "split": split, "group": split + "-group",
                    "target": "keep", "state": {"text": split + " document"},
                    "candidates": ["keep", "refer"],
                }
                handle.write(json.dumps(row) + "\n")

    def mock_numerics(self):
        # Exercise the real local checkpoint writer, reader, and replay. The tests
        # check which data reaches the numerical boundary, never model quality.
        self.base_dir.mkdir()
        pins = {}
        for name in base.FILE_HASHES:
            content = name.encode("utf-8")
            (self.base_dir / name).write_bytes(content)
            pins[name] = hashlib.sha256(content).hexdigest()
        pin_patch = patch.dict(base.FILE_HASHES, pins)
        pin_patch.start()
        self.addCleanup(pin_patch.stop)

        def fit(task, practice, source, output, **kwargs):
            (output / "test-only-bytes").write_bytes(b"mock numerical training")
            return {"optimizer_steps": 1}

        numerics = Mock()
        numerics.fit_bert.side_effect = fit
        numerics.load_scorer.return_value = Mock(return_value={"keep": 0.6, "refer": 0.4})
        module_patch = patch.dict(sys.modules, {"decision_workshop.bert": numerics})
        module_patch.start()
        self.addCleanup(module_patch.stop)
        return numerics

    def test_missing_split_fails_before_dependencies_or_base_are_needed(self):
        for splits, missing in [(("train", "test"), "dev"), (("dev", "test"), "train")]:
            with self.subTest(missing=missing):
                self.write_records(splits)
                with patch.dict(sys.modules, {"decision_workshop.bert": None}):
                    with self.assertRaisesRegex(ValueError, missing):
                        make(self.source, self.output, self.base_dir)
                self.assertFalse(self.output.exists())

    def test_existing_run_is_preserved(self):
        self.output.mkdir()
        sentinel = self.output / "existing"
        sentinel.write_bytes(b"keep previous run")
        with self.assertRaises(FileExistsError):
            make(self.source, self.output, self.base_dir)
        self.assertEqual(sentinel.read_bytes(), b"keep previous run")
        self.assertEqual(list(self.output.iterdir()), [sentinel])

    def test_cli_can_run_preflight_with_only_the_standard_library(self):
        self.write_records(("train", "test"))
        result = subprocess.run(
            [sys.executable, "-S", "-m", "decision_workshop", "make", str(self.source),
             "--out", str(self.output), "--base", str(self.base_dir)],
            cwd=Path(__file__).resolve().parents[2],
            capture_output=True, text=True, timeout=10, check=False,
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("split=dev", result.stderr)
        self.assertNotIn("Traceback", result.stderr)
        self.assertFalse(self.output.exists())

    def test_failure_after_training_leaves_no_visible_run(self):
        numerics = self.mock_numerics()
        numerics.load_scorer.side_effect = ValueError("cannot load test scorer")
        with self.assertRaisesRegex(ValueError, "cannot load test scorer"):
            make(self.source, self.output, self.base_dir)
        numerics.fit_bert.assert_called_once()
        self.assertFalse(self.output.exists())

    def test_publishes_run_and_keeps_test_examples_out_of_training_and_replay(self):
        numerics = self.mock_numerics()
        make(self.source, self.output, self.base_dir)

        practice = numerics.fit_bert.call_args.args[1]
        self.assertEqual([row["id"] for row in practice], ["train-id"])
        scorer = numerics.load_scorer.return_value
        self.assertEqual([call.args[0] for call in scorer.call_args_list],
                         [{"text": "dev document"}])
        self.assertTrue((self.output / "model" / "manifest.json").is_file())
        self.assertTrue((self.output / "tree.json").is_file())
        summary = json.loads((self.output / "report.json").read_text(encoding="utf-8"))
        self.assertEqual(summary["task"], "document_route")
        self.assertEqual(summary["split"], "dev")
        for backend in ("bert", "tree"):
            report = json.loads((self.output / f"{backend}-dev.json").read_text(encoding="utf-8"))
            self.assertEqual([case["id"] for case in report["cases"]], ["dev-id"])
            self.assertEqual(summary[backend]["model"], report["model"])
            expected_path = "model" if backend == "bert" else "tree.json"
            self.assertEqual(summary[backend]["checkpoint"], expected_path)
