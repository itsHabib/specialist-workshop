from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from decision_workshop.contracts import load_task, read_json, validate_candidates, validate_records, validate_state
from decision_workshop.scaffold import create_task


class ScaffoldTests(unittest.TestCase):
    def setUp(self):
        temporary = TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.out = self.root / "routing"
        self.question = "Which team should handle the case?"
        self.actions = [" billing = Handle billing ", "support=Handle cases with priority=high"]

    def test_default_task_and_separate_example(self):
        create_task(self.out, self.question, self.actions)
        self.assertEqual({p.name for p in self.out.iterdir()},
                         {"task.json", "record.example.json", "records.jsonl", "request.json"})
        task = load_task(self.out / "task.json")
        self.assertEqual(task, {
            "name": "routing", "version": "1", "question": self.question,
            "features": {"context": "text", "case_type": "category"}, "labels": ["billing", "support"],
            "descriptions": {"billing": "Handle billing", "support": "Handle cases with priority=high"},
        })
        self.assertEqual((self.out / "records.jsonl").read_bytes(), b"")
        example = read_json(self.out / "record.example.json")
        self.assertIn("synthetic", example["id"])
        validate_records(task, [example])
        request = read_json(self.out / "request.json")
        self.assertEqual(set(request), {"state", "candidates"})
        validate_state(task, request["state"])
        validate_candidates(task, request["candidates"])

    def test_custom_fields_determine_example_types(self):
        fields = ["count=number", "open=boolean", "queue=category", "note=text"]
        create_task(self.out, self.question, self.actions, fields)
        task = load_task(self.out / "task.json")
        self.assertEqual(task["features"], {"count": "number", "open": "boolean",
                                           "queue": "category", "note": "text"})
        row = read_json(self.out / "record.example.json")
        validate_records(task, [row])
        self.assertEqual(read_json(self.out / "request.json")["state"], row["state"])

    def test_invalid_input_leaves_no_directory(self):
        cases = [
            (self.question, ["one=First"], None, "at least two"),
            (self.question, ["one=First", "one=Again"], None, "duplicate"),
            (self.question, ["one=First", "missing value"], None, "ACTION=DESCRIPTION"),
            (self.question, ["one=First", "two="], None, "ACTION=DESCRIPTION"),
            (self.question, self.actions, ["x="], "FIELD=TYPE"),
            (self.question, self.actions, ["x=image"], "type"),
            (self.question, self.actions, ["x=text", "x=number"], "duplicate"),
            (" ", self.actions, None, "question"),
        ]
        for question, actions, fields, error in cases:
            with self.subTest(actions=actions, fields=fields, question=question):
                with self.assertRaisesRegex(ValueError, error):
                    create_task(self.out, question, actions, fields)
                self.assertEqual(list(self.root.iterdir()), [])

    def test_existing_paths_are_preserved(self):
        self.out.mkdir()
        with self.assertRaises(FileExistsError):
            create_task(self.out, self.question, self.actions)
        self.assertEqual(list(self.out.iterdir()), [])
        sentinel = self.out / "notes.txt"
        sentinel.write_text("keep", encoding="utf-8")
        with self.assertRaises(FileExistsError):
            create_task(self.out, self.question, self.actions)
        self.assertEqual(sentinel.read_text(encoding="utf-8"), "keep")
        with self.assertRaises(FileExistsError):
            create_task(sentinel, self.question, self.actions)

    def test_write_failure_leaves_no_partial_task(self):
        with patch.object(Path, "write_text", side_effect=OSError("simulated disk error")):
            with self.assertRaisesRegex(OSError, "disk error"):
                create_task(self.out, self.question, self.actions)
        self.assertEqual(list(self.root.iterdir()), [])

    def test_two_creators_publish_only_one_complete_task(self):
        with ThreadPoolExecutor(max_workers=2) as pool:
            attempts = [pool.submit(create_task, self.out, question, self.actions)
                        for question in (self.question, "Which queue owns this?")]
        errors = [attempt.exception() for attempt in attempts]
        self.assertEqual(errors.count(None), 1)
        self.assertEqual(sum(isinstance(error, FileExistsError) for error in errors), 1)
        task = load_task(self.out / "task.json")
        validate_records(task, [read_json(self.out / "record.example.json")])
        self.assertEqual(list(self.root.iterdir()), [self.out])
