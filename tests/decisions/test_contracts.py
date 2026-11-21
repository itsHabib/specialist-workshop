import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from decision_workshop.contracts import (
    load_task,
    read_json,
    read_records,
    validate_candidates,
    validate_records,
    validate_state,
    validate_task,
)


TASK = {
    "name": "next_action",
    "version": "1",
    "features": {"amount": "number", "category": "category", "active": "boolean"},
    "labels": ["proceed", "review"],
}
STATE = {"amount": 3.5, "category": "alpha", "active": True}


def row(**overrides):
    return {
        "id": "record-1", "group": "group-A", "split": "train", "state": STATE,
        "target": "review", "candidates": ["review", "proceed"], **overrides,
    }


class ContractTests(unittest.TestCase):
    def test_task_schema(self):
        validate_task(TASK)
        for invalid in [
            None, {**TASK, "version": 1}, {**TASK, "name": " "}, {**TASK, "notes": "x"},
            {k: v for k, v in TASK.items() if k != "labels"},
            {**TASK, "labels": []}, {**TASK, "labels": ["a", "a"]},
            {**TASK, "features": {"input": "map"}},
        ]:
            with self.subTest(task=invalid), self.assertRaises(ValueError):
                validate_task(invalid)

    def test_state_scalar_types(self):
        validate_state(TASK, STATE)
        validate_state(TASK, {**STATE, "amount": 1})
        for changes in [
            {"amount": True}, {"amount": float("nan")}, {"amount": float("inf")},
            {"amount": "3.2"}, {"amount": None}, {"amount": 10**400}, {"active": 1},
            {"category": {"detail": "value"}}, {"category": ["alpha"]},
            {"extra": "field"},
        ]:
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                validate_state(TASK, {**STATE, **changes})
        with self.assertRaisesRegex(ValueError, "missing.*amount"):
            validate_state(TASK, {"category": "alpha", "active": False})

    def test_optional_question_and_label_descriptions(self):
        question = {"question": "What should happen next?"}
        descriptions = {"proceed": "Continue with the input.", "review": "Review the input."}
        for metadata in ({}, question, {"descriptions": descriptions},
                         {**question, "descriptions": descriptions}):
            with self.subTest(metadata=metadata):
                validate_task({**TASK, **metadata})
        for bad_question in (None, 1, " \t "):
            with self.subTest(bad_question=bad_question), self.assertRaisesRegex(ValueError, "task.question"):
                validate_task({**TASK, "question": bad_question})
        for bad_descriptions in (None, [], {}, {"proceed": "Continue with the input."},
                                 {**descriptions, "extra": "Unexpected action."},
                                 {**descriptions, "review": "\n"}, {**descriptions, "proceed": 5}):
            with self.subTest(bad_descriptions=bad_descriptions), self.assertRaisesRegex(ValueError, "task.descriptions"):
                validate_task({**TASK, "descriptions": bad_descriptions})

    def test_text_accepts_unseen_notes_and_rejects_empty_or_structured_values(self):
        task = {**TASK, "features": {"note": "text"}}
        practice = row(state={"note": "The first case was reviewed."})
        new_note = "New information arrived.\nReview the second case."
        held_out = row(id="record-2", group="group-B", split="test", state={"note": new_note})
        validate_records(task, [practice, held_out])
        validate_state(task, {"note": new_note})
        for value in ("", "  ", None, 3, ["note"], {"note": "text"}):
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, "state.note"):
                validate_state(task, {"note": value})

    def test_preserve_missing_candidates_and_baseline_errors(self):
        records = [
            row(candidates=[], baselines={"legacy": None}),
            row(id="record-2", candidates=["proceed"], baselines={"legacy": "review"}),
        ]
        validate_records(TASK, records)
        self.assertEqual(records[1]["target"], "review")
        self.assertEqual(records[1]["candidates"], ["proceed"])

    def test_candidates(self):
        for candidates in [None, "proceed", [True], ["proceed", "proceed"], ["unknown"]]:
            with self.subTest(candidates=candidates), self.assertRaises(ValueError):
                validate_candidates(TASK, candidates)

    def test_unknown_historical_candidates_are_preserved(self):
        with TemporaryDirectory() as tmp:
            data_file = Path(tmp) / "final-only.jsonl"
            final_only = row(candidates=None)
            data_file.write_text(json.dumps(final_only), encoding="utf-8")
            self.assertEqual(read_records(data_file, TASK), [final_only])
            del final_only["candidates"]
            with self.assertRaisesRegex(ValueError, "missing.*candidates"):
                validate_records(TASK, [final_only])

    def test_read_json_is_strict_and_paths_are_reported_once(self):
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "input.json"
            path.write_text('["café"]', encoding="utf-8-sig")
            self.assertEqual(read_json(path), ["café"])
            path.write_text('{"a": {"b": 1, "b": 2}}', encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "duplicate JSON field") as ctx:
                load_task(path)
            self.assertEqual(str(ctx.exception).count(str(path)), 1)
            path.write_text('{"bad": Infinity}', encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "finite number"):
                read_json(path)

    def test_incompatible_records(self):
        for changes in [
            {"split": "verify"}, {"split": ["train"]}, {"target": "unknown"},
            {"group": ""}, {"id": None}, {"outcome": "good"},
            {"baselines": []}, {"baselines": {"old": "unknown"}},
            {"baselines": {"old": False}}, {"baselines": {"": None}},
        ]:
            with self.subTest(changes=changes), self.assertRaisesRegex(ValueError, "record 1"):
                validate_records(TASK, [row(**changes)])
        with self.assertRaisesRegex(ValueError, "missing.*group"):
            validate_records(TASK, [{k: v for k, v in row().items() if k != "group"}])

    def test_group_leakage_and_duplicate_ids(self):
        with self.assertRaisesRegex(ValueError, "record 2.*group.*train.*test"):
            validate_records(TASK, [row(), row(id="record-2", split="test")])
        with self.assertRaisesRegex(ValueError, "record 2.*duplicate record id"):
            validate_records(TASK, [row(), row(group="group-B")])
        validate_records(TASK, [row(), row(id="record-2", group="group-B", split="dev")])

    def test_evaluation_only_files_are_valid_but_empty_files_are_not(self):
        validate_records(TASK, [row(split="test")])
        with self.assertRaisesRegex(ValueError, "at least one record"):
            validate_records(TASK, [])

    def test_json_files_and_physical_line_numbers(self):
        with TemporaryDirectory() as tmp:
            task_file = Path(tmp) / "training task.json"
            data_file = Path(tmp) / "data.jsonl"
            task_file.write_text(json.dumps(TASK), encoding="utf-8-sig")
            self.assertEqual(load_task(task_file), TASK)
            good = json.dumps(row(split="dev"))
            data_file.write_text(good + "\n", encoding="utf-8-sig")
            self.assertEqual(read_records(data_file, TASK), [row(split="dev")])
            for second, message in [
                ("{bad", "line 3"),
                (good, "line 3.*duplicate record id"),
                (good.replace('"amount"', '"amount": 4, "amount"'),
                 "line 3.*duplicate JSON field"),
                (good.replace("3.5", "NaN"), "line 3.*finite number"),
            ]:
                with self.subTest(second=second):
                    data_file.write_text(good + "\n\n" + second + "\n", encoding="utf-8")
                    with self.assertRaisesRegex(ValueError, "data.jsonl: " + message):
                        read_records(data_file, TASK)
            task_file.write_text("{}", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "training task.json: task missing"):
                load_task(task_file)
