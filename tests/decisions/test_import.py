import errno
import json
import os
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from decision_workshop.contracts import load_task, read_records
from decision_workshop.importer import import_records

from tests.decisions.test_contracts import TASK, row


EXAMPLE = Path(__file__).resolve().parents[2] / "examples" / "decisions" / "software"
ADAPTER = "examples.decisions.software.host_adapter:iter_records"


class ImportTests(unittest.TestCase):
    def test_importable_host_adapter_and_denominators(self):
        task = load_task(EXAMPLE / "task.json")
        with TemporaryDirectory() as folder:
            out = Path(folder) / "nested" / "records.jsonl"
            summary = import_records(task, ADAPTER, EXAMPLE / "host-events.jsonl", out)
            self.assertEqual(summary, {"rows": 3, "groups_per_split": {"train": 1, "dev": 1, "test": 0},
                                       "candidate_sets_recorded": 2, "target_missing_from_candidates": 1})
            rows = read_records(out, task)
            self.assertEqual([r["candidates"] for r in rows], [["inspect_runner", "rerun_test"], None, []])
            self.assertEqual(rows[0]["state"]["duration_seconds"], 18)
            self.assertEqual(rows[-1]["target"], "triage_existing_failure")

    def test_private_adapter_module_can_live_outside_checkout(self):
        with TemporaryDirectory() as folder:
            source = Path(folder) / "event.json"
            source.write_text(json.dumps(row()), encoding="utf-8")
            module = Path(folder) / "synthetic_private_host.py"
            module.write_text("import json\ndef decisions(source):\n"
                              "    yield json.loads(source.read_text(encoding='utf-8'))\n", encoding="utf-8")
            out = Path(folder) / "records.jsonl"
            task_file = Path(folder) / "task.json"
            task_file.write_text(json.dumps(TASK), encoding="utf-8")
            env = {**os.environ, "PYTHONPATH": str(Path(__file__).resolve().parents[2])}
            run = subprocess.run(
                [sys.executable, "-S", "-m", "decision_workshop", "import", "--task", str(task_file),
                 "--adapter", "synthetic_private_host:decisions", "--source", str(source), "--out", str(out)],
                cwd=folder, env=env, text=True, capture_output=True, check=True, timeout=15,
            )
            self.assertEqual(json.loads(run.stdout)["rows"], 1)
            self.assertEqual(read_records(out, TASK), [row()])

    def test_invalid_adapter_rows_leave_no_partial_file(self):
        bad_rows = [
            (row(id="r2", split="dev"), "group.*splits"),
            (row(id="r2", candidates="proceed"), "candidates must be a list"),
            (row(id="r2", candidates=["other"]), "unknown candidate"),
            (row(id="r2", target="other"), "unknown target"),
            (row(id="r2", state={**row()["state"], "amount": True}), "finite number"),
            ({k: v for k, v in row(id="r2").items() if k != "split"}, "missing fields: split"),
            ({k: v for k, v in row(id="r2").items() if k != "group"}, "missing fields: group"),
            (row(), "duplicate record id"),
        ]
        with TemporaryDirectory() as folder:
            out = Path(folder) / "records.jsonl"
            for bad, reason in bad_rows:
                with self.subTest(reason=reason), self.assertRaisesRegex(ValueError, f"record 2: .*{reason}"):
                    import_records(TASK, lambda source: iter([row(), bad]), None, out)
                self.assertEqual(list(Path(folder).iterdir()), [])
            with self.assertRaisesRegex(ValueError, "at least one record"):
                import_records(TASK, lambda source: [], None, out)
            self.assertFalse(out.exists())

    def test_adapter_load_and_late_iteration_errors(self):
        def broken(source):
            yield row()
            raise ModuleNotFoundError("host-only library")

        with TemporaryDirectory() as folder:
            out = Path(folder) / "records.jsonl"
            for adapter, reason in [("not_a_spec", "module:function"), (ADAPTER + "_missing", "missing"),
                                    ("missing_host_lib:rows", "missing_host_lib"), (broken, "host-only library")]:
                with self.subTest(adapter=adapter), self.assertRaisesRegex(ValueError, f"adapter .*{reason}"):
                    import_records(TASK, adapter, None, out)
                self.assertEqual(list(Path(folder).iterdir()), [])

    def test_preserve_existing_output_including_empty_scaffold(self):
        with TemporaryDirectory() as folder:
            out = Path(folder) / "records.jsonl"
            for content in ["", "historical data\n"]:
                out.write_text(content, encoding="utf-8")
                with self.assertRaises(FileExistsError):
                    import_records(TASK, lambda source: self.fail("adapter ran"), None, out)
                self.assertEqual(out.read_text(encoding="utf-8"), content)

    def test_concurrent_writer_wins_without_clobber(self):
        with TemporaryDirectory() as folder:
            out = Path(folder) / "records.jsonl"
            def racing(source):
                yield row()
                other = import_records(TASK, lambda _: [row(id="winner")], None, out)
                self.assertEqual(other["rows"], 1)

            with self.assertRaises(FileExistsError):
                import_records(TASK, racing, None, out)
            self.assertEqual(read_records(out, TASK), [row(id="winner")])
            self.assertEqual(list(Path(folder).iterdir()), [out])

    def test_portable_publication_without_hard_links(self):
        with TemporaryDirectory() as folder, patch("decision_workshop.core.os.link",
                                                  side_effect=OSError(errno.ENOTSUP, "unsupported")):
            out = Path(folder) / "records.jsonl"
            import_records(TASK, lambda source: [row()], None, out)
            self.assertEqual(read_records(out, TASK), [row()])
            self.assertEqual(list(Path(folder).iterdir()), [out])
