import subprocess
import sys
import tempfile
from pathlib import Path

import pytest

from experiments.boost_workloop.taskpack import taskpack


def execute(files, command, stdin=""):
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        for name, content in files.items():
            path = root / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content, encoding="utf-8")
        argv = [sys.executable, "rectarea.py"]
        if "--json" in command:
            argv.append("--json")
        if "--input" in command:
            argv.extend(["--input", "withheld.txt"])
        run = subprocess.run(argv, cwd=root, input=stdin, text=True,
                             capture_output=True, timeout=5)
        return {"returncode": run.returncode, "stdout": run.stdout,
                "stderr": run.stderr, "error": None, "seconds": 0.0,
                "cleanup_confirmed": True}


@pytest.mark.parametrize("variant", [0, 1])
def test_manifest_is_worker_visible_and_self_contained(variant):
    manifest = taskpack.build_task(variant)
    assert set(manifest) == {"name", "goal", "files", "test_command", "entrypoint"}
    assert manifest["test_command"] == "python -m unittest discover -s tests -v"
    assert "final_requests" not in "".join(manifest["files"].values())
    assert ("test_comments_are_ignored" in manifest["files"]["tests/test_rectarea.py"]) == (variant == 1)


@pytest.mark.parametrize("variant", [0, 1])
def test_broken_starter_fails_and_reference_passes(variant):
    broken = taskpack.verify(taskpack.build_task(variant)["files"], execute, 20260920, variant)
    assert not broken["passed"]
    good_files = taskpack.reference_files(variant)
    assert taskpack.verify(good_files, execute, 20260920, variant)["passed"]

    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        for name, content in good_files.items():
            path = root / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content, encoding="utf-8")
        run = subprocess.run(manifest_command(taskpack.build_task(variant)), cwd=root,
                             text=True, capture_output=True)
        assert run.returncode == 0, run.stdout + run.stderr


def manifest_command(manifest):
    return [sys.executable, "-m", "unittest", "discover", "-s", "tests", "-v"]


def test_final_generation_is_seeded_and_oracle_covers_exact_cases():
    first = taskpack.final_requests(0, 7)
    assert first == taskpack.final_requests(0, 7)
    assert first != taskpack.final_requests(0, 8)
    assert first[0]["expected"] == "1"
    assert any("/" in request["expected"] for request in first)


def test_development_probe_is_useful_but_separate_from_final_seed():
    files = taskpack.build_task(0)["files"]
    probe = taskpack.development_probe(files, execute, 19)
    final = taskpack.verify(files, execute, 19)
    assert not probe["passed"] and not final["passed"]
    assert probe["checks"] == 5


def test_explicit_variant_cannot_be_downgraded_by_readme_tampering():
    files = taskpack.reference_files(1)
    files["README.md"] = taskpack.build_task(0)["files"]["README.md"]
    assert taskpack.verify(files, execute, 31, variant=1)["passed"]
    assert taskpack.development_probe(files, execute, 31, variant=1)["passed"]
