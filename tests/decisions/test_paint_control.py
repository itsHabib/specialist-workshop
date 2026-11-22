from concurrent.futures import ThreadPoolExecutor
from threading import Event

import pytest

from experiments.steering import paint_control as trial


def test_cycle_history_is_unchanged_by_same_cycle_retries_and_future_verdicts():
    cycles = {1: {11: [0]}, 2: {11: [0, 1], 2: [1]}, 3: {11: [0], 2: [0]}}
    changed = {1: {11: [0]}, 2: {11: [0, 0], 2: [0]}, 3: {11: [1], 2: [1]}}
    first = list(trial.snapshots(cycles))
    second = list(trial.snapshots(changed))
    assert first[1][1] == second[1][1]
    assert first[2][1][11]["last_fraction"] == 0.5
    assert first[2][1][11]["observed_cycles"] == 2
    assert second[2][1][11]["last_fraction"] == 0
    receipts = trial.grade(cycles, 1, ("numeric_id", "recent_failure"))
    assert [r["status"] for r in receipts] == ["all_pass", "repeated_names", "all_pass"]
    assert "ranks" not in receipts[1]
    assert receipts[1]["disagreeing_names"] == 1


def test_equal_tree_scores_use_numeric_id_even_when_features_differ():
    # The existing ranker sorts feature vectors for batch scoring. Its returned
    # tie order would put 11 first here. The frozen experiment uses numeric IDs.
    model = {"task": trial.TASK, "algorithm": trial.core.ALGORITHM,
             "training": {"categories": {}}, "tree": {"counts": {"prefer": 1, "defer": 2}}}
    low = dict.fromkeys(trial.FEATURES, 0)
    high = dict.fromkeys(trial.FEATURES, 5)
    assert trial.order({11: low, 2: high}, "tree", model) == [2, 11]


def test_completed_evaluation_keeps_receipt_and_stops_before_labels(tmp_path, monkeypatch):
    receipt = tmp_path / "holdout.json"
    receipt.write_bytes(b'{"first": true}\n')
    monkeypatch.setattr(trial, "_holdout_result", lambda *_: pytest.fail("attempted to rescore"))
    with pytest.raises(FileExistsError):
        trial.evaluate(tmp_path / "missing.csv", tmp_path)
    assert receipt.read_bytes() == b'{"first": true}\n'
    assert list(tmp_path.iterdir()) == [receipt]


def test_concurrent_evaluation_scores_once_and_failures_release_the_lock(tmp_path, monkeypatch):
    entered, release = Event(), Event()
    calls = []

    def score(*_):
        calls.append(True)
        entered.set()
        assert release.wait(timeout=5)
        return {"ok": True}

    def fail(*_):
        raise ValueError("invalid input")

    monkeypatch.setattr(trial, "_holdout_result", fail)
    with pytest.raises(ValueError, match="invalid input"):
        trial.evaluate(tmp_path / "data", tmp_path)
    assert not (tmp_path / "holdout.json").exists()

    monkeypatch.setattr(trial, "_holdout_result", score)
    with ThreadPoolExecutor(max_workers=1) as pool:
        pending = pool.submit(trial.evaluate, tmp_path / "data", tmp_path)
        try:
            assert entered.wait(timeout=5)
            with pytest.raises(OSError):
                trial.evaluate(tmp_path / "data", tmp_path)
        finally:
            release.set()
        pending.result(timeout=5)
    assert calls == [True]
    assert trial.contracts.read_json(tmp_path / "holdout.json") == {"ok": True}


def test_partial_freeze_leaves_destination_available_for_retry(tmp_path, monkeypatch):
    cycles = {1: {5: [1], 8: [0]}, trial.TRAIN_END + 1: {5: [1], 8: [0]}}
    monkeypatch.setattr(trial, "load_cycles", lambda *_: cycles)
    publish = trial.core.write_new

    def interrupted(path, body):
        if path.name == "freeze.json":
            raise OSError("disk full")
        publish(path, body)

    run = tmp_path / "run"
    monkeypatch.setattr(trial.core, "write_new", interrupted)
    with pytest.raises(OSError, match="disk full"):
        trial.freeze(tmp_path / "data", run)
    assert list(tmp_path.iterdir()) == []
    monkeypatch.setattr(trial.core, "write_new", publish)
    trial.freeze(tmp_path / "data", run)
    assert {p.name for p in run.iterdir()} == {"tree.json", "freeze.json"}
