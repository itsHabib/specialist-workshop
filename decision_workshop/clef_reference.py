"""Compare a completed Clef MLX inference receipt with two local maker checkpoints."""

import argparse
import json
import math
from pathlib import Path
import sys

from .checkpoint import FORMAT as BERT_ALGORITHM
from .clef_mlx_replay import FILE_HASHES, MODEL_ID, REVISION, UPSTREAM_ID, UPSTREAM_REVISION, file_hash, request_for
from .contracts import load_task, read_json, read_records
from .core import (ALGORITHM, decision_min_score, digest, load_model, predict,
                   training_rows, training_summary, write_new)


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _object(value, context):
    _require(isinstance(value, dict), f"{context} must be an object")
    return value


def _receipt_scores(receipt, task, held_out, hashes):
    receipt = _object(receipt, "receipt")
    _require(receipt.get("format") == "decision-workshop/clef-replay-v1", "unknown Clef receipt format")
    _require(receipt.get("sha256") == digest({k: v for k, v in receipt.items() if k != "sha256"}),
             "Clef receipt digest mismatch")
    _require(receipt.get("status") == "complete" and receipt.get("errors") == [], "incomplete or failed reference run")
    model = _object(receipt.get("model"), "reference model")
    for key, expected in (("id", MODEL_ID), ("revision", REVISION),
                          ("upstream_id", UPSTREAM_ID), ("upstream_revision", UPSTREAM_REVISION)):
        _require(model.get(key) == expected, f"unexpected reference model {key}")
    files = _object(model.get("files"), "reference model files")
    for filename, checksum in FILE_HASHES.items():
        metadata = _object(files.get(filename), filename)
        _require(metadata.get("sha256") == checksum, f"reference model file hash mismatch: {filename}")

    inputs = _object(receipt.get("inputs"), "receipt inputs")
    shipped_script = file_hash(Path(__file__).with_name("clef_mlx_replay.py"))
    recorded_script = inputs.get("replay_script_sha256")
    _require(recorded_script == shipped_script,
             f"replay_script_sha256 mismatch for shipped clef_mlx_replay.py "
             f"(expected {shipped_script}, recorded {recorded_script!r}); regenerate the receipt with this kit")
    environment = _object(receipt.get("environment"), "receipt environment")
    packages = _object(environment.get("packages"), "receipt environment.packages")
    for fields, source, keys in (
        ("environment", environment, ("os", "machine", "python")),
        ("environment.packages", packages, ("mlx", "mlx-lm", "mlx-vlm", "transformers", "huggingface_hub")),
    ):
        for key in keys:
            value = source.get(key)
            _require(isinstance(value, str) and value.strip(), f"receipt {fields}.{key} must be a nonblank string")
    for key, checksum in hashes.items():
        _require(inputs.get(key) == checksum, f"reference input hash mismatch: {key}")
    _require(inputs.get("port_loader_sha256") == FILE_HASHES["clef_mlx.py"], "reference loader hash mismatch")
    labels = task["labels"]
    expected_counts = {split: sum(row["split"] == split for row in held_out) for split in ("dev", "test")}
    _require(inputs.get("task") == task["name"] and inputs.get("choice_field") == "action"
             and inputs.get("trained_on_evaluation_rows") is False, "unexpected reference task or inference policy")
    _require(inputs.get("split_counts") == expected_counts, "reference split count mismatch")
    _require(inputs.get("options") == labels, "receipt action order differs from task")
    _require(isinstance(receipt.get("cases"), list), "receipt cases must be a list")
    expected = {row["id"]: row for row in held_out}
    scores = {}
    for case in receipt["cases"]:
        _object(case, "receipt case")
        row_id = case.get("id")
        _require(isinstance(row_id, str) and row_id in expected, f"unexpected reference row {row_id!r}")
        _require(row_id not in scores, f"duplicate reference row {row_id!r}")
        row = expected[row_id]
        _require(case.get("split") == row["split"], f"{row_id}: reference split mismatch")
        _require("error" in case and case["error"] is None, f"{row_id}: failed or incomplete reference inference")
        elapsed = case.get("elapsed_seconds")
        _require(type(elapsed) in (int, float) and math.isfinite(elapsed) and elapsed >= 0,
                 f"{row_id}: invalid reference elapsed time")
        request = request_for(task, row["state"])
        _require(digest(case.get("input")) == digest(request), f"{row_id}: reference model input mismatch")
        _require(list(case["input"]["questions"]["action"]["criteria"]) == labels,
                 f"{row_id}: reference input action order mismatch")
        probs = _object(case.get("probabilities"), f"{row_id} probabilities")
        _require(set(probs) == set(labels) and all(type(p) in (int, float) and 0 <= p <= 1 for p in probs.values())
                 and math.isclose(sum(probs.values()), 1.0, abs_tol=1e-5),
                 f"{row_id}: invalid probability distribution")
        _require(digest(case.get("raw_output")) == digest({"action": probs}), f"{row_id}: raw output mismatch")
        scores[row_id] = probs
    _require(scores.keys() == expected.keys(), f"missing reference rows: {sorted(expected.keys() - scores.keys())}")
    return scores


def _load_checkpoints(maker_dir, task, rows, training):
    maker_report = _object(read_json(maker_dir / "report.json"), "maker report")
    dev = sorted((row for row in rows if row["split"] == "dev"), key=lambda row: row["id"])
    dev_digest = digest(dev)
    _require(maker_report.get("task") == task["name"] and maker_report.get("version") == task["version"]
             and maker_report.get("question") == task.get("question") and maker_report.get("split") == "dev"
             and maker_report.get("training_rows") == training["rows"] and maker_report.get("development_rows") == len(dev),
             "maker report task or split mismatch")
    models = {}
    for name, filename, algorithm in (("bert", "model", BERT_ALGORITHM), ("tree", "tree.json", ALGORITHM)):
        model = load_model(maker_dir / filename)
        recorded = _object(maker_report.get(name), f"maker report {name}")
        _require(model["algorithm"] == algorithm and model["task"] == task and model["training"] == training,
                 f"{name} checkpoint task or training records mismatch")
        _require(recorded.get("checkpoint") == filename and recorded.get("model_sha256") == model["sha256"]
                 and recorded.get("evaluation_records_sha256") == dev_digest,
                 f"{name} checkpoint or development records disagree with maker report")
        models[name] = model
    return models


def _counts(cases, name, field):
    correct = wrong = review = 0
    for case in cases:
        action = case["models"][name][field]
        if action is None:
            review += 1
        elif action == case["target"]:
            correct += 1
        else:
            wrong += 1
    return {"correct": correct, "wrong": wrong, "review": review}


def compare(task_dir, receipt_path, maker_dir, *, min_score=None):
    """Verify the bindings and run all model preferences through the same review policy."""
    task_dir, maker_dir = Path(task_dir), Path(maker_dir)
    task_path, rows_path = task_dir / "task.json", task_dir / "records.jsonl"
    task, receipt = load_task(task_path), read_json(receipt_path)
    rows = read_records(rows_path, task)
    held_out = sorted((row for row in rows if row["split"] != "train"), key=lambda row: row["id"])
    _require({row["split"] for row in held_out} == {"dev", "test"}, "comparison requires dev and test groups")
    hashes = {"task_sha256": file_hash(task_path), "records_sha256": file_hash(rows_path)}
    scores = _receipt_scores(receipt, task, held_out, hashes)
    training = training_summary(task, training_rows(rows))
    checkpoints = _load_checkpoints(maker_dir, task, rows, training)
    cutoffs = {decision_min_score(model, min_score) for model in checkpoints.values()}
    _require(len(cutoffs) == 1, "comparison needs a common min_score. Pass one explicitly")
    min_score = cutoffs.pop()
    reference = {"algorithm": "clef-mlx-reference", "task": task, "training": training}
    splits = {}
    for split in ("dev", "test"):
        selected = [row for row in held_out if row["split"] == split]
        cases = []
        for row in selected:
            # Receipts are bound by record ID, even when two rows have identical states.
            probabilities = scores[row["id"]]
            reference_row = {**reference, "_score": lambda state: probabilities}
            models = {}
            for name, model in (("clef", reference_row), *checkpoints.items()):
                decision = predict(model, row["state"], row["candidates"], min_score=min_score)
                models[name] = {k: decision[k] for k in ("preferred", "action", "status", "reasons", "scores")}
            cases.append({k: row[k] for k in ("id", "group", "state", "target", "candidates")} | {"models": models})
        known = [case for case in cases if case["candidates"] is not None]
        available = sum(case["target"] in case["candidates"] for case in known)
        splits[split] = {
            "rows": len(cases), "groups": len({row["group"] for row in selected}),
            "evaluation_records_sha256": digest(selected),
            "reference_availability": {"present": available, "missing": len(known) - available,
                                       "unknown": len(cases) - len(known)},
            "models": {
                name: {"final": _counts(cases, name, "action"), "raw": _counts(cases, name, "preferred"),
                       "known_candidates": _counts(known, name, "action")} for name in ("clef", "bert", "tree")
            },
            "cases": cases,
        }
    return {
        "format": "decision-workshop/clef-comparison-v1", "task": task["name"], "version": task["version"],
        "inputs": hashes, "reference_receipt_sha256": receipt["sha256"], "min_score": min_score,
        "checkpoints": {name: model["sha256"] for name, model in checkpoints.items()},
        "reference_model": {"id": MODEL_ID, "revision": REVISION}, "splits": splits,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", type=Path, required=True, help="task directory")
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--maker", type=Path, required=True, help="completed maker run directory")
    parser.add_argument("--out", type=Path, required=True, help="new comparison JSON file")
    parser.add_argument("--min-score", type=float, help="override the shared checkpoint cutoff")
    args = parser.parse_args()
    try:
        report = compare(args.task, args.receipt, args.maker, min_score=args.min_score)
        write_new(args.out, report)
        print(json.dumps({split: data["models"] for split, data in report["splits"].items()}, indent=2))
    except (ValueError, OSError, ModuleNotFoundError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
