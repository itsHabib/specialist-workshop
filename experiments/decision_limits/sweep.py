"""Measure two local classifiers on matched, stratified numeric-band routing tasks.

Run from the project root: python -m experiments.decision_limits.sweep --base .state/base
"""

import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import random
from statistics import quantiles
from tempfile import TemporaryDirectory
import time

from decision_workshop.checkpoint import train_bert
from decision_workshop.contracts import validate_records
from decision_workshop.core import digest, load_model, predict, train, write_new


ACTION_COUNTS = (2, 4, 8, 16)
BUDGETS = (32, 128)
SEEDS = (17, 73)
THRESHOLDS = tuple(i / 20 for i in range(21))
MIN_COVERAGE = 0.80
MIN_ACCURACY = 0.95
BERT_RECIPE = {"epochs": 12, "batch_size": 8, "learning_rate": 0.0003}


def make_task(k):
    if k not in ACTION_COUNTS:
        raise ValueError(f"supported action counts are {ACTION_COUNTS}")
    labels = [f"route_{chr(97 + i)}" for i in range(k)]
    return {"name": f"numeric_bands_{k}", "version": "1", "features": {"value": "number"},
            "labels": labels, "question": "Which route should receive this value?",
            "descriptions": {label: f"Send the value to {label.replace('_', ' ')}." for label in labels}}


def make_rows(k, budget, seed):
    """Nested labeled budgets and the same unique values across all action counts."""
    labels = make_task(k)["labels"]
    if budget not in BUDGETS:
        raise ValueError(f"supported training budgets are {BUDGETS}")
    bands = []
    for band in range(16):
        values = list(range(band * 64, (band + 1) * 64))
        random.Random(f"numeric-bands-v1:{seed}:{band}").shuffle(values)
        bands.append(values)
    rows = []
    for split, start, groups in (("train", 0, budget // 16), ("dev", 8, 8), ("test", 16, 8)):
        for g in range(groups):
            group = f"s{seed}-{split}-{g}"
            for band in range(16):
                value = bands[band][start + g]
                rows.append({"id": f"{group}-{band}", "group": group, "split": split,
                             "state": {"value": value}, "target": labels[value * k // 1024],
                             "candidates": list(labels)})
    return rows


def score_cases(model, rows):
    cases = []
    for row in rows:
        # Only pre-decision state and the action vocabulary reach the classifier.
        output = predict(model, row["state"], row["candidates"])
        cases.append({"id": row["id"], "group": row["group"], "target": row["target"],
                      "state": row["state"], "scores": output["scores"],
                      "preferred": output["preferred"], "confidence": max(output["scores"].values())})
    return cases


def summarize_cases(cases, threshold):
    suggested = [c for c in cases if c["preferred"] is not None and c["confidence"] >= threshold]
    correct = sum(c["preferred"] == c["target"] for c in suggested)
    n, issued = len(cases), len(suggested)
    coverage = issued / n
    accuracy = correct / issued if issued else None
    return {"threshold": threshold, "total": n, "suggested": issued, "correct": correct,
            "wrong": issued - correct, "review": n - issued, "coverage": coverage,
            "accuracy": accuracy, "meets_target": coverage >= MIN_COVERAGE and
            accuracy is not None and accuracy >= MIN_ACCURACY}


def threshold_curve(cases):
    return [summarize_cases(cases, threshold) for threshold in THRESHOLDS]


def select_policy(curve):
    eligible = (p for p in curve if p["meets_target"])
    return max(eligible, default=None, key=lambda p: (p["coverage"], p["accuracy"], -p["threshold"]))


def _group_interval(cases, threshold):
    """Percentile bootstrap over whole sixteen-case groups, for one frozen policy."""
    grouped = {}
    for case in cases:
        grouped.setdefault(case["group"], []).append(case)
    totals = [summarize_cases(rows, threshold) for rows in grouped.values()]
    rng = random.Random(46091)
    coverage, accuracy = [], []
    for _ in range(1000):
        sample = rng.choices(totals, k=len(totals))
        issued = sum(x["suggested"] for x in sample)
        coverage.append(issued / sum(x["total"] for x in sample))
        if issued:
            accuracy.append(sum(x["correct"] for x in sample) / issued)

    def bounds(values):
        if len(values) < 2:
            return None
        cuts = quantiles(values, n=40, method="inclusive")
        return [cuts[0], cuts[-1]]

    return {"groups": len(grouped), "resamples": 1000, "coverage_95": bounds(coverage),
            "accuracy_95": bounds(accuracy), "accuracy_defined_resamples": len(accuracy)}


def _run(folder, base, backend, seed, k, budget):
    task = make_task(k)
    rows = make_rows(k, budget, seed)
    validate_records(task, rows)
    practice = [r for r in rows if r["split"] == "train"]
    support = Counter(row["target"] for row in practice)
    if len(support) != k or set(support.values()) != {budget // k}:
        raise ValueError("unbalanced practice data")
    started = time.perf_counter()
    write_new(folder / "task.json", task)
    (folder / "records.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf8")
    if backend == "bert":
        path = folder / "checkpoint"
        train_bert(task, practice, base, path, seed=seed, **BERT_RECIPE)
    else:
        path = folder / "tree.json"
        write_new(path, train(task, practice, max_depth=5, min_leaf=2))
    train_seconds = time.perf_counter() - started
    model = load_model(path)
    dev = score_cases(model, [r for r in rows if r["split"] == "dev"])
    curve = threshold_curve(dev)
    selected = select_policy(curve)
    write_new(folder / "development.json", {"cases": dev, "curve": curve, "selected": selected})
    # This policy is now frozen. Test outcomes cannot select or rescue a cutoff.
    test = score_cases(model, [r for r in rows if r["split"] == "test"])
    test_curve = threshold_curve(test)
    write_new(folder / "test.json", {"cases": test, "descriptive_curve": test_curve})
    selected_test = summarize_cases(test, selected["threshold"]) if selected else None
    result = {"id": f"{backend}-k{k}-n{budget}-s{seed}", "backend": backend, "seed": seed,
              "actions": k, "budget": budget, "examples_per_action": budget // k,
              "rows_sha256": digest(rows), "checkpoint": path.name, "checkpoint_sha256": model["sha256"],
              "dev_policy_status": "eligible" if selected else "no_eligible_dev_policy",
              "dev_selected": selected, "test_at_selected": selected_test,
              "dev_at_zero": curve[0], "test_at_zero": test_curve[0],
              "test_at_zero_interval": _group_interval(test, 0.0),
              "test_at_selected_interval": _group_interval(test, selected["threshold"]) if selected else None,
              "train_seconds": train_seconds, "elapsed_seconds": time.perf_counter() - started}
    write_new(folder / "result.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, required=True, help="verified local Google BERT-Tiny base directory")
    parser.add_argument("--out", type=Path, default=Path(".state/limits-sweep"))
    args = parser.parse_args()
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    import torch
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    args.out.mkdir(parents=True, exist_ok=False)
    protocol = {
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "actions": ACTION_COUNTS, "budgets": BUDGETS, "seeds": SEEDS,
        "tree": {"max_depth": 5, "min_leaf": 2}, "bert": BERT_RECIPE, "device": "cpu",
        "design": "For each seed independently permute each of 16 integer strata of size 64. "
                  "Every group has one value per stratum. Reserve 8 groups each for training, "
                  "development and test. The 32-row budget uses the first 2 training groups. "
                  "These finite-population samples have no repeated states within seed. Reuse "
                  "the values and IDs at every K. All K actions are always offered. Target index "
                  "is value*K//1024. Option descriptions identify each route without giving boundaries.",
        "thresholds": THRESHOLDS, "minimum_coverage": MIN_COVERAGE, "minimum_accuracy": MIN_ACCURACY,
        "policy": "Select an eligible dev threshold by maximum coverage, then accuracy, then smallest "
                  "threshold. With no eligible threshold the test-at-selected result is null. Report "
                  "threshold-zero results and descriptive test curves regardless. Ties defer.",
        "uncertainty": "95% percentile intervals from 1000 bootstrap draws of 8 whole groups with replacement; "
                       "conditional on one seed, fitted checkpoint, and the observed synthetic population.",
    }
    write_new(args.out / "protocol.json", protocol)
    start = time.perf_counter()
    results = []
    for k in ACTION_COUNTS:
        for budget in BUDGETS:
            for seed in SEEDS:
                for backend in ("tree", "bert"):
                    with TemporaryDirectory(dir=args.out, prefix=".fitting-") as temp:
                        folder = Path(temp) / "run"
                        folder.mkdir()
                        result = _run(folder, args.base, backend, seed, k, budget)
                        folder.rename(args.out / result["id"])
                    results.append(result)
                    print(result["id"], result["dev_policy_status"], f"{result['elapsed_seconds']:.1f}s", flush=True)
    write_new(args.out / "summary.json", {"protocol": "protocol.json", "protocol_sha256": digest(protocol),
              "elapsed_seconds": time.perf_counter() - start, "runs": results})


if __name__ == "__main__":
    main()
