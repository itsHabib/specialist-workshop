"""Change only the tree depth and validate on a third, previously unused cohort."""

import argparse
from pathlib import Path
import random

from decision_workshop.contracts import validate_records
from decision_workshop.core import train, write_new
from .sweep import make_rows, make_task, score_cases, select_policy, summarize_cases, threshold_curve


DEPTHS = (5, 8, 12, 16)


def fresh_rows(k, seed):
    task = make_task(k)
    feature, = task["features"]
    used = {r["state"][feature] for r in make_rows(k, 128, seed)}
    rng = random.Random(8111 + seed)
    rows = []
    for band in range(16):
        population = [v for v in range(64 * band, 64 * (band + 1)) if v not in used]
        for group, value in enumerate(rng.sample(population, 16)):
            rows.append({"id": f"fresh-{group}-{band}", "group": f"fresh-{group}",
                         "state": {feature: value}, "split": "test",
                         "target": task["labels"][value * k // 1024], "candidates": task["labels"]})
    validate_records(task, rows)
    return rows


def compare(k, budget, seed, out):
    task = make_task(k)
    rows = make_rows(k, budget, seed)
    dev = [r for r in rows if r["split"] == "dev"]
    fresh = fresh_rows(k, seed)
    validate_records(task, rows + fresh)
    folder = out / f"k{k}-n{budget}-s{seed}"
    candidates = []
    checkpoints = {}
    for depth in DEPTHS:
        model = train(task, rows, max_depth=depth)
        checkpoints[depth] = model
        cases = score_cases(model, dev)
        curve = threshold_curve(cases)
        candidates.append({"depth": depth, "dev_policy": select_policy(curve),
                           "dev_raw": summarize_cases(cases, 0), "checkpoint": model["sha256"]})
        write_new(folder / f"depth-{depth}.json", {"checkpoint": model, "dev": cases, "curve": curve})
    eligible = [c for c in candidates if c["dev_policy"] is not None]
    winner = max(eligible, default=None, key=lambda c: (
        c["dev_policy"]["coverage"], c["dev_policy"]["accuracy"], -c["depth"]))
    default_cases = score_cases(checkpoints[5], fresh)
    result = {"actions": k, "budget": budget, "seed": seed, "candidates": candidates,
              "default_fresh_raw": summarize_cases(default_cases, 0), "selected": winner}
    if winner:
        selected_cases = score_cases(checkpoints[winner["depth"]], fresh)
        result["selected_fresh"] = summarize_cases(selected_cases, winner["dev_policy"]["threshold"])
    else:
        selected_cases = []
        result["selected_fresh"] = None
    write_new(folder / "fresh.json", {"default": default_cases, "selected": selected_cases})
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    out = parser.parse_args().out
    out.mkdir(parents=True, exist_ok=False)
    results = [compare(k, budget, seed, out) for k in (2, 4, 8, 16)
               for budget in (32, 128) for seed in (17, 73)]
    write_new(out / "summary.json", results)


if __name__ == "__main__":
    main()
