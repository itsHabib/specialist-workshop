"""Compare two exact-Gini tie rules on reserved numeric bands at fixed depth five."""

import argparse
from pathlib import Path
import random

from decision_workshop.contracts import validate_records
from decision_workshop.core import digest, train, write_new
from .depth import fresh_rows
from .sweep import (ACTION_COUNTS, BUDGETS, SEEDS, make_rows, make_task, score_cases,
                    select_policy, summarize_cases, threshold_curve)


RESERVE_SEED = 62171


def reserved_rows(k, seed):
    labels = make_task(k)["labels"]
    prior = make_rows(k, max(BUDGETS), seed) + fresh_rows(k, seed)
    used = {r["state"]["value"] for r in prior}
    rng = random.Random(RESERVE_SEED + seed)
    rows = []
    for band in range(16):
        remaining = [v for v in range(64 * band, 64 * (band + 1)) if v not in used]
        if len(remaining) != 24:
            raise ValueError("the prescribed cohorts must leave 24 unused values in every stratum")
        for group, value in enumerate(rng.sample(remaining, 16)):
            group_id = f"reserve-s{seed}-g{group}"
            rows.append({"id": f"{group_id}-{band}", "group": group_id, "split": "test",
                         "state": {"value": value}, "target": labels[value * k // 1024],
                         "candidates": labels})
    return rows


def compare(k, budget, seed, root):
    task = make_task(k)
    rows = make_rows(k, budget, seed)
    practice = [row for row in rows if row["split"] == "train"]
    dev = [row for row in rows if row["split"] == "dev"]
    reserve = reserved_rows(k, seed)
    validate_records(task, rows + reserve)
    folder = root / f"k{k}-n{budget}-s{seed}"
    result = {"actions": k, "budget": budget, "seed": seed, "training_sha256": digest(practice),
              "development_sha256": digest(dev), "reserve_sha256": digest(reserve), "policies": {}}
    for name, balanced in (("default", False), ("balanced", True)):
        model = train(task, practice, max_depth=5, min_leaf=2, balanced_ties=balanced)
        write_new(folder / name / "tree.json", model)
        dev_cases = score_cases(model, dev)
        curve = threshold_curve(dev_cases)
        selected = select_policy(curve)
        write_new(folder / name / "development.json", {"cases": dev_cases, "curve": curve,
                                                       "selected": selected})
        cases = score_cases(model, reserve)
        write_new(folder / name / "reserve.json", {"cases": cases, "descriptive_curve": threshold_curve(cases)})
        result["policies"][name] = {
            "checkpoint_sha256": model["sha256"], "dev_selected": selected,
            "dev_policy_status": "eligible" if selected else "no_eligible_dev_policy",
            "reserve_at_zero": summarize_cases(cases, 0),
            "reserve_at_selected": summarize_cases(cases, selected["threshold"]) if selected else None,
        }
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    out = parser.parse_args().out
    out.mkdir(parents=True, exist_ok=False)
    protocol = {"max_depth": 5, "min_leaf": 2, "holdout_seed_offset": RESERVE_SEED,
                "cases_per_seed": 256, "groups_per_seed": 16,
                "design": "For each training seed exclude all 384 maximum-budget sweep values and all "
                          "256 values from the depth intervention. Sample 16 of the remaining 24 per "
                          "stratum without replacement. Pair default and balanced tie rules on identical "
                          "training, development and reserve records. Both use the sweep's fixed dev "
                          "threshold grid and eligibility rule; results are undefined for any policy "
                          "without an eligible dev threshold. The zero-cutoff result is descriptive."}
    write_new(out / "protocol.json", protocol)
    results = [compare(k, budget, seed, out) for k in ACTION_COUNTS
               for budget in BUDGETS for seed in SEEDS]
    write_new(out / "summary.json", {"protocol_sha256": digest(protocol), "runs": results})


if __name__ == "__main__":
    main()
