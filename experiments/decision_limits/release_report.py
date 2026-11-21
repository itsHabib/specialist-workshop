"""Summarize paired uncertainty, overlap, and the frozen hand-checks after a run."""

import argparse
from collections import defaultdict
import json
from pathlib import Path
import random
from statistics import mean

from decision_workshop import load_model
from .release import Episode, incumbent, optimal, oracle, quotes
from .release_run import learned


def paired_delta(traces, reference):
    by_id = {t["episode"]: t["total_cost"] for t in reference}
    if set(by_id) != {t["episode"] for t in traces}:
        raise ValueError("paired comparison needs identical episode IDs")
    delta = [trace["total_cost"] - by_id[trace["episode"]] for trace in traces]
    rng = random.Random(113)
    samples = sorted(mean(rng.choices(delta, k=len(delta))) for _ in range(2000))
    return {"episodes": len(delta), "mean": mean(delta), "bootstrap_95pct": [samples[49], samples[1949]]}


def hand_checks(root):
    models = {"oracle": oracle, "incumbent": incumbent,
              "tree": learned(load_model(root / "run/tree.json")),
              "bert": learned(load_model(root / "run/model"))}
    out = []
    for name, reliability, observations in (("sharp", 90, ()), ("noisy", 60, ()),
                                            ("after_positive", 90, (True,)), ("after_negative", 90, (False,))):
        ep = Episode(name, 25, reliability, 1, False, observations)
        state = ep.state(len(observations))
        costs = quotes(state)
        choices = {model: policy(name, state, list(costs))["action"] for model, policy in models.items()}
        out.append({"case": name, "state": state, "expected_costs": costs,
                    "optimal": optimal(costs), "choices": choices})
    return out


def diagnostics(root):
    traces = defaultdict(lambda: defaultdict(list))
    for line in (root / "episodes.jsonl").read_text(encoding="utf-8").splitlines():
        trace = json.loads(line)
        traces[trace["cohort"]][trace["policy"]].append(trace)
    contrasts = {cohort: {name: paired_delta(rows, policies["incumbent"])
                          for name, rows in policies.items() if name != "incumbent"}
                 for cohort, policies in traces.items()}
    rows = [json.loads(line) for line in (root / "task/records.jsonl").read_text(encoding="utf-8").splitlines()]
    def visible(row):
        return tuple(sorted(row["state"].items()))
    train_states = {visible(row) for row in rows if row["split"] == "train"}
    overlaps = {}
    for family in ("dev", "iid", "shift"):
        cohort = [row for row in rows if row["group"].startswith(family + "-")]
        overlaps[family] = {"rows": len(cohort), "seen_visible_states": sum(visible(r) in train_states for r in cohort)}
    return {"mean_cost_minus_incumbent": contrasts, "hand_checks": hand_checks(root),
            "visible_state_overlap_with_training": overlaps}


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, required=True)
    root = p.parse_args().out
    result = diagnostics(root)
    with (root / "diagnostics.json").open("x", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
        print(file=f)
