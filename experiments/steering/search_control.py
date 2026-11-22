"""Learn which deterministic ordering to use and when to end an actual search."""

import argparse
from collections import Counter
import hashlib
from pathlib import Path
import random
from statistics import mean, median
from time import perf_counter_ns

from decision_workshop.contracts import validate_records
from decision_workshop.core import predict, train, write_new
from . import scheduling
from .scheduling import make_queue, policy_for, search


RULES = ("spt", "edd", "wspt", "atc")
PRICE = 0.5
CAPS = (1, 5, 20)


def queues(split, count):
    for i in range(count):
        n = (9 + i % 2) if split == "larger" else 4 + i % 5
        yield f"control-{split}-{i}", make_queue(f"control-{split}", i, n=n)


def context(jobs):
    total = sum(j.duration for j in jobs)
    return {"size": len(jobs), "duration": total, "max_duration": max(j.duration for j in jobs),
            "mean_due_ratio": mean(j.due for j in jobs) / total,
            "due_spread": (max(j.due for j in jobs) - min(j.due for j in jobs)) / total,
            "max_weight": max(j.weight for j in jobs),
            "tight_fraction": mean(j.due <= total / 2 for j in jobs)}


def lower_bound(jobs):
    # No job can complete before its own duration even if it runs first.
    return sum(j.weight * max(0, j.duration - j.due) for j in jobs)


def stopping_state(ctx, rule, observed, bound):
    return {**ctx, "rule": rule, "attempts": len(observed), "initial_cost": observed[0],
            "remaining_gap": min(observed) - bound, "improvement": observed[0] - min(observed)}


def fit(name, labels, examples):
    state = examples[0][1]
    task = {"name": name, "version": "1", "labels": list(labels),
            "features": {k: "category" if isinstance(v, str) else "number" for k, v in state.items()}}
    rows = [{"id": str(i), "group": group, "split": "train", "state": x, "target": y,
             "candidates": list(labels)} for i, (group, x, y) in enumerate(examples)]
    validate_records(task, rows)
    return train(task, rows, balanced_ties=True)


def teach():
    strategies, stops = [], []
    for group, jobs in queues("train", 96):
        ctx = context(jobs)
        bound = lower_bound(jobs)
        final = {}
        for rule in RULES:
            attempts = search(jobs, policy_for(rule), budget=20)["attempts"]
            observed = [a["estimated_cost"] for a in attempts]
            scores = {cap: min(observed[:cap]) + PRICE * cap for cap in CAPS}
            final[rule] = scores[20]
            for cap in (1, 5):
                best_later = min(v for c, v in scores.items() if c > cap)
                target = "stop" if scores[cap] <= best_later else "continue"
                stops.append((group, stopping_state(ctx, rule, observed[:cap], bound), target))
        strategies.append((group, ctx, min(RULES, key=final.get)))
    return fit("choose_strategy", RULES, strategies), fit("end_search", ("stop", "continue"), stops)


def run_queue(jobs, rule, stopping, strategy_model=None, stop_model=None):
    start = perf_counter_ns()
    ctx = context(jobs)
    bound = lower_bound(jobs)
    if strategy_model is not None:
        rule = predict(strategy_model, ctx, list(RULES))["action"] or rule
    decisions = []

    def should_stop(observed):
        cap = len(observed)
        if cap not in (1, 5):
            return False
        next_cap = 5 if cap == 1 else 20
        if min(observed) - bound <= PRICE * (next_cap - cap):
            action = "bound"
        elif stopping == "learned":
            state = stopping_state(ctx, rule, observed, bound)
            action = predict(stop_model, state, ["stop", "continue"])["action"] or "continue"
        else:
            action = "continue"
        decisions.append({"after": cap, "best_seen": min(observed), "action": action})
        return action != "continue"

    adaptive = stopping in ("certified", "learned")
    result = search(jobs, policy_for(rule), budget=20 if adaptive else int(stopping),
                    stop_after=should_stop if adaptive else None)
    elapsed = perf_counter_ns() - start
    attempts = result["attempts"]
    cost = attempts[result["selected_attempt"] - 1]["realized_cost"]
    return {"rule": rule, "attempts": len(attempts), "cost": cost,
            "objective": cost + PRICE * len(attempts), "wall_ns": elapsed, "decisions": decisions,
            "selected_order": result["selected_order"]}


def experiment(out):
    out.mkdir(parents=True, exist_ok=False)
    write_new(out / "protocol.json", {"training_groups": 96, "dev_groups": 40, "iid_groups": 64,
              "larger_groups": 32, "attempt_price": PRICE, "allowed_stop_points": CAPS,
              "implementation_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in
                                        (Path(__file__), Path(scheduling.__file__))},
              "dev_selection": "lowest mean priced loss over rule x {1, 5, 20, certified}"})
    strategy_model, stop_model = teach()
    write_new(out / "strategy.json", strategy_model)
    write_new(out / "stop.json", stop_model)
    dev = list(queues("dev", 40))
    controls = [(rule, stop) for rule in RULES for stop in ("1", "5", "20", "certified")]
    dev_scores = {(rule, stop): mean(run_queue(j, rule, stop)["objective"] for _, j in dev)
                  for rule, stop in controls}
    best = min(controls, key=dev_scores.get)
    fixed20 = min(RULES, key=lambda r: dev_scores[r, "20"])
    write_new(out / "development.json", {"means": {f"{r}/{s}": v for (r, s), v in dev_scores.items()},
                                         "control": best, "rule_at_20": fixed20})
    configs = {"best_dev_control": (*best, None), "fixed20": (fixed20, "20", None),
               "certified": (fixed20, "certified", None), "learned_strategy": (fixed20, "20", strategy_model),
               "learned_stop": (fixed20, "learned", None), "both": (fixed20, "learned", strategy_model)}
    summary = {}
    for split, count in (("iid", 64), ("larger", 32)):
        results = []
        for group, jobs in queues(split, count):
            order = list(configs)
            random.Random(group).shuffle(order)
            policies = {}
            for label in order:
                rule, stop, selector = configs[label]
                policies[label] = run_queue(jobs, rule, stop, selector, stop_model)
            results.append({"group": group, "policies": policies})
        write_new(out / f"{split}.json", results)
        summary[split] = {}
        for label in configs:
            runs = [r["policies"][label] for r in results]
            differences = [run["objective"] - row["policies"]["best_dev_control"]["objective"]
                           for run, row in zip(runs, results, strict=True)]
            summary[split][label] = {"groups": count, "mean_priced_loss": mean(r["objective"] for r in runs),
                "mean_cost": mean(r["cost"] for r in runs), "mean_attempts": mean(r["attempts"] for r in runs),
                "wall_us_median": median(r["wall_ns"] for r in runs)/1000,
                "attempt_histogram": dict(Counter(r["attempts"] for r in runs)),
                "rule_histogram": dict(Counter(r["rule"] for r in runs)),
                "better_equal_worse": [sum(d < 0 for d in differences), differences.count(0),
                                       sum(d > 0 for d in differences)]}
    write_new(out / "summary.json", summary)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    experiment(parser.parse_args().out)
