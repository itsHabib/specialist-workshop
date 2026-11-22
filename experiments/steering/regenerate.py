"""Measure whether to pay for a second deterministic candidate-generation pass."""

import argparse
import hashlib
from pathlib import Path
from statistics import mean
from time import perf_counter_ns

from decision_workshop.core import predict, write_new
from . import scheduling, search_control
from .scheduling import make_queue, policy_for, search
from .search_control import context, fit, lower_bound


# Synthetic cost of regenerating the first-action shortlist, in weighted tardiness units.
FEE = 2


def visible(jobs, allowed, first_costs):
    return {**context(jobs), "first_cost": first_costs[0], "best_at_five": min(first_costs),
            "headroom": min(first_costs) - lower_bound(jobs),
            "max_omitted_weight": max(j.weight for j in jobs if j.id not in allowed),
            "min_omitted_due": min(j.due for j in jobs if j.id not in allowed)}


def run(jobs, rule, mode, model=None):
    start = perf_counter_ns()
    shortest = sorted(jobs, key=lambda j: (j.duration, j.due, -j.weight))[:2]
    allowed = {j.id for j in shortest}
    blocked = {j.id for j in jobs if j.id not in allowed}
    recovered = False
    observation = None

    def consider(costs):
        nonlocal recovered, observation
        if len(costs) != 5:
            return False
        observation = visible(jobs, allowed, costs)
        if mode == "always":
            recovered = True
        elif mode == "headroom":
            recovered = observation["headroom"] > FEE
        elif mode == "tree":
            recovered = predict(model, observation, ["keep", "expand"])["action"] == "expand"
        return recovered

    primary = search(jobs, policy_for(rule), budget=20, blocked_first=blocked, stop_after=consider)
    attempts = list(primary["attempts"])
    if recovered:
        widened = search(jobs, policy_for(rule), budget=15)
        attempts.extend(widened["attempts"])
    if len(attempts) != 20 or observation is None:
        raise ValueError("expected 20 actual search attempts and a decision after five")
    best = min(attempts, key=lambda a: a["estimated_cost"])
    elapsed = perf_counter_ns() - start
    return {"expanded": recovered, "observation": observation, "attempts": len(attempts),
            "distinct_schedules": len({tuple(a["order"]) for a in attempts}),
            "cost": best["realized_cost"], "priced_loss": best["realized_cost"] + FEE * recovered,
            "selected_order": best["order"], "wall_ns": elapsed}


def queues(split, count):
    for i in range(count):
        n = ((9 + i % 2) if split == "larger" else (6 + i % 3))
        yield f"regenerate-{split}-{i}", make_queue(f"regenerate-{split}", i, n=n)


def experiment(out, rule):
    out.mkdir(parents=True, exist_ok=False)
    write_new(out / "protocol.json", {"rule": rule, "fee": FEE, "first_budget": 5, "total_budget": 20,
              "first_moves": "two shortest jobs", "train": 96, "dev": 40, "iid": 64, "larger": 32,
              "implementation_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in
                                        (Path(__file__), Path(scheduling.__file__), Path(search_control.__file__))}})
    examples = []
    for group, jobs in queues("train", 96):
        plain = run(jobs, rule, "never")
        expanded = run(jobs, rule, "always")
        label = "expand" if expanded["priced_loss"] < plain["priced_loss"] else "keep"
        examples.append((group, plain["observation"], label))
    model = fit("regenerate", ("keep", "expand"), examples)
    write_new(out / "tree.json", model)
    controls = ("never", "always", "headroom")
    dev = list(queues("dev", 40))
    means = {name: mean(run(j, rule, name)["priced_loss"] for _, j in dev) for name in controls}
    baseline = min(controls, key=means.get)
    write_new(out / "development.json", {"means": means, "selected": baseline})
    summary = {}
    for split, count in (("iid", 64), ("larger", 32)):
        rows = []
        for group, jobs in queues(split, count):
            policies = {name: run(jobs, rule, name, model) for name in (*controls, "tree")}
            useful = policies["always"]["priced_loss"] < policies["never"]["priced_loss"]
            rows.append({"group": group, "beneficial_to_expand": useful, "policies": policies})
        write_new(out / f"{split}.json", rows)
        summary[split] = {}
        for name in (*controls, "tree"):
            expand = [r for r in rows if r["policies"][name]["expanded"]]
            summary[split][name] = {"groups": count, "expanded": len(expand),
                "beneficial_expansions": sum(r["beneficial_to_expand"] for r in expand),
                "beneficial_opportunities": sum(r["beneficial_to_expand"] for r in rows),
                "mean_cost": mean(r["policies"][name]["cost"] for r in rows),
                "mean_priced_loss": mean(r["policies"][name]["priced_loss"] for r in rows)}
    write_new(out / "summary.json", summary)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--rule", choices=("spt", "edd", "wspt", "atc"), default="atc")
    args = parser.parse_args()
    experiment(args.out, args.rule)
