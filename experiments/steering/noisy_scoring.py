"""A fixed independent replay comparing nominal and sampled-expected schedule costs."""

import argparse
from dataclasses import asdict, replace
import hashlib
import json
from pathlib import Path
import statistics
import time

from .scheduling import make_queue, optimal_first_costs, policy_for, rng_for, schedule_cost, search


FAMILY = "uncertainty-fresh-v1"
QUEUES = 64
BUDGETS = (5, 20)
MATCHED = (0.5, 0.75, 1.0, 1.5, 2.0)
REGIMES = {"matched": MATCHED, "no_noise": (1.0,), "faster": (0.5, 0.75)}
SCENARIOS = 32


def profile(jobs, rng, factors):
    # Assigned in observable feature order, so arbitrary IDs and caller order never choose the draw.
    return {j.id: max(1, round(j.duration * rng.choice(factors)))
            for j in sorted(jobs, key=lambda j: (j.duration, j.due, j.weight))}


def loss(jobs_by_id, order, durations):
    elapsed = total = 0
    for key in order:
        job = jobs_by_id[key]
        elapsed += durations[key]
        total += job.weight * max(0, elapsed - job.due)
    return total


def retained(costs, cap):
    return min(range(cap), key=costs.__getitem__)


def select_plans(jobs, orders, queue_index, *, robust):
    clock = time.perf_counter()
    by_id = {j.id: j for j in jobs}
    if robust:
        rng = rng_for(f"{FAMILY}/score/{queue_index}")
        profiles = [profile(jobs, rng, MATCHED) for _ in range(SCENARIOS)]
    else:
        profiles = [{j.id: j.duration for j in jobs}]
    setup_seconds = time.perf_counter() - clock
    costs = []
    snapshots = {}
    for order in orders:
        costs.append(sum(loss(by_id, order, d) for d in profiles) / len(profiles))
        if len(costs) in BUDGETS:
            pick = retained(costs, len(costs))
            snapshots[str(len(costs))] = {"selected_attempt": pick + 1,
                                         "predicted_cost": costs[pick],
                                         "selector_seconds": time.perf_counter() - clock}
    return {"profiles": profiles, "candidate_costs": costs, "by_budget": snapshots,
            "setup_seconds": setup_seconds}


def one_queue(i):
    jobs = make_queue(FAMILY, i)
    run = search(jobs, policy_for("atc"), budget=max(BUDGETS))
    orders = [a["order"] for a in run["attempts"]]
    selectors = {"nominal": select_plans(jobs, orders, i, robust=False),
                 "robust": select_plans(jobs, orders, i, robust=True)}
    # The observed outcomes are generated only after both selectors have committed to their choices.
    worlds = {}
    for name, factors in REGIMES.items():
        actual = profile(jobs, rng_for(f"{FAMILY}/world/{name}/{i}"), factors)
        world_jobs = [replace(j, actual=actual[j.id]) for j in jobs]
        optimum = min(optimal_first_costs(world_jobs, realized=True).values())
        scored = [schedule_cost(world_jobs, order, realized=True) for order in orders]
        worlds[name] = {"durations": actual, "true_optimum": optimum,
                        "candidate_realized_costs": scored,
                        "selected_costs": {method: {b: scored[p["selected_attempt"] - 1]
                                                   for b, p in result["by_budget"].items()}
                                           for method, result in selectors.items()}}
    return {"group": f"{FAMILY}-{i:03d}", "jobs": [asdict(j) for j in jobs], "search": run,
            "selectors": selectors, "worlds": worlds}


def aggregate(rows):
    summary = {}
    for regime in REGIMES:
        by_budget = {}
        for b in map(str, BUDGETS):
            worlds = [row["worlds"][regime] for row in rows]
            nominal = [w["selected_costs"]["nominal"][b] for w in worlds]
            robust = [w["selected_costs"]["robust"][b] for w in worlds]
            delta = [x - y for x,y in zip(robust,nominal)]
            rng = rng_for(f"{FAMILY}/bootstrap/{regime}/{b}")
            draws = sorted(statistics.mean(rng.choices(delta,k=len(delta))) for _ in range(2000))
            by_budget[b] = {
                "queues": len(rows), "nominal_mean": statistics.mean(nominal), "robust_mean": statistics.mean(robust),
                "robust_minus_nominal": statistics.mean(delta), "bootstrap_95pct": [draws[49], draws[1949]],
                "robust_won": sum(x < 0 for x in delta), "tied": delta.count(0), "robust_lost": sum(x > 0 for x in delta),
                "changed_order": sum(r["selectors"]["robust"]["by_budget"][b]["selected_attempt"] !=
                                     r["selectors"]["nominal"]["by_budget"][b]["selected_attempt"] for r in rows),
                "mean_true_optimum": statistics.mean(w["true_optimum"] for w in worlds),
                "selector_ms": {method: 1000 * statistics.mean(
                    r["selectors"][method]["by_budget"][b]["selector_seconds"] for r in rows)
                    for method in ("nominal","robust")},
                "search_ms": 1000 * statistics.mean(r["search"]["checkpoints"][b]["elapsed_seconds"] for r in rows),
            }
        summary[regime] = by_budget
    return summary


def run(out):
    out.mkdir(parents=True,exist_ok=False)
    rows = []
    with (out / "queues.jsonl").open("x",encoding="utf-8") as f:
        for i in range(QUEUES):
            row = one_queue(i)
            rows.append(row)
            print(json.dumps(row),file=f)
    manifest = {"family": FAMILY, "group_count":QUEUES,"budgets":BUDGETS,"scenario_count":SCENARIOS,
                "scenario_factors":MATCHED,"world_factors":REGIMES,
                "scorer_seed_namespace":FAMILY + "/score/","world_seed_namespace":FAMILY+"/world/",
                "sha256": {p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in (
                    Path(__file__), Path(__file__).with_name("scheduling.py"),out/"queues.jsonl")},
                "cohorts":aggregate(rows)}
    (out/"summary.json").write_text(json.dumps(manifest,indent=2)+"\n",encoding="utf-8")


if __name__ == "__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out",type=Path,required=True)
    run(p.parse_args().out)
