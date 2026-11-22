"""Compare job ordering policies in one bounded single-worker scheduling search."""

import argparse
from dataclasses import asdict, dataclass, replace
from functools import lru_cache
import hashlib
import heapq
import json
import math
from pathlib import Path
import random
import statistics
import time


BUDGETS = (1, 5, 20)
RULES = ("spt", "edd", "wspt", "atc")


@dataclass(frozen=True)
class Job:
    id: str
    duration: int
    due: int
    weight: int
    actual: int


def rng_for(key):
    return random.Random(int.from_bytes(hashlib.sha256(("queues-v1/" + key).encode()).digest(), "big"))


def make_queue(family, index, n=None):
    """The noisy family shares an IID queue's estimates but changes realized durations."""
    source = "iid" if family == "noisy" else family
    rng = rng_for(f"{source}/{index}")
    size = n if n is not None else (9 + index % 2 if family == "larger" else 4 + index % 5)
    durations = [rng.randint(1, 12) for _ in range(size)]
    total = sum(durations)
    jobs = []
    used = set()
    for i, duration in enumerate(durations):
        while True:
            due = rng.randint(max(1, total // 8), total)
            weight = rng.choice((1, 2, 3, 5, 8))
            if (duration, due, weight) not in used:
                used.add((duration, due, weight))
                break
        jobs.append(Job(f"job-{i:02d}", duration, due, weight, duration))
    if family == "noisy":
        noise = rng_for(f"realization/{index}")
        jobs = [replace(j, actual=max(1, round(j.duration * noise.choice((0.5, 0.75, 1., 1.5, 2.)))))
                for j in jobs]
    return tuple(jobs)


def schedule_cost(jobs, order, *, realized=False):
    by_id = {job.id: job for job in jobs}
    if len(order) != len(jobs) or len(by_id) != len(jobs) or set(order) != set(by_id):
        raise ValueError("order must contain every unique job ID exactly once")
    elapsed = loss = 0
    for identifier in order:
        job = by_id[identifier]
        elapsed += job.actual if realized else job.duration
        loss += job.weight * max(0, elapsed - job.due)
    return loss


def optimal_first_costs(jobs, elapsed=0, *, realized=False):
    """Exact optimum conditional on each next action; state is the remaining subset."""
    jobs = tuple(jobs)
    lengths = [j.actual if realized else j.duration for j in jobs]
    full = (1 << len(jobs)) - 1
    start = elapsed + sum(lengths)

    @lru_cache(None)
    def best(mask):
        if not mask:
            return 0
        now = start - sum(p for i, p in enumerate(lengths) if mask & (1 << i))
        return min(j.weight * max(0, now + lengths[i] - j.due) + best(mask ^ (1 << i))
                   for i, j in enumerate(jobs) if mask & (1 << i))

    return {j.id: j.weight * max(0, elapsed + lengths[i] - j.due) + best(full ^ (1 << i))
            for i, j in enumerate(jobs)}


def features(job, remaining, elapsed):
    slack = job.due - elapsed - job.duration
    work = sum(j.duration for j in remaining)
    return {"duration": job.duration, "weight": job.weight, "due_in": job.due - elapsed,
            "slack": slack, "remaining_work": work, "remaining_jobs": len(remaining),
            "due_rank": sum(j.due < job.due for j in remaining),
            "immediate_loss": job.weight * max(0, -slack)}


def policy_for(name, model=None):
    """Return a candidate ranker. Neither policy nor stop callback sees actual durations."""
    if name in ("tree", "bert"):
        from decision_workshop.steering import rank_candidates
        return lambda candidates: rank_candidates(model, candidates)
    if name not in RULES:
        raise ValueError(f"unknown policy {name}")

    def priority(f):
        if name == "spt":
            return -f["duration"]
        if name == "edd":
            return -f["due_in"]
        ratio = f["weight"] / f["duration"]
        if name == "wspt":
            return ratio
        avg = f["remaining_work"] / f["remaining_jobs"]
        return ratio * math.exp(-max(0, f["slack"]) / (2 * avg))

    def rank(candidates):
        scored = [(identifier, priority(f), tuple(f[k] for k in sorted(f))) for identifier, f in candidates]
        return [(identifier, score) for identifier, score, _ in sorted(scored, key=lambda x: (-x[1], x[2]))]
    return rank


def rank_codes(radices):
    """Enumerate complete permutations by summed deviations from greedy rank zero."""
    zero = (0,) * len(radices)
    heap = [(0, zero)]
    seen = {zero}
    while heap:
        total, negative = heapq.heappop(heap)
        digits = tuple(-x for x in negative)
        yield digits
        for i, bound in enumerate(radices):
            if digits[i] + 1 < bound:
                changed = negative[:i] + (negative[i] - 1,) + negative[i + 1:]
                if changed not in seen:
                    seen.add(changed)
                    heapq.heappush(heap, (total + 1, changed))


def search(jobs, policy, *, budget=20, blocked_first=(), stop_after=None):
    clock = time.perf_counter()
    by_id = {j.id: j for j in jobs}
    blocked = set(blocked_first)
    if (not jobs or budget < 1 or len(jobs) != len(by_id) or not blocked <= by_id.keys()
            or len(blocked) == len(jobs)):
        raise ValueError("search requires a positive budget, unique jobs, and an eligible first action")
    cache = {}
    attempts, checkpoints = [], {}
    selected = 0
    hindsight = None
    rank_time = 0.0
    radices = (len(jobs) - len(blocked), *range(len(jobs) - 1, 0, -1))
    for code in rank_codes(radices):
        remaining = tuple(jobs)
        elapsed = 0
        order = []
        for digit in code:
            key = frozenset(j.id for j in remaining)
            if key not in cache:
                eligible = [j for j in remaining if order or j.id not in blocked]
                if len(eligible) == 1:
                    cache[key] = [eligible[0].id]
                else:
                    candidates = [(j.id, features(j, remaining, elapsed)) for j in eligible]
                    t = time.perf_counter()
                    ranking = [identifier for identifier, _ in policy(candidates)]
                    rank_time += time.perf_counter() - t
                    if len(ranking) != len(eligible) or set(ranking) != {j.id for j in eligible}:
                        raise ValueError("policy must rank every eligible job exactly once")
                    cache[key] = ranking
            ident = cache[key][digit]
            order.append(ident)
            elapsed += by_id[ident].duration
            remaining = tuple(j for j in remaining if j.id != ident)
        estimate = schedule_cost(jobs, order)
        actual = schedule_cost(jobs, order, realized=True)
        if not attempts or estimate < attempts[selected]["estimated_cost"]:
            selected = len(attempts)
        hindsight = actual if hindsight is None else min(hindsight, actual)
        attempts.append({"order": order, "rank_code": code, "estimated_cost": estimate, "realized_cost": actual,
                         "elapsed_seconds": time.perf_counter() - clock})
        count = len(attempts)
        if count in BUDGETS:
            pick = attempts[selected]
            checkpoints[str(count)] = {
                "selected_attempt": selected + 1, "estimated_cost": pick["estimated_cost"],
                "realized_cost": pick["realized_cost"], "best_visited_realized_cost": hindsight,
                "elapsed_seconds": time.perf_counter() - clock, "ranking_seconds": rank_time,
            }
        stop = stop_after is not None and stop_after(tuple(x["estimated_cost"] for x in attempts))
        if count >= budget or stop:
            break
    return {"attempts": attempts, "checkpoints": checkpoints, "selected_attempt": selected + 1,
            "selected_order": attempts[selected]["order"], "elapsed_seconds": time.perf_counter() - clock}


def training_examples(count=64):
    for i in range(count):
        queue = make_queue("train", i)
        rng = rng_for(f"training-prefix/{i}")
        shuffled = rng.sample(list(queue), len(queue))
        cut = rng.randint(1, len(queue) - 2)
        for remaining, elapsed in ((queue, 0), (shuffled[cut:], sum(j.duration for j in shuffled[:cut]))):
            costs = optimal_first_costs(remaining, elapsed)
            for j in remaining:
                yield {"group": f"train-{i:03d}", "features": features(j, remaining, elapsed),
                       "preferred": costs[j.id] == min(costs.values())}


def forbidden_optima(jobs):
    costs = optimal_first_costs(jobs)
    return tuple(k for k, v in costs.items() if v == min(costs.values())) if len(set(costs.values())) > 1 else ()


def measure(family, index, policies):
    prefix = "iid" if family == "omitted" else family
    jobs = make_queue(prefix, index)
    blocked = forbidden_optima(jobs) if family == "omitted" else ()
    actual = optimal_first_costs(jobs, realized=True)
    estimate = optimal_first_costs(jobs)
    selected = {name: search(jobs, fn, blocked_first=blocked) for name, fn in policies.items()}
    return {"group": f"{prefix}-{index:03d}", "index": index, "family": family,
            "jobs": [asdict(j) for j in jobs], "blocked_first": list(blocked),
            "realized_optimum": min(actual.values()), "estimated_optimum": min(estimate.values()),
            "attainable_actual": min(v for k, v in actual.items() if k not in blocked), "policies": selected}


def strategy_delta(results, name):
    return statistics.mean((ck["estimated_cost"] - row["estimated_optimum"]) / max(1, row["estimated_optimum"])
                           for row in results for ck in row["policies"][name]["checkpoints"].values())


def aggregate(results):
    def at_budget(result, budget):
        if str(budget) in result["checkpoints"]:
            return result["checkpoints"][str(budget)]
        attempts = result["attempts"]
        if len(attempts) >= budget:
            raise ValueError("missing recorded search checkpoint")
        # All legal permutations were exhausted before this cap.
        pick = attempts[result["selected_attempt"] - 1]
        return {**pick, "elapsed_seconds": result["elapsed_seconds"],
                "best_visited_realized_cost": min(a["realized_cost"] for a in attempts)}

    summary = {}
    for name in results[0]["policies"]:
        summary[name] = {}
        for budget in BUDGETS:
            ck = [at_budget(row["policies"][name], budget) for row in results]
            exact = [r["realized_optimum"] for r in results]
            costs = [x["realized_cost"] for x in ck]
            summary[name][str(budget)] = {
                "count": len(ck), "mean_cost": statistics.mean(costs),
                "mean_regret": statistics.mean(x - y for x, y in zip(costs, exact)),
                "optimal": sum(x == y for x, y in zip(costs, exact)),
                "mean_attainable_regret": statistics.mean(x - r["attainable_actual"] for x, r in zip(costs, results)),
                "mean_best_visited_actual": statistics.mean(x["best_visited_realized_cost"] for x in ck),
                "mean_seconds": statistics.mean(x["elapsed_seconds"] for x in ck),
                "exhausted_before_budget": sum(len(row["policies"][name]["attempts"]) < budget for row in results),
            }
    return summary


def run(out, base):
    from decision_workshop.steering import fit_rankers
    out.mkdir(parents=True, exist_ok=False)
    controls = {name: policy_for(name) for name in RULES}
    dev = [measure("dev", i, controls) for i in range(16)]
    rule = min(RULES, key=lambda name: strategy_delta(dev, name))
    models = fit_rankers("scheduling_candidate", list(training_examples()), out / "rankers", base)
    policies = {**controls, **{name: policy_for(name, m) for name, m in models.items()}}
    totals = {}
    with (out / "queues.jsonl").open("x", encoding="utf-8") as file:
        for row in dev:
            print(json.dumps(row), file=file)
        for family, size in (("iid", 24), ("larger", 12), ("noisy", 24), ("omitted", 24)):
            cohort = []
            for i in range(size):
                row = measure(family, i, policies)
                print(json.dumps(row), file=file, flush=True)
                cohort.append(row)
            totals[family] = aggregate(cohort)
    summary = {"dev_control": rule, "dev_normalized_regret": {n: strategy_delta(dev, n) for n in RULES},
               "budgets": list(BUDGETS), "cohorts": totals}
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--base", type=Path, required=True)
    args = p.parse_args()
    run(args.out, args.base)
