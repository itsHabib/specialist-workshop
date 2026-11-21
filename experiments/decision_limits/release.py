"""A synthetic, finite-horizon release decision with an exact Bayes oracle."""

import hashlib
import math
import random
from collections import Counter
from dataclasses import dataclass


ACTIONS = ("publish", "rollback", "review", "diagnose")
HORIZON = 2
SIZES = {"train": 96, "dev": 24, "iid": 80, "shift": 48}
TASK = {
    "name": "release_decision", "version": "1",
    "question": ("Choose the lowest expected total cost. Each diagnostic independently reports "
                 "the true fault status at the stated accuracy; prior and accuracy are percentages. "
                 "At most two diagnostics may be purchased in total."),
    "features": {"prior_fault_pct": "number", "diagnostic_accuracy_pct": "number",
                 "diagnostic_cost": "number", "positive_checks": "number", "negative_checks": "number"},
    "labels": list(ACTIONS),
    "descriptions": {"publish": "End now. Cost 30 if faulty, zero if healthy.",
                     "rollback": "End now. Cost 10 if healthy, zero if faulty.",
                     "review": "End now with human review. Cost 3 regardless of fault.",
                     "diagnose": "Pay the diagnostic cost to observe another positive or negative result."},
}


@dataclass(frozen=True)
class Episode:
    id: str
    prior: int
    accuracy: int
    price: int
    faulty: bool
    checks: tuple

    def state(self, step):
        observed = self.checks[:step]
        return {"prior_fault_pct": self.prior, "diagnostic_accuracy_pct": self.accuracy,
                "diagnostic_cost": self.price, "positive_checks": sum(observed),
                "negative_checks": len(observed) - sum(observed)}


def episodes(family):
    """Independent seeds per episode; only the shift family changes reliability."""
    reliability = (54, 62, 68) if family == "shift" else (72, 84, 93)
    for i in range(SIZES[family]):
        name = f"{family}-{i:03d}"
        seed = int.from_bytes(hashlib.sha256(f"release-v1/{name}".encode()).digest(), "big")
        rng = random.Random(seed)
        prior = rng.randint(4, 94)
        accuracy = rng.choice(reliability)
        price = rng.choice((1, 2))
        faulty = rng.random() < prior / 100
        checks = tuple((rng.random() < accuracy / 100) == faulty for _ in range(HORIZON))
        yield Episode(name, prior, accuracy, price, faulty, checks)


def posterior(state):
    p = state["prior_fault_pct"] / 100
    q = state["diagnostic_accuracy_pct"] / 100
    positives, negatives = state["positive_checks"], state["negative_checks"]
    bad = p * q**positives * (1 - q)**negatives
    good = (1 - p) * (1 - q)**positives * q**negatives
    return bad / (bad + good)


def expected_costs(p, q, price, left):
    """Bellman action values under optimal subsequent decisions."""
    costs = {"publish": 30 * p, "rollback": 10 * (1 - p), "review": 3.0}
    if left:
        w = p * q + (1 - p) * (1 - q)
        yes = min(expected_costs(p * q / w, q, price, left - 1).values())
        no = min(expected_costs(p * (1 - q) / (1 - w), q, price, left - 1).values())
        costs["diagnose"] = price + w * yes + (1 - w) * no
    return costs


def quotes(state):
    left = HORIZON - state["positive_checks"] - state["negative_checks"]
    return expected_costs(posterior(state), state["diagnostic_accuracy_pct"] / 100,
                          state["diagnostic_cost"], left)


def optimal(costs):
    best = min(costs.values())
    return [a for a in ACTIONS if a in costs and abs(costs[a] - best) < 1e-9]


def records():
    for family in SIZES:
        split = "test" if family in ("iid", "shift") else family
        for ep in episodes(family):
            for step in range(HORIZON + 1):
                state = ep.state(step)
                costs = quotes(state)
                yield {"id": f"{ep.id}-s{step}", "group": ep.id, "split": split,
                       "state": state, "target": optimal(costs)[0], "candidates": list(costs)}


def incumbent(row_id, state, candidates):
    if not state["positive_checks"] + state["negative_checks"]:
        action = "diagnose"
    else:
        action = "rollback" if state["positive_checks"] else "publish"
    return {"preferred": action, "action": action if action in candidates else None}


def oracle(row_id, state, candidates):
    costs = quotes(state)
    choice = optimal({a: costs[a] for a in candidates})[0]
    return {"preferred": choice, "action": choice}


def rollout(episode, policy, *, omit=False):
    """Replay a policy(row_id, state, candidates) returning action and preferred."""
    trace = []
    total = 0
    for step in range(HORIZON + 1):
        state = episode.state(step)
        costs = quotes(state)
        unrestricted = optimal(costs)
        omitted = (unrestricted[0] if omit and step == 0 and len(unrestricted) == 1
                   and unrestricted[0] != "review" else None)
        options = {a: v for a, v in costs.items() if a != omitted}
        best = optimal(options)
        row_id = f"{episode.id}-s{step}"
        response = policy(row_id, state, list(options))
        choice = response["action"]
        if choice is not None and choice not in options:
            raise ValueError(f"policy returned unavailable executable action: {choice}")
        action = choice or "review"
        regret = max(0.0, options[action] - min(options.values()))
        loss = {"publish": 30 * episode.faulty, "rollback": 10 * (not episode.faulty),
                "review": 3, "diagnose": episode.price}[action]
        trace.append({
            "id": row_id, "state": state, "posterior": posterior(state), "candidates": list(options),
            "oracle_actions": best, "oracle_expected_costs": options, "omitted_action": omitted,
            "preferred": response.get("preferred"), "choice": choice, "executed": action,
            "deferred": choice is None, "reasons": response.get("reasons", []),
            "optimal": action in best, "expected_regret": regret, "incurred_cost": loss,
        })
        total += loss
        if action != "diagnose":
            return {"episode": episode.id, "faulty": episode.faulty, "total_cost": total,
                    "terminal_action": action, "steps": trace}
    raise AssertionError("diagnostic escaped its horizon")


def summarize(traces):
    n = len(traces)
    if not n:
        raise ValueError("summary requires episodes")
    steps = [s for trace in traces for s in trace["steps"]]
    finals = [trace["steps"][-1] for trace in traces]
    reviews = sum(s["executed"] == "review" for s in finals)
    correct = sum(s["optimal"] and s["executed"] != "review" for s in finals)
    costs = [trace["total_cost"] for trace in traces]
    per_step = {}
    for i in range(HORIZON + 1):
        visited = [trace["steps"][i] for trace in traces if len(trace["steps"]) > i]
        per_step[str(i)] = {"visits": len(visited), "optimal": sum(s["optimal"] for s in visited),
                            "executed": dict(Counter(s["executed"] for s in visited))}
    return {
        "episodes": n, "total_cost": sum(costs), "mean_cost": sum(costs) / n,
        "mean_expected_regret": math.fsum(s["expected_regret"] for s in steps) / n,
        "harmful_publishes": sum(t["terminal_action"] == "publish" and t["faulty"] for t in traces),
        "healthy_rollbacks": sum(t["terminal_action"] == "rollback" and not t["faulty"] for t in traces),
        "diagnostics": sum(s["executed"] == "diagnose" for s in steps),
        "unnecessary_diagnostics": sum(s["executed"] == "diagnose" and not s["optimal"] for s in steps),
        "reviews": reviews, "oracle_justified_reviews": sum(s["executed"] == "review" and s["optimal"] for s in finals),
        "policy_deferrals": sum(s["deferred"] for s in finals),
        "unrestricted_optimum_omitted": sum(s["omitted_action"] is not None for s in steps),
        "terminal_correct_suggestions": correct, "terminal_wrong_suggestions": n - reviews - correct,
        "coverage": (n - reviews) / n, "accuracy_when_suggested": correct / (n - reviews) if n > reviews else None,
        "per_step": per_step,
    }
