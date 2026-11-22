"""Evaluate a bounded decision sequence and derive training labels from replayed outcomes."""

from copy import deepcopy
from fractions import Fraction
import math
import random

from .contracts import validate_candidates, validate_records, validate_state, validate_task


OUTCOMES = ("verified", "failed", "review", "incorrect")


def run_episode(episode, policy, *, max_steps):
    """Offer observations to the policy, execute available actions and verify the result.

    The caller owns the episode's lifetime. Its observe(), apply(action), is_terminal()
    and verify(stop_reason) methods define one application operation. Exceptions from
    any callback propagate. A policy may return an action, None, or a predict() result.
    """
    if type(max_steps) is not int or max_steps < 0:
        raise ValueError("max_steps must be a nonnegative integer")
    steps = []
    while not episode.is_terminal():
        if len(steps) == max_steps:
            stop_reason = "step_limit"
            break
        state, candidates = episode.observe()
        if not isinstance(state, dict) or not isinstance(candidates, list):
            raise ValueError("observe must return a state dictionary and a candidate list")
        if (any(not isinstance(a, str) or not a.strip() for a in candidates)
                or len(set(candidates)) != len(candidates)):
            raise ValueError("candidates must be distinct nonempty action names")
        step = {"state": deepcopy(state), "candidates": list(candidates), "executed": False}
        if candidates:
            response = policy(deepcopy(state), list(candidates))
        else:
            response = {"action": None, "reasons": ["no_candidates"]}
        if isinstance(response, dict):
            if "action" not in response:
                raise ValueError("policy decision must contain an action")
            action = response["action"]
            if "status" in response and response["status"] != ("review" if action is None else "suggest"):
                raise ValueError("policy status is inconsistent with action")
            step["decision"] = deepcopy(response)
        else:
            action = response
        if action is not None and (not isinstance(action, str) or not action.strip()):
            raise ValueError("policy action must be a nonempty string or None")
        step["action"] = action
        steps.append(step)
        if action is None:
            stop_reason = "review"
            break
        if action not in step["candidates"]:
            stop_reason = "unavailable_action"
            break
        episode.apply(action)
        step["executed"] = True
    else:
        stop_reason = "terminal"
    outcome = episode.verify(stop_reason)
    if not isinstance(outcome, dict) or outcome.get("status") not in OUTCOMES:
        raise ValueError("verifier must return a verified, failed, review or incorrect outcome")
    return {"steps": steps, "stop_reason": stop_reason, "outcome": deepcopy(outcome)}


def _finite_value(value):
    if type(value) not in (int, float) or (type(value) is float and not math.isfinite(value)):
        raise ValueError("outcome values must be finite numbers")
    return value


def replay_alternatives(open_episode, continuation, score, *, max_steps):
    """Force each possible next action on a fresh copy of one training situation.

    open_episode returns a context manager at the same prefix, with the same hidden
    world and external event schedule. The caller supplies a common continuation
    policy and a terminal scoring rule. Only the visible state is passed to policies.
    """
    if type(max_steps) is not int or max_steps < 1:
        raise ValueError("paired replays require a positive step budget")
    with open_episode() as source:
        if source.is_terminal():
            raise ValueError("cannot branch a terminal episode")
        state, candidates = deepcopy(source.observe())
    if not candidates:
        raise ValueError("paired replays require available candidates")
    replays = {}
    values = {}
    for action in candidates:
        first = True

        def branch(visible, offered):
            nonlocal first
            if first:
                if visible != state or offered != candidates:
                    raise ValueError("counterfactual replay started at a different observation")
                first = False
                return action
            return continuation(visible, offered)

        with open_episode() as episode:
            trace = run_episode(episode, branch, max_steps=max_steps)
        if first or not trace["steps"][0]["executed"]:
            raise ValueError("counterfactual replay did not execute its first action")
        replays[action] = trace
        values[action] = _finite_value(score(trace))
    return {"state": state, "candidates": candidates, "replays": replays, "values": values}


def outcome_labels(task, trials, *, action_order=None):
    """Convert complete, paired training replays into ordinary workshop training rows.

    Each trial supplies id, group, visible state, candidates and a finite higher-is-
    better terminal value per candidate. Equally observed worlds receive equal weight
    per trial. Ties use the supplied priority or the task label order. No test/dev
    evidence may contribute to targets.
    """
    validate_task(task)
    priority = task["labels"] if action_order is None else list(action_order)
    validate_candidates(task, priority)
    if set(priority) != set(task["labels"]):
        raise ValueError("action_order must include each task label")
    rows = []
    keys = []
    totals = {}
    required = {"id", "group", "state", "candidates", "values"}
    for trial in trials:
        if not isinstance(trial, dict) or set(trial) - {"split"} != required:
            raise ValueError("expected id, group, state, candidates and values per trial")
        if trial.get("split", "train") != "train":
            raise ValueError("outcome labels require training trials only")
        state = trial["state"]
        actions = trial["candidates"]
        values = trial["values"]
        validate_state(task, state)
        validate_candidates(task, actions)
        if not actions or not isinstance(values, dict) or set(values) != set(actions):
            raise ValueError("every available candidate needs a paired outcome value")
        key = (tuple(sorted(state.items())), frozenset(actions))
        scores = totals.setdefault(key, {})
        for action, value in values.items():
            # Exact sums make ties independent of trial order, including mixed signs.
            scores[action] = scores.get(action, 0) + Fraction(_finite_value(value))
        rows.append({"id": trial["id"], "group": trial["group"], "split": "train",
                     "state": dict(state), "candidates": list(actions)})
        keys.append(key)
    winners = {key: max((a for a in priority if a in scores), key=scores.__getitem__)
               for key, scores in totals.items()}
    for row, key in zip(rows, keys, strict=True):
        row["target"] = winners[key]
    validate_records(task, rows)
    return rows


def compare_verified(candidate, baseline, *, seed=86307, replicates=5000):
    """Compare verified job fractions on matched (workload group, case ID) keys.

    Each job has equal weight. Resampling whole groups preserves dependence
    between cases from the same workload. The interval is the percentile
    bootstrap of the paired difference on these observed groups.
    """
    if not candidate or candidate.keys() != baseline.keys():
        raise ValueError("comparison needs identical, nonempty (group, case ID) keys")
    if type(replicates) is not int or replicates < 2:
        raise ValueError("at least two bootstrap replicates are required")
    grouped = {}
    wins = losses = 0
    for key, status in candidate.items():
        if (not isinstance(key, tuple) or len(key) != 2
                or any(not isinstance(part, str) or not part.strip() for part in key)):
            raise ValueError("expected (workload group, case ID) keys")
        other = baseline[key]
        if status not in OUTCOMES or other not in OUTCOMES:
            raise ValueError("unknown verifier status in paired comparison")
        delta = int(status == "verified") - int(other == "verified")
        wins += delta > 0
        losses += delta < 0
        group = grouped.setdefault(key[0], [0, 0])
        group[0] += delta
        group[1] += 1
    groups = [grouped[name] for name in sorted(grouped)]
    rng = random.Random(seed)
    samples = []
    for _ in range(replicates):
        selected = rng.choices(groups, k=len(groups))
        changed, jobs = map(sum, zip(*selected))
        samples.append(changed / jobs)
    samples.sort()
    return {"jobs": len(candidate), "groups": len(groups), "candidate_only_verified": wins,
            "baseline_only_verified": losses, "difference": (wins - losses) / len(candidate),
            "bootstrap95": [samples[math.ceil(.025 * replicates) - 1], samples[math.ceil(.975 * replicates) - 1]]}
