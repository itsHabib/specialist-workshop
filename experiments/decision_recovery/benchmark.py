"""Fit recovery decisions from paired training rollouts and check frozen policies on fresh jobs."""

import argparse
from collections import Counter, defaultdict, deque
from contextlib import contextmanager
from dataclasses import asdict, replace
import hashlib
import json
from pathlib import Path
import random
import time

from decision_workshop.core import digest, load_model, predict, train, write_new
from decision_workshop.contracts import read_json
from .policy import ACTIONS, TASK, known_action, status_first
from decision_workshop.sequence import compare_verified, outcome_labels, replay_alternatives, run_episode
from .worker import Episode, Scenario

MAX_STEPS = 16
COHORTS = ("ordinary", "shift", "free")
SPLITS = {"train": (64, 11713), "dev": (32, 21061), "test": (96, 31817)}
TREE_GRID = [(depth, leaf) for depth in (2, 4, 6) for leaf in (2, 8)]
RULE_GRID = [(slack, failed) for slack in (-2, 0, 1, 2, 4) for failed in (0, 1, 2)]
BASELINES = ("status_first", "same_id_retry", "fitted_rule")
PROTOCOL = {
    "splits": SPLITS,
    "seeded_controlled_fault_mix": ["before", "checkpoint", "lost_reply", "none"],
    "cohorts": COHORTS,
    "max_steps": MAX_STEPS,
    "tree_grid": TREE_GRID,
    "rule_grid": RULE_GRID,
    "tie_priority": ACTIONS,
    "prefix_depth": 2,
    "rule_fit_split": "train",
    "selection_split": "dev",
    "baseline_priority_on_tie": BASELINES,
    "selection": [
        "most verified",
        "fewest incorrect",
        "fewest duplicate effects",
        "least elapsed",
        "declared order",
    ],
    "outcome_utility": {"verified": 1, "incorrect_or_duplicate": -1, "other": 0},
    "bootstrap": {
        "seed": 86307,
        "replicates": 5000,
        "unit": "workload family",
        "sorted_indices": [124, 4874],
    },
}


def worlds(split, cohort="ordinary", *, seed=None):
    if cohort not in COHORTS:
        raise ValueError("unknown recovery cohort")
    count, default_seed = SPLITS[split]
    rng = random.Random(default_seed if seed is None else seed)
    namespace = split if seed is None else f"{split}-{seed}"
    for index in range(count):
        group = f"{namespace}-{index:03d}"
        n = rng.choice((2, 3, 4, 5))
        batches = tuple(tuple(rng.randint(-9, 15) for _ in range(rng.randint(2, 5))) for _ in range(n))
        batch_cost = rng.choice((1, 2))
        lookup_cost = rng.choice((1, 2, 3))
        deadline = max(2, n * batch_cost + rng.choice((-3, -1, 0, 1, 3)))
        outage = rng.choice((0, 0, 0, 2, 4, 6))
        wait_cost = rng.choice((1, 2))
        retry_fault = rng.choice(("none", "none", "none", "before", "checkpoint", "lost_reply"))
        for fault in ("before", "checkpoint", "lost_reply", "none"):
            case_id = f"{group}-{fault}"
            scenario = Scenario(
                request_id=case_id,
                batches=batches,
                fault=fault,
                deadline=deadline,
                batch_cost=batch_cost,
                lookup_cost=lookup_cost,
                outage_until=outage,
                wait_cost=wait_cost,
                retry_fault=retry_fault,
            )
            if cohort == "shift":
                scenario = replace(scenario, outage_until=outage + 4)
            elif cohort == "free":
                scenario = replace(scenario, outage_until=0, lookup_cost=0)
            yield group, case_id, scenario


def same_id_retry(state, candidates):
    return "retry"


def cutoff_policy(margin, failures):
    def choose(state, candidates):
        known = known_action(state)
        if known in candidates:
            return known
        full_work = state["batches"] * state["batch_cost"]
        slack = state["remaining"] - full_work
        if state["lookup_cost"] == 0 or slack < 0:
            return "lookup"
        if state["lookup_failures"] <= failures and slack >= state["lookup_cost"] + margin:
            return "lookup"
        return "retry"

    return choose


@contextmanager
def opened(scenario, prefix=()):
    ep = Episode(scenario)
    try:
        for action in prefix:
            ep.apply(action)
        yield ep
    finally:
        ep.close()


def value(trace):
    outcome = trace["outcome"]
    if outcome["status"] == "incorrect" or outcome["duplicate_effects"]:
        return -1
    return int(outcome["status"] == "verified")


def alternatives(scenario, prefix):
    return replay_alternatives(
        lambda: opened(scenario, prefix), status_first, value, max_steps=MAX_STEPS - len(prefix)
    )


def training_trials(cases, out):
    trials = []
    with (out / "training-alternatives.jsonl").open("x", encoding="utf-8") as log:
        for group, case_id, scenario in cases:
            queue, seen = deque([()]), set()
            while queue:
                prefix = queue.popleft()
                with opened(scenario, prefix) as ep:
                    if ep.is_terminal():
                        continue
                    state, candidates = ep.observe()
                key = digest([state, sorted(candidates)])
                if key in seen:
                    continue
                seen.add(key)
                result = alternatives(scenario, prefix)
                trial_id = f"{case_id}-{len(seen):02d}"
                trials.append(
                    {
                        "id": trial_id,
                        "group": group,
                        "state": state,
                        "candidates": candidates,
                        "values": result["values"],
                    }
                )
                branches = {
                    a: {
                        "actions": [step["action"] for step in trace["steps"]],
                        "status": trace["outcome"]["status"],
                        "elapsed": trace["outcome"]["elapsed"],
                        "duplicate_effects": trace["outcome"]["duplicate_effects"],
                    }
                    for a, trace in result["replays"].items()
                }
                log.write(
                    json.dumps(
                        {"id": trial_id, "case_id": case_id, "prefix": prefix, "branches": branches},
                        separators=(",", ":"),
                    )
                    + "\n"
                )
                if len(prefix) < 2:
                    queue.extend(prefix + (a,) for a in candidates)
    return trials


def episode(policy, scenario):
    started = time.perf_counter_ns()
    with opened(scenario) as ep:
        trace = run_episode(ep, policy, max_steps=MAX_STEPS)
    return trace, time.perf_counter_ns() - started


def totals(outcomes):
    statuses = Counter(o["status"] for o in outcomes)
    if set(statuses) - {"verified", "failed", "review", "incorrect"}:
        raise ValueError("unrecognized terminal status in policy outcomes")
    return {
        "jobs": len(outcomes),
        **{s: statuses[s] for s in ("verified", "review", "failed", "incorrect")},
        **{
            k: sum(o[k] for o in outcomes)
            for k in (
                "elapsed",
                "lookups",
                "lookup_failures",
                "waits",
                "recovery_attempts",
                "repeated_batches",
                "batches_processed",
                "duplicate_effects",
            )
        },
    }


def selection_key(metrics):
    return (metrics["verified"], -metrics["incorrect"], -metrics["duplicate_effects"], -metrics["elapsed"])


def select(candidates, world_list):
    scores = {
        name: totals([episode(policy, sc)[0]["outcome"] for _, _, sc in world_list])
        for name, policy in candidates.items()
    }
    return max(candidates, key=lambda name: selection_key(scores[name])), scores


def predictor(model):
    return lambda state, candidates: predict(model, state, candidates)


def paired(result, reference):
    comparison = compare_verified(
        result, reference, seed=PROTOCOL["bootstrap"]["seed"], replicates=PROTOCOL["bootstrap"]["replicates"]
    )
    comparison["tree_only_verified"] = comparison.pop("candidate_only_verified")
    return comparison


def evaluate_group(cohort, world_list, policies, ref, out):
    outputs = defaultdict(list)
    joined = defaultdict(dict)
    paired_examples = {}
    with (out / "episodes" / f"{cohort}.jsonl").open("x", encoding="utf-8") as stream:
        for group, case_id, sc in world_list:
            for name, policy in policies.items():
                trace, host_ns = episode(policy, sc)
                outcome = trace["outcome"]
                outputs[name].append(outcome)
                if (group, case_id) in joined[name]:
                    raise ValueError("duplicate outcome for policy and case")
                joined[name][(group, case_id)] = outcome["status"]
                row = {
                    "cohort": cohort,
                    "group": group,
                    "case_id": case_id,
                    "scenario": asdict(sc),
                    "policy": name,
                    "host_ns": host_ns,
                    **trace,
                }
                stream.write(json.dumps(row, separators=(",", ":")) + "\n")
                if cohort == "dev":
                    paired_examples[name, case_id] = row
    return {
        "policies": {name: totals(oo) for name, oo in outputs.items()},
        "paired": paired(joined["tree"], joined[ref]),
    }, paired_examples


def source_hashes():
    root = Path(__file__).resolve().parents[2]
    linked = (
        "experiments/decision_recovery/benchmark.py",
        "experiments/decision_recovery/worker.py",
        "experiments/decision_recovery/policy.py",
        "decision_workshop/sequence.py",
        "decision_workshop/tree.py",
        "decision_workshop/core.py",
    )
    return {relative: hashlib.sha256((root / relative).read_bytes()).hexdigest() for relative in linked}


def develop(out):
    out.mkdir(parents=True, exist_ok=False)
    (out / "episodes").mkdir()
    write_new(out / "protocol.json", {**PROTOCOL, "files_sha256": source_hashes()})
    write_new(out / "task.json", TASK)
    practice = list(worlds("train"))
    development = list(worlds("dev"))
    write_new(out / "train-worlds.json", [asdict(s) for _, _, s in practice])
    trials = training_trials(practice, out)
    write_new(out / "trials.json", trials)
    labeled = outcome_labels(TASK, trials)
    rule_candidates = {f"margin={m},failures={n}": cutoff_policy(m, n) for m, n in RULE_GRID}
    rule_name, rule_metrics = select(rule_candidates, practice)
    baselines = {
        "status_first": status_first,
        "same_id_retry": same_id_retry,
        "fitted_rule": rule_candidates[rule_name],
    }
    best, baselines_dev = select(baselines, development)
    trees = {
        f"depth={depth},leaf={leaf}": train(TASK, labeled, max_depth=depth, min_leaf=leaf)
        for depth, leaf in TREE_GRID
    }
    winner, trees_dev = select({k: predictor(v) for k, v in trees.items()}, development)
    write_new(out / "tree.json", trees[winner])
    freeze = {
        "selected_tree": winner,
        "model_sha256": trees[winner]["sha256"],
        "selected_baseline": best,
        "cutoff_fit": rule_name,
        "rule_train": rule_metrics,
        "baselines_dev": baselines_dev,
        "trees_dev": trees_dev,
        "task_sha256": digest(TASK),
        "train_trials_sha256": digest(trials),
        "train_trials": len(trials),
    }
    write_new(out / "selection.json", freeze)
    policies = {**baselines, "tree": predictor(load_model(out / "tree.json"))}
    development_report, dev_cases = evaluate_group("dev", development, policies, best, out)
    first = None
    contrast = None
    for _, case_id, sc in development:
        model_case = dev_cases["tree", case_id]
        baseline_case = dev_cases[best, case_id]
        if model_case["outcome"]["status"] != "verified":
            if first is None:
                first = case_id
            if baseline_case["outcome"]["status"] == "verified" and contrast is None:
                contrast = case_id
    diagnostic = {}
    selected_cases = {key for key in (first, contrast) if key}
    for _, cid, sc in development:
        if cid in selected_cases:
            branches = alternatives(sc, ())
            diagnostic[cid] = {
                a: {
                    "status": trace["outcome"]["status"],
                    "elapsed": trace["outcome"]["elapsed"],
                    "value": branches["values"][a],
                }
                for a, trace in branches["replays"].items()
            }
    write_new(
        out / "dev-errors.json",
        {
            "first_miss": first,
            "first_paired_loss": contrast,
            "initial_paired_branches_with_status_first_continuation": diagnostic,
            "cases": {
                i: {n: dev_cases[n, i] for n in ("tree", best)}
                for i in dict.fromkeys([first, contrast])
                if i is not None
            },
        },
    )
    write_new(out / "development.json", development_report)


def finalize(out):
    protocol = read_json(out / "protocol.json")
    if protocol["files_sha256"] != source_hashes():
        raise ValueError("frozen experiment source has changed since development")
    freeze = read_json(out / "selection.json")
    fitted = next(
        (cutoff_policy(m, n) for m, n in RULE_GRID if f"margin={m},failures={n}" == freeze["cutoff_fit"])
    )
    model = load_model(out / "tree.json")
    if model["sha256"] != freeze["model_sha256"]:
        raise ValueError("frozen model digest differs from development selection")
    best = freeze["selected_baseline"]
    policies = {
        "status_first": status_first,
        "same_id_retry": same_id_retry,
        "fitted_rule": fitted,
        "tree": predictor(model),
    }
    summary = {
        "selection_sha256": digest(freeze),
        "selected_baseline": best,
        "cohorts": {},
        "development": read_json(out / "development.json"),
    }
    for cohort in COHORTS:
        summary["cohorts"][cohort], _ = evaluate_group(cohort, worlds("test", cohort), policies, best, out)
    write_new(out / "summary.json", summary)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path(".state/recovery-benchmark"))
    parser.add_argument("--stage", choices=("develop", "finalize", "all"), default="all")
    args = parser.parse_args()
    if args.stage in ("develop", "all"):
        develop(args.out)
    if args.stage in ("finalize", "all"):
        finalize(args.out)
