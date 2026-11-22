"""Independently verify SQLite recovery evidence without consulting a worker's answer key."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path


class AuditError(ValueError):
    """A purported run disagrees with its independent evidence."""


def require(condition, explanation):
    if not condition:
        raise AuditError(explanation)


def read_ledger(path):
    """Reopen the persisted database read-only on a connection independent of the worker."""
    uri = Path(path).resolve().as_uri() + "?mode=ro"
    conn = sqlite3.connect(uri, uri=True)
    try:
        conn.row_factory = sqlite3.Row
        return {
            "requests": [dict(row) for row in conn.execute(
                "SELECT request_id, input_json, checkpoint_batches, subtotal, result FROM requests")],
            "completions": [dict(row) for row in conn.execute(
                "SELECT request_id, result FROM completions")],
        }
    finally:
        conn.close()


def inspect_ledger(request_id, batches, requests, completions):
    """Check original inputs, persisted progress and exactly-once completion on a one-job database."""
    work = [list(batch) for batch in batches]
    expected = sum(sum(batch) for batch in work)
    require(len(requests) <= 1, "more than one request row in a one-job database")
    require(all(r["request_id"] == request_id for r in [*requests, *completions]),
            "durable row belongs to a different operation")
    require(len(completions) <= 1, "duplicate completion effects")
    if not requests:
        require(not completions, "completion exists without a request")
        return False

    row = requests[0]
    persisted = json.loads(row["input_json"])
    require(persisted == work, "persisted input differs from the caller's batches")
    prefix = row["checkpoint_batches"]
    require(type(prefix) is int and 0 <= prefix <= len(work), "invalid saved checkpoint")
    require(row["subtotal"] == sum(sum(batch) for batch in work[:prefix]),
            "checkpoint subtotal differs from the original input prefix")
    if row["result"] is None:
        require(not completions, "completion was published before the result committed")
        return False
    require(prefix == len(work) and row["result"] == expected, "incorrect committed result")
    require(len(completions) == 1 and completions[0]["result"] == expected,
            "committed result lacks its correct completion effect")
    return True


def check_outcome(out):
    """Independently reconstruct caller, progress and effect evidence from a per-job receipt."""
    batches = out["batches"]
    ledger = out["ledger"]
    key = out["request_id"]
    total = sum(n for batch in batches for n in batch)
    committed = inspect_ledger(key, batches, ledger["requests"], ledger["completions"])
    caller = out["caller"]
    correct_reply = (isinstance(caller, dict) and caller.get("request_id") == key
                     and type(caller.get("result")) is int and caller["result"] == total)
    require(out["expected_result"] == total, "reported expected result disagrees with original batches")
    require(out["event_count"] == len(ledger["completions"]), "reported effect count disagrees with ledger")
    require(out["duplicate_effects"] == 0, "duplicate effect reported")
    if caller is not None:
        require(committed and correct_reply, "false success: caller result lacks matching durable evidence")
    verified = bool(caller is not None and correct_reply and committed and out["elapsed"] <= out["deadline"])
    require((out["status"] == "verified") == verified, "verified status disagrees with reply, ledger or deadline")
    if "on_time" in out:
        require(out["on_time"] == (out["elapsed"] <= out["deadline"]), "inconsistent deadline result")
    if "stop_reason" in out:
        status = "verified" if verified else "failed"
        if caller is None and out["stop_reason"] == "review" and out["elapsed"] <= out["deadline"]:
            status = "review"
        require(out["status"] == status, "terminal outcome bucket lacks matching caller evidence")
    if "work" in ledger:
        done = ledger["work"]
        require(all(type(i) is int and 0 <= i < len(batches) for i in done), "unknown batch in work log")
        require(len(done) == out["batches_processed"], "processed batch count disagrees with work log")
        require(len(done) - len(set(done)) == out["repeated_batches"], "incorrect repeated batch count")
    return verified


def paired_interval(group_differences):
    """Resample workloads as units, preserving their paired hidden-world variants."""
    import random

    means = [sum(ds) / len(ds) for _, ds in sorted(group_differences.items())]
    require(bool(means), "a paired interval needs at least one workload")
    n = len(means)
    rng = random.Random(86307)
    replicates = sorted(sum(rng.choices(means, k=n)) / n for _ in range(5000))
    return [replicates[124], replicates[4874]]


def paired_outcomes(rows, challenger, baseline):
    """Join on original case IDs. Reject missing, duplicated or mismatched worlds."""
    from collections import defaultdict

    require(challenger != baseline, "pair needs two different policy names")
    require(len({r["cohort"] for r in rows}) == 1, "mixing cohorts in a paired interval")
    cases = defaultdict(dict)
    for row in rows:
        policies = cases[row["case_id"]]
        require(row["policy"] not in policies, "duplicate episode for one case and policy")
        policies[row["policy"]] = row
    groups = defaultdict(list)
    counts = {"win": 0, "tie": 0, "loss": 0}
    for policies in cases.values():
        require(challenger in policies and baseline in policies, "unpaired case")
        new, old = policies[challenger], policies[baseline]
        require(new["group"] == old["group"] and new["scenario"] == old["scenario"],
                "paired policies did not start in the same world")
        d = int(check_outcome(new["outcome"])) - int(check_outcome(old["outcome"]))
        groups[new["group"]].append(d)
        counts[{1: "win", 0: "tie", -1: "loss"}[d]] += 1
    group_mean = sum(sum(ds) / len(ds) for ds in groups.values()) / len(groups)
    return {"cases": len(cases), "groups": len(groups), **counts,
            "group_mean_difference": group_mean, "bootstrap_95pct": paired_interval(groups)}


def latest_safe_lookup(state, candidates):
    """Spend slack on lookup while a complete retry remains possible; keep probing when retry is too late."""
    if state["known"] == "committed":
        return "return_result"
    if state["known"] == "checkpoint":
        return "resume"
    if state["known"] == "missing":
        return "retry"
    remaining = state["remaining"]
    full_retry = state["batches"] * state["batch_cost"]
    if full_retry <= remaining < full_retry + state["lookup_cost"]:
        return "retry"
    if state["lookup_cost"] == 0 and state["lookup_failures"] > 0:
        return "wait"
    return "lookup"


def live_lost_reply(path):
    """Compare a settled database with the caller's response before and after recovery."""
    from .worker import Episode, Scenario

    case = Scenario("audit-operation", ((7, -2), (-8,), (3,)), "lost_reply", deadline=4)
    episode = Episode(case, path=path)
    try:
        before = read_ledger(path)
        require(episode.caller is None and episode.timeouts == 1, "lost reply did not reach the caller")
        require(inspect_ledger(case.request_id, case.batches, **before), "no durable result after lost reply")
        episode.apply("lookup")
        episode.apply("return_result")
        outcome = episode.verify("terminal")
        require(check_outcome(outcome), "lost reply failed independent recovery check")
    finally:
        episode.close()
    after = read_ledger(path)
    require(before == after, "retrieving the result changed the durable output")
    return {"caller_before": None, "caller_after": outcome["caller"],
            "durable_before": before, "durable_after": after}


def check_case(row, model=None):
    """Verify the original request, charged ticks, visible decisions and each retained effect."""
    from decision_workshop.core import predict

    sc, out = row["scenario"], row["outcome"]
    require(sc["request_id"] == row["case_id"] == out["request_id"], "caller request changed")
    require(sc["batches"] == out["batches"] and sc["deadline"] == out["deadline"], "workload changed")
    verified = check_outcome(out)
    actions = [s["action"] for s in row["steps"] if s["executed"]]
    require(all(s["action"] in s["candidates"] for s in row["steps"] if s["executed"]),
            "policy executed an unavailable action")
    require(out["lookups"] == actions.count("lookup") and out["waits"] == actions.count("wait")
            and out["recovery_attempts"] == actions.count("retry") + actions.count("resume"),
            "observed action counts differ from receipt")
    initial = {"none": len(sc["batches"]), "lost_reply": len(sc["batches"]),
               "before": 0, "checkpoint": 1}[sc["fault"]]
    ticks = ((out["batches_processed"] - initial) * sc["batch_cost"]
             + out["lookups"] * sc["lookup_cost"] + out["waits"] * sc["wait_cost"])
    require(out["elapsed"] == ticks, "elapsed ticks differ from actual work and operations")
    if row["policy"] == "tree":
        for step in row["steps"]:
            expected = predict(model, step["state"], step["candidates"])
            require(step["decision"] == expected and step["action"] == expected["action"],
                    "recorded decision does not come from frozen checkpoint")
    remaining_work = (len(sc["batches"]) - initial) * sc["batch_cost"]
    possible = remaining_work <= sc["deadline"]
    require(not verified or possible, "verified job violates the physical work bound")
    return possible


def counts(rows):
    from collections import Counter

    counts = Counter(row["outcome"]["status"] for row in rows)
    return {"jobs": len(rows), **{s: counts[s] for s in ("verified", "incorrect", "failed", "review")},
            **{k: sum(r["outcome"][k] for r in rows)
               for k in ("repeated_batches", "duplicate_effects", "elapsed", "batches_processed", "lookups")}}


def run_extra(original, *, reuse):
    from time import perf_counter_ns
    from decision_workshop.sequence import run_episode
    from .worker import Episode, Scenario

    name = "reuse_retry" if reuse else "latest_safe_lookup"
    started = perf_counter_ns()
    ep = Episode(Scenario(**original["scenario"]), reuse_progress=reuse)
    try:
        policy = (lambda state, available: "retry") if reuse else latest_safe_lookup
        trace = run_episode(ep, policy, max_steps=16)
    finally:
        ep.close()
    out = {k: original[k] for k in ("cohort", "group", "case_id", "scenario")}
    return {**out, "policy": name, "host_ns": perf_counter_ns() - started, **trace}


def verify_training(root, model, freeze, protocol):
    from collections import defaultdict
    from decision_workshop.core import digest

    trials = json.loads((root / "trials.json").read_text())
    branches = {t["id"]: t for t in _lines(root / "training-alternatives.jsonl")}
    originals = {s["request_id"] for s in json.loads((root / "train-worlds.json").read_text())}
    grouped = defaultdict(lambda: defaultdict(int))
    require(digest(trials) == freeze["train_trials_sha256"], "unbound training trials")
    require(len(branches) == len(trials) == freeze["train_trials"], "missing training branches")
    def pool(t):
        return tuple(sorted(t["state"].items())), tuple(sorted(t["candidates"]))

    for t in trials:
        receipt = branches[t["id"]]
        require(receipt["case_id"] in originals and t["group"].startswith("train-"), "nontraining world")
        require(set(t["values"]) == set(receipt["branches"]) == set(t["candidates"]), "missing alternative")
        for action, branch in receipt["branches"].items():
            value = -1 if branch["status"] == "incorrect" or branch["duplicate_effects"] else int(branch["status"] == "verified")
            require(t["values"][action] == value and branch["actions"][0] == action, "wrong training target")
            grouped[pool(t)][action] += value
    labeled = []
    ties = 0
    for t in trials:
        sums = grouped[pool(t)]
        best = max(sums.values())
        preferred = [a for a in protocol["tie_priority"] if sums.get(a) == best]
        ties += len(preferred) > 1
        labeled.append({**{k: t[k] for k in ("id", "group", "candidates", "state")},
                        "split": "train", "target": preferred[0]})
    require(digest(sorted(labeled, key=lambda t: t["id"])) == model["training"]["records_sha256"],
            "frozen checkpoint training hash disagrees with independently derived targets")
    return {"trials": len(trials), "unique_observation_sets": len(grouped),
            "training_rows_with_tied_best": ties, "model_sha256": model["sha256"]}


def indistinguishable_pair():
    """Two hidden worlds can need opposite first moves at exactly the same observation."""
    from dataclasses import replace
    from decision_workshop.sequence import run_episode
    from .worker import Episode, Scenario

    pending = Scenario("hidden-pair", ((4, -2), (7,)), "before", deadline=4)
    completed = replace(pending, fault="lost_reply", retry_fault="lost_reply")
    state = None
    result = {}
    for name, sc in (("pending", pending), ("committed", completed)):
        result[name] = {}
        for first in ("lookup", "retry", "wait"):
            ep = Episode(sc)
            try:
                visible = ep.observe()
                if state is None:
                    state = visible
                require(visible == state, "hidden alternatives leaked at initial observation")
                ep.apply(first)
                trace = run_episode(ep, latest_safe_lookup, max_steps=15)
                good = check_outcome(trace["outcome"])
                require(good == ((name == "pending") == (first == "retry")), "paired tradeoff failed")
                result[name][first] = {"verified": good, "elapsed": trace["outcome"]["elapsed"],
                                       "durable_effects": trace["outcome"]["event_count"]}
            finally:
                ep.close()
    return {"visible": state, "first_actions": result,
            "hidden_difference": "initial fault phase and next worker response"}


def _lines(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def run(root, out):
    from collections import defaultdict
    from hashlib import sha256
    from decision_workshop.core import digest, load_model, write_new

    root, out = Path(root), Path(out)
    out.mkdir(parents=True, exist_ok=False)
    freeze, protocol, report = [json.loads((root / (k + ".json")).read_text())
                                for k in ("selection", "protocol", "summary")]
    model = load_model(root / "tree.json")
    require(digest(freeze) == report["selection_sha256"] and model["sha256"] == freeze["model_sha256"],
            "selection does not identify frozen checkpoint")
    code_root = Path(__file__).resolve().parents[2]
    for relative, expected_digest in protocol["files_sha256"].items():
        require(sha256((code_root / relative).read_bytes()).hexdigest() == expected_digest,
                f"source differs from frozen protocol: {relative}")
    verified_training = verify_training(root, model, freeze, protocol)
    chosen = freeze["selected_baseline"]
    def sorted_score(item):
        return item["verified"], -item["incorrect"], -item["duplicate_effects"], -item["elapsed"]
    for selected, table, priority in (
        (chosen, freeze["baselines_dev"], protocol["baseline_priority_on_tie"]),
        (freeze["selected_tree"], freeze["trees_dev"], [f"depth={d},leaf={l}" for d,l in protocol["tree_grid"]]),
        (freeze["cutoff_fit"], freeze["rule_train"], [f"margin={m},failures={n}" for m,n in protocol["rule_grid"]]),
    ):
        require(max(priority,key=lambda k: sorted_score(table[k])) == selected, "selection objective mismatch")
    result = {"training": verified_training, "baseline": chosen, "source_sha256": sha256(Path(__file__).read_bytes()).hexdigest(),
              "physical_lost_reply": live_lost_reply(out / "lost-reply.sqlite"),
              "indistinguishable_pair": indistinguishable_pair(), "cohorts": {}}
    final_jobs = {}
    for cohort in ("dev", "ordinary", "shift", "free"):
        source = root / "episodes" / (cohort + ".jsonl")
        rows = _lines(source)
        original = [r for r in rows if r["policy"] == "same_id_retry"]
        n = protocol["splits"]["dev" if cohort == "dev" else "test"][0]
        require(len(original) == 4*n and len(rows) == 4*len(original), "incorrect episode coverage")
        physical = {}
        faults = defaultdict(list)
        for row in original:
            sc = row["scenario"]
            faults[row["group"]].append(sc["fault"])
            physical[row["case_id"]] = {k:v for k,v in sc.items() if k not in ("lookup_cost","outage_until")}
        require(all(set(fs) == {"before","checkpoint","lost_reply","none"} and len(fs)==4 for fs in faults.values()),
                "unbalanced workload fault mix")
        if cohort == "ordinary":
            final_jobs = {r["case_id"]: r["scenario"] for r in original}
        elif cohort in ("shift", "free"):
            require(set(physical) == set(final_jobs), "cohorts do not share identical workloads")
            for row in original:
                old = final_jobs[row["case_id"]]
                expected = ({**old, "outage_until": old["outage_until"]+4} if cohort == "shift"
                            else {**old, "outage_until": 0, "lookup_cost": 0})
                require(expected == row["scenario"], "unexpected cross-cohort scenario change")
        extra = []
        for r in original:
            extra.extend([run_extra(r, reuse=False), run_extra(r, reuse=True)])
        for r in [*rows, *extra]:
            check_case(r, model)
        with (out / (cohort + ".jsonl")).open("x", encoding="utf-8") as f:
            for r in extra:
                f.write(json.dumps(r,separators=(",",":"))+"\n")
        joined = rows + extra
        pair = paired_outcomes(rows, "tree", chosen)
        reported = (report["development"] if cohort == "dev" else report["cohorts"][cohort])
        for name, expected_metrics in reported["policies"].items():
            got = counts([r for r in rows if r["policy"] == name])
            require(all(expected_metrics[k] == v for k,v in got.items()), "published policy totals differ")
        m = reported["paired"]
        require((m["tree_only_verified"],m["baseline_only_verified"],m["bootstrap95"],m["difference"])
                == (pair["win"],pair["loss"],pair["bootstrap_95pct"],pair["group_mean_difference"]),
                "published comparison disagrees with raw paired outcomes")
        result["cohorts"][cohort] = {
            "raw_sha256": sha256(source.read_bytes()).hexdigest(), "validated_rows": len(joined),
            "max_possible_by_minimum_work": sum(check_case(r) for r in original),
            "totals": {p: counts([r for r in joined if r["policy"] == p]) for p in dict.fromkeys(r["policy"] for r in joined)},
            "mean_local_episode_us": {p: sum(r["host_ns"] for r in joined if r["policy"] == p) / len(original) / 1000
                                      for p in dict.fromkeys(r["policy"] for r in joined)},
            "tree_vs_selected": pair,
            "tree_vs_latest_safe_lookup": paired_outcomes(joined, "tree", "latest_safe_lookup"),
            "reuse_vs_unchanged_same_id_retry": paired_outcomes(joined, "reuse_retry", "same_id_retry"),
        }
    write_new(out / "summary.json", result)
    return result


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True, help="benchmark directory with frozen receipts")
    parser.add_argument("--out", type=Path, required=True, help="new directory for audit and additional actual runs")
    args = parser.parse_args()
    audited = run(args.run, args.out)
    for cohort, metrics in audited["cohorts"].items():
        print(cohort, {name: v["verified"] for name,v in metrics["totals"].items()})
