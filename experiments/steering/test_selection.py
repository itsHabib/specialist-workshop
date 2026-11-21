"""Rank portable regression tests for changes to small synthetic integer utilities.

Run from the checkout root: python -m experiments.steering.test_selection --base .state/base
"""
import argparse
from collections import Counter
from dataclasses import dataclass
from functools import lru_cache
import hashlib
from pathlib import Path
import platform
import random
import statistics
import sys
import time

from decision_workshop.core import write_new
from decision_workshop.steering import fit_rankers, rank_candidates

# Golden Python implementations. Each replacement changes the marked expression.
# Values of x range over non-negative counts; y is a positive divisor or limit.
CATALOG = {
    "train": {
        "pages": ("if x == 0: return 0\nreturn (x+y-1)//y", "(x+y-1)//y", ("x//y", "(x+y)//y")),
        "excess": ("if x < y: return 0\nreturn x-y", "x-y", ("x-y+1", "max(0,x-2*y)")),
        "bundle": ("if x < y: return x\nreturn x-(x//y)", "x-(x//y)", ("x-((x+1)//y)", "x-(x%y)")),
        "rollover": ("if x < y: return x\nreturn x%y", "x%y", ("(x+1)%y", "min(x,y)")),
        "tickets": ("if x == 0: return 0\nreturn min(x,y)", "min(x,y)", ("min(x,y+1)", "max(x,y)")),
        "buffer": ("if x < y: return y\nreturn x+y", "x+y", ("x+y-1", "max(x,y)")),
    },
    "dev": {
        "index": ("if x == 0: return 0\nreturn (x-1)//y", "(x-1)//y", ("x//y", "(x+1)//y")),
        "discount": ("if x <= y: return x\nreturn y+(x-y)//2", "y+(x-y)//2", ("y+(x-y+1)//2", "x//2")),
        "remainder": ("if x >= y: return 0\nreturn y-x", "y-x", ("y-x-1", "y")),
    },
    "test": {
        "padding": ("if x == 0: return 0\nreturn (-x)%y", "(-x)%y", ("y-x%y", "x%y")),
        "overflow": ("if x < y: return 0\nreturn 1+(x-y)//y", "1+(x-y)//y", ("1+(x-y+1)//y", "(x-y)//y")),
        "allowance": ("if x <= y: return y-x\nreturn 0", "y-x", ("max(0,y-x-1)", "y")),
        "countdown": ("if x <= y: return 0\nreturn (x-y+1)//2", "(x-y+1)//2", ("(x-y)//2", "x-y")),
        "segments": ("if x < 2: return 0\nreturn (x-1)%y", "(x-1)%y", ("x%y", "(x-1)//y")),
        "bump": ("if x < y: return x+1\nreturn y", "x+1", ("x", "min(x+2,y)")),
    },
}
BUDGETS = (1, 5, 20)
RULES = ("line_and_assertion", "coverage_per_cost", "boundary_per_cost")
PROTOCOL = {
    "budgets": BUDGETS, "seed": 19031, "tests_per_suite": 28, "implementations_per_family": 3,
    "splits": {s: list(items) for s,items in CATALOG.items()},
    "rule_selection": "Maximize summed development detections at 1, 5, 20; then minimize total modeled cost to failure or budget; use declaration order for ties.",
    "cost_model": "Illustrative units: one per test plus call count times a deterministic fixture-load factor (one to five).",
    "oracle": "Run every test against each mutant; evaluate each policy by actually executing selected tests until first failure or budget exhaustion.",
    "conditions": ["full", "renamed", "shuffled", "failing_options_removed"],
    "time": "Both ranking and prefix CPU use the median of 3 batch means. Adaptive calibration requires at least 500,000 process CPU nanoseconds per batch. Modeled fixture cost is separate.",
}


def make_function(name, body):
    code = "def operation(x, y):\n" + "".join("    "+line+"\n" for line in body.splitlines())
    namespace = {}
    exec(compile(code, f"<synthetic/{name}>", "exec"), namespace)
    return namespace["operation"]


LIBRARY = {name: make_function(name, body) for families in CATALOG.values() for name,(body,_,_) in families.items()}


@lru_cache(maxsize=None)
def baseline_call(name, x, y):
    function = LIBRARY[name]
    lines = set()
    def trace(frame, event, arg):
        if event == "line" and frame.f_code is function.__code__:
            lines.add(frame.f_lineno)
        return trace
    previous = sys.gettrace()
    try:
        sys.settrace(trace)
        expected = function(x, y)
    finally:
        sys.settrace(previous)
    return expected, frozenset(lines)


@dataclass(frozen=True)
class Regression:
    calls: tuple
    bucket: int
    cost: int


def regression_suite(name, split):
    # Use helpers from this split so held-out utility code never runs in a training example.
    helpers = sorted(set(CATALOG[split]) - {name})
    rng = random.Random(f"19031:{name}")
    tests = []
    for slot in range(PROTOCOL["tests_per_suite"]):
        count = rng.randint(1, 6)
        bucket = rng.choice((1,1,1,2,4))
        calls = []
        for j in range(count):
            component = name if slot < 20 and (j == 0 or rng.random() < 0.58) else rng.choice(helpers)
            y = rng.randrange(2,9)
            x = rng.choice((0, 1, y-1, y, y+1, 2*y, 2*y+1, rng.randrange(35)))
            expected, _ = baseline_call(component, x, y)
            calls.append((component,x,y,expected//bucket))
        cost = 1 + count * rng.randrange(1,6)
        tests.append(Regression(tuple(calls),bucket,cost))
    return tests


def passes(test, library):
    # Each stored value is a golden assertion. A coarse assertion observes only a bucket.
    for component,x,y,expected in test.calls:
        if library[component](x,y)//test.bucket != expected:
            return False
    return True


def features(test, changed, changed_line):
    calls = [(x,y) for name,x,y,_ in test.calls if name == changed]
    hits = [(x,y) for x,y in calls if changed_line in baseline_call(changed,x,y)[1]]
    edges = sum(x in (0,1,y-1,y,y+1,2*y) for x,y in hits)
    return {"changed_function_calls": len(calls), "changed_line_calls": len(hits),
            "distinct_line_inputs": len(set(hits)), "line_boundary_inputs": edges,
            "assertion_bucket": test.bucket, "fixture_cost_units": test.cost, "calls": len(test.calls)}


def rule_score(rule, f):
    n = f["distinct_line_inputs"]
    strength = 1/f["assertion_bucket"]
    cost = f["fixture_cost_units"]
    if rule == "line_and_assertion":
        return (n > 0, strength, n, -cost)
    if rule == "coverage_per_cost":
        return n * strength / cost
    return (n + f["line_boundary_inputs"]) * strength / cost


def opaque(s):
    return hashlib.sha256(s.encode()).hexdigest()[:14]


def patches(split):
    for name,(body,marked,replacements) in CATALOG[split].items():
        tests = regression_suite(name, split)
        if not all(passes(t, LIBRARY) for t in tests):
            raise RuntimeError("inconsistent golden test")
        changed_line = next(i for i,line in enumerate(body.splitlines(),2) if marked in line)
        for idx,replacement in enumerate((*replacements, f"({marked})")):
            group = f"{name}-{idx}"
            library = {**LIBRARY, name: make_function(name,body.replace(marked,replacement,1))}
            order = list(range(len(tests)))
            random.Random("discovery:"+group).shuffle(order)
            cases = []
            for slot in order:
                test = tests[slot]
                cases.append({"id": opaque(group+":"+str(slot)), "test": test, "features": features(test,name,changed_line),
                              "failed": not passes(test,library)})
            yield {"group": group, "family": name, "invariant": idx == 2, "cases": cases,
                   "library": library, "old": body, "new": body.replace(marked,replacement,1)}


def candidate_inputs(cases):
    return [(case["id"],case["features"]) for case in cases]


def ordering(policy, candidates, models):
    # Only the caller-visible ID and pre-change features cross the ranker boundary.
    if policy == "discovery":
        return [identifier for identifier,_ in candidates]
    if policy in RULES:
        base = sorted(candidates,key=lambda item: tuple(item[1][k] for k in sorted(item[1])))
        return [identifier for identifier,_ in sorted(base,key=lambda item:rule_score(policy,item[1]),reverse=True)]
    return [identifier for identifier,_ in rank_candidates(models[policy],candidates)]


def cpu_batch(operation, windows=3):
    """Median of per-execution CPU batch means; no window uses a zero/short reading."""
    iterations = 1
    samples = []
    while len(samples) < windows:
        start = time.process_time_ns()
        for _ in range(iterations):
            value = operation()
        elapsed = time.process_time_ns() - start
        if elapsed < 500_000:
            iterations *= max(2, min(32, (500_000 + max(elapsed,1) - 1) // max(elapsed,1)))
            samples.clear()
            continue
        samples.append(elapsed / iterations)
    return value, statistics.median(samples), iterations


def execute(order, by_id, library, budget):
    def run_prefix():
        cost = 0
        for index,test_id in enumerate(order[:budget],1):
            test = by_id[test_id]["test"]
            cost += test.cost
            if not passes(test, library):
                return test_id,index,cost
        return None,min(len(order),budget),cost
    result, cpu_ns, iterations = cpu_batch(run_prefix)
    first,tried,cost = result
    return {"first_failure":first,"tests_executed":tried,"modeled_units":cost,
            "host_cpu_ns":cpu_ns,"timing_batch_executions":iterations}



def choose_rule(dev):
    scores = {}
    for rule in RULES:
        detections, cost = 0, 0
        for patch in dev:
            cases = patch["cases"]
            order = ordering(rule,candidate_inputs(cases),{})
            by_id = {r["id"]:r for r in cases}
            for budget in BUDGETS:
                result = execute(order,by_id,patch["library"],budget)
                detections += result["first_failure"] is not None
                cost += result["modeled_units"]
        scores[rule] = {"detections": detections, "modeled_units": cost}
    selected = max(RULES,key=lambda r:(scores[r]["detections"],-scores[r]["modeled_units"]))
    return selected,scores


def variants(patch):
    cases = patch["cases"]
    yield "full", cases
    renamed = [{**r, "id":opaque("renamed:"+r["id"])} for r in cases]
    yield "renamed", renamed
    shuffled = list(cases)
    random.Random("shuffle:"+patch["group"]).shuffle(shuffled)
    yield "shuffled", shuffled
    if not patch["invariant"]:
        yield "failing_options_removed", [r for r in cases if not r["failed"]]


def summarize(rows):
    hit = [r for r in rows if r["first_failure"] is not None]
    reachable = sum(r["reachable"] for r in rows)
    return {"patches":len(rows),"reachable":reachable,"detected":len(hit),"missed_reachable":reachable-len(hit),
            "unreachable":len(rows)-reachable, "equivalent_edits":sum(r["invariant"] for r in rows),
            "mean_tests_executed":statistics.mean(r["tests_executed"] for r in rows),
            "mean_modeled_units_to_stop":statistics.mean(r["modeled_units"] for r in rows),
            "mean_host_cpu_us_to_stop":statistics.mean(r["host_cpu_ns"]/1000 for r in rows),
            "mean_modeled_units_to_detection":statistics.mean(r["modeled_units"] for r in hit) if hit else None,
            "mean_host_cpu_us_to_detection":statistics.mean(r["host_cpu_ns"]/1000 for r in hit) if hit else None,
            "mean_rank_cpu_ms":statistics.mean(r["rank_cpu_ns"]/1e6 for r in rows)}


def run(base,out):
    out.mkdir(exist_ok=False,parents=True)
    write_new(out/"protocol.json",{**PROTOCOL,"source_sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                                   "python":platform.python_version(),"machine":platform.machine()})
    training = list(patches("train"))
    dev = list(patches("dev"))
    rule,scores = choose_rule(dev)
    write_new(out/"rule-selection.json",{"selected":rule, "development":scores})
    practice = [{"group":p["group"],"features":r["features"],"preferred":r["failed"]}
                for p in training for r in p["cases"]]
    models = fit_rankers("synthetic_regression_test_priority",practice,out/"rankers",base)
    evidence, oracle = [], []
    for patch in patches("test"):
        for condition,cases in variants(patch):
            by_id = {r["id"]:r for r in cases}
            failing = [r["id"] for r in cases if r["failed"]]
            oracle.append({"group":patch["group"],"family":patch["family"],"invariant":patch["invariant"],
                           "old":patch["old"],"new":patch["new"],"condition":condition,"available":list(by_id),
                           "exhaustive_failures":failing,"cases":[{"id":r["id"],"features":r["features"],"calls":r["test"].calls}
                                                                 for r in cases]})
            for policy in ("discovery",rule,"tree","bert"):
                candidates = candidate_inputs(cases)
                order,rank_cpu,ranking_repeats = cpu_batch(lambda: ordering(policy,candidates,models))
                if set(order) != set(by_id) or len(order) != len(by_id):
                    raise ValueError("ranker lost or introduced candidates")
                for budget in BUDGETS:
                    result = execute(order,by_id,patch["library"],budget)
                    expected = next((item for item in order[:budget] if by_id[item]["failed"]),None)
                    if result["first_failure"] != expected:
                        raise RuntimeError("execution trace disagrees with exhaustive test oracle")
                    evidence.append({"group":patch["group"],"condition":condition,"invariant":patch["invariant"],
                                     "reachable":bool(failing),"budget":budget,"policy":policy,"rank_cpu_ns":rank_cpu,
                                     "ranking_batch_executions":ranking_repeats, "ordered_prefix":order[:budget], **result})
    summary = {}
    for condition in ("full","renamed","shuffled","failing_options_removed"):
        summary[condition] = {}
        for policy in ("discovery",rule,"tree","bert"):
            summary[condition][policy] = {
                budget:summarize([r for r in evidence if (r["condition"],r["policy"],r["budget"])==(condition,policy,budget)])
                for budget in BUDGETS}
    write_new(out/"oracle.json",oracle)
    write_new(out/"traces.json",evidence)
    write_new(out/"summary.json",{"selected_rule":rule,"training_candidates":len(practice),
                                 "training_failures":sum(x["preferred"] for x in practice),"results":summary})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base",type=Path,default=Path(".state/base"))
    parser.add_argument("--out",type=Path,default=Path(".state/steering-tests"))
    args=parser.parse_args()
    run(args.base,args.out)
