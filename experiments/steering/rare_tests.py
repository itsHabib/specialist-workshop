"""Frozen rankers on singleton boundary regressions in executable phase counters."""
import argparse
import hashlib
from pathlib import Path
import random

from decision_workshop.contracts import read_json
from decision_workshop.core import load_model, write_new
from .test_selection import (
    BUDGETS, LIBRARY, Regression, candidate_inputs, cpu_batch, execute, features,
    make_function, opaque, ordering, passes, summarize,
)

OLD_OFFSETS = (26,19,15,4)
FRESH_OFFSETS = (24,20,13,7)
PROTOCOL = {
    "initial_offsets": OLD_OFFSETS, "fresh_offsets": FRESH_OFFSETS, "candidate_inputs": list(range(3,31)),
    "divisor": 32, "initial_directions": [1,0], "fresh_directions": [1,-1,0], "budgets": BUDGETS,
    "protocol": "Change the phase numerator by +1 or -1, or preserve it. Each input 3..30 is a distinct single exact test.",
    "queue": "Shuffle once per offset with seed rare-queue:<offset>. Stable queue order breaks identical feature ties.",
    "probe": "On fresh offsets prioritize any test whose clean result differs at x-1 or x+1; keep both directions tied. No mutation direction input.",
}


def suite(offset):
    name = f"phase_{offset}"
    body = f"return (x+{offset})//y"
    clean = make_function(name,body)
    LIBRARY[name] = clean
    rows = []
    for x in range(3,31):
        # The requirement counts scheduled events. Here only the first can occur.
        expected = sum(1 for due in range(32-offset,x+1,32))
        test = Regression(((name,x,32,expected),),1,4)
        if not passes(test,{name:clean}):
            raise ValueError("golden phase requirement disagrees with clean program")
        rows.append({"id":opaque(f"{name}:{x}"),"test":test,"features":features(test,name,2)})
    random.Random("rare-queue:"+str(offset)).shuffle(rows)
    if len({tuple(r["features"].items()) for r in rows}) != 1:
        raise ValueError("synthetic collision failed")
    return name,body,rows


def clean_probes(clean,cases):
    scores = {}
    for row in cases:
        _,x,y,expected = row["test"].calls[0]
        left, right = clean(x-1,y), clean(x+1,y)
        scores[row["id"]] = int(left != expected or right != expected)
    return scores


def available(cases,offset,edit):
    yield "full", cases
    renamed = [{**row,"id":opaque("new:"+row["id"])} for row in cases]
    yield "renamed", renamed
    shuffled = list(cases)
    random.Random(f"rare-reshuffle:{offset}").shuffle(shuffled)
    yield "shuffled", shuffled
    if edit:
        yield "failing_options_removed", [row for row in cases if not row["failed"]]


def run(prior,out):
    out.mkdir(parents=True,exist_ok=False)
    selected = read_json(prior/"rule-selection.json")["selected"]
    rankers = {k:load_model(prior/"rankers"/filename) for k,filename in (("tree","tree.json"),("bert","bert"))}
    protocol = {**PROTOCOL,"selected_rule":selected,"checkpoint_sha256":{k:v["sha256"] for k,v in rankers.items()},
                "source_sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    write_new(out/"protocol.json",protocol)
    evidence,oracle,construction = [],[],[]
    for arm,offsets in (("collision",OLD_OFFSETS),("clean_probes",FRESH_OFFSETS)):
        for offset in offsets:
            name,clean_body,rows = suite(offset)
            probe = {}
            if arm == "clean_probes":
                probe,cpu,iterations = cpu_batch(lambda: clean_probes(LIBRARY[name],rows))
                if sum(probe.values()) != 2:
                    raise ValueError("expected two candidate clean transition neighbors")
                construction.append({"offset":offset,"candidates":len(rows),"extra_clean_calls":2*len(rows),
                                     "host_cpu_ns":cpu,"timing_batch_executions":iterations})
            for edit in ((1,0) if arm == "collision" else (1,-1,0)):
                group = f"offset-{offset}-{edit:+d}"
                changed = f"return (x+{offset+edit})//y" if edit else f"return ((x+{offset})//y)"
                library = {name:make_function(name,changed)}
                labeled = [{**r,"failed":not passes(r["test"],library)} for r in rows]
                triggered = [r["test"].calls[0][1] for r in labeled if r["failed"]]
                expected = [31-offset] if edit == 1 else [32-offset] if edit == -1 else []
                if triggered != expected:
                    raise RuntimeError("executed singleton evidence disagrees with threshold calculation")
                for condition,cases in available(labeled,offset,edit):
                    candidates = candidate_inputs(cases)
                    by_id = {r["id"]:r for r in cases}
                    failures = {r["id"] for r in cases if r["failed"]}
                    # Match both versions of each opaque ID to the same PRE-CHANGE feature.
                    if probe:
                        probe_scores = {r["id"]:probe[opaque(f"{name}:{r['test'].calls[0][1]}")] for r in cases}
                    oracle.append({"arm":arm,"patch":group,"clean":clean_body,"changed":changed,"condition":condition,
                                   "expected_exposing_inputs":expected,"exhaustive_failures":sorted(failures),
                                   "cases":[{"id":r["id"],"x":r["test"].calls[0][1],"features":r["features"],
                                             "clean_probe":probe_scores[r["id"]] if probe else None} for r in cases]})
                    policies = ("discovery", selected,"tree","bert") + (("clean_transition",) if probe else ())
                    for policy in policies:
                        if policy == "clean_transition":
                            rank = lambda: sorted((i for i,_ in candidates),key=probe_scores.__getitem__,reverse=True)
                        else:
                            rank = lambda: ordering(policy,candidates,rankers)
                        order,rank_cpu,iterations = cpu_batch(rank)
                        for budget in BUDGETS:
                            result = execute(order,by_id,library,budget)
                            first = next((i for i in order[:budget] if i in failures),None)
                            if first != result["first_failure"]:
                                raise ValueError("actual prefix disagrees with exhaustive oracle")
                            evidence.append({"arm":arm,"group":group,"condition":condition,"policy":policy,
                                             "budget":budget,"reachable":bool(failures),"invariant":not edit,
                                             "rank_cpu_ns":rank_cpu,"ranking_batch_executions":iterations,
                                             "order":order[:budget],**result})
    report = {}
    for arm,condition,policy,budget in sorted({(r["arm"],r["condition"],r["policy"],r["budget"]) for r in evidence}):
        subset = [r for r in evidence if (r["arm"],r["condition"],r["policy"],r["budget"])==(arm,condition,policy,budget)]
        report.setdefault(arm,{}).setdefault(condition,{}).setdefault(policy,{})[budget] = summarize(subset)
    write_new(out/"oracle.json",oracle)
    write_new(out/"traces.json",evidence)
    write_new(out/"summary.json",{"results":report,"probe_construction":construction})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prior",type=Path,default=Path(".state/steering-tests"))
    parser.add_argument("--out",type=Path,default=Path(".state/steering-rare"))
    args=parser.parse_args()
    run(args.prior,args.out)
