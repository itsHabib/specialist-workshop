"""Train paired routing/recovery specialists and measure exact CSV imports."""

import argparse
from collections import Counter, defaultdict
from functools import lru_cache
from hashlib import sha256
from importlib.metadata import version
import json
from pathlib import Path
import platform
import random

from decision_workshop.core import write_new
from .fit import RECOVERY_TASK, train_policies
from .host import attempt, jobs


ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
SEEDS = (17, 29, 43)
RULE = {"decode_error": "switch_encoding", "shape_error": "switch_delimiter"}


def code_identity():
    files = [HERE / leaf for leaf in ("run.py", "host.py", "fit.py", "PROTOCOL.md")]
    files.extend(sorted((ROOT / "decision_workshop").glob("*.py")))
    return {p.relative_to(ROOT).as_posix(): sha256(p.read_bytes()).hexdigest() for p in files}


def fixture_identity(cohort):
    entries = []
    for job in cohort:
        entries.append([job.id, job.family, sha256(job.payload).hexdigest(), job.expected,
                        job.state, job.route_target])
    encoded = json.dumps(entries, sort_keys=True, separators=(",", ":")).encode()
    return sha256(encoded).hexdigest()


def header_route(job):
    state = job.state
    difference = state["comma_count"] - state["semicolon_count"]
    if difference == 0:
        return state["declared_hint"]
    return "comma" if difference > 0 else "semicolon"


def replay(job, policies):
    # Reuse identical parses for this immutable file. Every policy pays for its own calls.
    @lru_cache(maxsize=6)
    def parse(route, action):
        status, rows = attempt(job, route, action)
        return status, status == "ok" and rows == job.expected

    for router, choose in {**policies["routes"], "header": header_route}.items():
        route = choose(job)
        first_status, first_exact = parse(route, None)
        # Only the grading ceiling and the labeled oracle arm see this outcome lookup.
        best = None
        if first_status != "ok":
            best = next((a for a in RECOVERY_TASK["labels"] if parse(route, a)[1]), None)
        attainable = first_exact or best is not None
        learned = policies["repairs"] if router != "header" else {}
        handlers = {**learned, ("control", "diagnostic", "none"): lambda status: None,
                    ("control", "diagnostic", "rule"): RULE.get,
                    ("control", "diagnostic", "oracle"): lambda status: best}
        for (source, observation, repair), handler in handlers.items():
            action = handler(first_status) if first_status != "ok" else None
            final_status, final_exact = parse(route, action)
            yield {"job_id": job.id, "family": job.family, "router": router, "route": route,
                   "training_source": source, "observation": observation, "repair": repair,
                   "first_status": first_status, "first_exact": first_exact, "action": action,
                   "final_status": final_status, "final_exact": final_exact, "attainable": attainable}


def metrics(row):
    first, final = row["first_exact"], row["final_exact"]
    return {"jobs": 1, "first_exact": first, "final_exact": final,
            "flagged": row["first_status"] != "ok", "repair_calls": row["action"] is not None,
            "recovered": final and not first, "regressed": first and not final,
            "attainable": row["attainable"], "parser_accepted_wrong": row["final_status"] == "ok" and not final}


def primary_contrast(pairs):
    """Pair two fixed tree recovery arms by file. Keep seed repeats separate."""
    by_family = defaultdict(list)
    for (family, _), outcomes in pairs.items():
        by_family[family].append(int(outcomes["tree"]) - int(outcomes["hint"]))
    combined = [value for values in by_family.values() for value in values]
    rng = random.Random(814)
    draws = sorted(sum(sum(rng.choices(values, k=len(values))) for values in by_family.values())
                   / len(combined) for _ in range(2000))

    def counts(values):
        return {"wins": values.count(1), "losses": values.count(-1), "ties": values.count(0),
                "rate_difference": sum(values) / len(values)}

    return {"all": counts(combined), "by_family": {f: counts(v) for f, v in by_family.items()},
            "paired_95_percentile_interval": [draws[49], draws[1949]]}


def run(out, base, seeds):
    if len(set(seeds)) != len(seeds):
        raise ValueError("each seed must be unique")
    out.mkdir(parents=True, exist_ok=False)
    code = code_identity()
    policies = train_policies(out, base, seeds=seeds)
    receipt = out / "fits" / "receipt.json"
    seal = {"seeds": seeds, "code": code, "fits_sha256": sha256(receipt.read_bytes()).hexdigest(),
            "runtime": {"python": platform.python_version(), "architecture": platform.machine(),
                        **{p: version(p) for p in ("torch", "transformers", "safetensors")}}}
    write_new(out / "freeze.json", seal)
    cohorts = {s: jobs(s) for s in ("fit", "collect", "dev", "test")}
    inputs = [j for cohort in cohorts.values() for j in cohort]
    if len({j.payload for j in inputs}) != len(inputs) or len({j.id for j in inputs}) != len(inputs):
        raise ValueError("partitions share a file or identifier")
    totals = defaultdict(Counter)
    pairs = defaultdict(lambda: defaultdict(dict))
    with (out / "cases.jsonl").open("x", encoding="utf-8") as stream:
        for split in ("dev", "test"):
            for job in cohorts[split]:
                for seed in seeds:
                    for row in replay(job, policies[seed]):
                        row.update(seed=seed, split=split)
                        stream.write(json.dumps(row, separators=(",", ":")) + "\n")
                        arm = tuple(row[k] for k in ("router", "training_source", "observation", "repair"))
                        for family in (job.family, "all"):
                            totals[split, family, seed, *arm].update(metrics(row))
                        if split == "test" and arm in (("tree", "tree", "coarse", "tree"),
                                                        ("tree", "hint", "coarse", "tree")):
                            pairs[seed][job.family, job.id][row["training_source"]] = row["final_exact"]
    if code_identity() != code or sha256(receipt.read_bytes()).hexdigest() != seal["fits_sha256"]:
        raise ValueError("inputs changed during evaluation")
    fields = ("split", "family", "seed", "router", "training_source", "observation", "repair")
    report = {"seeds": seeds, "fixture_sha256": {s: fixture_identity(c) for s, c in cohorts.items()},
              "cases_sha256": sha256((out / "cases.jsonl").read_bytes()).hexdigest(),
              "primary_paired_contrast": {str(seed): primary_contrast(pairs[seed]) for seed in seeds},
              "tables": [{**dict(zip(fields, key)), **counts} for key, counts in sorted(totals.items())]}
    write_new(out / "summary.json", report)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True, type=Path, help="new local run directory")
    parser.add_argument("--base", default=Path(".state/base"), type=Path)
    parser.add_argument("--seeds", nargs="+", default=SEEDS, type=int)
    args = parser.parse_args()
    result = run(args.out, args.base, args.seeds)
    print(json.dumps(result["primary_paired_contrast"], indent=2))
