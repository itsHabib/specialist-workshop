"""Audit the recorded decisions and claimed totals without loading any model."""

import argparse
import csv
import gzip
import io
import json
import random
import re
from collections import Counter, defaultdict
from hashlib import sha256
from itertools import product
from pathlib import Path

from .host import jobs


ROUTERS = ("hint", "tree", "bert")
ACTIONS = ("switch_delimiter", "switch_encoding")
LEARNED = tuple(product(ROUTERS, ("coarse", "diagnostic"), ("tree", "bert")))
CONTROLS = tuple(("control", "diagnostic", name) for name in ("rule", "none", "oracle"))
ARMS = (tuple((router, *policy) for router in ROUTERS for policy in LEARNED + CONTROLS)
        + tuple(("header", *policy) for policy in CONTROLS))
POLICY_KEYS = ("router", "training_source", "observation", "repair")
METRICS = ("jobs", "first_exact", "final_exact", "flagged", "repair_calls", "recovered",
           "regressed", "attainable", "parser_accepted_wrong")


def decode(job, route, action):
    """Independently parse wire bytes and compare all recovered text to the source."""
    if route not in ("comma", "semicolon") or action not in (None, *ACTIONS):
        raise ValueError(f"unexpected route or repair for {job.id}: {(route, action)}")
    delimiter = {"comma": ",", "semicolon": ";"}[route]
    if action == "switch_delimiter":
        delimiter = ";" if delimiter == "," else ","
    codec = "cp1252" if action == "switch_encoding" else "utf-8"
    try:
        text = job.payload.decode(codec)
    except UnicodeDecodeError:
        return "decode_error", False
    fields = ["ticket", "description", "units"]
    reader = csv.DictReader(io.StringIO(text, newline=""), delimiter=delimiter, strict=True)
    try:
        if reader.fieldnames != fields:
            return "shape_error", False
        rows = []
        for row in reader:
            if None in row or any(value is None for value in row.values()):
                return "shape_error", False
            if (not row["ticket"].startswith("T") or not row["description"]
                    or not re.fullmatch("[0-9]+", row["units"])):
                return "value_error", False
            rows.append(tuple(row[name] for name in fields))
    except csv.Error:
        return "shape_error", False
    if not rows:
        return "shape_error", False
    return "ok", tuple(rows) == job.expected


def verify_decision(row, outcomes):
    """Check a conditional two-attempt receipt against all available outcomes."""
    first, first_exact = outcomes[None]
    action = row["action"]
    if action not in outcomes or (first == "ok" and action is not None):
        raise ValueError("unexpected repair action or repair after parser success")
    final, final_exact = outcomes[action]
    attainable = first_exact or (first != "ok" and any(outcomes[a][1] for a in ACTIONS))
    for key, value in (("first_status", first), ("final_status", final),
                       ("first_exact", first_exact), ("final_exact", final_exact),
                       ("attainable", attainable)):
        if row[key] != value or type(row[key]) is not type(value):
            raise ValueError(f"incorrect {key}: {row[key]!r}, expected {value!r}")
    if row["repair"] == "none" and action is not None:
        raise ValueError("no-retry control performed a repair")
    if row["repair"] == "rule":
        rule = {"decode_error": "switch_encoding", "shape_error": "switch_delimiter"}
        if action != rule.get(first):
            raise ValueError("rule control used the wrong action for the error")
    if row["repair"] == "oracle":
        oracle = next((a for a in ACTIONS if first != "ok" and outcomes[a][1]), None)
        if action != oracle or final_exact != attainable:
            raise ValueError("oracle control disagrees with enumerated actions")
    return {"jobs": 1, "first_exact": first_exact, "final_exact": final_exact, "flagged": first != "ok",
            "repair_calls": action is not None, "recovered": final_exact and not first_exact,
            "regressed": first_exact and not final_exact, "attainable": attainable,
            "parser_accepted_wrong": final == "ok" and not final_exact}


def observation_ceiling(outcomes):
    """Hindsight bounds for a constant action, one action per error, and case truth."""
    first_exact = per_case = count = 0
    fixes = defaultdict(Counter)
    for attempts in outcomes:
        count += 1
        first, exact = attempts[None]
        first_exact += exact
        per_case += exact
        if first != "ok":
            successes = [action for action in ACTIONS if attempts[action][1]]
            fixes[first].update(successes)
            per_case += bool(successes)
    all_errors = sum(fixes.values(), Counter())
    return {"jobs": count, "first_exact": first_exact,
            "coarse": first_exact + max(all_errors.values(), default=0),
            "diagnostic": first_exact + sum(max(group.values(), default=0) for group in fixes.values()),
            "per_case": per_case}


def paired_contrast(jobs_in_order, outcomes):
    """Rebuild the reported tree-trained versus hint-trained paired contrast."""
    def describe(differences):
        counts = Counter(differences)
        return {"wins": counts[1], "losses": counts[-1], "ties": counts[0],
                "rate_difference": (counts[1] - counts[-1]) / len(differences)}

    groups = defaultdict(list)
    for job in jobs_in_order:
        delta = int(outcomes[job.id, "tree"]) - int(outcomes[job.id, "hint"])
        groups[job.family].append(delta)
    # Sample whole files within each family in generator order. The seed,
    # sample count and percentile ranks are fixed in the published protocol.
    rng = random.Random(814)
    size = len(jobs_in_order)
    bootstrap = []
    for _ in range(2000):
        net = 0
        for group in groups.values():
            n = len(group)
            net += sum(group[int(rng.random() * n)] for _ in range(n))
        bootstrap.append(net / size)
    bootstrap.sort()
    return {"all": describe([d for group in groups.values() for d in group]),
            "by_family": {family: describe(group) for family, group in groups.items()},
            "paired_95_percentile_interval": [bootstrap[49], bootstrap[1949]]}


def check(rows, summary, splits):
    """Require every expected job × seed × policy once, including referrals."""
    by_job = {(split, job.id): job for split, cohort in splits.items() for job in cohort}
    if len(by_job) != sum(map(len, splits.values())):
        raise ValueError("duplicate job ID in input fixture")
    seeds = summary["seeds"]
    if not seeds or any(type(seed) is not int for seed in seeds) or len(seeds) != len(set(seeds)):
        raise ValueError("empty or repeated seed")
    total_expected = len(by_job) * len(seeds) * len(ARMS)
    seen, routes, actions, outcomes = set(), {}, {}, {}
    contrast = defaultdict(dict)
    totals = defaultdict(Counter)
    for n, row in enumerate(rows, 1):
        try:
            split, seed, job_id = row["split"], row["seed"], row["job_id"]
            job = by_job[split, job_id]
            policy = tuple(row[k] for k in POLICY_KEYS)
            if seed not in seeds or type(seed) is not int or policy not in ARMS or row["family"] != job.family:
                raise ValueError("unexpected seed, policy or family")
            key = (split, job_id, seed, policy)
            if key in seen:
                raise ValueError("duplicate job/seed/policy")
            seen.add(key)
            route = row["route"]
            route_key = (split, job_id, seed, row["router"])
            if routes.setdefault(route_key, route) != route:
                raise ValueError("different route for the same frozen router and job")
            if row["router"] == "hint" and route != job.state["declared_hint"]:
                raise ValueError("hint router ignored supplied delimiter hint")
            if row["router"] == "header":
                header = job.payload.splitlines()[0]
                commas, semicolons = header.count(b","), header.count(b";")
                chosen = job.state["declared_hint"]
                if commas != semicolons:
                    chosen = "comma" if commas > semicolons else "semicolon"
                if route != chosen:
                    raise ValueError("header router ignored delimiter counts")
            attempt_key = (split, job_id, route)
            if attempt_key not in outcomes:
                outcomes[attempt_key] = {a: decode(job, route, a) for a in (None, *ACTIONS)}
            counts = verify_decision(row, outcomes[attempt_key])
            if split == "test" and policy in (("tree", "hint", "coarse", "tree"),
                                               ("tree", "tree", "coarse", "tree")):
                contrast[seed][job_id, policy[1]] = counts["final_exact"]
            if row["repair"] in ("tree", "bert") and row["first_status"] != "ok":
                signal = "failed" if row["observation"] == "coarse" else row["first_status"]
                learned_key = (seed, *policy[1:], signal)
                if actions.setdefault(learned_key, row["action"]) != row["action"]:
                    raise ValueError("learned repair changed for identical observable input")
            for family in (job.family, "all"):
                totals[split, family, seed, *policy].update(counts)
        except (ValueError, KeyError, TypeError) as error:
            raise ValueError(f"receipt {n}: {error}") from error
    if len(seen) != total_expected:
        raise ValueError(f"expected {total_expected} unique receipts, found {len(seen)}")
    compared = set()
    for table in summary["tables"]:
        key = tuple(table[k] for k in ("split", "family", "seed", *POLICY_KEYS))
        if key not in totals or key in compared:
            raise ValueError(f"unexpected or duplicate summary table {key}")
        compared.add(key)
        for metric in METRICS:
            if type(table[metric]) is not int or table[metric] != totals[key][metric]:
                raise ValueError(f"{key} {metric}: expected {totals[key][metric]}, got {table[metric]}")
    if compared != totals.keys():
        raise ValueError("summary omitted one or more expected tables")
    if "test" in splits:
        expected_contrast = {str(seed): paired_contrast(splits["test"], contrast[seed]) for seed in seeds}
        if json.dumps(summary.get("primary_paired_contrast"), sort_keys=True) != json.dumps(
                expected_contrast, sort_keys=True):
            raise ValueError("primary paired contrast differs from independent per-file reconstruction")
    test = defaultdict(list)
    for (split, job_id, seed, router), route in routes.items():
        if split == "test":
            test[seed, router].append(outcomes[split, job_id, route])
    ceilings = [{"seed": seed, "router": router, **observation_ceiling(test[seed, router])}
                for seed, router in sorted(test)]
    return {"verified_receipts": len(seen), "verified_tables": len(totals),
            "oracle_parses": len(outcomes) * 3, "test_observation_ceilings": ceilings}


def audit(run):
    splits = {name: jobs(name) for name in ("fit", "collect", "dev", "test")}
    all_jobs = [job for cohort in splits.values() for job in cohort]
    if len({job.id for job in all_jobs}) != len(all_jobs) or len({job.payload for job in all_jobs}) != len(all_jobs):
        raise ValueError("source jobs or byte payloads overlap")
    summary = json.loads((run / "summary.json").read_text(encoding="utf-8"))
    freeze = json.loads((run / "freeze.json").read_text(encoding="utf-8"))
    if summary["seeds"] != freeze["seeds"]:
        raise ValueError("summary does not cover the frozen seeds")
    fits_path = run / "fits" / "receipt.json"
    if not fits_path.is_file():
        fits_path = run / "fits.json"
    if sha256(fits_path.read_bytes()).hexdigest() != freeze["fits_sha256"]:
        raise ValueError("training receipt differs from frozen fits SHA-256")
    hashes = {}
    for split, cohort in splits.items():
        manifest = [(j.id, j.family, sha256(j.payload).hexdigest(), j.expected, j.state, j.route_target)
                    for j in cohort]
        encoded = json.dumps(manifest, separators=(",", ":"), sort_keys=True).encode()
        hashes[split] = sha256(encoded).hexdigest()
    if summary["fixture_sha256"] != hashes:
        raise ValueError("fixture SHA-256 differs from the generated cohorts")
    raw = run / "cases.jsonl"
    trace_file = raw.open("rb") if raw.is_file() else gzip.open(run / "cases.jsonl.gz", "rb")
    digest = sha256()

    def records(stream):
        for line in stream:
            digest.update(line)
            yield json.loads(line)

    with trace_file as stream:
        checked = check(records(stream), summary, {k: splits[k] for k in ("dev", "test")})
    if digest.hexdigest() != summary["cases_sha256"]:
        raise ValueError("uncompressed receipts differ from the recorded SHA-256")
    return checked


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True, type=Path)
    print(json.dumps(audit(parser.parse_args().run), indent=2))
