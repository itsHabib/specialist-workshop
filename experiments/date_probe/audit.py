"""Replay probe choices and exact date outcomes with the Python standard library."""

import argparse
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime
from functools import lru_cache
import gzip
from hashlib import sha256
import json
import math
from pathlib import Path
import random

from .host import batches


POLICIES = ("none", "largest", "oracle", "empirical_fixed", "empirical_adaptive",
            "tree_fixed", "tree_adaptive", "bert_fixed", "bert_adaptive", "empirical_lookahead")
BUDGETS = (1, 2, 3)
FORMATS = {"mdy": "%m/%d/%Y", "dmy": "%d/%m/%Y"}


@dataclass(frozen=True)
class CheckedFeed:
    id: str
    channel: str
    notice: bool
    rows: int
    disputed: int
    wrong: bool
    free_bit: bool | None

    @property
    def gain(self):
        return self.disputed if self.wrong else 0


def inspect(feed):
    """Try both complete interpretations. Check the purchased witness against the source."""
    header, *wire = feed.payload.decode("ascii").splitlines()
    if not wire or header != "feed_id=" + feed.id or len(wire) != len(feed.expected):
        raise ValueError(f"invalid feed header or row count: {feed.id}")
    parses = {}
    for order, fmt in FORMATS.items():
        try:
            parses[order] = tuple(datetime.strptime(s, fmt).date().isoformat() for s in wire)
        except ValueError:
            pass
    if not parses or feed.hint not in FORMATS:
        raise ValueError(f"no feasible date convention or invalid hint: {feed.id}")
    raw, reference = feed.witness
    positions = [i for i, value in enumerate(wire) if value == raw]
    if not positions:
        raise ValueError(f"unmatched witness: {feed.id}")
    supported = {order for order, values in parses.items() if all(values[i] == reference for i in positions)}
    if not supported or any(parses[order] != feed.expected for order in supported):
        raise ValueError(f"witness disagrees with source rows: {feed.id}")
    baseline = parses.get(feed.hint, next(iter(parses.values())))
    disputed = sum(a != b for a, b in zip(*parses.values())) if len(parses) == 2 else 0
    wrong = baseline != feed.expected
    gain = sum(a != b for a, b in zip(baseline, feed.expected))
    if gain != (disputed if wrong else 0):
        raise ValueError(f"source has inconsistent convention: {feed.id}")
    free_bit = (feed.hint not in parses) if len(parses) == 1 else None
    return CheckedFeed(feed.id, feed.channel, feed.notice, len(wire), disputed, wrong, free_bit)


def probabilities(receipt):
    scores = {}
    for row in receipt["states"]:
        key = row["notice"], row["peer_outcome"]
        if type(key[0]) is not bool or key in scores:
            raise ValueError("invalid or duplicate score state")
        scores[key] = row["scores"]
        for kind in ("tree", "bert", "empirical"):
            p = scores[key][kind]
            if type(p) not in (float, int) or not math.isfinite(p) or not 0 <= p <= 1:
                raise ValueError("invalid frozen probability")
    expected = {(n, signal) for n in (False, True) for signal in ("unseen", "right", "wrong", "mixed")}
    if set(scores) != expected:
        raise ValueError("incomplete frozen score table")
    return scores


def signal_for(feed, history):
    bits = {wrong for channel, wrong in history if channel == feed.channel}
    if len(bits) == 2:
        return "mixed"
    if bits:
        return "wrong" if True in bits else "right"
    return "unseen"


def verify_training(cohort, receipt):
    """Rebuild the finite empirical table exclusively from the designated fit batches."""
    counts = defaultdict(Counter)
    seen_groups = set()
    for batch in cohort:
        files = [inspect(f) for f in batch.feeds]
        free = [(f.channel, f.free_bit) for f in files if f.free_bit is not None]
        unresolved = [f for f in files if f.disputed]
        if unresolved:
            seen_groups.add(batch.id)
        for target in unresolved:
            history = free.copy()
            peers = [f for f in unresolved if f != target and f.channel == target.channel][:2]
            for peer in [None, *peers]:
                if peer is not None:
                    history.append((peer.channel, peer.wrong))
                counts[target.notice, signal_for(target, history)][target.wrong] += 1
    total = sum(counts.values(), Counter())
    if total.total() != receipt["training_rows"] or len(seen_groups) != receipt["groups"]:
        raise ValueError("unexpected training population")
    for row in receipt["states"]:
        cell = counts[row["notice"], row["peer_outcome"]]
        if (row["train_rows"], row["train_prefer"]) != (cell.total(), cell[True]):
            raise ValueError("empirical training counts disagree with labeled source files")
        notice_group = sum((v for k, v in counts.items() if k[0] == row["notice"]), Counter())
        level, pool = next((name, c) for name, c in [("state", cell), ("notice", notice_group), ("global", total)] if c)
        if row["empirical_source"] != level or row["scores"]["empirical"] != (1 + pool[True]) / (2 + pool.total()):
            raise ValueError("incorrect empirical chance")
    return total.total()


def probability(feed, history, scores, scorer):
    return scores[feed.notice, signal_for(feed, history)][scorer]


def plan(roster, history, scores, horizon):
    """Independently evaluate the expected gain of each root across all probe outcome branches."""
    @lru_cache(maxsize=None)
    def best(available, bits, k):
        if not k or not available:
            return None, 0.0
        choices = []
        for i in available:
            feed = roster[i]
            p = probability(feed, bits, scores, "empirical")
            value = p * feed.disputed
            if k > 1:
                remaining = tuple(j for j in available if j != i)
                for observed, weight in [(True, p), (False, 1 - p)]:
                    next_bits = bits | {(feed.channel, observed)}
                    value += weight * best(remaining, next_bits, k - 1)[1]
            choices.append((i, value))
        peak = max(value for _, value in choices)
        return next((i, value) for i, value in choices if peak - value <= 1e-10)

    return roster[best(tuple(range(len(roster))), frozenset(history), horizon)[0]]


def check_case(row, feeds, scores):
    eligible = [f for f in feeds if f.disputed]
    budget, policy = row["budget"], row["policy"]
    free = [(f.channel, f.free_bit) for f in feeds if f.free_bit is not None]
    history = list(free)
    spent = min(len(eligible), budget) if policy != "none" else 0
    if type(row["steps"]) is not list or len(row["steps"]) != spent:
        raise ValueError("wrong number of paid probes")
    total = sum(f.rows for f in feeds)
    native = total - sum(f.gain for f in feeds)
    oracle = native + sum(sorted((f.gain for f in eligible), reverse=True)[:budget])
    recovered = helpful = 0
    for i, step in enumerate(row["steps"]):
        if policy == "largest":
            chosen = max(eligible, key=lambda f: f.disputed)
        elif policy == "oracle":
            chosen = max(eligible, key=lambda f: f.gain)
        elif policy == "empirical_lookahead":
            chosen = plan(eligible, history, scores, budget - i)
        else:
            scorer, mode = policy.split("_")
            seen = free if mode == "fixed" else history
            chosen = max(eligible, key=lambda f: f.disputed * probability(f, seen, scores, scorer))
        if (step["feed_id"] != chosen.id or step["wrong"] is not chosen.wrong
                or type(step["gain"]) is not int or step["gain"] != chosen.gain):
            raise ValueError(f"probe {i + 1} disagrees with independently chosen feed or decoded outcome")
        eligible.remove(chosen)
        history.append((chosen.channel, chosen.wrong))
        recovered += chosen.gain
        helpful += chosen.gain > 0
    final = native + recovered
    expected = {"rows": total, "native_exact": native, "final_exact": final, "remaining_wrong": total - final,
                "oracle_exact": oracle, "ambiguous_feeds": sum(f.disputed > 0 for f in feeds)}
    for key, value in expected.items():
        if type(row[key]) is not int or row[key] != value:
            raise ValueError(f"incorrect {key} for {row['batch_id']}")
    return {"rows": total, "native_exact": native, "final_exact": final, "remaining_wrong": total - final,
            "oracle_exact": oracle, "probes": spent, "helpful_probes": helpful, "regret": oracle - final, "batches": 1}


def paired(cohort, observed):
    result = {}
    families = defaultdict(list)
    for batch in cohort:
        fixed, adaptive = (observed[batch.id, name] for name in ("empirical_fixed", "empirical_adaptive"))
        families[batch.family].append(adaptive - fixed)
    for family, changes in families.items():
        rng = random.Random(713)
        n = len(changes)
        boot = sorted(sum(changes[int(rng.random() * n)] for _ in range(n)) / n for _ in range(2000))
        result[family] = {"batches": n, "wins": sum(d > 0 for d in changes), "ties": changes.count(0),
                          "losses": sum(d < 0 for d in changes), "additional_exact_rows": sum(changes),
                          "mean_rows_per_batch": sum(changes) / n, "paired_95_interval": [boot[49], boot[1949]]}
    return result


def check(rows, summary, splits, scores):
    source = {(split, batch.id): (batch.family, tuple(inspect(f) for f in batch.feeds))
              for split, cohort in splits.items() for batch in cohort}
    totals, seen, measured = defaultdict(Counter), set(), {}
    for row in rows:
        split, ident, budget, policy = (row[k] for k in ("split", "batch_id", "budget", "policy"))
        key = split, ident, budget, policy
        if ((split, ident) not in source or type(budget) is not int or budget not in BUDGETS
                or policy not in POLICIES or key in seen):
            raise ValueError(f"unexpected or duplicate receipt: {key}")
        seen.add(key)
        family, feeds = source[split, ident]
        if row["family"] != family:
            raise ValueError(f"incorrect batch family: {key}")
        counts = check_case(row, feeds, scores)
        totals[split, family, budget, policy].update(counts)
        if split == "test" and budget == 2 and policy in ("empirical_fixed", "empirical_adaptive"):
            measured[ident, policy] = counts["final_exact"]
    expected = len(source) * len(BUDGETS) * len(POLICIES)
    if len(seen) != expected:
        raise ValueError(f"missing cases: found {len(seen)} of {expected}")
    counted = set()
    for table in summary["tables"]:
        key = tuple(table[k] for k in ("split", "family", "budget", "policy"))
        if key in counted or key not in totals:
            raise ValueError(f"unexpected summary row: {key}")
        counted.add(key)
        for name, count in totals[key].items():
            if type(table[name]) is not int or table[name] != count:
                raise ValueError(f"wrong summary {name}: {key}")
    if counted != set(totals):
        raise ValueError("missing summary rows")
    if "test" in splits and summary.get("primary_paired") != paired(splits["test"], measured):
        raise ValueError("incorrect primary paired contrast or interval")
    return {"verified_receipts": len(seen), "verified_tables": len(totals),
            "verified_probe_selections": sum(t["probes"] for t in totals.values()),
            "independently_parsed_feeds": sum(len(feeds) for _, feeds in source.values())}


def audit(run):
    with (run / "summary.json").open() as stream:
        summary = json.load(stream)
    fit_path = run / "fits" / "receipt.json"
    if not fit_path.is_file():
        fit_path = run / "fits.json"
    fit_bytes = fit_path.read_bytes()
    freeze = json.loads((run / "freeze.json").read_bytes())
    if (freeze["fits_sha256"] != sha256(fit_bytes).hexdigest() or set(freeze["policies"]) != set(POLICIES)
            or freeze["budgets"] != list(BUDGETS)):
        raise ValueError("score receipt, policies or budgets differ from the freeze")
    fit_receipt = json.loads(fit_bytes)
    score_table = probabilities(fit_receipt)
    splits = {s: batches(s) for s in ("fit", "dev", "test")}
    ids, payloads, batch_ids = set(), set(), set()
    for cohort in splits.values():
        for batch in cohort:
            if batch.id in batch_ids:
                raise ValueError("repeated batch ID")
            batch_ids.add(batch.id)
            for feed in batch.feeds:
                if feed.id in ids or feed.payload in payloads:
                    raise ValueError("overlapping feed identity or bytes")
                ids.add(feed.id)
                payloads.add(feed.payload)
    raw = run / "cases.jsonl"
    source = raw.open("rb") if raw.exists() else gzip.open(run / "cases.jsonl.gz", "rb")
    digest = sha256()

    def records():
        for line in source:
            digest.update(line)
            yield json.loads(line)

    with source:
        result = check(records(), summary, {k: splits[k] for k in ("dev", "test")}, score_table)
    if digest.hexdigest() != summary["cases_sha256"]:
        raise ValueError("uncompressed trace SHA-256 mismatch")
    result["verified_training_rows"] = verify_training(splits["fit"], fit_receipt)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    print(json.dumps(audit(parser.parse_args().run), indent=2))
