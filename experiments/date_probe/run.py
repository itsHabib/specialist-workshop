"""Train decision models and replay a batch-wide budget for trusted date witnesses."""

import argparse
from collections import Counter, defaultdict
from hashlib import sha256
from importlib.metadata import version
import json
from pathlib import Path
import platform
import random

from decision_workshop.core import write_new
from .host import batches, free_history, observe, probe
from .learn import fit, state
from .policy import Candidate, plan


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
METHODS = ("none", "largest", "empirical_fixed", "empirical_adaptive", "empirical_lookahead",
           "tree_fixed", "tree_adaptive", "bert_fixed", "bert_adaptive", "oracle")
BUDGETS = (1, 2, 3)


def source_digests():
    names = ("PROTOCOL.md", "host.py", "learn.py", "policy.py", "run.py")
    paths = [HERE / name for name in names] + sorted((ROOT / "decision_workshop").glob("*.py"))
    return {p.relative_to(ROOT).as_posix(): sha256(p.read_bytes()).hexdigest() for p in paths}


def exact(rows, expected):
    if len(rows) != len(expected):
        raise ValueError("imported row count differs from the source")
    return sum(a == b for a, b in zip(rows, expected))


class Arrival:
    def __init__(self, batch):
        self.batch = batch
        self.by_id = {f.id: f for f in batch.feeds}
        self.native = {}
        self.roster = []
        self.initial = free_history(batch)
        self.rows = sum(len(f.expected) for f in batch.feeds)
        self.native_exact = 0
        self.misses = {}
        for feed in batch.feeds:
            parsed, differences, resolved = observe(feed)
            self.native[feed.id] = parsed
            correct = exact(parsed, feed.expected)
            self.native_exact += correct
            if not resolved:
                self.roster.append(Candidate(feed.id, feed.channel, feed.notice, differences))
                # The offline grader and its oracle control are the only consumers.
                self.misses[feed.id] = len(parsed) - correct

    def replay(self, method, budget, scorers):
        pending, history = list(self.roster), list(self.initial)
        delivered, steps = dict(self.native), []
        for i in range(min(budget, len(pending)) if method != "none" else 0):
            if method == "oracle":
                selected = max(pending, key=lambda c: self.misses[c.id])
            elif method == "largest":
                selected = max(pending, key=lambda c: c.differences)
            else:
                learner, mode = method.split("_")
                scorer = scorers[learner]

                def chance(c, observations):
                    return scorer(state(c, observations))

                if mode == "lookahead":
                    selected = plan(pending, history, chance, budget - i)
                else:
                    seen = history if mode == "adaptive" else self.initial
                    selected = max(pending, key=lambda c: c.differences * chance(c, seen))
            pending.remove(selected)
            feed = self.by_id[selected.id]
            delivered[feed.id], wrong = probe(feed)
            history.append((feed.channel, wrong))
            gain = exact(delivered[feed.id], feed.expected) - exact(self.native[feed.id], feed.expected)
            if gain != wrong * selected.differences:
                raise ValueError("witness outcome and corrected row count disagree")
            steps.append({"feed_id": feed.id, "wrong": wrong, "gain": gain})
        final = sum(exact(delivered[f.id], f.expected) for f in self.batch.feeds)
        upper = self.native_exact + sum(sorted(self.misses.values(), reverse=True)[:budget])
        return {"batch_id": self.batch.id, "family": self.batch.family, "budget": budget, "policy": method,
                "steps": steps, "rows": self.rows, "native_exact": self.native_exact, "final_exact": final,
                "remaining_wrong": self.rows - final, "oracle_exact": upper, "ambiguous_feeds": len(self.roster)}


def paired(differences):
    rng = random.Random(713)
    n = len(differences)
    draws = sorted(sum(rng.choices(differences, k=n)) / n for _ in range(2000))
    return {"batches": n, "wins": sum(v > 0 for v in differences), "ties": differences.count(0),
            "losses": sum(v < 0 for v in differences), "additional_exact_rows": sum(differences),
            "mean_rows_per_batch": sum(differences) / n, "paired_95_interval": [draws[49], draws[1949]]}


def run(out, base):
    out.mkdir(parents=True, exist_ok=False)
    code = source_digests()
    scorers, _ = fit(out, base)
    fit_digest = sha256((out / "fits/receipt.json").read_bytes()).hexdigest()
    write_new(out / "freeze.json", {"code": code, "fits_sha256": fit_digest, "budgets": BUDGETS,
                                    "policies": METHODS, "python": platform.python_version(),
                                    "packages": {n: version(n) for n in ("torch", "transformers", "safetensors")}})
    cohorts = {split: batches(split) for split in ("fit", "dev", "test")}
    feeds = [f for cohort in cohorts.values() for batch in cohort for f in batch.feeds]
    if len(feeds) != len({f.id for f in feeds}) or len(feeds) != len({f.payload for f in feeds}):
        raise ValueError("feeds overlap between or within cohorts")
    totals, grouped = defaultdict(Counter), defaultdict(list)
    with (out / "cases.jsonl").open("x", encoding="utf-8") as output:
        for split in ("dev", "test"):
            for batch in cohorts[split]:
                arrival = Arrival(batch)
                for budget in BUDGETS:
                    finals = {}
                    for method in METHODS:
                        row = arrival.replay(method, budget, scorers)
                        row["split"] = split
                        output.write(json.dumps(row, separators=(",", ":")) + "\n")
                        finals[method] = row["final_exact"]
                        totals[split, batch.family, budget, method].update(
                            batches=1, rows=row["rows"], native_exact=row["native_exact"],
                            final_exact=row["final_exact"], remaining_wrong=row["remaining_wrong"],
                            oracle_exact=row["oracle_exact"], probes=len(row["steps"]),
                            helpful_probes=sum(s["gain"] > 0 for s in row["steps"]),
                            regret=row["oracle_exact"] - row["final_exact"])
                    if split == "test" and budget == 2:
                        grouped[batch.family].append(finals["empirical_adaptive"] - finals["empirical_fixed"])
    if code != source_digests():
        raise ValueError("source changed during evaluation")
    summary = {"cases_sha256": sha256((out / "cases.jsonl").read_bytes()).hexdigest(),
               "primary_paired": {family: paired(v) for family, v in grouped.items()},
               "tables": [{"split": split, "family": family, "budget": b, "policy": p, **counts}
                          for (split, family, b, p), counts in sorted(totals.items())]}
    write_new(out / "summary.json", summary)
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True, help="new local output directory")
    parser.add_argument("--base", type=Path, default=Path(".state/base"))
    args = parser.parse_args()
    print(json.dumps(run(args.out, args.base)["primary_paired"], indent=2))
