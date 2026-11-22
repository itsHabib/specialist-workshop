"""Show what each policy bought on one archived batch and what the witness revealed."""

import argparse
import gzip
import json
from pathlib import Path

from .host import batches


def explain(run, batch_id, budget, policies):
    split = batch_id.partition("-")[0]
    if split not in ("dev", "test"):
        raise ValueError(f"expected a development or test batch, got {batch_id!r}")
    batch = next((b for b in batches(split) if b.id == batch_id), None)
    if batch is None:
        raise ValueError(f"batch {batch_id!r} is absent from the fixture")
    feeds = {f.id: f for f in batch.feeds}
    raw = run / "cases.jsonl"
    stream = raw.open(encoding="utf-8") if raw.is_file() else gzip.open(run / "cases.jsonl.gz", "rt")
    receipts = {}
    with stream:
        for line in stream:
            row = json.loads(line)
            if row["batch_id"] == batch_id and row["budget"] == budget and row["policy"] in policies:
                receipts[row["policy"]] = row
    missing = set(policies) - receipts.keys()
    if missing:
        raise ValueError(f"missing policies for {batch_id!r}: {sorted(missing)}")
    lines = [f"{batch.id}  |  {batch.family} channels  |  budget {budget}"]
    for policy in policies:
        row = receipts[policy]
        lines.append(f"\n{policy}  |  {row['native_exact']} → {row['final_exact']} exact dates of {row['rows']}")
        for i, step in enumerate(row["steps"], 1):
            feed = feeds[step["feed_id"]]
            wire, truth = feed.witness
            verdict = "contradicted" if step["wrong"] else "confirmed"
            lines.append(f"  {i}. {feed.id}  |  {wire} → {truth}  |  {verdict} {feed.hint} hint"
                         f"  |  {step['gain']} dates corrected")
    return "\n".join(lines)


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("batch_id", nargs="?", default="test-037")
    cli.add_argument("--budget", type=int, choices=(1, 2, 3), default=2)
    cli.add_argument("--policy", nargs="+", default=("largest", "empirical_adaptive", "empirical_lookahead"))
    cli.add_argument("--run", type=Path, default=Path(__file__).resolve().parent / "evidence")
    args = cli.parse_args()
    try:
        print(explain(args.run, args.batch_id, args.budget, args.policy))
    except ValueError as error:
        cli.error(str(error))
