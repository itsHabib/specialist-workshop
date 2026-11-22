"""Select a fixed release cohort and replay recorded Clef scores through the simulator."""

import argparse
from itertools import islice
import json
from pathlib import Path

from decision_workshop.clef_mlx_replay import file_hash
from decision_workshop.clef_reference import _receipt_scores
from decision_workshop.contracts import load_task, read_json, read_records
from decision_workshop.core import predict, write_new

from .release import episodes, rollout, summarize


COHORTS = {"dev": 2, "iid": 16, "shift": 8}


def prepare(source, packet):
    task = load_task(source / "task.json")
    rows = read_records(source / "records.jsonl", task)
    groups = {ep.id for family, size in COHORTS.items() for ep in islice(episodes(family), size)}
    selection = [r for r in rows if r["group"] in groups]
    if len(selection) != 3 * len(groups):
        raise ValueError("expected three observation prefixes per selected episode")
    packet.mkdir(parents=True, exist_ok=False)
    write_new(packet / "task.json", task)
    with (packet / "records.jsonl").open("x", encoding="utf8") as out:
        for row in selection:
            out.write(json.dumps(row) + "\n")


def evaluate(packet, receipt_path, out):
    task = load_task(packet / "task.json")
    rows = read_records(packet / "records.jsonl", task)
    receipt = read_json(receipt_path)
    scores = _receipt_scores(receipt, task, rows, {
        "task_sha256": file_hash(packet / "task.json"),
        "records_sha256": file_hash(packet / "records.jsonl"),
    })
    states = {r["id"]: r["state"] for r in rows}
    base = {"algorithm": "clef-mlx-reference", "task": task, "training": {"categories": {}}}

    def decision(row_id, state, candidates):
        if states[row_id] != state:
            raise ValueError(f"simulator state differs from recorded input on {row_id}")
        model = {**base, "_score": lambda _: scores[row_id]}
        return predict(model, state, candidates, min_score=0.0)

    result = {"receipt_sha256": receipt["sha256"], "cohorts": {}}
    for family in ("iid", "shift"):
        traces = [rollout(ep, decision) for ep in islice(episodes(family), COHORTS[family])]
        result["cohorts"][family] = {"metrics": summarize(traces), "episodes": traces}
    write_new(out, result)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prep = commands.add_parser("prepare")
    prep.add_argument("--source", type=Path, required=True)
    prep.add_argument("--packet", type=Path, required=True)
    score = commands.add_parser("score")
    score.add_argument("--packet", type=Path, required=True)
    score.add_argument("--receipt", type=Path, required=True)
    score.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "prepare":
        prepare(args.source, args.packet)
    else:
        evaluate(args.packet, args.receipt, args.out)


if __name__ == "__main__":
    main()
