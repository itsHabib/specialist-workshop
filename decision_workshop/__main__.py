"""Train and run local decision policies. prepare-base downloads public weights."""

import argparse
import json
from pathlib import Path
import sys

from .contracts import load_task, read_json, read_records
from .core import load_model, predict, train, write_new
from .evaluation import evaluate


def training_input(task, args):
    if args.outcomes is None:
        return read_records(args.data, task)
    from .sequence import outcome_labels
    trials = read_json(args.outcomes)
    if not isinstance(trials, list):
        raise ValueError("outcomes file must contain an array of paired training trials")
    return outcome_labels(task, trials)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser("prepare-base", help="download and verify the pinned public Google BERT files")
    prepare.add_argument("--out", type=Path, required=True)
    starter = commands.add_parser("init", help="create a new decision definition and empty data file")
    starter.add_argument("task_directory", type=Path)
    starter.add_argument("--question", required=True)
    starter.add_argument("--action", action="append", required=True, metavar="ID=DESCRIPTION")
    starter.add_argument("--field", action="append", metavar="NAME=TYPE")
    importer = commands.add_parser("import", help="validate records from a local adapter and write new JSONL")
    importer.add_argument("--task", type=Path, required=True)
    importer.add_argument("--adapter", required=True, metavar="MODULE:FUNCTION")
    importer.add_argument("--source", type=Path, help="source path passed to the adapter (default: None)")
    importer.add_argument("--out", type=Path, required=True, help="new records file")
    maker = commands.add_parser("make", help="build a decision model and replay it against a tree baseline")
    maker.add_argument("task_directory", type=Path, help="folder containing task.json and records.jsonl")
    maker.add_argument("--out", type=Path, required=True, help="new run directory")
    maker.add_argument("--base", type=Path, default=Path(".state/base"))
    neural = commands.add_parser("train-bert", help="fine-tune verified BERT and its decision head offline")
    neural.add_argument("--task", type=Path, required=True)
    neural.add_argument("--base", type=Path, required=True)
    neural.add_argument("--model", type=Path, required=True, help="new checkpoint directory")
    for command in (maker, neural):
        command.add_argument("--epochs", type=int, default=12)
        command.add_argument("--batch-size", type=int, default=8)
        command.add_argument("--learning-rate", type=float, default=0.0003)
        command.add_argument("--seed", type=int, default=17)
    trainer = commands.add_parser("train", help="fit a CPU decision tree from local labeled records")
    trainer.add_argument("--task", type=Path, required=True)
    for command in (trainer, neural):
        source = command.add_mutually_exclusive_group(required=True)
        source.add_argument("--data", type=Path, help="local labeled JSONL records")
        source.add_argument("--outcomes", type=Path, help="JSON array of paired action outcomes from training jobs")
    trainer.add_argument("--model", type=Path, required=True, help="new JSON file")
    trainer.add_argument("--max-depth", type=int, default=5)
    trainer.add_argument("--min-leaf", type=int, default=2)
    trainer.add_argument("--balanced-ties", action="store_true",
                         help="prefer more evenly sized children when two splits have equal Gini gain")
    for command in (trainer, neural, maker):
        command.add_argument("--min-score", type=float, default=0.0,
                             help="store this decision cutoff in the checkpoint (default: 0.0)")
    evaluator = commands.add_parser("evaluate", help="replay a held-out split")
    evaluator.add_argument("--model", type=Path, required=True)
    evaluator.add_argument("--data", type=Path, required=True)
    evaluator.add_argument("--split", choices=("dev", "test"), default="dev")
    evaluator.add_argument("--report", type=Path, help="new JSON file including per-case decisions")
    host = commands.add_parser("compare-host", help="compare verified host outcomes on declared arrivals")
    host.add_argument("--input", type=Path, required=True, help="arrival IDs, paired receipts, metric and direction")
    host.add_argument("--report", type=Path, required=True, help="new JSON file including per-arrival outcomes")
    predictor = commands.add_parser("predict", help="score one request")
    predictor.add_argument("--model", type=Path, required=True)
    predictor.add_argument("--input", type=Path, required=True, help="JSON with state and candidates")
    for command in (evaluator, predictor):
        command.add_argument("--min-score", type=float,
                             help="override the checkpoint cutoff (required for older checkpoints without one)")
    args = parser.parse_args()
    try:
        if args.command == "init":
            from .scaffold import create_task
            create_task(args.task_directory, args.question, args.action, args.field)
            output = {"task": str(args.task_directory / "task.json"),
                      "add_reviewed_records_to": str(args.task_directory / "records.jsonl")}
        elif args.command == "import":
            from .importer import import_records
            output = import_records(load_task(args.task), args.adapter, args.source, args.out)
        elif args.command == "make":
            from .maker import make
            result = make(args.task_directory, args.out, args.base, epochs=args.epochs, seed=args.seed,
                          batch_size=args.batch_size, learning_rate=args.learning_rate, min_score=args.min_score)
            output = {"run": str(args.out), "checkpoint": str(args.out / "model"),
                      "report": str(args.out / "report.json"), "split": result["split"],
                      "min_score": result["min_score"], "tree_diagnostics": result["tree"]["diagnostics"],
                      "bert": result["bert"]["model"], "tree": result["tree"]["model"],
                      "with_recorded_candidates": {
                          name: result[name]["model_with_recorded_candidates"] for name in ("bert", "tree")
                      }}
        elif args.command == "prepare-base":
            from .base import MODEL_ID, REVISION, prepare_base
            path = prepare_base(args.out)
            output = {"model": MODEL_ID, "revision": REVISION, "path": str(path)}
        elif args.command == "train-bert":
            from .checkpoint import train_bert
            task = load_task(args.task)
            model = train_bert(task, training_input(task, args), args.base, args.model,
                               epochs=args.epochs, seed=args.seed, batch_size=args.batch_size,
                               learning_rate=args.learning_rate, min_score=args.min_score)
            output = {"model": str(args.model), "sha256": model["sha256"], "training_rows": model["training"]["rows"],
                      "min_score": model["min_score"], "base": model["base"]["model"], "optimization": model["optimization"]}
        elif args.command == "train":
            task = load_task(args.task)
            records = training_input(task, args)
            model = train(task, records, args.max_depth, args.min_leaf,
                          balanced_ties=args.balanced_ties, min_score=args.min_score)
            write_new(args.model, model)
            output = {"model": str(args.model), "sha256": model["sha256"], "training_rows": model["training"]["rows"],
                      "min_score": model["min_score"], "tree_diagnostics": model["tree_diagnostics"]}
        elif args.command == "evaluate":
            model = load_model(args.model)
            records = read_records(args.data, model["task"])
            report = evaluate(model, records, args.split, min_score=args.min_score)
            if args.report:
                write_new(args.report, report)
            output = {key: value for key, value in report.items() if key != "cases"}
        elif args.command == "compare-host":
            from .outcomes import compare_host_outcomes
            bundle = read_json(args.input)
            if not isinstance(bundle, dict) or set(bundle) != {"arrivals", "baseline", "candidate", "metric", "direction"}:
                raise ValueError("host comparison requires arrivals, baseline, candidate, metric and direction")
            if not all(isinstance(bundle[field], list) for field in ("arrivals", "baseline", "candidate")):
                raise ValueError("arrivals, baseline and candidate must be arrays")
            report = compare_host_outcomes(**bundle)
            write_new(args.report, report)
            output = {key: value for key, value in report.items() if key != "cases"}
        else:
            model = load_model(args.model)
            request = read_json(args.input)
            if not isinstance(request, dict) or set(request) != {"state", "candidates"}:
                raise ValueError("a prediction request must contain state and candidates")
            output = predict(model, request["state"], request["candidates"], min_score=args.min_score)
        print(json.dumps(output, indent=2, allow_nan=False))
    except ModuleNotFoundError as exc:
        print(f"error: {exc}. For the BERT trainer install requirements-decisions.txt.", file=sys.stderr)
        return 1
    except (ValueError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
