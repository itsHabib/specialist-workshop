"""Train and run local decision policies. prepare-base downloads public weights."""

import argparse
import json
from pathlib import Path
import sys

from .contracts import load_task, read_json, read_records
from .core import load_model, predict, train, write_new
from .evaluation import evaluate


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
    maker = commands.add_parser("make", help="build a decision model and replay it against a tree baseline")
    maker.add_argument("task_directory", type=Path, help="folder containing task.json and records.jsonl")
    maker.add_argument("--out", type=Path, required=True, help="new run directory")
    maker.add_argument("--base", type=Path, default=Path(".state/base"))
    neural = commands.add_parser("train-bert", help="fine-tune verified BERT and its decision head offline")
    neural.add_argument("--task", type=Path, required=True)
    neural.add_argument("--data", type=Path, required=True)
    neural.add_argument("--base", type=Path, required=True)
    neural.add_argument("--model", type=Path, required=True, help="new checkpoint directory")
    for command in (maker, neural):
        command.add_argument("--epochs", type=int, default=12)
        command.add_argument("--batch-size", type=int, default=8)
        command.add_argument("--learning-rate", type=float, default=0.0003)
        command.add_argument("--seed", type=int, default=17)
    trainer = commands.add_parser("train", help="fit a CPU decision tree from local labeled records")
    trainer.add_argument("--task", type=Path, required=True)
    trainer.add_argument("--data", type=Path, required=True)
    trainer.add_argument("--model", type=Path, required=True, help="new JSON file")
    trainer.add_argument("--max-depth", type=int, default=5)
    trainer.add_argument("--min-leaf", type=int, default=2)
    trainer.add_argument("--balanced-ties", action="store_true",
                         help="prefer more evenly sized children when two splits have equal Gini gain")
    evaluator = commands.add_parser("evaluate", help="replay a held-out split")
    evaluator.add_argument("--model", type=Path, required=True)
    evaluator.add_argument("--data", type=Path, required=True)
    evaluator.add_argument("--split", choices=("dev", "test"), default="dev")
    evaluator.add_argument("--report", type=Path, help="new JSON file including per-case decisions")
    predictor = commands.add_parser("predict", help="score one request")
    predictor.add_argument("--model", type=Path, required=True)
    predictor.add_argument("--input", type=Path, required=True, help="JSON with state and candidates")
    for command in (evaluator, predictor):
        command.add_argument("--min-score", type=float, default=0.0,
                             help="refer a prediction for review below this score")
    args = parser.parse_args()
    try:
        if args.command == "init":
            from .scaffold import create_task
            create_task(args.task_directory, args.question, args.action, args.field)
            output = {"task": str(args.task_directory / "task.json"),
                      "add_reviewed_records_to": str(args.task_directory / "records.jsonl")}
        elif args.command == "make":
            from .maker import make
            result = make(args.task_directory, args.out, args.base, epochs=args.epochs, seed=args.seed,
                          batch_size=args.batch_size, learning_rate=args.learning_rate)
            output = {"run": str(args.out), "checkpoint": str(args.out / "model"),
                      "report": str(args.out / "report.json"), "split": result["split"],
                      "bert": result["bert"]["model"], "tree": result["tree"]["model"]}
        elif args.command == "prepare-base":
            from .base import MODEL_ID, REVISION, prepare_base
            path = prepare_base(args.out)
            output = {"model": MODEL_ID, "revision": REVISION, "path": str(path)}
        elif args.command == "train-bert":
            from .checkpoint import train_bert
            task = load_task(args.task)
            model = train_bert(task, read_records(args.data, task), args.base, args.model,
                               epochs=args.epochs, seed=args.seed, batch_size=args.batch_size,
                               learning_rate=args.learning_rate)
            output = {"model": str(args.model), "sha256": model["sha256"], "training_rows": model["training"]["rows"],
                      "base": model["base"]["model"], "optimization": model["optimization"]}
        elif args.command == "train":
            task = load_task(args.task)
            records = read_records(args.data, task)
            model = train(task, records, args.max_depth, args.min_leaf, balanced_ties=args.balanced_ties)
            write_new(args.model, model)
            output = {"model": str(args.model), "sha256": model["sha256"], "training_rows": model["training"]["rows"]}
        elif args.command == "evaluate":
            model = load_model(args.model)
            records = read_records(args.data, model["task"])
            report = evaluate(model, records, args.split, min_score=args.min_score)
            if args.report:
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
