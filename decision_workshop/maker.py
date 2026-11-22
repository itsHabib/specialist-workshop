"""Build and compare local specialists from a decision definition and reviewed records."""

import json
import os
from pathlib import Path
import tempfile

from .checkpoint import train_bert
from .contracts import load_task, read_records
from .core import load_model, train, write_new
from .evaluation import evaluate


def make(task_dir: Path, output: Path, base: Path, *, epochs=12, batch_size=8,
         learning_rate=0.0003, seed=17, min_score=0.0):
    """Publish both checkpoints and their development comparisons as one run."""
    task_dir = Path(task_dir)
    output = Path(output)
    if os.path.lexists(output):
        raise FileExistsError(f"output already exists: {output}")

    task = load_task(task_dir / "task.json")
    rows = read_records(task_dir / "records.jsonl", task)
    practice = [r for r in rows if r["split"] == "train"]
    development = [r for r in rows if r["split"] == "dev"]
    if not practice or not development:
        raise ValueError("make requires training and development records (split=train and split=dev)")

    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=output.parent, prefix=".making-decisions-") as tmp:
        stage = Path(tmp) / "run"
        stage.mkdir()
        inputs = stage / "inputs"
        write_new(inputs / "task.json", task)
        with (inputs / "records.jsonl").open("x", encoding="utf-8") as handle:
            for row in rows:
                if row["split"] in ("train", "dev"):
                    handle.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
        neural = train_bert(task, practice, base, stage / "model", epochs=epochs,
                            seed=seed, batch_size=batch_size, learning_rate=learning_rate,
                            min_score=min_score)
        tree = train(task, practice, min_score=min_score)
        write_new(stage / "tree.json", tree)
        report = {
            "task": task["name"], "version": task["version"], "question": task.get("question"),
            "split": "dev", "training_rows": len(practice), "development_rows": len(development),
            "base": neural["base"], "min_score": min_score,
        }
        for name, checkpoint in (("bert", "model"), ("tree", "tree.json")):
            # Replay the saved artifact through the same loader a work application uses.
            replay = evaluate(load_model(stage / checkpoint), development, "dev")
            detail = f"{name}-dev.json"
            write_new(stage / detail, replay)
            report[name] = {
                "checkpoint": checkpoint, "report": detail, "model_sha256": replay["model_sha256"],
                "evaluation_records_sha256": replay["evaluation_records_sha256"],
                "model": replay["model"],
                "model_with_recorded_candidates": replay["model_with_recorded_candidates"],
                "raw_model_preference": replay["raw_model_preference"],
                "brier_multiclass": replay["brier_multiclass"],
            }
        report["tree"]["diagnostics"] = tree["tree_diagnostics"]
        write_new(stage / "report.json", report)
        if os.path.lexists(output):
            raise FileExistsError(f"output already exists: {output}")
        stage.rename(output)
    return report
