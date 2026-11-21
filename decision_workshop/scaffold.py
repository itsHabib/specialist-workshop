"""Create an editable local decision task and an empty training dataset."""

import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Optional

from .contracts import validate_records, validate_task


def _assignments(items, flag, syntax):
    result = {}
    for item in items:
        if not isinstance(item, str):
            raise ValueError(f"{flag} expects {syntax}")
        name, separator, value = item.partition("=")
        name, value = name.strip(), value.strip()
        if not separator or not name or not value:
            raise ValueError(f"{flag} expects {syntax}, received {item!r}")
        if name in result:
            raise ValueError(f"duplicate {flag} name: {name!r}")
        result[name] = value
    return result


def create_task(directory: Path, question: str, actions: list[str],
                fields: Optional[list[str]] = None) -> None:
    """Create a complete task in a fresh directory, leaving records.jsonl empty."""
    directory = Path(directory).expanduser()
    if os.path.lexists(directory):
        raise FileExistsError(f"task path already exists: {directory}; choose a new directory")
    descriptions = _assignments(actions, "--action", "ACTION=DESCRIPTION")
    if len(descriptions) < 2:
        raise ValueError("supply at least two --action ACTION=DESCRIPTION options")
    features = _assignments(fields or [], "--field", "FIELD=TYPE") or {"context": "text"}
    sample_values = {
        "text": "Replace this synthetic example with the facts available before the decision.",
        "category": "example_category", "number": 1, "boolean": False,
    }
    for name, kind in features.items():
        if kind not in sample_values:
            raise ValueError(f"--field {name!r} needs a type from {', '.join(sample_values)}")
    labels = list(descriptions)
    task = {
        "name": directory.name, "version": "1", "question": question,
        "features": features, "labels": labels, "descriptions": descriptions,
    }
    validate_task(task)
    example = {
        "id": "synthetic-example-1", "group": "synthetic-example-group", "split": "train",
        "state": {name: sample_values[kind] for name, kind in features.items()},
        "target": labels[0], "candidates": labels,
    }
    validate_records(task, [example])
    request = {"state": example["state"], "candidates": labels}
    directory.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=directory.parent, prefix=".decision-task-") as scratch:
        stage = Path(scratch) / "task"
        stage.mkdir()
        for filename, payload in (("task.json", task), ("record.example.json", example),
                                  ("request.json", request)):
            (stage / filename).write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
                                         encoding="utf-8")
        (stage / "records.jsonl").touch()
        if os.path.lexists(directory):
            raise FileExistsError(f"task path already exists: {directory}; choose a new directory")
        try:
            stage.rename(directory)
        except OSError as exc:
            if os.path.lexists(directory):
                raise FileExistsError(f"task path already exists: {directory}; choose a new directory") from exc
            raise
