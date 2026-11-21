"""Validate decision tasks, prediction inputs, and labeled JSONL records."""

import json
import math
from pathlib import Path


_FEATURE_TYPES = ("number", "category", "boolean", "text")
_SPLITS = ("train", "dev", "test")


def _object(value, where):
    if not isinstance(value, dict):
        raise ValueError(f"{where} must be an object")


def _fields(value, required, where, optional=()):
    _object(value, where)
    missing = set(required).difference(value)
    extra = set(value).difference(required, optional)
    if missing:
        raise ValueError(f"{where} missing fields: {', '.join(sorted(missing))}")
    if extra:
        names = sorted(repr(name) for name in extra)
        raise ValueError(f"{where} has unknown fields: {', '.join(names)}")


def _string(value, where):
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{where} must be a nonempty string")


def _strings(values, where):
    if not isinstance(values, list):
        raise ValueError(f"{where} must be a list")
    seen = set()
    for value in values:
        _string(value, where)
        if value in seen:
            raise ValueError(f"{where} contains duplicate {value!r}")
        seen.add(value)


def validate_task(task):
    """Check the task definition, including feature types and label uniqueness."""
    _fields(task, ("name", "version", "features", "labels"), "task",
            optional=("question", "descriptions"))
    _string(task["name"], "task.name")
    _string(task["version"], "task.version")
    _object(task["features"], "task.features")
    for feature, kind in task["features"].items():
        _string(feature, "feature name")
        if kind not in _FEATURE_TYPES:
            raise ValueError(f"unknown feature type for {feature!r}: {kind!r}")
    _strings(task["labels"], "task.labels")
    if not task["labels"]:
        raise ValueError("task.labels must include at least one label")
    if "question" in task:
        _string(task["question"], "task.question")
    if "descriptions" in task:
        descriptions = task["descriptions"]
        _fields(descriptions, task["labels"], "task.descriptions")
        for label in task["labels"]:
            _string(descriptions[label], f"task.descriptions[{label!r}]")


def validate_state(task, state):
    """Check a state against the fields and scalar types of the task."""
    validate_task(task)
    _check_state(task, state)


def _check_state(task, state):
    _fields(state, task["features"], "state")
    for feature, kind in task["features"].items():
        value = state[feature]
        if kind in ("category", "text"):
            _string(value, f"state.{feature}")
        elif kind == "boolean":
            if type(value) is not bool:
                raise ValueError(f"state.{feature} must be a boolean")
        else:
            try:
                finite = type(value) in (int, float) and math.isfinite(value)
            except OverflowError:
                finite = False
            if not finite:
                raise ValueError(f"state.{feature} must be a finite number")


def validate_candidates(task, candidates):
    """Allow an empty list, but require unique labels from the task vocabulary."""
    validate_task(task)
    _check_candidates(task, candidates)


def _check_candidates(task, candidates):
    _strings(candidates, "candidates")
    for label in candidates:
        if label not in task["labels"]:
            raise ValueError(f"unknown candidate {label!r}")


def _validate_row(task, row):
    _fields(row, ("id", "group", "split", "state", "target", "candidates"),
            "record", optional=("baselines",))
    _string(row["id"], "id")
    _string(row["group"], "group")
    if row["split"] not in _SPLITS:
        raise ValueError(f"unknown split {row['split']!r}")
    _check_state(task, row["state"])
    if row["candidates"] is not None:
        _check_candidates(task, row["candidates"])
    if row["target"] not in task["labels"]:
        raise ValueError(f"unknown target {row['target']!r}")
    if "baselines" in row:
        baselines = row["baselines"]
        _object(baselines, "baselines")
        for name, label in baselines.items():
            _string(name, "baseline name")
            if label is not None and label not in task["labels"]:
                raise ValueError(f"baseline {name!r} has unknown label {label!r}")


def _check_dataset(task, numbered_rows):
    ids = set()
    group_splits = {}
    for location, row in numbered_rows:
        try:
            _validate_row(task, row)
            if row["id"] in ids:
                raise ValueError(f"duplicate record id {row['id']!r}")
            ids.add(row["id"])
            split = group_splits.setdefault(row["group"], row["split"])
            if split != row["split"]:
                raise ValueError(f"group {row['group']!r} occurs in splits "
                                 f"{split!r} and {row['split']!r}")
        except ValueError as exc:
            raise ValueError(f"{location}: {exc}") from exc
    if not ids:
        raise ValueError("dataset must include at least one record")


def validate_records(task, rows):
    """Accept any nonempty split selection and prevent group leakage across splits."""
    validate_task(task)
    if not isinstance(rows, list):
        raise ValueError("records must be a list")
    _check_dataset(task, ((f"record {i}", row) for i, row in enumerate(rows, 1)))


def _unique_keys(pairs):
    obj = {}
    for key, value in pairs:
        if key in obj:
            raise ValueError(f"duplicate JSON field {key!r}")
        obj[key] = value
    return obj


def _reject_nonfinite(token):
    raise ValueError(f"expected a finite number, found {token}")


def _decode_json(content):
    return json.loads(content, object_pairs_hook=_unique_keys, parse_constant=_reject_nonfinite)


def read_json(path: Path):
    """Read a UTF-8 JSON value and reject duplicate keys at every depth."""
    path = Path(path).expanduser()
    try:
        return _decode_json(path.read_text(encoding="utf-8-sig"))
    except (OSError, ValueError) as exc:
        raise ValueError(f"{path}: {exc}") from exc


def load_task(path: Path) -> dict:
    """Read and validate one task definition from a UTF-8 JSON file."""
    task = read_json(path)
    try:
        validate_task(task)
    except ValueError as exc:
        raise ValueError(f"{path}: {exc}") from exc
    return task


def read_records(path: Path, task) -> list[dict]:
    """Read JSONL, preserving physical line numbers in record-level errors."""
    path = Path(path).expanduser()
    validate_task(task)
    rows = []

    def numbered_rows(stream):
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                row = _decode_json(line)
            except ValueError as exc:
                raise ValueError(f"line {line_number}: {exc}") from exc
            yield f"line {line_number}", row
            rows.append(row)

    try:
        with path.open(encoding="utf-8-sig") as stream:
            _check_dataset(task, numbered_rows(stream))
    except (OSError, ValueError) as exc:
        raise ValueError(f"{path}: {exc}") from exc
    return rows
