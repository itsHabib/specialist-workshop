"""Train a local preference score for candidates supplied by a deterministic program."""

import json
import math
import os
from pathlib import Path
import tempfile

from .checkpoint import train_bert
from .contracts import validate_records, validate_state
from .core import load_model, predict, train, write_new


LABELS = ["prefer", "defer"]


def fit_rankers(name, examples, out, base):
    """Fit from training groups only. Each row carries numeric features and a boolean target."""
    examples = list(examples)
    if (not examples or not isinstance(examples[0], dict)
            or not isinstance(examples[0].get("features"), dict) or not examples[0]["features"]):
        raise ValueError("training requires candidates with numeric features")
    task = {"name": name, "version": "1", "features": {f: "number" for f in examples[0]["features"]},
            "labels": LABELS.copy(), "question": "Should the program try this candidate early?"}
    rows = []
    for i, example in enumerate(examples):
        if (not isinstance(example, dict) or set(example) != {"group", "features", "preferred"}
                or type(example["preferred"]) is not bool):
            raise ValueError("expected group, features and a boolean preferred target")
        rows.append({"id": f"train-{i}", "group": example["group"], "split": "train",
                     "state": example["features"], "candidates": LABELS.copy(),
                     "target": "prefer" if example["preferred"] else "defer"})
    validate_records(task, rows)
    if len({row["target"] for row in rows}) != 2:
        raise ValueError("training requires both preferred and deferred examples")
    out = Path(out)
    if os.path.lexists(out):
        raise FileExistsError(f"output already exists: {out}")
    out.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=out.parent, prefix=".fit-rankers-") as tmp:
        stage = Path(tmp) / "rankers"
        stage.mkdir()
        tree = train(task, rows, balanced_ties=True)
        write_new(stage / "tree.json", tree)
        neural = train_bert(task, rows, base, stage / "bert", epochs=4, batch_size=16,
                            learning_rate=0.0003, seed=17)
        write_new(stage / "training.json", {"task": task, "records": rows,
                                           "checkpoints": {"bert": neural["sha256"], "tree": tree["sha256"]}})
        models = {"tree": load_model(stage / "tree.json"), "bert": load_model(stage / "bert")}
        stage.rename(out)
    return models


def validate_ranker(model):
    """Check whether the checkpoint can score numeric candidate features."""
    task = model["task"]
    if task["labels"] != LABELS or set(task["features"].values()) != {"number"}:
        raise ValueError("expected a numeric candidate preference checkpoint")


def rank_candidates(model, candidates, *, native_ties=False):
    """Rank supplied IDs. With native_ties, every score tie keeps the input order."""
    validate_ranker(model)
    task = model["task"]
    fields = sorted(task["features"])
    seen = set()
    items = []
    for identifier, features in candidates:
        if not isinstance(identifier, str) or not identifier.strip() or identifier in seen:
            raise ValueError("each candidate needs a distinct, nonempty string ID")
        validate_state(task, features)
        seen.add(identifier)
        values = tuple(features[f] for f in fields)
        # Python compares 1 and 1.0 as equal, but the tokenizer sees different text.
        key = (values, json.dumps(values, allow_nan=False))
        items.append((key, identifier, features))
    # Keep neural batch membership stable across incoming order and ID changes.
    feature_order = sorted(items, key=lambda item: item[0])
    distinct = {key: state for key, _, state in feature_order}
    scorer = model.get("_score")
    if scorer is None:
        distributions = [predict(model, state, LABELS)["scores"] for state in distinct.values()]
    else:
        distributions = scorer.score_many(list(distinct.values()))
    scores = {}
    for key, dist in zip(distinct, distributions, strict=True):
        if (set(dist) != set(LABELS) or not all(0 <= v <= 1 for v in dist.values())
                or not math.isclose(sum(dist.values()), 1, abs_tol=1e-5)):
            raise ValueError("invalid candidate preference distribution")
        scores[key] = dist["prefer"]
    tie_order = items if native_ties else feature_order
    ranked = [(identifier, scores[key]) for key, identifier, _ in tie_order]
    return sorted(ranked, key=lambda item: -item[1])
