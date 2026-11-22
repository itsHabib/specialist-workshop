"""Local training, checkpoint I/O, and advisory decisions for one task."""

from contextlib import contextmanager
import errno
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile
import warnings

from .contracts import read_json, validate_candidates, validate_state, validate_task
from .tree import feature_diagnostics, fit, predict_counts


ALGORITHM = "classification-tree-gini-v1"
FORMAT = "decision-workshop/1"


def digest(value):
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


@contextmanager
def new_output_file(path):
    """Publish UTF-8 text at a fresh path only once the writer finishes."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent,
                                         delete=False)
    temporary = Path(handle.name)
    try:
        with handle:
            yield handle
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, path)
        except OSError as exc:
            # Some removable media cannot hard-link. Serialize cooperating writers
            # via a sibling lock so readers still only see a complete JSON file.
            unsupported = (errno.EPERM, errno.EACCES, errno.ENOSYS, errno.ENOTSUP)
            if exc.errno not in unsupported and getattr(exc, "winerror", None) != 1:
                raise
            lock = path.with_name("." + path.name + ".publishing")
            lock.touch(exist_ok=False)
            try:
                if os.path.lexists(path):
                    raise FileExistsError(f"output already exists: {path}")
                os.replace(temporary, path)
            finally:
                lock.unlink()
    finally:
        temporary.unlink(missing_ok=True)


def write_new(path, value):
    """Publish complete JSON to a new path. Refuse to replace an existing run."""
    with new_output_file(path) as handle:
        json.dump(value, handle, indent=2, allow_nan=False)
        handle.write("\n")


def training_rows(rows):
    """Keep evaluation labels and incumbent outputs outside the numerical trainer."""
    practice = sorted((row for row in rows if row["split"] == "train"), key=lambda row: row["id"])
    if not practice:
        raise ValueError("training requires records assigned to train")
    return practice


def training_summary(task, practice):
    label_counts = dict.fromkeys(task["labels"], 0)
    for row in practice:
        label_counts[row["target"]] += 1
    seen_categories = {
        key: sorted({row["state"][key] for row in practice})
        for key, kind in task["features"].items() if kind == "category"
    }
    return {
        "rows": len(practice), "records_sha256": digest(practice),
        "group_hashes": sorted({digest(row["group"]) for row in practice}),
        "id_hashes": sorted(digest(row["id"]) for row in practice),
        "label_counts": label_counts, "categories": seen_categories,
    }


def validate_min_score(value):
    if type(value) not in (int, float) or not 0 <= value <= 1:
        raise ValueError("min_score must be a number between 0 and 1")


def decision_min_score(model, override):
    """Resolve the operating cutoff without guessing for a legacy checkpoint."""
    if override is None:
        if "min_score" not in model:
            raise ValueError("checkpoint has no recorded min_score. Pass min_score explicitly (--min-score in the CLI)")
        override = model["min_score"]
    validate_min_score(override)
    return override


def train(task, rows, max_depth=5, min_leaf=2, *, balanced_ties=False, min_score=0.0):
    """Fit a tree to structured features from validated training records."""
    if not 0 <= max_depth <= 30 or min_leaf < 1:
        raise ValueError("max_depth must be 0..30 and min_leaf must be positive")
    validate_min_score(min_score)
    practice = training_rows(rows)
    structured = {key: kind for key, kind in task["features"].items() if kind != "text"}
    model = {
        "format": FORMAT, "algorithm": ALGORITHM, "task": task, "min_score": min_score,
        "settings": {"max_depth": max_depth, "min_leaf": min_leaf},
        "training": training_summary(task, practice),
        "tree": fit(structured, practice, task["labels"],
                    max_depth=max_depth, min_leaf=min_leaf, balanced_ties=balanced_ties),
    }
    diagnostics = feature_diagnostics(task["features"], practice, model["tree"])
    model["tree_diagnostics"] = diagnostics
    for message in diagnostics["warnings"]:
        warnings.warn(message, stacklevel=2)
    if balanced_ties:
        model["settings"]["balanced_ties"] = True
    return {**model, "sha256": digest(model)}


def load_model(path):
    """Load a versioned JSON artifact. The digest detects unintended file changes."""
    if Path(path).is_dir():
        from .checkpoint import load_bert
        return load_bert(Path(path))
    model = read_json(path)
    if not isinstance(model, dict) or model.get("format") != FORMAT or model.get("algorithm") != ALGORITHM:
        raise ValueError("unsupported model format or algorithm")
    body = {key: value for key, value in model.items() if key != "sha256"}
    if model.get("sha256") != digest(body):
        raise ValueError("model digest mismatch")
    validate_task(model["task"])
    if "min_score" in model:
        validate_min_score(model["min_score"])
    return model


def predict(model, state, candidates, *, min_score=None):
    """Apply the supplied or saved cutoff and check the preference against the current candidates."""
    task = model["task"]
    validate_state(task, state)
    if candidates is not None:
        validate_candidates(task, candidates)
    min_score = decision_min_score(model, min_score)
    labels = task["labels"]
    if model["algorithm"] == ALGORITHM:
        counts = predict_counts(model["tree"], state)
        total = sum(counts.values())
        scores = {label: counts.get(label, 0) / total for label in labels}
        details = {"leaf_counts": counts}
    else:
        scores = model["_score"](state)
        valid_scores = all(math.isfinite(v) and 0 <= v <= 1 for v in scores.values())
        normalized = math.isclose(sum(scores.values()), 1, abs_tol=1e-5)
        if set(scores) != set(labels) or not valid_scores or not normalized:
            raise ValueError("decision model returned an invalid distribution")
        details = {}
    best = max(scores.values())
    winners = [label for label in labels if scores[label] == best]
    preferred = winners[0] if len(winners) == 1 else None
    training_categories = model["training"]["categories"]
    unfamiliar = sorted(key for key, values in training_categories.items() if state[key] not in values)
    reasons = []
    if best < min_score:
        reasons.append("below_min_score")
    if unfamiliar:
        reasons.append("unseen_category")
    if preferred is None:
        reasons.append("tied_preferences")
    if candidates is None:
        reasons.append("candidates_unrecorded")
    elif not candidates:
        reasons.append("no_candidates")
    elif preferred is not None and preferred not in candidates:
        reasons.append("preferred_action_missing")
    return {
        "action": None if reasons else preferred, "preferred": preferred,
        "status": "review" if reasons else "suggest", "reasons": reasons,
        "unseen_features": unfamiliar,
        "scores": scores, **details,
    }
