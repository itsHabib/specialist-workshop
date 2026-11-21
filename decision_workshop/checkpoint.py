"""Bind learned decision weights and the local training recipe in a reloadable checkpoint."""

import hashlib
import os
from pathlib import Path
import tempfile

from .base import FILE_HASHES, MODEL_ID, REVISION, verify_base
from .contracts import read_json, validate_task
from .core import digest, training_rows, training_summary, write_new

FORMAT = "decision-workshop/bert-choices-v1"


def _files(folder):
    hashes = {}
    for path in sorted(folder.iterdir()):
        if path.name == "manifest.json":
            continue
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"unexpected checkpoint entry: {path.name}")
        hashes[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
    return hashes


def train_bert(task, rows, base, output, *, epochs, seed, batch_size, learning_rate):
    from .bert import fit_bert

    output = Path(output)
    if os.path.lexists(output):
        raise FileExistsError(f"output already exists: {output}")
    base = verify_base(Path(base))
    practice = training_rows(rows)
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=output.parent, prefix=".decision-training-") as tmp:
        stage = Path(tmp) / "checkpoint"
        stage.mkdir()
        stats = fit_bert(task, practice, base, stage, epochs=epochs, seed=seed,
                         batch_size=batch_size, learning_rate=learning_rate)
        model = {
            "format": FORMAT, "algorithm": FORMAT, "task": task,
            "base": {"model": MODEL_ID, "revision": REVISION, "files": FILE_HASHES},
            "training": training_summary(task, practice),
            "optimization": stats,
            "artifacts": _files(stage),
        }
        model["sha256"] = digest(model)
        write_new(stage / "manifest.json", model)
        if os.path.lexists(output):
            raise FileExistsError(f"output already exists: {output}")
        stage.rename(output)
    return model


def load_bert(folder):
    """Verify checkpoint contents before initializing the offline inference runtime."""
    manifest = folder / "manifest.json"
    if manifest.is_symlink():
        raise ValueError("unexpected symlink in checkpoint")
    model = read_json(manifest)
    if not isinstance(model, dict) or model.get("format") != FORMAT or model.get("algorithm") != FORMAT:
        raise ValueError("unsupported neural decision checkpoint")
    if model.get("sha256") != digest({k: v for k, v in model.items() if k != "sha256"}):
        raise ValueError("checkpoint manifest digest mismatch")
    if model.get("base") != {"model": MODEL_ID, "revision": REVISION, "files": FILE_HASHES}:
        raise ValueError("unexpected pretrained base identity")
    if not model.get("artifacts") or model["artifacts"] != _files(folder):
        raise ValueError("checkpoint files differ from recorded hashes")
    validate_task(model["task"])
    from .bert import load_scorer
    model["_score"] = load_scorer(folder, model["task"])
    return model
