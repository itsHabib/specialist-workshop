"""Replay a local Clef-Flash reference on held-out decisions using Apple MLX."""

import argparse
import hashlib
import importlib.metadata
import importlib.util
import math
import os
from pathlib import Path
import platform
import sys
import tempfile
import time

from .contracts import load_task, read_records
from .core import digest, write_new


MODEL_ID = "mlx-community/clef-flash-4bit"
REVISION = "b5adbe2c3a95200d4bf04a0727c8ec457157d5e1"
UPSTREAM_ID = "Cloudflare/clef-flash"
# Publisher revision with a byte-identical joint head. The conversion card
# identifies its source model but does not specify the source commit.
UPSTREAM_REVISION = "22b75507981345858dd152fbd9a6f8c0c4068b0a"
FILE_HASHES = {
    "chat_template.jinja": "a4aee8afcf2e0711942cf848899be66016f8d14a889ff9ede07bca099c28f715",
    "clef_mlx.py": "852223c944819a32fad5cf798d9d1dff30419820eaf5ad1f10cb9698afec97d5",
    "config.json": "c323f36eb35fb9349b2bde60b3e16ee3d86316e20a5166c91f4a0585286f0ec1",
    "generation_config.json": "45707f8467bf4e54e112c6095a1b2d1f2d63651cf86816344cfff21f02dda0d4",
    "joint_head.safetensors": "19cdcec8c81dc9212be320fff47462ab342fbc1278be4368fb3da71241cf5ba0",
    "joint_head_config.json": "77efe959a38b5b17b241543e129e695f3c77465ece55a25985279bd8176279a0",
    "model-00001-of-00002.safetensors": "39894f085d844004c8e5c2394e45b5ab66bce4637ecdf23d92bf7c85cb66238f",
    "model-00002-of-00002.safetensors": "40258d3d4f7225a0acbb45f4d34693465b98c8e2c0a4b0a512e536750379fcf3",
    "model.safetensors.index.json": "4518e4d98baed265cb25c5adb45839d5b69e0b9700f4bd573533392097bbb874",
    "processor_config.json": "d89ef49ce9cd37fbf510158e13c1ef063d9286411c1ec9049932dbe0487143b1",
    "tokenizer.json": "a5cd9732badce41de57e6efce8302930ded1c1188c5f81feb2bd6c24c4a1941f",
    "tokenizer_config.json": "7d884d6eaf8142a3343a933ca5ab6ccc2fa93d52591cdbbe2ea33559bb2b503a",
}


def file_hash(path):
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def request_for(task, state):
    descriptions = task.get("descriptions", {})
    criteria = {label: descriptions.get(label, label.replace("_", " ")) for label in task["labels"]}
    question = task.get("question", task["name"].replace("_", " "))
    return {"model": "clef-flash", "state": state, "questions": {
        "action": {"type": "choice", "instructions": question, "criteria": criteria}
    }}


def replay(task_dir, model_dir, out):
    if os.path.lexists(out):
        raise FileExistsError(f"output already exists: {out}")
    task_file = task_dir / "task.json"
    records_file = task_dir / "records.jsonl"
    task = load_task(task_file)
    rows = read_records(records_file, task)
    held_out = sorted((r for r in rows if r["split"] in ("dev", "test")),
                      key=lambda row: (row["split"], row["id"]))
    split_counts = {s: sum(r["split"] == s for r in held_out) for s in ("dev", "test")}
    if not all(split_counts.values()):
        raise ValueError("comparison requires nonempty dev and test splits")
    inputs = {
        "task": task["name"], "task_sha256": file_hash(task_file),
        "records_sha256": file_hash(records_file), "port_loader_sha256": FILE_HASHES["clef_mlx.py"],
        "replay_script_sha256": file_hash(Path(__file__)), "choice_field": "action",
        "options": task["labels"], "split_counts": split_counts,
        "model_request_fields": ["model", "state", "questions"], "trained_on_evaluation_rows": False,
    }
    # HF environment settings must precede importing the inference stack.
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["HF_HUB_DISABLE_IMPLICIT_TOKEN"] = "1"
    files = {}
    cases = []
    start = time.perf_counter()
    # Isolate loader inputs from unrelated files in a cache or removable drive.
    with tempfile.TemporaryDirectory(prefix="clef-flash-") as tmp:
        snapshot = Path(tmp)
        for name, expected in FILE_HASHES.items():
            path = (model_dir / name).resolve(strict=True)
            if file_hash(path) != expected:
                raise ValueError(f"wrong file digest for {MODEL_ID} at {REVISION}: {name}")
            files[name] = {"sha256": expected, "size": path.stat().st_size}
            (snapshot / name).symlink_to(path)
        spec = importlib.util.spec_from_file_location("clef_mlx", snapshot / "clef_mlx.py")
        port = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = port
        spec.loader.exec_module(port)
        tick = time.perf_counter()
        model = port.load(snapshot, backend="vlm")
        load_seconds = time.perf_counter() - tick
        for row in held_out:
            request = request_for(task, row["state"])
            tick = time.perf_counter()
            raw = model.predict(request, truncate=False)
            elapsed = time.perf_counter() - tick
            scores = raw["action"]
            if (set(raw) != {"action"} or set(scores) != set(task["labels"])
                    or not all(type(p) in (float, int) and 0 <= p <= 1 for p in scores.values())
                    or not math.isclose(sum(scores.values()), 1, abs_tol=1e-5)):
                raise ValueError(f"invalid model distribution for row {row['id']}")
            cases.append({"id": row["id"], "split": row["split"], "input": request,
                          "raw_output": raw, "probabilities": scores, "elapsed_seconds": elapsed,
                          "error": None})
    receipt = {
        "format": "decision-workshop/clef-replay-v1", "status": "complete",
        "model": {"id": MODEL_ID, "revision": REVISION, "upstream_id": UPSTREAM_ID,
                  "upstream_revision": UPSTREAM_REVISION, "upstream_revision_scope": "joint head hash",
                  "variant": "Clef-Flash 9B, MLX 4-bit", "files": files},
        "environment": {"os": platform.mac_ver()[0], "machine": platform.machine(),
                        "python": platform.python_version(),
                        "packages": {p: importlib.metadata.version(p) for p in
                                     ("mlx", "mlx-lm", "mlx-vlm", "transformers", "huggingface_hub",
                                      "torch", "torchvision")}},
        "inputs": inputs, "load_seconds": load_seconds, "elapsed_seconds": time.perf_counter() - start,
        "cases": cases, "errors": [],
    }
    write_new(out, {**receipt, "sha256": digest(receipt)})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True, help="local pinned MLX model files")
    parser.add_argument("--out", type=Path, required=True, help="new JSON receipt")
    args = parser.parse_args()
    try:
        replay(args.task, args.model_dir, args.out)
    except (ValueError, OSError, ImportError) as exc:
        parser.exit(1, f"error: {exc}\n")


if __name__ == "__main__":
    main()
