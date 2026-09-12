"""One process owns the accelerator for each finite, recorded operation."""

import argparse
import fcntl
import json
import subprocess
import sys
import time
from pathlib import Path

from workshop.domain import Skill, check_training_length, digest, grade, messages_for, summarize
from workshop.runtime import LocalModel, RemoteModel
from workshop.store import STATE, MODEL_ID, MODEL_REVISION, atomic_json, load_job, read_json


def evaluate(job, directory, skill):
    rows = []
    model = load_model(job["model"], job.get("adapter_path"))
    for row in skill.test:
        raw, elapsed, tokens = model.predict(skill, row.input)
        result = grade(skill, row.input, raw, row.expected)
        result.update(id=row.id, input=row.input, source=row.source, expected=row.expected,
                      elapsed_ms=elapsed, input_tokens=tokens)
        rows.append(result)
        job.update(progress=len(rows), total=len(skill.test), metrics=summarize(rows), rows=rows)
        atomic_json(directory / "job.json", job)


def train(job, directory, skill):
    from huggingface_hub import snapshot_download
    from transformers import AutoTokenizer
    model_path = snapshot_download(MODEL_ID, revision=MODEL_REVISION)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    data = directory / "data"
    data.mkdir()
    for name, examples in (("train", skill.train), ("valid", skill.validation)):
        with (data / f"{name}.jsonl").open("w") as stream:
            for row in examples:
                output = dict(bucket=row.expected, evidence=row.evidence)
                messages = messages_for(skill, row.input, output)
                check_training_length(tokenizer, messages, row.id)
                stream.write(json.dumps(dict(messages=messages)) + "\n")
    adapter = directory / "adapter"
    command = [sys.executable, "-m", "mlx_lm", "lora", "--model", model_path,
               "--train", "--data", str(data), "--adapter-path", str(adapter),
               "--iters", str(job["iterations"]), "--batch-size", "2", "--num-layers", "8",
               "--learning-rate", "0.00002", "--max-seq-length", "1024", "--mask-prompt",
               "--steps-per-report", "10", "--steps-per-eval", "50", "--val-batches", str(min(4, len(skill.validation) // 2)),
               "--save-every", str(job["iterations"]), "--seed", "17"]
    # The recipe is fixed before the external test is evaluated.
    atomic_json(directory / "recipe.json", dict(command=command, model=MODEL_ID,
                revision=MODEL_REVISION, training_hash=digest([r.model_dump() for r in skill.train])))
    with (directory / "training.log").open("w") as log:
        subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True, timeout=1800)
    weights = adapter / "adapters.safetensors"
    if not weights.is_file():
        raise RuntimeError("Training returned without an adapter checkpoint.")
    import hashlib
    job.update(adapter_path=str(adapter), adapter_sha256=hashlib.sha256(weights.read_bytes()).hexdigest(),
               progress=job["iterations"], total=job["iterations"])


def run_job(job_id):
    job = load_job(job_id)
    directory = STATE / "jobs" / job_id
    skill = Skill.model_validate(read_json(directory / "skill.json"))
    job.update(status="running", started_at=time.time())
    atomic_json(directory / "job.json", job)
    try:
        if job["kind"] == "train":
            train(job, directory, skill)
        if job["kind"] == "evaluate":
            evaluate(job, directory, skill)
        job["status"] = "completed"
    except Exception as exc:
        job.update(status="failed", error=str(exc))
    job["finished_at"] = time.time()
    atomic_json(directory / "job.json", job)


def load_model(model_name, adapter_path=None):
    if model_name == "reference":
        return RemoteModel()
    return LocalModel(adapter_path)


def infer(request_path):
    request = read_json(Path(request_path))
    skill = Skill.model_validate(request["skill"])
    model = load_model(request["model"], request.get("adapter_path"))
    raw, elapsed, tokens = model.predict(skill, request["input"])
    result = grade(skill, request["input"], raw)
    result.update(elapsed_ms=elapsed, input_tokens=tokens, model=request["model"],
                  model_id=MODEL_ID, model_revision=MODEL_REVISION)
    if request["model"] == "reference":
        result.update(model_id=model.model, model_revision=None)
    atomic_json(Path(request_path).with_suffix(".result.json"), result)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("operation", choices=["job", "infer"])
    parser.add_argument("subject")
    args = parser.parse_args()
    STATE.mkdir(parents=True, exist_ok=True)
    with (STATE / "accelerator.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if args.operation == "job":
            run_job(args.subject)
            return
        infer(args.subject)


if __name__ == "__main__":
    main()
