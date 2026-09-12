"""Local UI and a small reusable skill / inference / experiment API."""

from contextlib import asynccontextmanager
import json
import os
import signal
import subprocess
import sys
import threading
import time

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict, Field
from starlette.middleware.trustedhost import TrustedHostMiddleware

from workshop.domain import Example, Skill, digest, input_key
from workshop import store

operation_lock = threading.Lock()
data_lock = threading.Lock()
children = set()


@asynccontextmanager
async def lifespan(app):
    store.initialize()
    for job in store.jobs():
        if job["status"] in ("queued", "running") and not store.runtime_busy():
            job.update(status="interrupted", error="Server restarted; this run was not completed.")
            store.atomic_json(store.STATE / "jobs" / job["id"] / "job.json", job)
    yield
    for child in list(children):
        if child.poll() is None:
            os.killpg(child.pid, signal.SIGTERM)
            child.wait(timeout=10)


app = FastAPI(title="Specialist Workshop", version="0.1.0", lifespan=lifespan)
app.add_middleware(TrustedHostMiddleware, allowed_hosts=["127.0.0.1", "localhost", "testserver"])


@app.middleware("http")
async def local_requests(request: Request, call_next):
    if request.method == "POST":
        origin = request.headers.get("origin")
        if origin and origin != str(request.base_url).rstrip("/"):
            return JSONResponse({"detail": "Cross-origin writes are disabled."}, status_code=403)
        if request.headers.get("content-type", "").split(";")[0] != "application/json":
            return JSONResponse({"detail": "Use application/json."}, status_code=415)
        body = await request.body()
        if len(body) > 4_000_000:
            return JSONResponse({"detail": "Request exceeds 4 MB."}, status_code=413)
    response = await call_next(request)
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["Content-Security-Policy"] = "default-src 'self'; style-src 'self'; img-src 'self' data:; frame-ancestors 'none'"
    return response


def skill_or_404(skill_id):
    try:
        return store.load_skill(skill_id)
    except (ValueError, FileNotFoundError):
        raise HTTPException(404, "Skill not found.")


def job_or_404(job_id):
    try:
        return store.load_job(job_id)
    except (ValueError, FileNotFoundError):
        raise HTTPException(404, "Run not found.")


def adapter_for(skill, model, checkpoint):
    if model != "specialist":
        return None
    if not checkpoint:
        raise HTTPException(409, "Train and select a completed checkpoint first.")
    job = job_or_404(checkpoint)
    if job["kind"] != "train" or job["status"] != "completed" or job["skill_id"] != skill.id:
        raise HTTPException(409, "Checkpoint does not belong to this skill or is incomplete.")
    if job["model_id"] != store.MODEL_ID or job["model_revision"] != store.MODEL_REVISION:
        raise HTTPException(409, "Checkpoint uses a different base model than the active runtime.")
    import hashlib
    from pathlib import Path
    weights = Path(job["adapter_path"]) / "adapters.safetensors"
    if not weights.is_file() or hashlib.sha256(weights.read_bytes()).hexdigest() != job["adapter_sha256"]:
        raise HTTPException(409, "Checkpoint weights are missing or no longer match the recorded hash.")
    trained_skill = Skill.model_validate(store.read_json(store.STATE / "jobs" / checkpoint / "skill.json"))
    from workshop.domain import prompt_for
    if prompt_for(trained_skill) != prompt_for(skill):
        raise HTTPException(409, "Skill contract changed since this checkpoint was trained.")
    return job["adapter_path"]


def reference_guard(model):
    if model == "reference" and not (os.environ.get("WORKSHOP_REFERENCE_URL") and os.environ.get("WORKSHOP_REFERENCE_MODEL")):
        raise HTTPException(409, "Configure a reference endpoint and model on the server first.")


@app.get("/api/status")
def status():
    return dict(busy=operation_lock.locked() or store.runtime_busy(), model=store.MODEL_ID, revision=store.MODEL_REVISION,
                grader_version="v2-label-schema-separated",
                reference_model=os.environ.get("WORKSHOP_REFERENCE_MODEL"),
                reference_configured=bool(os.environ.get("WORKSHOP_REFERENCE_URL") and os.environ.get("WORKSHOP_REFERENCE_MODEL")),
                mode="local", training="QLoRA · MLX · Apple Silicon")


@app.get("/api/skills")
def skills():
    return [store.skill_summary(Skill.model_validate(store.read_json(path)))
            for path in sorted((store.STATE / "skills").glob("*.json"))]


@app.get("/api/skills/{skill_id}")
def skill_detail(skill_id: str):
    return skill_or_404(skill_id).model_dump()


@app.post("/api/skills", status_code=201)
def create_skill(skill: Skill):
    with data_lock:
        path = store.STATE / "skills" / f"{skill.id}.json"
        if path.exists():
            raise HTTPException(409, "Skill already exists. Use a new id for a new contract.")
        store.atomic_json(path, skill.model_dump())
    return store.skill_summary(skill)


class Correction(BaseModel):
    model_config = ConfigDict(extra="forbid")
    input: str = Field(min_length=1, max_length=16000)
    expected: str = Field(min_length=1, max_length=80)
    evidence: str = Field(min_length=1, max_length=2000)


@app.post("/api/skills/{skill_id}/corrections", status_code=201)
def correct(skill_id: str, correction: Correction):
    with data_lock:
        skill = skill_or_404(skill_id)
        key = input_key(correction.input)
        reserved = {input_key(row.input) for row in skill.test + skill.validation}
        if key in reserved:
            raise HTTPException(409, "This input is reserved for evaluation. Add a new practice example instead.")
        row = Example(id=store.new_id("correction"), source="user-correction", **correction.model_dump())
        payload = skill.model_dump()
        payload["train"] = [r.model_dump() for r in skill.train if input_key(r.input) != key] + [row.model_dump()]
        try:
            updated = Skill.model_validate(payload)
        except ValueError as exc:
            raise HTTPException(422, str(exc))
        store.atomic_json(store.STATE / "skills" / f"{skill.id}.json", updated.model_dump())
    return dict(example=row.model_dump(), skill=store.skill_summary(updated))


class Inference(BaseModel):
    model_config = ConfigDict(extra="forbid")
    skill_id: str
    input: str = Field(min_length=1, max_length=16000)
    model: str = Field(default="base", pattern="^(base|specialist|reference)$")
    checkpoint: str | None = None


@app.post("/api/infer")
def infer(request: Inference):
    skill = skill_or_404(request.skill_id)
    reference_guard(request.model)
    adapter = adapter_for(skill, request.model, request.checkpoint)
    if store.runtime_busy():
        raise HTTPException(409, "Another process is using the local runtime.")
    if not operation_lock.acquire(blocking=False):
        raise HTTPException(409, "An experiment is using the local runtime. Wait for it to finish.")
    identity = store.new_id("inference")
    path = store.STATE / "inferences" / f"{identity}.json"
    child = None
    try:
        store.atomic_json(path, dict(**request.model_dump(), skill=skill.model_dump(), adapter_path=adapter))
        with path.with_suffix(".log").open("w") as log:
            child = subprocess.Popen([sys.executable, "-m", "workshop.worker", "infer", str(path)],
                                     cwd=store.ROOT, stdout=log, stderr=log, start_new_session=True)
            children.add(child)
            child.wait(timeout=180)
        if child.returncode:
            raise HTTPException(500, "Inference failed. Inspect the local inference log: " + identity)
        result = store.read_json(path.with_suffix(".result.json"))
        result.update(id=identity, checkpoint=request.checkpoint, skill_id=skill.id,
                      skill_fingerprint=digest(skill.model_dump()), input=request.input)
        store.atomic_json(path.with_suffix(".result.json"), result)
        return result
    except subprocess.TimeoutExpired:
        os.killpg(child.pid, signal.SIGTERM)
        child.wait(timeout=10)
        raise HTTPException(504, "Inference timed out after 180 seconds.")
    finally:
        if child:
            children.discard(child)
        operation_lock.release()


class RunRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    skill_id: str
    kind: str = Field(pattern="^(train|evaluate)$")
    model: str = Field(default="base", pattern="^(base|specialist|reference)$")
    checkpoint: str | None = None
    iterations: int = Field(default=150, ge=10, le=300)


def monitor(child, job_id, log):
    try:
        code = child.wait(timeout=2400)
        job = store.load_job(job_id)
        if code or job["status"] in ("queued", "running"):
            job.update(status="failed", error="Worker exited unexpectedly; inspect run log.", finished_at=time.time())
            store.atomic_json(store.STATE / "jobs" / job_id / "job.json", job)
    except subprocess.TimeoutExpired:
        os.killpg(child.pid, signal.SIGTERM)
        child.wait(timeout=10)
        job = store.load_job(job_id)
        job.update(status="failed", error="Run exceeded the 40-minute limit.", finished_at=time.time())
        store.atomic_json(store.STATE / "jobs" / job_id / "job.json", job)
    finally:
        log.close()
        children.discard(child)
        operation_lock.release()


@app.post("/api/runs", status_code=202)
def start_run(request: RunRequest):
    skill = skill_or_404(request.skill_id)
    if request.kind == "train" and request.model != "base":
        raise HTTPException(422, "This recipe trains a fresh adapter from the base model.")
    reference_guard(request.model)
    adapter = adapter_for(skill, request.model, request.checkpoint)
    if store.runtime_busy():
        raise HTTPException(409, "Another process is using the local runtime.")
    if not operation_lock.acquire(blocking=False):
        raise HTTPException(409, "Another operation is using the runtime.")
    identity = store.new_id("job")
    directory = store.STATE / "jobs" / identity
    job = dict(**request.model_dump(), id=identity, status="queued", created_at=time.time(),
               progress=0, total=request.iterations if request.kind == "train" else len(skill.test), adapter_path=adapter,
               model_id=store.MODEL_ID, model_revision=store.MODEL_REVISION,
               skill_fingerprint=digest(skill.model_dump()),
               test_fingerprint=digest([r.model_dump() for r in skill.test]),
               training_count=len(skill.train), validation_count=len(skill.validation))
    from workshop.domain import prompt_for
    job.update(grader_version="v2-label-schema-separated", prompt_fingerprint=digest(prompt_for(skill)),
               inference_settings=dict(temperature=0, max_tokens=220, seed=17))
    if request.model == "reference":
        job.update(model_id=os.environ["WORKSHOP_REFERENCE_MODEL"], model_revision=None)
    log = None
    try:
        store.atomic_json(directory / "skill.json", skill.model_dump())
        store.atomic_json(directory / "job.json", job)
        log = (directory / "worker.log").open("w")
        child = subprocess.Popen([sys.executable, "-m", "workshop.worker", "job", identity],
                                 cwd=store.ROOT, stdout=log, stderr=log, start_new_session=True)
        children.add(child)
        threading.Thread(target=monitor, args=(child, identity, log), daemon=True).start()
    except Exception:
        if log:
            log.close()
        operation_lock.release()
        raise
    return job


@app.get("/api/runs")
def runs():
    return [{k: v for k, v in job.items() if k != "rows"} for job in store.jobs()]


@app.get("/api/runs/{job_id}")
def run_detail(job_id: str):
    job = job_or_404(job_id)
    directory = store.STATE / "jobs" / job_id
    log_path = directory / "training.log"
    if not log_path.exists():
        log_path = directory / "worker.log"
    job["log"] = log_path.read_text(errors="replace")[-16000:] if log_path.exists() else ""
    return job


@app.get("/")
def index():
    return FileResponse(store.ROOT / "static" / "index.html")


app.mount("/static", StaticFiles(directory=store.ROOT / "static"), name="static")
