"""Atomic artifacts; all mutable experiment data stays in one local state root."""

import json
import os
from pathlib import Path
import tempfile
import time
import uuid

from workshop.domain import Skill, digest

ROOT = Path(__file__).resolve().parents[1]
STATE = Path(os.environ.get("WORKSHOP_STATE", ROOT / ".state")).resolve()
MODEL_ID = "mlx-community/Qwen2.5-1.5B-Instruct-4bit"
MODEL_REVISION = "8b403126fc14f14cfc99bb4cfa72ecbc129ea677"


def atomic_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(dir=path.parent, prefix=".write-")
    try:
        with os.fdopen(fd, "w") as stream:
            json.dump(value, stream, indent=2, ensure_ascii=False)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def read_json(path):
    return json.loads(path.read_text())


def initialize():
    for name in ("skills", "jobs", "inferences"):
        (STATE / name).mkdir(parents=True, exist_ok=True)
    for seed in sorted((ROOT / "skills").glob("*/skill.json")):
        skill = Skill.model_validate(read_json(seed))
        target = STATE / "skills" / f"{skill.id}.json"
        if not target.exists():
            atomic_json(target, skill.model_dump())


def load_skill(skill_id):
    # Validate before using client-controlled identifiers as path components.
    import re
    if not re.fullmatch(r"[a-z][a-z0-9-]{1,48}", skill_id):
        raise ValueError("invalid skill id")
    return Skill.model_validate(read_json(STATE / "skills" / f"{skill_id}.json"))


def new_id(prefix):
    return f"{prefix}-{uuid.uuid4().hex[:12]}"


def skill_summary(skill):
    from workshop.domain import prompt_for
    from collections import Counter
    majority = max(Counter(row.expected for row in skill.test).values()) / len(skill.test)
    return dict(id=skill.id, name=skill.name, description=skill.description, labels=skill.labels,
                train_count=len(skill.train), validation_count=len(skill.validation),
                test_count=len(skill.test), fingerprint=digest(skill.model_dump()),
                test_fingerprint=digest([r.model_dump() for r in skill.test]),
                prompt_fingerprint=digest(prompt_for(skill)),
                majority_baseline=majority,
                provenance=skill.provenance)


def jobs():
    return sorted([read_json(p) for p in (STATE / "jobs").glob("*/job.json")],
                  key=lambda j: j["created_at"], reverse=True)


def runtime_busy():
    import fcntl
    STATE.mkdir(parents=True, exist_ok=True)
    with (STATE / "accelerator.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return True
        fcntl.flock(lock, fcntl.LOCK_UN)
        return False


def load_job(job_id):
    import re
    if not re.fullmatch(r"job-[0-9a-f]{12}", job_id):
        raise ValueError("invalid job id")
    return read_json(STATE / "jobs" / job_id / "job.json")
