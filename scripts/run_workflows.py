"""Serial, recorded experiments using the existing trainer and model runtime."""
import argparse
from collections import Counter
import fcntl
import gc
import json
from pathlib import Path
import statistics
import subprocess
import sys
import tempfile
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from workshop.arithmetic import arithmetic_rule
from workshop.domain import Skill, digest, grade, messages_for, prompt_for, summarize
from workshop.practice import Practice, diagnostic_rule
from workshop.runtime import LocalModel
from workshop import store
from workshop.worker import train

DEST = store.ROOT / "experiments" / "workflows"


class Frontier:
    """Tool-free CLI calls under existing subscription; no delegate or tools."""
    def predict(self, skill, text):
        messages = messages_for(skill, text)
        command = ["claude", "-p", "--output-format", "json", "--tools", "", "--strict-mcp-config",
                   "--mcp-config", '{"mcpServers":{}}', "--no-session-persistence", "--effort", "low",
                   "--setting-sources", "", "--disable-slash-commands", "--max-budget-usd", "0.50",
                   "--system-prompt", messages[0]["content"]]
        start = time.perf_counter()
        # Temporary cwd excludes portfolio instructions from the model context.
        with tempfile.TemporaryDirectory() as tmp:
            process = subprocess.run(command, input=messages[1]["content"], text=True, capture_output=True,
                                     cwd=tmp, timeout=180)
        if process.returncode:
            raise RuntimeError(process.stderr[-500:])
        envelope = json.loads(process.stdout)
        if envelope.get("is_error"):
            raise RuntimeError(str(envelope.get("result")))
        self.last_usage = {k: envelope.get(k) for k in ("modelUsage", "usage", "total_cost_usd", "duration_ms")}
        return envelope["result"], round((time.perf_counter() - start) * 1000), None


def subset(items, key):
    counts, selected = Counter(), []
    for item in items:
        expected = key(item)
        limit = 2 if hasattr(item, "input") else 1
        if counts[expected] < limit:
            selected.append(item)
            counts[expected] += 1
    return selected


def classify(skill, policy, model=None):
    cases = subset(skill.test, lambda x: x.expected) if policy == "frontier" else skill.test
    majority = Counter(row.expected for row in skill.train).most_common(1)[0][0]
    rows = []
    for row in cases:
        usage = None
        if model:
            raw, elapsed, tokens = model.predict(skill, row.input)
            usage = getattr(model, "last_usage", None)
        else:
            start = time.perf_counter()
            label = arithmetic_rule(row.input) if policy == "rules" else majority
            raw, tokens = json.dumps({"bucket": label, "line": 2}), None
            elapsed = (time.perf_counter() - start) * 1000
        result = grade(skill, row.input, raw, row.expected)
        result.update(id=row.id, input=row.input, expected=row.expected, elapsed_ms=elapsed, input_tokens=tokens, usage=usage)
        rows.append(result)
    return dict(metrics=summarize(rows), rows=rows)


def episodes(skill, policy, model=None):
    scenarios = store.read_json(DEST / "scenarios.json")["test"]
    if policy == "frontier":
        scenarios = subset(scenarios, lambda x: x["expected"])
    majority = Counter(s["expected"] for s in store.read_json(DEST / "scenarios.json")["train"]).most_common(1)[0][0]
    rows = []
    for scenario in scenarios:
        predictions = []
        with tempfile.TemporaryDirectory(prefix="workshop-practice-") as tmp:
            env = Practice(scenario, tmp)
            while not env.done:
                observation = env.observe()
                usage = None
                if model:
                    raw, elapsed, tokens = model.predict(skill, observation)
                    usage = getattr(model, "last_usage", None)
                    parsed = grade(skill, observation, raw)
                    action = parsed["output"]["bucket"] if parsed["valid"] and parsed["evidence_valid"] else "invalid-output"
                else:
                    start = time.perf_counter()
                    action = diagnostic_rule(observation)
                    if policy == "majority" and action not in ("read-log", "read-history"):
                        action = majority
                    elapsed, tokens = (time.perf_counter() - start) * 1000, None
                    raw = json.dumps({"bucket": action})
                predictions.append(dict(raw=raw, elapsed_ms=elapsed, input_tokens=tokens, usage=usage))
                env.step(action)
            outcome = env.outcome()
            rows.append(dict(id=scenario["id"], family=scenario["family"], **outcome, trace=env.trace, predictions=predictions))
    latencies = [p["elapsed_ms"] for row in rows for p in row["predictions"]]
    metrics = {name: sum(row[name] for row in rows) for name in ("success", "harmful", "invalid", "escalated", "unnecessary_escalation", "corrections_needed", "steps")}
    metrics.update(count=len(rows), success_rate=metrics["success"] / len(rows), median_ms=statistics.median(latencies))
    return dict(metrics=metrics, rows=rows)


def run(skill_id, policy):
    skill = Skill.model_validate(store.read_json(store.ROOT / "skills" / skill_id / "skill.json"))
    manifest = store.read_json(DEST / "manifest.json")
    if digest(skill.model_dump()) != manifest["skills"][skill_id]:
        raise ValueError("Frozen skill changed; create a new experiment instead")
    if digest(store.read_json(DEST / "scenarios.json")) != manifest["scenarios_sha"]:
        raise ValueError("Frozen scenarios changed")
    model = None
    checkpoint = None
    if policy == "train":
        identity = store.new_id("job")
        directory = store.STATE / "jobs" / identity
        job = dict(id=identity, kind="train", model="base", skill_id=skill_id, iterations=200,
                   status="running", created_at=time.time(), started_at=time.time(), model_id=store.MODEL_ID,
                   model_revision=store.MODEL_REVISION, skill_fingerprint=digest(skill.model_dump()),
                   test_fingerprint=digest([r.model_dump() for r in skill.test]), prompt_fingerprint=digest(prompt_for(skill)),
                   training_count=len(skill.train), validation_count=len(skill.validation))
        store.atomic_json(directory / "skill.json", skill.model_dump())
        store.atomic_json(directory / "job.json", job)
        try:
            train(job, directory, skill)
            job["status"] = "completed"
        except Exception as exc:
            job.update(status="failed", error=str(exc))
            raise
        finally:
            job["finished_at"] = time.time()
            store.atomic_json(directory / "job.json", job)
        store.atomic_json(DEST / f"{skill_id}-checkpoint.json", {k: v for k, v in job.items() if k != "adapter_path"})
        print(json.dumps({"checkpoint": identity, "status": job["status"]}), flush=True)
        return
    if policy == "specialist":
        checkpoint = store.read_json(DEST / f"{skill_id}-checkpoint.json")
        job = store.load_job(checkpoint["id"])
        import hashlib
        weights = Path(job["adapter_path"]) / "adapters.safetensors"
        if hashlib.sha256(weights.read_bytes()).hexdigest() != checkpoint["adapter_sha256"]:
            raise ValueError("Checkpoint hash mismatch")
        model = LocalModel(job["adapter_path"])
    if policy == "base":
        model = LocalModel()
    if policy == "frontier":
        model = Frontier()
    start = time.perf_counter()
    result = classify(skill, policy, model) if skill_id == "blox-arithmetic" else episodes(skill, policy, model)
    result.update(skill_id=skill_id, policy=policy, model_id=store.MODEL_ID if policy in ("base", "specialist") else policy,
                  revision=store.MODEL_REVISION if model and policy != "frontier" else None,
                  skill_sha=digest(skill.model_dump()), checkpoint=checkpoint,
                  wall_seconds=round(time.perf_counter() - start, 3),
                  latency_scope="generation only for MLX; full CLI invocation for frontier; model loading excluded")
    store.atomic_json(DEST / f"{skill_id}-{policy}.json", result)
    print(json.dumps({"skill": skill_id, "policy": policy, **result["metrics"]}), flush=True)
    del model
    gc.collect()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("skill", choices=["blox-arithmetic", "ci-diagnostic"])
    parser.add_argument("policy", choices=["rules", "majority", "base", "train", "specialist", "frontier"])
    args = parser.parse_args()
    store.initialize()
    output = DEST / f"{args.skill}-{args.policy}.json"
    if output.exists() or (args.policy == "train" and (DEST / f"{args.skill}-checkpoint.json").exists()):
        raise ValueError("Result exists; retain it rather than silently overwrite")
    if args.policy == "frontier":
        run(args.skill, args.policy)
        return
    with (store.STATE / "accelerator.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        run(args.skill, args.policy)


if __name__ == "__main__":
    main()
