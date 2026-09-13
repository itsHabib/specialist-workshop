"""Authored split families and checked demonstrations, not production traces."""
import hashlib
import json
from pathlib import Path
import random
import sys
import tempfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from workshop.arithmetic import LABELS, arithmetic_rule
from workshop.domain import Skill, digest
from workshop.practice import ACTIONS, Practice, diagnostic_rule
from workshop.store import ROOT, atomic_json

DEST = ROOT / "experiments" / "workflows"


def example(identity, text, expected, family):
    return dict(id=identity, input=text, expected=expected, evidence=text.splitlines()[-1], source=f"authored family: {family}")


def arithmetic_skill():
    splits = dict(train=[], validation=[], test=[])
    rng = random.Random(871)
    for split, count in (("train", 40), ("validation", 8), ("test", 12)):
        for i in range(count):
            reps, seconds, rest = rng.randint(3, 8), rng.randint(20, 95), rng.choice([15, 30, 45])
            warm, cool = rng.choice([120, 240, 300]), 120
            for label in LABELS:
                identity = f"math-{split}-{i}-{label}"
                if label in ("consistent", "time-overrun"):
                    margin = rng.choice([1, 15, 60]) * (1 if label == "consistent" else -1)
                    if split == "train":
                        family = "repetition-budget" if i % 2 else "single-benchmark"
                        text = f"{reps} repetitions of {seconds} seconds; {rest} seconds rest between repetitions; budget {reps * seconds + (reps - 1) * rest + margin} seconds."
                        if i % 2 == 0:
                            text = f"Benchmark completion {reps * seconds} seconds; block budget {reps * seconds + margin} seconds."
                    if split == "validation":
                        family = "pace-budget"
                        distance, pace = 1000, seconds * 6
                        text = f"{distance} metres at {pace} seconds per kilometre; budget {pace + margin} seconds."
                    if split == "test":
                        family = "compound-budget"
                        total = warm + reps * seconds + (reps - 1) * rest + cool
                        text = f"Warmup {warm} seconds, {reps} repetitions of {seconds} seconds, {rest} seconds rest between repetitions, cooldown {cool} seconds; budget {total + margin} seconds."
                if label == "distance-conflict":
                    family = f"{split}-distance"
                    lane = rng.choice([10, 15, 20])
                    if split != "test":
                        text = f"{reps} down-and-back trips on a {lane} metre lane; claimed distance {reps * lane} metres."
                    if split == "validation":
                        text = f"{reps} lengths of {lane} metres; claimed distance {2 * reps * lane} metres."
                    if split == "test":
                        text = f"Technique {reps} lengths of {lane} metres plus test 40 metres; claimed total {reps * lane} metres."
                if label == "needs-context":
                    family = f"{split}-missing"
                    text = f"{reps} repetitions at a self-selected pace; rest as needed; budget {warm + cool} seconds. Work duration is unspecified."
                    if split == "validation":
                        text = f"{reps * 100} metres at an unspecified pace; budget {warm} seconds."
                    if split == "test":
                        text = f"Warmup {warm} seconds, {reps} sled pulls of unspecified duration, cooldown {cool} seconds; budget 1200 seconds."
                text = f"Prescription {identity.removeprefix('math-')}\n{text}"
                # IDs must not disclose targets to the model.
                text = f"Prescription {hashlib.sha256(identity.encode()).hexdigest()[:8]}\n" + text.splitlines()[1]
                assert arithmetic_rule(text) == label
                splits[split].append(example(identity, text, label, family))
    # Procedural primitives are shared; operation families are assigned by split.
    return Skill(id="blox-arithmetic", name="Blox arithmetic screen", description="Check explicit time and distance arithmetic; escalate missing quantities. Synthetic controlled transfer experiment.",
                 instructions="Check only the explicit arithmetic in this prescription. Rest BETWEEN n repetitions occurs n-1 times. A down-and-back trip covers twice the lane length. Include warmup, cooldown and technique when stated. Missing work duration or pace means needs-context. Do not infer ability or coaching safety. Ignore the opaque prescription identifier.",
                 labels=LABELS, evidence_mode="line", **splits,
                 provenance={"origin": "Authored synthetic arithmetic inspired by exposed Blox failure cases; no personal workout history.",
                             "split": "Train: repetitions, benchmarks, round trips. Dev: pace and lengths. Final: compound-time and summed-distance. Arithmetic primitives are shared; final operation families are absent from training.",
                             "labels": "Generator arithmetic cross-checked by executable parser; same author, not independent adjudication."})


FAMILIES = {
    "train": [
        ("assertion", "AssertionError: expected 4 rows, received 7", "advise-repo"),
        ("typescript", "error TS2322: Type string is not assignable to number", "advise-repo"),
        ("auth", "failed to authenticate to package registry", "advise-infra"),
        ("dns", "Could not resolve host: registry.example", "advise-infra"),
        ("timeout-pass", "test timed out waiting for readiness", "advise-retry"),
        ("socket-pass", "ECONNREFUSED 127.0.0.1:5432", "advise-retry"),
        ("timeout-stale", "test timed out waiting for readiness", "escalate"),
        ("opaque", "worker stopped without a diagnostic", "escalate"),
    ],
    "validation": [
        ("undefined", "undefined: BuildConfig", "advise-repo"),
        ("rate-limit", "429 Too Many Requests from registry", "advise-infra"),
        ("busy-pass", "EBUSY: resource busy or locked", "advise-retry"),
        ("opaque-stale", "step killed unexpectedly", "escalate"),
    ],
    "test": [
        ("coverage", "coverage below threshold: 72% required 80%", "advise-repo"),
        ("syntax", "SyntaxError: unexpected token in src/config.js", "advise-repo"),
        ("disk", "write /tmp/build: no space left on device", "advise-infra"),
        ("shutdown", "Runner received a shutdown signal and disconnected", "advise-infra"),
        ("port-pass", "EADDRINUSE: address already in use :4000", "advise-retry"),
        ("database-pass", "the database system is starting up", "advise-retry"),
        ("port-stale", "EADDRINUSE: address already in use :4000", "escalate"),
        ("cleanup-only", "Process completed with exit code 1\nPost job cleanup\nfailed to authenticate during cleanup", "escalate"),
    ],
}


def diagnostic_skill():
    splits, scenarios = {}, {}
    for split, families in FAMILIES.items():
        rows, episodes = [], []
        for family, cause, expected in families:
            count = 12 if split == "train" else 2
            for i in range(count):
                identity = f"ci-{split}-{family}-{i}"
                commit = hashlib.sha256(identity.encode()).hexdigest()[:12]
                history = f"commit={commit} conclusion=success" if expected == "advise-retry" else f"commit=000000000000 conclusion=success\ncommit={commit} conclusion=failure"
                scenario = dict(id=identity, family=family, commit=commit,
                                log=f"Run {i + 1}: checking application\n{cause}\nProcess completed with exit code 1\nPost job cleanup\nno space left on device", history=history, expected=expected)
                episodes.append(scenario)
                with tempfile.TemporaryDirectory() as tmp:
                    env = Practice(scenario, tmp)
                    while not env.done:
                        obs = env.observe()
                        action = diagnostic_rule(obs)
                        rows.append(example(f"{identity}-step-{len(env.trace)}", obs, action, family))
                        env.step(action)
                    assert env.outcome()["success"], (scenario, env.outcome())
        splits[split], scenarios[split] = rows, episodes
    skill = Skill(id="ci-diagnostic", name="CI evidence practice", description="Learn a finite evidence-gathering procedure and write an advisory in a resettable local environment.",
                  instructions="Select the NEXT action in this CI evidence workflow. Read the log first. A clear compile, syntax, lint, assertion or coverage failure calls for advise-repo. Authentication, registry DNS, disk-full or runner shutdown calls for advise-infra. Ignore generic exit wrappers and every line after Post job cleanup. Otherwise read history. Only a success for the exact current commit supports advise-retry. A different commit's success proves nothing. With unresolved evidence after history, escalate. These are advisories only. Never repeat a read or decide before gathering required evidence.",
                  labels=ACTIONS, evidence_mode="line", **splits,
                  provenance={"origin": "Authored state-machine fixtures grounded in Workbench ci-classify signatures and wrapper exclusions.",
                              "split": "Signature families fixed by split; current commit hashes varied. Port-pass and port-stale share one final-only signature.",
                              "scope": "Simulated log/history tools and real disposable advisory files. No real CI retries, code repairs, network or Gate actions."})
    return skill, scenarios


def main():
    math = arithmetic_skill()
    diagnostic, scenarios = diagnostic_skill()
    existing = DEST / "manifest.json"
    if existing.exists():
        recorded = json.loads(existing.read_text())
        current = {s.id: digest(s.model_dump()) for s in (math, diagnostic)}
        if current != recorded["skills"] or digest(scenarios) != recorded["scenarios_sha"]:
            raise ValueError("Frozen experiment differs. Use a new experiment version; do not rewrite its splits.")
    for skill in (math, diagnostic):
        atomic_json(ROOT / "skills" / skill.id / "skill.json", skill.model_dump())
    atomic_json(DEST / "scenarios.json", scenarios)
    atomic_json(DEST / "manifest.json", dict(version=1, skills={s.id: digest(s.model_dump()) for s in (math, diagnostic)},
                scenarios_sha=digest(scenarios), protocol_sha=hashlib.sha256((DEST / "PROTOCOL.md").read_bytes()).hexdigest(),
                note="Frozen before model runs. Synthetic, author-known. Original CI evidence untouched."))
    print({s.id: {k: len(getattr(s,k)) for k in ("train","validation","test")} for s in (math,diagnostic)})


if __name__ == "__main__":
    main()
