"""Build synthetic practice; vendor the untouched Workbench evaluation separately."""

import hashlib
import json
from pathlib import Path
import random

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "skills/ci-triage/workbench-evaluation.jsonl"

TRAIN = {
    "real-break": [
        "src/worker.ts(18,4): error TS2345: Argument of type 'string' is not assignable to parameter of type 'number'.",
        "error[E0308]: mismatched types: expected u64, found String",
        "internal/parser.go:82: cannot use request (variable of type string) as int value",
        "AssertionError: expected status 202, received 400",
        "FAIL TestLeaseExpiry: expected lease to be released, got active",
        "error: unused variable: `request_id` [-D unused-variables]",
        "src/router.ts:17:5 error Unexpected any. Specify a different type @typescript-eslint/no-explicit-any",
        "[warn] Code style issues found in the above file. Run Prettier with --write to fix.",
        "ERROR: Coverage for functions (71.2%) does not meet global threshold (80%)",
        "Diff in /workspace/src/pool.rs:47: formatting does not match rustfmt",
        "go.mod: invalid go version 'latest': must match format 1.23.0",
        "ERROR: package-lock.json is not up to date with package.json. Run npm install.",
        "Security advisory RUSTSEC-2025-0999: vulnerable dependency in Cargo.lock must be upgraded",
        "ValueError: invalid literal for int() with base 10: 'abc' in parse_customer_id",
        "error: struct `Job` has no field named `deadline`",
        "FAIL tests/test_invoice.py::test_rounding - assert 120 == 119",
        "Schema validation failed: required property 'version' missing in .golangci.yml",
        "ModuleNotFoundError: No module named 'project.handlers.removed_handler'",
        "error: conflicting implementations of trait Serialize for type Record",
        "eslint: 4 errors, 0 warnings. src/delivery.ts exceeds max-params limit",
    ],
    "infra": [
        "Runner provisioning failed: no hosted runners available for this organization",
        "Error: ENOSPC: no space left on device, write '/runner/_work/_temp/output'",
        "Error: Failed to exchange GitHub App installation token: HTTP 401 Unauthorized",
        "npm ERR! code E429 registry.npmjs.org: Too Many Requests",
        "Error: failed to download provider: registry.terraform.io returned 503 Service Unavailable",
        "fatal: unable to access 'https://github.com/example/repo/': Could not resolve host: github.com",
        "setup-go: cannot find /runner/_work/project/go.mod specified in cache-dependency-path",
        "Error: missing required CI secret SERVICE_API_KEY; setup action cannot authenticate",
        "The runner has received a shutdown signal. This job will not be continued.",
        "Error: failed to pull base image: Docker Hub anonymous pull rate limit exceeded",
        "actions/checkout failed: repository token has expired",
        "Error: CI service account is suspended; contact organization administrator",
        "npm ERR! network request to https://registry.npmjs.org/left-pad failed: ETIMEDOUT",
        "TLS certificate verification failed for corporate package proxy; runner trust store missing CA",
        "Runner disk quota exceeded while unpacking toolchain archive",
        "Error: could not obtain cloud credentials from OIDC provider: audience mismatch in CI identity configuration",
        "Failed to start job: self-hosted runner is offline",
        "Setup action refused to run: API key must be configured in repository secrets",
        "Error: artifact storage service unavailable (HTTP 502) before build started",
        "git checkout failed: Permission denied (publickey). Runner deploy key is revoked.",
    ],
    "flake": [
        "Error: connect ECONNREFUSED 127.0.0.1:5432; postgres health check still starting",
        "FATAL: the database system is starting up; retry connection after readiness probe",
        "Error: listen EADDRINUSE: address already in use :::4310 from previous test process",
        "Error: EBUSY: resource busy or locked, unlink 'C:\\temp\\suite.lock'",
        "Test checkout timed out on attempt 1; retry attempt 2 passed with the same commit",
        "Postgres socket /tmp/.s.PGSQL.5432 not found yet; container startup in progress",
        "Intermittent race in integration test: passes in isolation and on immediate unchanged rerun",
        "Timeout waiting for browser on first attempt; test passed on retry",
        "Temporary connection reset during test request; retry succeeded",
        "FAIL TestConcurrentRefresh: scheduler race; unchanged rerun PASS",
        "redis connection refused while service container is becoming healthy",
        "Windows cleanup failed: file is being used by another process; next attempt succeeded",
        "Error: socket hang up mid-test; local test service restarted and retry passed",
        "Health probe: service not ready yet. Startup took longer than the test wait.",
        "Test attempt 1 failed waiting for selector; attempt 2 passed without changes",
        "Bind failed: ephemeral test port is still occupied by the preceding suite",
        "Deadlock victim in parallel fixture transaction; retry succeeded",
        "Database reports recovery is in progress immediately after container launch",
        "Temporary test artifact lock contention; a clean retry completed successfully",
        "Playwright browser launch exceeded timeout under runner load; unchanged retry green",
    ],
}

VALID = {
    "real-break": [
        "error TS2322: Type 'undefined' is not assignable to type 'User' in src/session.ts",
        "FAILED test_total: expected decimal 40.50 but received 41.00",
        "ruff check: F401 imported but unused in service/api.py",
    ],
    "infra": [
        "Failed to prepare worker: runner image cannot mount its configured build volume",
        "Package mirror responded HTTP 504 to dependency installation request",
        "CI app authentication denied: installation was removed from this organization",
    ],
    "flake": [
        "Cannot connect to MySQL yet: initialization has not finished; subsequent check connected",
        "Suite encountered an intermittent timing failure; same revision succeeded on second run",
        "Temporary directory locked by test antivirus scan; retry cleanup succeeded",
    ],
}


def make_rows(catalog, variants, prefix):
    rows = []
    for label, lines in catalog.items():
        for index, line in enumerate(lines):
            for variant in range(variants):
                lead = ["", "[CI] Booting job\n[INFO] dependencies already cached\n",
                        "=== build and verify ===\nCompleted 12 checks successfully.\n",
                        "[INFO] checkout complete\n[INFO] running verification\nWarning: a newer CLI release is available\n"][variant]
                tail = ["", "\nProcess completed with exit code 1.",
                        "\nELIFECYCLE Command failed with exit code 1.\nCleaning up workspace.",
                        "\nmake: *** [verify] Error 1\nUploading diagnostics\nJob finished."][variant]
                rows.append(dict(id=f"{prefix}-{label}-{index}-{variant}", input=lead + line + tail,
                                 expected=label, evidence=line, source="authored synthetic practice; not production"))
    random.Random(17).shuffle(rows)
    return rows


def main():
    directory = ROOT / "skills" / "ci-triage"
    directory.mkdir(parents=True, exist_ok=True)
    raw = SOURCE.read_bytes()
    # Preserve exact bytes and labels. Neither external test labels nor logs enter training.
    (directory / "workbench-evaluation.jsonl").write_bytes(raw)
    test = [dict(id=f"wb-{i:02d}", input=row["input"], expected=row["expected"],
                 source=row["meta"]) for i, row in enumerate(json.loads(s) for s in raw.splitlines())]
    skill = dict(id="ci-triage", name="CI failure triage", evidence_mode="line",
                 description="Read a failed build log. Identify a repository break, environment failure, or likely flake, and quote the evidence.",
                 instructions="Classify a failed CI log. Ask: what would happen if the same commit ran unchanged? Look above generic exit-code wrappers for the actual cause. This is advisory triage, not permission to retry or merge.",
                 labels={"real-break": "A code, test, config, or dependency change in the repository is needed.",
                         "infra": "The CI environment, credentials, runner, or dependency registry needs repair.",
                         "flake": "A transient test or service-startup failure could pass on an unchanged retry."},
                 train=make_rows(TRAIN, 4, "practice"), validation=make_rows(VALID, 1, "validation"), test=test,
                 provenance={"repository": "itsHabib/workbench", "revision": "86638fa7e0df7a4203c1939fcdced576ef365510",
                             "path": "cmd/gate/docs/features/ci-classify/eval/ci-lines-v2.jsonl",
                             "sha256": hashlib.sha256(raw).hexdigest(),
                             "evaluation": "51 existing labeled cases: 41 CI excerpts and 10 isolated lines. Historical labels, not independently re-adjudicated.",
                             "training": "240 authored synthetic examples from 60 error signatures. 9 distinct validation signatures.",
                             "limits": "Not representative random traffic. Repeated families in the test set. Quote presence does not prove causal correctness."})
    (directory / "skill.json").write_text(json.dumps(skill, indent=2, ensure_ascii=False))
    print("Wrote 240 training, 9 validation, 51 external evaluation cases.")


if __name__ == "__main__":
    main()
