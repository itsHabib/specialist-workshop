# Specialist Workshop

A local experiment workbench for finding and training useful specialists. Start with [the Monday walkthrough](docs/MONDAY.md) or open [Task packages](http://127.0.0.1:8787/packages). Import a task contract and data, compare baselines, train real weights, inspect mistakes, and save corrections as new immutable versions.

The original workshop remains a working POC for teaching a language model one narrow task. Bring a skill package, submit an input, inspect structured output, save corrections, train an actual QLoRA adapter, and compare recorded results.

Worked examples: [bounded analytics planning](docs/ANALYTICS-EXPERIMENT.md) separates interpretation from deterministic execution. [The practice lab](http://127.0.0.1:8787/practice) runs a resettable CI evidence workflow. Two further trained adapters were measured against rules and a frontier reference; neither earned rollout. See [workflow experiments](docs/WORKFLOW-EXPERIMENTS.md).

Measured first result: label accuracy improved from 33.3% to 54.9%, below the 56.9% majority-label baseline. This proves the learning loop runs; the classifier needs better data. See [the full experiment](docs/EXPERIMENT.md).

The first skill comes from Workbench: classify a failed CI log as `real-break`, `infra`, or `flake`, with a verbatim evidence quote. The original skill interface accepts other text-classification tasks; the package interface also supports custom JSON-object schemas and installed semantic graders. It does not call Gate, retry jobs, send mail, or alter Workbench state.

## Run

The local training backend requires Apple Silicon. The development machine used Python 3.14. Task packages use pinned Qwen3-4B (about 2.5 GB); the original classifier uses Qwen2.5-1.5B (about 1 GB). Models download from Hugging Face on first use; subsequent local inference and training use the cache.

```sh
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/uvicorn workshop.app:app --host 127.0.0.1 --port 8787
```

Open [Task packages](http://127.0.0.1:8787/packages) and follow [the walkthrough](docs/MONDAY.md). `requirements-lock.txt` records the current experiment environment. `requirements.txt` is the smaller installation contract. [Readiness evidence and review](docs/READINESS.md) describe what was checked.

The [original classification workshop](http://127.0.0.1:8787) also remains available:

1. Run a base-model inference or evaluation.
2. Train a specialist. The default recipe runs 150 supervised steps over practice examples.
3. Select its checkpoint and evaluate it on the reserved cases.
4. Inspect failures in **Experiments**. Add new practice examples through **Teach a correction**.
5. Import a different task through **Skill library**, using the included starter template.

Training writes real `.safetensors` adapter weights. There are no fake model calls, prefilled scores, retrieval-as-training substitutions, or automatic remote fallbacks. A poor result remains visible.

## Original classification API

```sh
curl http://127.0.0.1:8787/api/infer \
  -H 'Content-Type: application/json' \
  -d '{"skill_id":"ci-triage","input":"error TS2322: Type string is not assignable to number","model":"base"}'
```

The result includes raw output, parsed output, contract validity, quote validity, generation latency, model identity, and a durable inference ID. CI triage uses numbered source lines internally: the model selects a line, and the platform returns that exact original line as evidence. Other skills can choose direct quote generation. Correctness is **unknown** for unlabeled live inputs; a quote matching the input does not establish a correct diagnosis.

For a trained model, set `model` to `specialist` and provide a completed training run ID as `checkpoint`. Checkpoints are checked against the skill contract, base model revision, and weights hash.

| Endpoint | Purpose |
| --- | --- |
| `GET /api/skills` | List skills and data counts |
| `POST /api/skills` | Import a complete skill package |
| `POST /api/skills/{id}/corrections` | Add or replace a practice example |
| `POST /api/infer` | Run a single input |
| `POST /api/runs` | Start `train` or `evaluate` |
| `GET /api/runs/{id}` | Read progress, raw logs, and per-case results |
| `GET /openapi.json` | Machine-readable API contract |

See [the starter skill](static/skill-template.json) for the package format. The original skill interface uses the output contract: `{bucket, evidence}`. Inputs, label definitions, instructions, and separate train/validation/test datasets are customizable. Arbitrary uploaded code and tool execution are outside this POC.

## Optional reference provider

Configure an OpenAI-compatible chat-completions API on the server:

```sh
export WORKSHOP_REFERENCE_URL=https://your-provider.example/v1
export WORKSHOP_REFERENCE_MODEL=your-model-id
export WORKSHOP_REFERENCE_KEY=your-key
```

The endpoint receives data only when the caller explicitly chooses `reference`. It can be a local compatible server or a cloud provider; the configured model identity is recorded. Credentials never go to the browser. No provider is configured by default. There are no measured frontier-model results in the initial POC.

## Data and experiment boundaries

- Practice data: 240 authored synthetic examples built from 60 CI-error signatures. Validation: 9 distinct authored signatures. These are demonstrations, not independently verified production traces.
- External diagnostic data: Workbench's existing 51 labeled cases, preserved byte for byte with source revision and SHA-256. There are 41 CI excerpts and 10 isolated lines; historical labels have not been independently re-adjudicated.
- The external cases never enter gradient updates. Exact normalized input overlap across splits is rejected. Semantic duplicates and related incident families still require human review.
- The initial 0.5B result informed the decision to try 1.5B; direct quote failures informed the line-selection representation. These are exploratory diagnostic results, not an untouched final qualification or evidence of production reliability.
- Every run snapshots the skill and data. Corrections affect future training; they cannot alter a previous checkpoint or score.
- Latency is per-request generation, excluding model load. Total request time includes loading and is separately visible in the UI. No invented dollar-cost or speedup claims; local energy and hardware costs are unmeasured.

Local artifacts live under `.state/` (gitignored), or `WORKSHOP_STATE`. Keep the model cache and adapter directories to reproduce a checkpoint. Training and inference share an exclusive local accelerator lock; additional operations are rejected while it is occupied.

## Development

```sh
.venv/bin/python -m pytest -q
node --check static/app.js
node --check static/packages.js
node --check static/practice.js
node --check static/analytics.js
```

This is a single-user localhost application. Run one Uvicorn worker. It is not ready for exposure to other machines: hosted use requires authentication, tenant isolation, quotas, and a durable job service.

Architecture and the connection to the DeepSeek book: [design notes](docs/DESIGN.md). Initial measurements and limitations: [experiment report](docs/EXPERIMENT.md).
