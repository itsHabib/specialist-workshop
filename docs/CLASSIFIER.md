# Original classifier and API

The earlier text-classification interface remains available at `http://127.0.0.1:8787/`. It uses a pinned Qwen2.5-1.5B model; task packages use Qwen3-4B. These details and measurements describe the original classifier experiment.



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

See [the starter skill](../static/skill-template.json) for the package format. The original skill interface uses the output contract: `{bucket, evidence}`. Inputs, label definitions, instructions, and separate train/validation/test datasets are customizable. Arbitrary uploaded code and tool execution are outside this POC.

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

Architecture and the connection to the DeepSeek book: [design notes](DESIGN.md). Initial measurements and limitations: [experiment report](EXPERIMENT.md).
