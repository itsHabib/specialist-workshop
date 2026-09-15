# Bounded judgment: a RoxIQ analytics planner

This experiment tests a better specialist task: interpreting a question and
composing a typed analysis plan. Arithmetic and query validation remain in code.
The earlier arithmetic/CI specimens are preserved in
[WORKFLOW-EXPERIMENTS.md](WORKFLOW-EXPERIMENTS.md); their poor task selection is not
evidence against specialization in general.

Open **http://127.0.0.1:8787/analytics**. Ask a question, choose the base model,
structured prompt or trained adapter, inspect the selected result IDs, metrics
and comparison scope, and view deterministic fixture calculations. Ambiguous
requests produce a clarification reason; unsupported evidence produces no query.
The explanations/questions displayed by the UI are templates, not separately
trained or evaluated model prose.

## Measured outcome

No specialist earned rollout. On the 24 frozen requests, the original strict
JSON evaluation produced:

| Policy | Correct intent | Valid schema/context | Unjustified executions | Median latency |
| --- | ---: | ---: | ---: | ---: |
| Always clarify metric | 1/24 | 24/24 | 0 | deterministic |
| Qwen3-4B base | 7/24 | 18/24 | 6 | 3,339.5 ms generation |
| Base plus checklist | 7/24 | 16/24 | 5 | 2,997.5 ms generation |
| Trained adapter, strict raw JSON | 0/24 | 0/24 | 0 | 2,797.5 ms generation |
| Frontier reference | 24/24 | 24/24 | 0 | 3,168.5 ms whole CLI |

The adapter's original zero concealed an integration error: the pinned tokenizer
inserts an empty `<think>...</think>` channel into assistant training messages,
and all 24 completions reproduced it before their JSON. Inspecting the actual
serialized training target traced this to the model template. The platform now
previews that boundary before training, preserves raw output, and separately
scores content after recognizing only an empty leading channel.

A **post-hoc replay of the retained completions**, without rerunning the model or
changing labels, gives the adapter **12/24 correct, 21/24 valid, and six
unjustified executions**. Base/checklist/reference scores remain unchanged.
[The replay](../experiments/analytics/transport-replay.json) is a diagnostic under
a revised completion protocol, not a new untouched qualification. The original
strict scores remain visible. Removing an empty transport channel does not repair
wrong identities, scopes or decisions. Nonempty reasoning and arbitrary malformed
prefixes remain invalid. The always-clarify control also unnecessarily abstained
on 13 executable requests.

The 120-step training took 537.46 seconds with a reported peak MLX allocation of
5.578 GB, updating 3.670 million parameters (0.091% of the model). Development
loss fell from 3.473 to 0.095. This was a short run with fewer updates than one
nominal pass over 160 examples; it does not establish an optimized recipe.
The reference's 24 requests reported $0.3022495 in API-equivalent CLI usage,
not a verified incremental invoice. Different timing scopes do not establish
a serving speed advantage. Local hardware/energy costs were not measured.

This experiment supplied a useful platform requirement: make model templates,
raw outputs, semantic targets and decoder versions inspectable. The next task
can reuse that machinery through [task packages](MONDAY.md), while preserving
this negative result.

## Why this resembles the Shopify task

Shopify's Flow specialist translates merchant intent into a composed automation,
with semantic and syntactic evaluation kept separate. Its initial data was built
backwards from validated workflows. The production system used Qwen3-32B and
subsequent corrective feedback, so our small local study is not a comparable
performance test. [Shopify Flow](https://shopify.engineering/fine-tuning-agent-shopify-flow).

Shopify's GraphQL account similarly emphasizes an explicit quality rubric and
annotator agreement before using a judge at scale.
[Sidekick's continual learning loop](https://shopify.engineering/sidekicks-continual-learning-loop).
Here a separate agent derived intended query plans from a blind packet before
scoring. Executable JSON alone does not establish that the plan answers the user.
These sources motivated the task shape and evidence separation, not a claim that
we reproduced their scale or production economics.

## Actual RoxIQ capabilities and the fixture boundary

Read-only inspection at RoxIQ revision
`44b148fd3c93a6c804287274733f367dbfff393d` established these seams:

- `internal/model/model.go`: athlete/result identity, event/season metadata,
  nullable split times, running aggregates and four comparison scopes.
- `internal/api/handler.go`: result retrieval, result-bound percentile responses,
  scope mapping and pooled percentile query arguments.
- `internal/api/compare.go`: selected-result comparison with explicit identity
  checks, rather than treating a result ID as an athlete ID.
- `migrations/012_sex_scope_percentiles.up.sql` and
  `014_total_time_percentiles.up.sql`: pooled demographic, division, sex and
  global references. There is no event or season percentile filter, and these
  references omit `run_total` and `best_run_lap`.

The POC exposes a subset through five fields: action, result IDs, metrics, scope,
and clarification/unavailability reason. `value` retrieves times; `delta`
subtracts the first selected result from the second; `benchmark` compares times
with pooled p50. Metric order is irrelevant. Result order matters for delta;
value/benchmark selection order is normalized for semantic scoring.

A request about whether the whole field was faster at one event is unsupported
by this contract. Event/season metadata selects a result; it cannot be used to
silently pretend the pooled percentile table has an event/season filter. Exact
percentile-rank estimation, population field changes, causal physiology, complete
live coverage and arbitrary SQL are outside the implemented surface.

The catalog and numeric values are synthetic fixtures shaped by those real
contracts. There is no production data access. All available result IDs and
nullable-field availability are supplied explicitly to the model. The executor
uses allowlisted fields and typed IDs; it computes values independently of model
output. Its pooled references are consistent across results sharing a cohort.
Fixture execution is not proof that a live query has adequate samples or data.

## Frozen data and independent intent adjudication

There are 160 authored training pairs, six development cases and 24 final
requests. The final set exercises unfamiliar paraphrases and compositions:
multiple metrics, scope negation, explicit athlete override, duplicate names,
missing profile/race/metric/cohort, missing splits, unavailable season results,
unavailable percentile metrics, schema distractors and reversed subtraction.
Final result IDs do not appear in training or development. Shared vocabulary,
event names and semantic primitives are intentional; identities and phrasing
are not diverse production traffic. This is a narrow authored transfer diagnostic.

The separate coordinating agent received only `contract.txt`, `test-inputs.json`
and source-contract pointers. It independently returned every intended five-field
plan and caveats. It did not see training/development pairs, proposed final
labels or predictions. No packet edits were required. The resulting
[test-adjudicated.json](../experiments/analytics/test-adjudicated.json) was hashed
before scoring; the receipt is
[adjudication.json](../experiments/analytics/adjudication.json).
These are independently derived **model judgments**, not human expert ground
truth or a user trial. Final labels never entered gradient updates.

Models receive only request and context plus the same contract. Family tags,
case IDs and expected plans are excluded. Training streams contain only the
practice pairs and development validation messages; recorded manifests also hash
final inputs for continuity, which does not put those inputs into training.

## Comparisons and what is measured

The base prompt already includes a detailed schema, metric meanings, scope
semantics, identity rules and capability limits. The structured-prompt condition
adds an explicit identity → race → metric → cohort → availability checklist;
it is not a broad prompt-search baseline. The specialist uses the base prompt.
The reference is the existing tool-free Claude CLI at low effort, reported as
`claude-opus-5[1m]`, with ancillary Haiku usage recorded by the CLI. No final-case
retries or post-hoc response repair are allowed.

Exact intent match and schema/context validity are separate. A plan can be valid
but select the wrong analysis. Unjustified execution counts a ready plan for a
request whose independent target was clarification or unavailable evidence.
Needless abstention counts the reverse. For every valid output the runner records
both its deterministic fixture result and the independently intended result.
Correction burden is case count, not timed human effort. These measures do not
assess natural-language answer quality, explanation fidelity or live availability.

The always-clarify control demonstrates why avoiding all execution is not useful.
There is no hand-coded natural-language answer table. Deterministic code validates
and executes the model's selection; it does not solve the interpretation task.

## Training and evidence

The model is `mlx-community/Qwen3-4B-Instruct-2507-4bit`, pinned to
`50d427756c6b1b2fe0c0a10f67fbda1fc8e82c1b`. Training updates actual LoRA weights:
120 supervised steps, batch one, last eight layers, learning rate 2e-5, seed 29,
assistant-only loss, 2048-token limit. The largest training message is 912 tokens;
no target was truncated. Greedy local generation allows 260 output tokens.
Training fits the available Apple Silicon Mac; this is much smaller than the
32B Shopify example and uses a tiny authored dataset without customer feedback.

The fixed protocol, input manifests, independent test receipt, training recipe,
weight digest, losses and every policy's raw output are under
[experiments/analytics](../experiments/analytics). Weights stay in
`.state/analytics/adapter/` outside Git. The shared accelerator lock serializes
local model operations. No RL, GRPO, teacher distillation or foundation-model
pretraining was performed.

Local latency measures generation, excluding model load; browser requests include
loading. Frontier latency measures the entire CLI invocation, with the CLI's own
request duration separately recorded. API-equivalent CLI usage is an estimate
under existing authenticated access, not a verified incremental invoice. Local
energy and amortized hardware costs were not measured; no zero-cost claim is made.

## Run and API

```sh
.venv/bin/uvicorn workshop.app:app --host 127.0.0.1 --port 8787
curl http://127.0.0.1:8787/api/analytics/infer \
  -H 'Content-Type: application/json' \
  -d '{"question":"Between Harbor and Summit, how did my running and transition times change?","policy":"specialist"}'
```

`GET /api/analytics` returns the fixed demo context and recorded comparisons.
`POST /api/analytics/infer` accepts a question and `base`, `structured` or
`specialist`. It returns the raw model response, checked plan, fixture calculation,
model identity and generation latency. For a new free-text request, intent
correctness is deliberately null. The demo uses its displayed fixture context;
selecting an example question does not import that evaluation case's context.

The dedicated analytics runner extends the existing local workshop rather than
adding a generic orchestration framework. Training/evaluation scripts refuse to
overwrite prior evidence. The original classifier and resettable CI environment
remain available. Nothing is promoted into RoxIQ, Ivy, Workbench or Gate.
