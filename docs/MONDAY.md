# Monday: bring one recurring work problem

Start at **http://127.0.0.1:8787/packages**. The goal is a repeatable experiment
that can find a worthwhile specialist task—not a mandatory fine-tuning win.
The generic package path supports contracts, JSON schemas, examples/provenance,
family-separated splits, real supervised LoRA, checkpoint inference, comparisons,
corrections as new versions, and a resettable environment interface.

Use actual company data only in a separately approved work environment with an
approved state directory and provider. This repository contains
synthetic examples and existing personal research fixtures. This workflow does
not connect employer systems or authorize moving company data here.

## 1. Decide whether the problem fits

Write: “Given **these inputs**, choose/compose **this bounded output**, using
**these tools**, and succeed when **this independently checkable intent/outcome**
holds.” Then identify who can supply credible corrections.

Good candidates involve recurring semantic judgment inside stable boundaries:
interpreting an operational request, choosing the intended analysis, composing
an allowlisted workflow, or resolving which missing information to request.
There should be enough examples/corrections to learn from, a defensible semantic
check as well as mechanical validation, and a plausible serving benefit.

Reject or defer tasks that code already solves reliably, lack trustworthy
feedback, have too little representative data, or are already adequate with a
base model/rules. A synthetic demo does not substitute for that evidence. Do not
train a language model to do arithmetic that belongs in a calculator.

The [RoxIQ planning specimen](ANALYTICS-EXPERIMENT.md) illustrates the right shape:
interpret athlete/race/metric/cohort intent, then execute deterministic queries.
Its current adapter is not qualified for production. The starter support package
is deliberately tiny and proves the plumbing only.

## 2. Make the task a package

Download the starter from the page, or copy
[examples/support-intake.package.json](../examples/support-intake.package.json).
Edit ordinary JSON: `id`, `name`, `instructions`, `output_schema`, `provenance`,
and the `train`, `development`, `final` examples. Each example has:

```json
{
  "id": "example-001",
  "family": "one-incident-or-intent-family",
  "input": {"message": "An original, approved example"},
  "expected": {"owner": "accounts", "urgency": "normal", "reason": "access"},
  "provenance": "Source and who checked this intended outcome"
}
```

Inputs are JSON objects or strings; outputs are JSON objects checked against a
self-contained JSON Schema. The instructions must describe the output contract
the model should produce; the schema is the mechanical checker and is not
silently added to the model prompt. External schema references and uploaded
executable code are rejected.

Keep an entire incident/task family in one split. Do not randomly distribute
paraphrases of one incident across training and final evaluation. Import rejects
duplicate IDs, normalized input overlap and repeated family IDs across splits.
Those checks cannot discover all semantic duplicates or verify that your family
labels and intended outputs are honest; reviewed dataset design remains necessary.

Use development data for iteration. Have someone independently adjudicate the
final intended outcomes before seeing predictions. Record their provenance;
do not confuse an LLM judge with human expert ground truth. The package's exact
JSON grader treats the supplied expected objects as the semantic target; a valid
schema does not make those labels trustworthy.

## 3. Import and inspect the actual model boundary

Browser import starts no inference or training. It returns an immutable version.
In the CLI, from the workshop directory:

```sh
.venv/bin/python -m workshop.experiment import examples/support-intake.package.json
```

Copy the returned `reference` into a shell variable, for example the checked-in
starter's reference:

```sh
TASK_REF='support-intake@dd72cbdfb154394c1a5aceefe8f808e4f82ba7356902f02e230c1f393b425a94'
.venv/bin/python -m workshop.experiment preview "$TASK_REF"
```

`preview` shows the actual serialized assistant target, tokenizer-template hash,
token length, pinned model and completion protocol. This matters: the pinned
Qwen template adds an empty thinking channel during SFT. The substrate records
raw output separately and recognizes only that exact empty channel boundary.
It does not extract arbitrary JSON from prose, rewrite labels or discard nonempty
reasoning to manufacture a pass.

The current package backend is pinned Qwen3-4B on Apple Silicon, with greedy
260-token generation and a 4,000-token inference limit. Training rejects examples
above 2,048 tokens. Changing model/backend or arbitrary decoding recipes still
requires a small reviewed code/configuration extension; there is no claim of
universal model support. The original classification UI separately retains its
pinned 1.5B backend.

## 4. Establish baselines, then train only if justified

Run the base and majority controls before interpreting an adapter:

```sh
.venv/bin/python -m workshop.experiment run "$TASK_REF" evaluate --policy base
.venv/bin/python -m workshop.experiment run "$TASK_REF" evaluate --policy majority
.venv/bin/python -m workshop.experiment run "$TASK_REF" train --steps 120
```

The majority control emits the most frequent complete training output, not an
invented language-understanding rule table. It is not useful for every task.
An environment can provide a deterministic baseline where appropriate.

The recipe is real supervised LoRA: batch one, last eight layers, learning rate
2e-5, seed 29, assistant-only loss. Each run snapshots the package, train/dev
message hashes, recipe, tokenization preview and complete adapter files. The
model weights are actual `.safetensors`. Model downloads and local compute need
available disk/memory; local hardware/energy costs are not measured as zero.

A reference API is optional and explicitly selected. Only configure it in an
approved environment for the data you will send:

```sh
# Configure WORKSHOP_REFERENCE_URL, WORKSHOP_REFERENCE_MODEL and
# WORKSHOP_REFERENCE_KEY for your approved compatible provider first.
.venv/bin/python -m workshop.experiment run "$TASK_REF" evaluate --policy reference
```

No remote fallback occurs. The package/run never stores the key. Provider usage
is retained when returned; dollar cost is unknown unless independently supplied
by the provider/measurement. Existing analytics reference records separately
preserve CLI-reported API-equivalent estimates, not invoices.

## 5. Exercise the checkpoint and inspect every failure

Copy the completed training run ID into `CHECKPOINT`, then use it explicitly:

```sh
CHECKPOINT='pkg-<the completed training run id>'
.venv/bin/python -m workshop.experiment run "$TASK_REF" evaluate \
  --policy specialist --checkpoint "$CHECKPOINT"
.venv/bin/python -m workshop.experiment run "$TASK_REF" infer \
  --policy specialist --checkpoint "$CHECKPOINT" --input-file input.json
```

`input.json` contains the input object/string, not the entire package. The browser
provides the same operations and shows raw outputs and individual scored cases.
A fresh input has `correct: null`; the system cannot establish its semantic
correctness without a trusted target/review.

Compare completed evaluation IDs using:

```sh
.venv/bin/python -m workshop.experiment compare pkg-<base-run> pkg-<adapter-run>
```

`evaluate` uses development cases by default. Iterate on that split, then freeze
the package, checkpoint and policy before explicitly running final evaluation:

```sh
.venv/bin/python -m workshop.experiment run "$TASK_REF" qualify \
  --policy specialist --checkpoint "$CHECKPOINT"
```

The browser exposes the same choice under **Final evaluation**. `qualify` records
final-case results; it does not automatically approve a model. Once you inspect
final failures, treat those cases as exposed and reserve fresh cases for a new
qualification claim. Older `evaluate` records remain labeled as final.

Comparisons reject different evaluated splits, case hashes, final-set identities,
graders or operations. Checkpoints bind the full adapter manifest, including configuration,
and the exact reserved development/final/episode sets. Moving old training examples into a
new development or final split cannot evaluate an old checkpoint. The current conservative
policy requires the same reserved evaluation identity for checkpoint reuse.

Read `valid` (schema/context), `raw_valid` (before transport decoding), `correct`
(expected intent), `accepted` (valid and matches the stored target), `abstained`,
`invalid`, latency and the complete raw response. “Accepted” is an evaluation
oracle metric, not live permission to act. If old records lack explicit raw-valid
fields, they are unmeasured; early smoke records with zero from missing fields
are retained and annotated rather than treated as a real format score.

Inspect misselected identities, omitted requirements, unjustified execution,
needless abstention and correction burden. A higher aggregate score can conceal
worse mistakes. Local generation timing excludes model loading; provider timing
may have a different scope. Preserve that distinction before claiming savings.

Before deciding whether a specialist is worthwhile, compare the **complete cost
of finishing tasks**, including retries, failed tasks, verification and fallback,
at agreed quality and latency. Three cheap attempts can beat one expensive call.
The package evaluator is single-attempt. The existing retry script accepts any
imported package and a compatible checkpoint, with development as its default:

```sh
.venv/bin/python scripts/retry_economics.py --package "$TASK_REF" \
  --checkpoint "$CHECKPOINT" --policies local-base local-specialist \
  --attempts 3 --output .state/my-development-comparison
.venv/bin/python scripts/summarize_retry.py .state/my-development-comparison
```

Together with the import and training commands above, this is the path from a
clean clone to a comparison using your own adapter. Supply a new output directory
for each run. `--split final` explicitly selects all final cases when you are ready.
No checkpoint or MLX is needed for API-only policies: `--policies gpt-5.6-luna`
uses `OPENAI_API_KEY`; Astra is also supported. API calls require that explicit
policy selection and have a default $3 reservation budget (`--api-budget-usd`).
`--attempts` applies to every selected model; use 1 for a one-shot control. The
report includes first-attempt and complete-policy scores from the same trajectories.

Prices are the recorded September 15, 2026 rates, not live billing quotes. Local
cost stays unknown. No fallback or human-review cost is inferred. The current
model/decoding presets remain fixed; this adds no backend configuration framework.
Partial attempts are saved and the run fails if a request or budget check fails;
failed runs cannot produce a completed comparison report. Keep private task
outputs under `.state/`. Follow the [economics protocol](ECONOMICS.md), and never
use expected answers to choose a retry's winning output.

## 6. Correct practice data without rewriting history

In the page, enter a corrected output, practice family and provenance, then save.
The input/family cannot come from development or final evaluation. A correction
creates a new package version; old packages, runs, weights and held-out sets stay
unchanged. Old compatible checkpoints remain old checkpoints—they have not
learned the new correction until a new training run completes.

CLI equivalent:

```sh
.venv/bin/python -m workshop.experiment correct "$TASK_REF" correction.json
```

Use the returned new reference for the next training/evaluation run. Write a
short decision: keep the plain baseline, gather better examples, change the task
contract in a new experiment, or advance a candidate to a small reviewed trial.
Do not keep training simply to force a win on an already exposed final set.

## When the task needs tools or a different grader

Most changes to a JSON-output task are package/data changes. Two installed
graders are available: `json-exact-v1` with optional unordered top-level arrays,
and `roxiq-plan-v1` with source-contract validation and order-aware plan equality.
A genuinely new semantic equivalence rule needs a small trusted Python grader,
a registered ID, adversarial tests, and an implementation/version change.
No uploaded package can execute arbitrary grader code.

For multi-step work, [workshop/environments.py](../workshop/environments.py)
defines the small boundary: construct/reset from scenario data into a disposable
directory, `observe()`, `step(output)`, `done`, `outcome()`. The CI implementation
is a real specimen with fixed file writes and final-state checks. Its scenario
set comes from a package, rather than application-specific HTTP handlers:

```sh
.venv/bin/python -m workshop.experiment import examples/ci-practice.package.json
# Use the returned reference:
.venv/bin/python -m workshop.experiment run "$CI_TASK_REF" episodes --policy rules
```

A new tool environment requires a reviewed adapter, registration in `FACTORIES`
and the package allowlist, and tests for reset, outcomes and unacceptable side
effects. The application/runner then supplies the loop. This is intentionally a
small trusted extension point, not a general sandbox for arbitrary uploaded code.

The same exclusive lock covers inference/training. Numerical child processes
inherit the lock descriptor, so abrupt supervisor death cannot advertise a free
accelerator while training continues. Interrupted/failed runs stay visible and
are not promoted. This is a single-user local runner, not a distributed scheduler.

## Readiness and limits

The browser/API/CLI path has been exercised with three distinct packages,
actual support-task training and checkpoint inference, analytics checkpoint
adoption after exact training-stream verification, environment episodes, and
correction versioning. Tiny support results were poor and remain visible. Those
checks establish repeatable mechanics, not a useful trained support model.

The analytics comparison retains a real base/checklist/reference/adapter study
with blind model adjudication. Its formatting failure and separate transport
replay are documented in [ANALYTICS-EXPERIMENT.md](ANALYTICS-EXPERIMENT.md).
No specimen earns production rollout. The useful deliverable is the experiment
substrate for finding a suitable workload next.

Supported now: supervised LoRA, structured inference, frozen package versions,
provided-label semantic checks, reviewed environment adapters and explicit
reference calls. Not implemented: automated expert label quality, generic
arbitrary tools, hosted tenancy, job scheduling across machines, customer rollout,
RL/GRPO, or guaranteed serving economics. Those should follow evidence of value.
