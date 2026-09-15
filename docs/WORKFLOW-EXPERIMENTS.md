# Two workflow experiments, September 12

The workshop can train real adapters and run resettable, graded trajectories.
Neither new adapter earns a product rollout. The deterministic baselines win
both controlled tasks; the real-text replay exposes an additional extraction gap.
The original CI classification result remains **17/51 → 28/51**, below the
**29/51 majority baseline**, in [EXPERIMENT.md](EXPERIMENT.md).

Open **http://127.0.0.1:8787/practice** for the working CI environment, manual or
model actions, reset, retained failures and recorded comparisons. It uses the
same inference/checkpoint API as the original workshop. Both new skill packages
also appear in the skill library and can be exported or supplied through the API.

## Results

| Task | Training-majority baseline | Rules | Untrained Qwen 1.5B | Fresh trained adapter |
| --- | ---: | ---: | ---: | ---: |
| Blox arithmetic, 48 synthetic final cases | 12/48 (25%) | 48/48 (100%) | 12/48 (25%) | 22/48 (45.8%) |
| CI diagnostic procedure, 16 synthetic final episodes | 4/16 (25%) | 16/16 (100%) | 0/16 (0%) | 4/16 (25%) |

Arithmetic uses label correctness; all four policies also had valid output
contracts and source-line selections on the synthetic cases. The CI result
checks the final advisory file, required evidence reads, invalid actions and
protected repository bytes. The rule policy is an authored baseline over this
finite environment, not a claimed deployment of Workbench's shipping floor.
The majority policy follows the rule policy's evidence-gathering path and then
always advises repository investigation; ties in training counts choose the
first label. It therefore receives the benefit of correct evidence gathering.

Accuracy alone conceals a bad tradeoff: the Blox adapter labelled **11 of 36
non-consistent cases as consistent**. The untrained model falsely cleared none,
but mainly predicted time-overrun and was otherwise unhelpful. Neither model
selected needs-context on any final case; there were 12 missing-quantity cases.
The rule baseline identified all 12. A higher aggregate score is insufficient.

| CI policy | Unacceptable episodes* | Invalid actions | Escalations | Unnecessary escalations | Episodes needing correction |
| --- | ---: | ---: | ---: | ---: | ---: |
| Majority | 12 | 0 | 0 | 0 | 12 |
| Rules | 0 | 0 | 4 | 0 | 0 |
| Base | 16 | 15 | 0 | 0 | 16 |
| Adapter | 8 | 2 | 4 | 4 | 12 |

*The retained `harmful` field counts the union of invalid actions, premature or
wrong non-escalation dispositions, and protected-file violations. It is a count
of unacceptable attempts in this fixture, not actual production harm. Invalid
actions overlap that count; unnecessary conservative escalation is counted
separately. Every repository fixture remained intact. Correction burden here
means failed episodes requiring review, not measured human editing time.

The adapter reduced invalid actions but still confused new signatures. It
advised repository investigation for disk-full and port-conflict cases, and
sometimes recommended retry from cleanup-only evidence without reading history.
Different opaque commit hashes also produced inconsistent behavior within a
family. Training did not reliably teach semantic transfer.

## Existing frontier reference

The logged-in, tool-free Claude CLI supplied a reference on a subset fixed in
[the protocol](../experiments/workflows/PROTOCOL.md): first two arithmetic cases
per label, first CI episode per disposition. Each step was a fresh call with the
same task system prompt and observation as the local model. Tool and MCP access,
project settings and skills were disabled. The selected default model reported
`claude-opus-5[1m]`; CLI usage also reports ancillary Haiku calls. Low effort was
used. No prompt search or frontier retry was performed.

| Matched subset | Majority | Rules | Base | Adapter | Frontier reference |
| --- | ---: | ---: | ---: | ---: | ---: |
| Arithmetic, 8 cases | 2/8 | 8/8 | 2/8 | 4/8 | 8/8 |
| CI, 4 episodes | 1/4 | 4/4 | 0/4 | 1/4 | 2/4 |

The frontier reference treated both selected port-conflict episodes as
infrastructure failures instead of consulting history. One required a retry
advisory and the other escalation. These tiny, authored subsets establish
neither a general frontier ranking nor small-model superiority.

## What the environment actually does

`Practice` creates a fresh temporary directory containing a protected repository
fixture, a failure log and run history. `read-log` and `read-history` reveal
previously hidden evidence and append events. Terminal actions write
`advisory.json`; the grader reads that file and hashes the protected repository.
Unknown actions, repeated/out-of-order reads, premature decisions, wrong
advisories and episodes longer than five steps cannot pass. Reset reconstructs
an empty disposable environment. The browser statelessly replays the submitted
action sequence, while the evaluation runner retains each complete trajectory.

There are **96 train episodes / 240 checked next-action demonstrations**, eight
development episodes / 20 demonstrations, and 16 final episodes / 40 oracle
steps. Compile/assertion, authentication/DNS and ambiguous timeout/socket families
supply training. Development has undefined-symbol, rate-limit, busy and opaque
failures. Final families include coverage, syntax, disk, shutdown, port,
database-startup and cleanup-only failures. A passing run must match the exact
commit before a retry advisory is accepted. Related port variants remain in the
same final split. Generic wrappers and cleanup noise are deliberate traps.

This is a finite model of evidence handling. It does **not** execute real CI,
repair source code, infer a genuine flaky-test probability, change checks or
approval policy, call Gate, or grant authority. Tool containment is enforced by
the authored action surface; passing cannot demonstrate safe arbitrary shell use.
The final-state verifier was tested with deliberate advisory/protected-file
tampering, premature actions, invalid names, stale history and reset isolation.

One inherited contract limitation matters: action demonstrations reuse the
workshop's `{bucket,line}` format, and their supporting line is the final
`Advisory: none` status line. It is present but does not prove the diagnosis.
The episode score measures action sequence and final state, **not causal quote
quality**. A future action-specific contract should remove this unused field or
teach and independently check decisive evidence. No such fix was smuggled into
a rerun after seeing these final results.

## Blox: computation versus extracting quantities

The synthetic package has 160 practice rows, 32 development rows and 48 final
rows. Practice uses repeated work, single benchmarks and round-trip distances;
development uses pace-derived duration and single-length distances; final uses
compound time budgets and combined technique/test distance. Missing-quantity
families also differ by split. Shared arithmetic primitives and authored wording
are deliberate. These are procedural transfer cases, not unseen athlete briefs.

The narrow parser/calculator recognizes this supported language. Its 100% score
is an upper-bound demonstration for explicit quantities, not coverage of
arbitrary generated workouts. The generator and calculator share an author;
hand-computed boundary tests add checks but are not independent adjudication.

Four **already exposed** retained Blox defects were replayed after the synthetic
base result, separately from final scoring. They cover contradictory shuttle
distance, a benchmark exceeding its block, repeated sled work exceeding its
allocation, and omitted technique distance. Original excerpts, source candidate
IDs, manual normalizations and outputs are in
[blox-retained-diagnostics.json](../experiments/workflows/blox-retained-diagnostics.json).
The fifth retained issue (an overstated written time audit that still fits its
slot) falls outside these four labels and was excluded explicitly.

| Retained real text, 4 known defects | Correct label | Correct plus valid output |
| --- | ---: | ---: |
| Parser on raw prose | 0/4 — abstained on all four | 0/4 |
| Calculator on human-normalized quantities | 4/4 | 4/4 |
| Base model on raw prose | 2/4 | 2/4 |
| Trained adapter on raw prose | 2/4 | 1/4 |

The normalization step was authored manually, not learned or automated. Its
labor was not timed. These cases are not training gradients, a fresh holdout,
a coaching assessment or an exhaustive plan audit. Passing the narrow arithmetic
check would not clear loads, treadmill instructions, suitability or safety.

## Training, latency and cost

Both runs used the existing supervised QLoRA worker: pinned
`mlx-community/Qwen2.5-1.5B-Instruct-4bit` revision
`8b403126fc14f14cfc99bb4cfa72ecbc129ea677`, 200 iterations, batch two, last eight
layers, learning rate 2e-5, seed 17, assistant-only loss, 1024-token limit.
The recipe was fixed before the final evaluations and was not tuned afterward.
The accelerator lock serialized local numerical execution; the existing
subscription reference used no local accelerator. Browser and CPU checks ran
alongside training, so timings are ordinary workstation observations.

| Run | Checkpoint | Training wall time | Peak MLX allocation | Median final generation |
| --- | --- | ---: | ---: | ---: |
| Blox | `job-218412b484d0` | 151.7 s | 2.527 GB | base 256 ms / adapter 322 ms |
| CI procedure | `job-eb7578d38146` | 343.0 s | 3.128 GB | base 301 ms / adapter 579.5 ms |

MLX allocation is not total system memory. Local generation latency excludes
model loading; the browser pays that load cost for each step. There is no local
energy, amortized hardware or dollar-cost measurement. Neither adapter was
faster than its base in these runs.

Frontier arithmetic calls had 10,688 ms median whole-CLI time and 1,912 ms median
CLI-reported request duration; CI calls had 4,014 ms and 1,638 ms respectively.
Those differ from local generation scope and should not be turned into a serving
speedup claim. The 16 reference calls reported **$0.145463 API-equivalent usage**
($0.071792 arithmetic + $0.073671 CI), including ancillary calls. This is a CLI
estimate under the existing subscription, not an incremental invoice.

Exact weight hashes, training data hashes, recipe values and telemetry are in
[training.json](../experiments/workflows/training.json) and the two training
logs. Weights remain under `.state/jobs/`, outside Git. All raw predictions,
traces, source hashes, skill packages and the frozen manifest are retained.
Original CI artifacts have not been rescored or overwritten.

## What earns the next product trial

**No trained adapter from this experiment earns integration.** The strongest
next Blox trial is an explicit quantity contract alongside generated prose:
work duration/distance, repetitions, rest semantics, warmup/cooldown and budget,
with deterministic checks and unknown fields rejected for human review. The
unproven part is extracting or generating quantities that faithfully represent
the prescription. That trial needs new briefs and extraction-level checks.
This task does not change Blox's generator or validator.

The CI environment is useful as a workshop specimen, but its hand-authored rule
policy already solves the defined workload. More platform or RL is not justified
by this result. A next specialization study should target a measured residual
that rules cannot handle, repair the action/evidence contract, and collect
independently checked trajectories from genuinely different incidents. Ivy was
deferred because no reviewed learner-error/hint dataset was established; learner
ownership and teaching quality cannot be inferred from synthetic labels.

The DeepSeek book's chapter 7 distinction remains useful: verified trajectories
can supply supervised cold-start data; reliable outcome rewards can later support
RL. Here the trajectories train by supervised imitation of a rule policy.
There is no teacher-model distillation, RL, GRPO or architecture-from-scratch claim.

## Reproduce and inspect

Run the app using README instructions, then open `/practice`. Rules need no
model; model actions require the local runtime and a completed checkpoint.
The checked-in JSON packages can be imported into another workshop instance.

```sh
.venv/bin/python -m pytest -q
node --check static/app.js
node --check static/practice.js
# Each policy has a recorded result; scripts refuse silent overwrite.
# Use a fresh copy of the experiment outputs for a distinct rerun.
.venv/bin/python scripts/run_workflows.py blox-arithmetic train
.venv/bin/python scripts/run_workflows.py ci-diagnostic train
```

A fresh run must preserve the frozen input manifest and record distinct results;
do not delete this evidence to make a rerun fit. The artifact names here describe
one fixed experiment, not a general-purpose experiment registry.

Validation: 53 tests passed, both JavaScript syntax checks passed, and browser
checks covered a successful manual episode, premature retry rejection, reset,
390px layout without horizontal overflow, comparison inspection, and a real
trained-model `read-log → advise-repo` episode on a labelled practice case.
That hands-on success is a functional check on a familiar case, not new performance
evidence. No independent reviewer panel is configured for this repository.
