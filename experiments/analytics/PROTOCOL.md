# Bounded judgment experiment: RoxIQ request to typed plan

Fixed before model predictions. This is a task-selection correction, not an
extension of the arithmetic or deterministic CI-training recipes. Preserve those
negative results without using them as evidence against specialist-model value.

Task: interpret a natural-language request plus a frozen RoxIQ-shaped result
catalog. Select result identities, up to two metrics, operation and comparison
scope; clarify ambiguous intent or refuse unsupported evidence. Five-field typed
plans compose a read-only fixture executor. No SQL generation, production reads,
customer actions, live scrape use or model arithmetic. Nulls and absent IDs remain
unknown. Percentile reference is pooled, never event/season-specific.

Data: 160 authored training pairs, six development cases, 24 final inputs with
unfamiliar paraphrases/compositions, schema distractors, ambiguous references,
null/absent data and unsupported causal or event-field requests. Final intended
plans are independently authored by a separate coordinating agent from only the
frozen input and contract packet. Reconcile ambiguity against the source contract
before scoring; retain the adjudication. Final judgments never enter training.
This is independent model adjudication, not human expert validation. Shared
vocabulary and conceptual primitives are intentional; no random-row split or
production-quality claim. Test family descriptions are absent from model input.

Local model: mlx-community/Qwen3-4B-Instruct-2507-4bit, pinned revision
50d427756c6b1b2fe0c0a10f67fbda1fc8e82c1b. It fits the available Mac with batch one.
Shopify Flow used Qwen3-32B, proprietary artifacts, much more training and real
corrective feedback; this small local experiment is not a comparable test of its
performance. Download uses available local disk, not a paid training service.

Comparisons: Qwen base; the same base plus an explicit interpretation checklist;
a fresh supervised adapter with the baseline prompt; existing tool-free Claude
reference with the baseline prompt on all 24 final inputs. Decode greedily for
MLX, max 260 output tokens. Do not hand-code an intent-answer table. An always-
clarify control measures vacuous abstention; deterministic validation measures
executability, separately from exact intended-plan correctness. Metric ordering
is irrelevant; result order matters only for delta. No post-hoc label repair.

Training: 120 supervised LoRA steps, batch one, last eight layers, learning rate
2e-5, seed 29, mask prompt, max sequence length 2048. Reject overlength samples;
never truncate targets. Monitor development loss, preserve adapter/data/code
hashes, do not tune on final results. Evaluate final only once per policy.

Measure exact intent match, schema/context validity, unsafe executable plans for
requests needing clarification/unavailable evidence, needless abstention, and
field-level failure slices. Execute validated plans on deterministic fixture
values and compare selected outcome against the independently intended plan;
execution success alone is not semantic success. Preserve raw inputs/outputs,
per-case failure details, latency scope and actual CLI usage estimates. Local
energy/hardware dollar cost remains unmeasured. No frontier distillation or RL.

Interpretation: improvement must beat a reasonable structured prompt at matched
quality to justify a next product trial. A negative result narrows task/data/model
choices; it does not disprove focused specialists. No model auto-promotion.
