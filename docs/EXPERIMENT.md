# Initial specialization experiment — 2026-09-12

The platform works end to end and trained real adapter weights. The first CI specialist improved over the matched small-model baseline, but **did not beat a constant majority-label predictor**. It is not qualified for automated CI diagnosis or any authorization decision.

## Final matched comparison

Both runs used the same Qwen2.5-1.5B 4-bit base revision, skill prompt, numbered-line representation, 51 external inputs, greedy decoding, 220-token output limit, and v2 scorer. The specialist adds the locally trained adapter.

| Metric | Base model | Specialist |
| --- | ---: | ---: |
| Correct labels | 17 / 51 (33.3%) | 28 / 51 (54.9%) |
| Valid output schema | 49 / 51 (96.1%) | 51 / 51 (100%) |
| Nonempty source evidence | 48 / 51 (94.1%) | 51 / 51 (100%) |
| Raw response is JSON without a code fence | 23 / 51 (45.1%) | 51 / 51 (100%) |
| Median generation time, excluding model load | 1.093 s | 2.353 s |
| Correct labels on 41 CI excerpts | 12 / 41 | 18 / 41 |
| Correct labels on 10 isolated lines | 5 / 10 | 10 / 10 |

Always choosing `real-break` scores **29 / 51 (56.9%)**. It is shown in the UI. The specialist's 21.6 percentage-point improvement over the model baseline is a learning-loop demonstration, not evidence that this model is a useful classifier. There is no speed advantage in this measurement and no frontier-model evaluation.

Selecting an existing source line guarantees a mechanically faithful quote when the index is valid. This does not establish that the selected line is causally relevant. There are no independently adjudicated gold evidence spans for the external set.

## Exact evidence

- Base run: `job-0c21bf292f2f` ([raw outputs](../artifacts/experiments/job-0c21bf292f2f.json)).
- Training run: `job-0ef50463c8e3` ([loss telemetry](../artifacts/experiments/job-0ef50463c8e3.log)).
- Specialist run: `job-5405ebec4cab` ([raw outputs](../artifacts/experiments/job-5405ebec4cab.json)).
- All completed runs: [index](../artifacts/experiments/index.json), including unsuccessful earlier experiments.
- Base model: `mlx-community/Qwen2.5-1.5B-Instruct-4bit`, revision `8b403126fc14f14cfc99bb4cfa72ecbc129ea677`.
- Adapter SHA-256: `906c3dc24dded32a6dba10ffa1456016ea3885a2d3e0a7751c4ec5c0adfe69fc`.
- Skill snapshot: `db48dc6e83a8110605b72dd41024331893d9dc92c37755ba7a34d035364b30f3`.
- Test fingerprint: `1a0f19e8c4087c96c6e9b345fc77bd08802642e1ae7e736397bcf2999e1eb7ff`.
- Prompt fingerprint: `eef0cbd50fa5996b6ab5c5267abfd2ed1d07675c3eec03984f87977e7364a5d9`.

The raw source corpus lives in `skills/ci-triage/workbench-evaluation.jsonl`, with the original Workbench license and SHA-256 provenance in the skill package. Exported result rows refer to its ordered case IDs. Full local snapshots, recipes, and adapters remain under `.state/jobs/` and are excluded from Git.

## Recipe and experimental history

The final adapter used 240 authored synthetic demonstrations, 9 separate validation examples, 200 steps, batch size 2, learning rate 2e-5, LoRA on the last 8 layers, and masked assistant-output loss. The local training process took 196.4 seconds; MLX reported a peak allocation of 2.527 GB. That is not total system memory usage or an energy-cost estimate.

Earlier attempts used a 0.5B model and then a 1.5B model with direct quote generation. They struggled with copying and raw JSON formatting. The next representation asked the model to select a numbered input line, which the service resolves to the original text. A 2e-4 learning rate produced unstable losses and poor predictions; the final recipe lowered it to 2e-5. All preceding measured failures remain in the exported history.

The first scorer conflated raw JSON adherence with label accuracy. The current scorer separates those measures and mechanically unwraps one complete JSON code fence. Earlier strict-format metrics are archived rather than presented as directly comparable to v2. The final base and specialist results above both use v2.

The CI corpus was never used for gradient updates. However, observed errors informed model and representation choices. These are **exploratory diagnostic results**, not untouched final test results. Repeated incident families, synthetic practice, and unrevalidated historical labels further limit the claim.

## What was learned

Supervision improved short-input behavior substantially, while transfer to noisy CI excerpts remained weak. The gap between 10/10 isolated lines and 18/41 real excerpts points to the next useful work: production-like examples, clearer label adjudication, and a genuinely fresh evaluation set. It does not justify more platform infrastructure or a claim that the model beats frontier models.

The DeepSeek book's chapters 7–8 informed the staged approach: supervised adaptation first, checked teacher demonstrations as a possible next data source, and reward-based optimization only after the grader is trustworthy. This POC implements QLoRA; it does not implement GRPO, teacher distillation, or foundation-model pretraining.

## Live application check

After the final evaluation, the trained checkpoint handled a new three-line assertion failure through the browser and `/api/infer`. It returned `real-break` with the exact assertion line in 0.80 seconds of generation time. A browser correction then added this practice example for future training, leaving the measured checkpoint and its 51-case evaluation unchanged. This is functional evidence from one example, not a second accuracy benchmark.
