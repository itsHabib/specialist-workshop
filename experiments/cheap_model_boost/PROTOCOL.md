# Cheap-model boost: geometry pilot

Question: does a same-model critic help more than spending the same number of
calls on solo retries? All material is synthetic; no private work examples.

- Models: native Claude `sonnet`, `haiku` aliases (resolved identities recorded),
  local Ollama `llama3.2:1b` (digest frozen). No stronger-model help.
- Task: exact filled area of self-intersecting polygonal walks under two winding
  rules. Four public development examples; a separate frozen final set. These
  small cases do not cover every maximum-bound input.
- Shared initial candidate. Solo gets two further repair calls with development
  feedback. Boost gets one critic call and one repair call. Three calls per
  policy, five actual calls per model because the initial candidate is shared.
  Critic sees original code and raw development results, not solo repairs.
- Every call is made even when development passes. Final cases are evaluated
  only after both candidates are sealed; never forwarded to any model.
- Model order shuffled with fixed seed. One task and one trial per model:
  descriptive pilot, not statistical evidence of general uplift.
- Hard limits: 15 calls total; 240 seconds per call; Claude native budget
  $0.80/call (10 hosted calls, nominal $8 cap; timeout usage may be unknown).
  Local output capped at 6,000 tokens, context 16,384, temperature zero.
- Score exact full-case correctness, per-case count, format/transport failures,
  model tokens, reported estimated dollars and wall time separately. Compare
  costs including the shared initial call in BOTH policy totals. Equal call
  count is not equal token, time or dollar budget.
- Continue only if receipts and cleanup are trustworthy. A pilot benefit means
  boost passes the complete hidden set while solo fails. Both fail: no rescue;
  both pass: no quality advantage; boost alone fails: harm. Never infer greater
  intelligence from extra prose, consensus or self-report.

Candidate Python runs in the existing networkless, resource-limited Docker
sandbox. Model clients run on the host with tools disabled; this is not an
end-to-end isolated model environment. Local HTTP cancellation may leave an
Ollama generation finishing server-side; inspect/unload after a timeout. Raw
native metadata stays in ignored `.runs/`; publish sanitized receipts.

Run from this repository:

```sh
python3 experiments/cheap_model_boost/run.py freeze .runs/cheap-geometry-01
python3 experiments/cheap_model_boost/run.py run .runs/cheap-geometry-01
```

Do not restart a reserved run automatically. Inspect its call receipts first;
create a fresh labeled run if another trial is justified. This runner is a
bounded experiment, not a durable job scheduler.
