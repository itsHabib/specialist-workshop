# What actually helped

**We have a working repair tool, not evidence of a general intelligence boost.**
Haiku repaired a broken exact-geometry library and CLI in about two minutes for
$0.21 in reported model cost. A separate run demonstrated the useful mechanism:
visible tests passed, an external development check caught a missing requirement,
and the same model repaired it. [Run it](README.md).

## Results, including the failures

One synthetic task family: exact rectangle-union area; variant 1 adds comments,
file input and compact JSON. Every worker could inspect, edit, run tests and retry.
Final acceptance used the original regression tests and 14 external CLI cases.

| Intervention | Model | Final acceptance | First valid candidate / run duration | Calls | Reported model cost |
|---|---|---|---|---:|---:|
| Ordinary loop, v0 | Local Qwen 2.5 7B | Failed | — / 291s | 12 | Unpriced |
| Tests after each edit, v0 | Qwen 7B | Failed | — / 458s | 12 | Unpriced |
| Same feedback loop, v0 | Haiku | Passed | 40s / 110s | 8 | $0.210 |
| Frozen prose lesson, v1 | Qwen 7B | Failed; stopped repeating | — / 270s | 6 | Unpriced |
| Verified geometry source available, v1 | Qwen 7B | Failed; stalled on comments | — / 290s | 7 | Unpriced |
| Fresh worker after stalled Qwen, v0 | Haiku | Passed; call allowance reached | 35s / 89s | 8 | $0.289 |
| Fresh worker after reuse, v1 | Haiku | Failed compact JSON; numeric results correct | — / 67s | 6 | $0.167 |
| External development check after that failure | Haiku | Passed | 23s / 45s | 5 | $0.119 |
| Actual `--check` CLI, original broken v1 source | Haiku | Passed | 113s / 122s | 6 | $0.210 |

[Machine-readable results](receipts/summary.json) · [Prompts, outputs and snapshots](receipts/README.md).
Haiku resolved to `claude-haiku-4-5-20251001`. Each receipt contains actual usage.
“First valid” is when the source was produced, **evaluated afterward**; it is not
when verification finished. The v0 rescue produced correct code but did not
self-complete before its call allowance. A timeout or budget stop is incomplete
execution, not a model capability ceiling.

## Lessons worth using

1. **Get the worker a real failing observation.** Repeated edits without running
   tests were useless. Automatic tests exposed errors, though feedback alone did
   not rescue this local model.
2. **Test the contract beyond the existing suite.** Haiku's v1 rescue passed all
   five visible tests but emitted spaced JSON. A caller-owned development check
   exposed the mismatch; the next edit fixed it. Keep final grading separate.
3. **Reuse correct work.** A verified geometry component got Qwen past the math.
   It still missed CLI details. Prior acceptance covers the old contract, not
   the new integration. Source reuse is useful even when prose lessons fail.
4. **Escalate with evidence.** A fresh Haiku continuation solved the stalled v0
   task. Preserve the current patch, raw failure and original tests; count both
   attempts. Changing models is not proof that more agents improve one model.
5. **Keep lessons provisional.** A generated lesson changed Qwen's approach but
   did not produce a correct program. The later lesson export is a candidate
   for future use, not a promoted policy or learned model weights.

## What this does not establish

This was adaptive product development, not a completed controlled comparison.
The proposed helper/solo and lesson/no-lesson matrices were not completed. No
live helper run here demonstrates same-model team uplift. The harness changed
between trials; the last CLI run pins generation commit `672fb66`. Earlier
plans contain partial source hashes, not complete immutable runner bundles.

The compact-JSON failure was discovered in final seed 90149, which was then
spent. Repair used a separate development probe, followed by seed 771239. The
full CLI trial used public development checks and seed 771251. Shared canonical
cases and one task family limit generalization. Two local calls overlapped
between the initial runs; their latency difference is not a speed comparison.

All recorded worker and lesson calls totaled **$1.030127 in known hosted-model
cost**, plus unpriced local inference. That excludes the supervising agent's
engineering work, hardware and evaluation compute. Rescue cost includes its
failed parent when assessing economics; the v1 success also inherited the
$0.210 practice solution, local reuse attempt and $0.167 continuation. Calling
only the final $0.119 repair the cost of solving the whole task would be false.

**Keep:** the loop, concrete checks, focused continuation and verified source
reuse. Use `/boost` on actual work and record whether these changed the outcome.
Next evidence worth collecting is several unrelated engineering repairs with
acceptance fixed first, compared against an equally resourced ordinary loop.
