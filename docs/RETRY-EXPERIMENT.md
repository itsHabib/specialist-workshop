# Retry economics: a first measured comparison

On 2026-09-15, GPT-5.6 Luna matched GPT-6 Astra's 12/12 outcomes on a small analytics
sample after one validator-guided repair, at **38.5× lower estimated API cost**.
This supports comparing complete policies, including retries. It does not establish
that the trained local specialist is competitive: that adapter remained at 7/12.

## Results

Every policy received the same 12 inputs and original task instructions. A retry
received its previous answer and the installed schema/context validator's error.
The policy stopped at its first valid answer, even when that answer was wrong.
Gold answers scored the completed trajectory and never selected an attempt.

| Policy | One-shot correct | Correct with retries | Total calls | Estimated API cost, complete policy | Median / p95 task seconds |
| --- | ---: | ---: | ---: | ---: | ---: |
| GPT-6 Astra, one shot | 12/12 | 12/12 (no retries allowed) | 12 | $0.1380500 | 2.15 / 4.11 |
| GPT-5.6 Luna, up to 3 | 11/12 | 12/12 | 13 | $0.0035842 | 1.75 / 4.16 |
| Local Qwen3-4B base, up to 3 | 3/12 | 5/12 | 17 | Unknown local cost | 2.32 / 6.95 |
| Existing Qwen3-4B adapter, up to 3 | 7/12 | 7/12 | 16 | Unknown local cost | 1.74 / 5.65 |

Astra cost approximately $0.011504 per successful task; Luna with repair cost
$0.000299. Those denominators include costs of failed attempts and all assigned
tasks. Luna's one-shot policy cost $0.0032602 in total, with one failure. All API
calls combined cost approximately **$0.1416342**, below the $3 reservation budget.
These are usage-based estimates, not reconciled invoice charges.

The repaired Luna case, `final-19`, asked for a benchmark whose requested metric
had no supported pooled reference. Its first response proposed that unavailable
benchmark. The validator returned `No supported pooled reference for metric`.
Luna then returned the contract's unsupported/unavailable answer, which matched
the held-out target. This is a repair using feedback the workflow can supply.

The base model recovered two tasks. Six of its final answers were valid but wrong;
three adapter answers were also valid but wrong. A schema/context checker cannot
catch every intent error. Repeating until the checker passes therefore does not
guarantee semantic success. The adapter spent four extra calls without recovering
a task in this run.

## Scope and accounting

- This was a **diagnostic replay of previously exposed cases**, selected before
  calls as every other case in the original 24-case final set. The 12-case subset
  and policy were saved before execution. There was no fresh held-out qualification,
  new training, or claim of statistically established equivalence.
- One-shot scores are the first attempts of the same trajectories, not independent
  reruns. Only Luna, base and adapter had retry budgets. No frontier retry or
  specialist-to-frontier fallback policy was tested.
- API aliases were `gpt-6-astra` and `gpt-5.6-luna`, both at low reasoning effort,
  with a 2,048-output-token cap. Provider-returned model identities, responses and
  usage are retained. Aliases can change. Local inference used the existing pinned
  model/checkpoint, seed 29, greedy decoding, a 260-token output cap, and a 4,000-token
  input limit. Different token caps mean this is a comparison of these particular
  configurations, not model families in general.
- Local task latency excludes startup: 2.35 seconds for base and 0.80 seconds for
  adapter. Complete local generation occupied 38.08 and 28.27 seconds respectively.
  Hardware, energy, training, setup and maintenance costs remain unpriced. The
  timing sample has only 12 tasks; its p95 is the maximum observation.
- There were no human reviews, paid tools or fallbacks inside the measured policies.
  Agent development and analysis time is outside the inference scorecard. Quality
  and latency requirements for a production workload have not been agreed or met.
- Pricing was checked against the official [Astra](https://developers.openai.com/api/docs/models/gpt-6-astra)
  and [Luna](https://developers.openai.com/api/docs/models/gpt-5.6-luna) pages on the
  run date, including the 1.25× input cache-write rate. The initial raw ledger omitted
  that premium on 1,068 Luna tokens. The separate analysis adds $0.0000534 without
  modifying raw outputs, usage, or the original ledger.

## Inspect and reproduce

[Raw run and trajectories](../artifacts/retry-economics/run-2026-09-15/run.json),
[corrected accounting](../artifacts/retry-economics/run-2026-09-15/analysis.json),
[executed source](../artifacts/retry-economics/run-2026-09-15/executed_script.py),
[package snapshot](../artifacts/retry-economics/run-2026-09-15/package.json), and
[local runtime/checkpoint identity](../artifacts/retry-economics/run-2026-09-15/local-runtime.json)
retain the evidence. The executed source hash is checked before recomputing costs:

```sh
.venv/bin/python scripts/summarize_retry.py artifacts/retry-economics/run-2026-09-15
```

The [current runner](../scripts/retry_economics.py) accepts an imported package,
checkpoint, policies, split and attempt/API budgets. It defaults to development
cases; see the [clean-clone walkthrough](MONDAY.md) to train and compare your own
adapter. Local weights and state for this historical run are excluded from Git.
The archived executed source and 12-case subset preserve the original recipe;
new runs use all cases in the selected split and do not rewrite this evidence.

No credentials or model weights are published. Four offline tests exercise stop
behavior, non-oracle feedback, failure-inclusive cost denominators, and usage pricing;
the full suite passed with **89 tests** and two dependency deprecation warnings.

The next useful experiment is a fresh, larger task set with frozen policies and a
stronger deployable verifier where the task supports one. Luna plus bounded repair
is a promising inexpensive baseline; the current adapter needs better task quality
before its cost can make it competitive.
