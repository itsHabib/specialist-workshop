# Recover a batch job after a worker exception

This local experiment tests the workshop's sequence evaluator. An application requests the sum of a batch of integers. Its worker saves progress in SQLite. It may throw the same timeout before processing, after a checkpoint, or after committing the result but before replying. An uninterrupted job is the control. The successful user outcome is a correct reply by the deadline, a matching persisted result and one completion event for the original request. [Sequence integration guide](DECISION-SEQUENCES.md)

## Comparison specified before final evaluation

Use 64 training workload groups, 32 development groups and 96 untouched final groups. Each workload includes all four job conditions. Keep all variants of a workload together and replay each policy on its own copy of the same inputs and event schedule. The ordinary, longer-outage and free-status cohorts reuse the final workloads. Report them separately. Their overlap makes a combined count unsuitable for estimating independent sample size.

Training uses incomplete jobs. From a reachable training prefix, force each available action in turn and finish with the same coded policy. A verified job receives value 1. An incorrect result or duplicate effect receives −1. Other outcomes receive 0. Pool trials when their visible features and available actions coincide. Retain the alternative outcomes and ties. The policy sees conditions available to its caller, including the time remaining. The hidden fault phase stays inside the fixture and the independent checker.

Fit a decision tree from scratch and compare six tree settings on development jobs. Compare the strongest development-selected tree with three coded policy families, including a rule whose threshold is fitted using training outcomes. The primary comparison is the fraction of all jobs that recover automatically with a verified result by deadline. Resample entire held-out workload groups when computing a paired interval. Advance the tree only if its 95% interval for the improvement on ordinary jobs clears zero and the tested tree has zero false successes and duplicate effects. Report deferred jobs, failed jobs, latency and repeated work separately. Costs and deadlines in this fixture use simulated ticks.

Source inspection before final evaluation exposed a separate executor experiment. The original retry accepts a request ID and checks its saved input, then recomputes the batches. Try reading progress and a committed result in that same worker-side query. A bounded retry can then reuse them even during an outage of the caller's status endpoint. Compare the two executors under the same retry policy. Record this arm separately because it changes what an action does.

## Frozen policies on the original executor

| Status lookup condition | Status-first rule | Blind keyed retry | Fitted coded rule | Trained tree | Tree minus fitted rule, 95% interval in percentage points |
| --- | ---: | ---: | ---: | ---: | ---: |
| Ordinary | 195 | 225 | 271 | 225 | −11.98 [−15.63, −8.59] |
| Longer status outages | 147 | 225 | 250 | 225 | −6.51 [−10.16, −3.39] |
| Free, reliable status | 337 | 225 | 337 | 225 | −29.17 [−35.16, −23.44] |

Each cell counts verified replies by deadline out of 384 jobs, including 96 uninterrupted controls. The original tree learned to retry on every decision it reached. On ordinary jobs it missed 46 that the fitted rule recovered and gained zero. Across all three conditions all primary policies recorded zero reviews, incorrect successes and duplicate effects. Ordinary jobs caused 592 repeated batches for the tree and 269 for the fitted rule. Median measured host time per ordinary job was 0.237 ms for the tree and 0.198 ms for the fitted rule. This includes the local in-memory worker setup, execution and verification.

The original tree used depth 2 and minimum leaf size 2. Its 1,084 training trials covered 489 distinct combinations of observation and available actions. In 363 combinations the best pooled terminal value tied across actions. In three combinations individual hidden worlds had no shared best action. These are different limits. A tie can make the training label sensitive to the task's priority order. Conflicting outcomes can make the hidden situation genuinely undecidable from the current observation. The receipts keep every measured alternative. The public training command recreated the selected tree byte for byte.

## Change what the executor does on retry

Reusing the saved progress improved ordinary verified recoveries from **225 to 337 of 384** under the same bounded retry policy. That gained 112 jobs and lost zero. The paired difference was +29.17 percentage points with a 95% interval of +23.18 to +35.16. Repeated batches fell from 592 to zero. Mean recovery time fell from 4.60 to 2.32 simulated ticks. The remaining 47 jobs needed more unprocessed batch time than the deadline allowed. The corrected worker recovered all 337 jobs feasible under that bound in each of the three matched cohorts. This relies on the worker having its own local access to checkpoint and result storage during a caller-side status outage.

An additional coded challenger reserved enough time for a full retry when choosing to look up status. It recovered 256 ordinary jobs on the original executor. The fitted coded rule recovered 271 and the tree recovered 225. Both choices of coded control lead to the same conclusion about the original tree.

The independent audit checked 5,120 primary policy episodes across development and final cohorts and 2,560 additional coded-policy and executor replays. It recomputed outcomes from original batches, caller replies and ledger rows. Worker tests reopened a durable checkpoint in a fresh process after abrupt exit. Two concurrent processes returned the same correct result and produced one committed completion event. Concurrent execution could still repeat computation before either process committed. [Independent audit](../experiments/decision_recovery/audit.py) · [Worker tests](../tests/decisions/test_recovery_worker.py)

## A trace you can inspect by hand

The two batches `[2, 3]` and `[-1, 9]` sum to 13. The worker commits 13 and throws `WorkerTimeout` before the caller gets its answer. The status-first handler looks up the result and returns 13 in one recovery tick. A blind same-ID retry with the original executor also returns 13, but reprocesses both batches and takes four recovery ticks. Both traces contain one durable completion event. The clock starts when recovery begins. Setup, work before the timeout and host execution time are separate.

Three deliberately damaged traces exercise the checker. A wrong total, a duplicated completion event and a successful-looking response without a commit each receive `incorrect`. The SQLite worker commits its result and completion event in one transaction. Systems that perform effects outside the database need their own verification of those effects.

## One model correction chosen using development evidence

On development job `dev-003-lost_reply`, a full retry cost ten ticks with nine left. The original tree retried and missed the deadline. The coded handler spent two ticks looking up the committed result. Test one additional numeric feature, `retry_slack = remaining - batches * batch_cost`, calculated only from facts already visible to both policies. Refit the same six tree settings on the same training outcomes and select on the original development groups. Freeze that tree and compare it with the original tree and coded winner on 96 fresh ordinary workload groups using seed 735269. Each group contains the same four fault conditions. Report both paired advantages and intervals. Improvement over the original tree alone qualifies the feature for further study.

| Policy on the fresh cohort | Verified by deadline, all 384 jobs | Verified, 288 interrupted jobs | Repeated batches |
| --- | ---: | ---: | ---: |
| Original tree | 246 | 150 | 556 |
| Tree with retry slack | 262 | 166 | 476 |
| Development-selected coded rule | 285 | 189 | 306 |

The new feature recovered 16 jobs the original tree missed, and lost zero that it recovered. The paired advantage was 4.17 percentage points with a 95% interval of +2.08 to +6.51. The coded rule recovered another 23 jobs. The corrected tree's paired difference against that rule was −5.99 points, with an interval of −9.64 to −2.86. Each policy recorded zero incorrect successes, duplicate effects and reviews. Its remaining jobs missed the deadline. The model improved from a development diagnosis while the coded handler remained the better choice for this fixture.

## Reproduce locally

Use Python 3.12 or newer. The tree benchmark and example caller use the standard library. Choose fresh directories for each run so you can compare versions.

```sh
python -m experiments.decision_recovery.benchmark --out .state/recovery-benchmark
python -m experiments.decision_recovery.slack --baseline-run .state/recovery-benchmark --out .state/recovery-slack
python -m experiments.decision_recovery.audit --run .state/recovery-benchmark --out .state/recovery-audit
python -m experiments.decision_recovery.demo --model .state/recovery-benchmark/tree.json
python -m experiments.decision_recovery.demo --model .state/recovery-slack/tree.json
python -m experiments.decision_recovery.demo
python -m unittest discover -s tests/decisions -p 'test_recovery*.py'
```

The demo loads the model once at startup. When a checkpoint is missing, unreadable or belongs to another task, it records the reason and uses the coded status-first policy. The four demonstration requests exercise all three timeouts and uninterrupted completion. A model that deliberately defers sends the job to review. [Example caller](../experiments/decision_recovery/demo.py)
