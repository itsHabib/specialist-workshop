# Measure the cost of finishing the task

A specialist can be worthwhile even when it needs two or three attempts to match
a frontier model's first answer. Qualify the complete policy by delivered quality,
total cost, latency and human effort. One-shot accuracy is a useful diagnostic;
it does not settle the economic comparison.

For illustration only: a frontier answer costing $0.03 versus three specialist
attempts costing $0.002 each is $0.03 versus $0.006, or 5× cheaper. That comparison
requires comparable delivered quality, acceptable latency and no omitted review,
tool or fallback costs. These are hypothetical amounts, not measured model prices.

## A bounded comparison

Freeze the policies on development cases, then run them on the same reserved
final tasks. Include a frontier one-shot control, a cheaper general model with
up to three attempts, and a specialist with up to three attempts. A frontier retry
control can show whether extra attempts also improve the reference. Declare any
frontier fallback as its own policy and charge its calls to that policy.

Agree the required success rate, unacceptable errors, maximum latency and human
effort before evaluation. Policies need not use equal call counts; they must meet
the same task requirements. Count tasks that fail or exhaust their budget.

Retries must use feedback available in the real workflow: for example, a schema
error, a failing executable check, or a priced reviewer. Define how the policy
stops and selects its final answer. The evaluator's expected answer may score the
finished task, but must never tell the policy which attempt to keep. Oracle
best-of-three is only a diagnostic ceiling, not delivered success.

Record the actual retry strategy. Repeating an identical greedy prompt may yield
the same failure; do not assume independent attempts or calculate success from
`1 - (1 - one_shot_accuracy)^attempts` without evidence.

## The scorecard

Report these together for each complete policy:

| Measure | Accounting rule |
| --- | --- |
| Delivered success and unacceptable errors | Score the selected final outcome on every assigned task; retain failures and abstentions |
| Cost per successful task | Total cost across **all** assigned tasks, including failures, divided by successfully delivered tasks; undefined if none succeed |
| Cost per assigned task | Total cost divided by assigned tasks, alongside the success rate |
| Attempts and fallback use | Include failed calls, verification, repair and escalation |
| End-to-end latency | Median and p95 task time, including retries, tools, verification and declared model-loading policy |
| Human effort | Review and correction minutes, with any dollar conversion stated separately |

Keep raw responses, feedback, token usage, checkpoint and policy identities,
stop/selection reasons and final outcomes. Price provider usage using a dated
price source or actual billing evidence. Report unknown costs as unknown.
Local inference is not automatically free: state electricity, hardware and
utilization assumptions. Show marginal running cost and training/setup costs
amortized over an explicit workload separately. Savings must survive the stated
quality and latency requirements, not just lower token prices.

## What Workshop measures today

The package `evaluate` command makes **one prediction per final case**. Its
`accepted` field uses the held-out target after inference; it is not a deployable
retry verifier. The separate environment runner supports multi-step episodes,
which are not independent retries. Existing run timing and provider usage do not
constitute a complete cost ledger; local dollar cost remains unknown.

The recorded support and analytics failures remain valid observations of those
tested configurations. A [first diagnostic retry comparison](RETRY-EXPERIMENT.md) now records Luna
recovering one failure and matching Astra on 12 exposed cases at lower API cost.
The local adapter did not recover a task. Fresh economic qualification remains
unmeasured. The standalone experiment script does not change the package evaluator
or historical scores.

The next experiment should use one task with a credible deployable verifier,
freeze a bounded repair policy on development data, and record the complete
scorecard on fresh reserved cases before adding broader platform machinery.
