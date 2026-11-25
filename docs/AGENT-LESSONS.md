# What helped our agents finish work

Experiments with cheaper models, Fleet coordination and durable jobs. These are
lessons from recorded runs; a useful repair is not proof of general intelligence uplift.

## Give the worker a failure it can act on

A Haiku worker passed its visible tests but missed a CLI requirement. A separate
development check exposed the mismatch; the next edit fixed it. A fresh run on the
broken geometry library and CLI passed the regression suite and 14 external cases
in 122 seconds for $0.21 in reported worker model cost. That excludes engineering,
evaluation and earlier experiments. **Make the contract executable; keep final
acceptance separate from development feedback.** [Runs and costs](../experiments/boost_workloop/RESULTS.md).

## Save working code alongside advice

A prose geometry lesson changed a local 7B model's approach without producing a
correct program. Verified source reuse got the mathematics right, but integration
still failed. **Keep tested components, fixtures and failure examples. Recheck
reuse against the new contract.** This observation comes from one task family.

## A good handoff is a strong baseline

A controller registry and native messages plus a task file both delivered working
CSV import apps and recovered through planned worker replacement. We have not
shown a productivity or cost advantage for the controller. **Compare new machinery
against a well-run simple workflow.** [Controller evidence](https://github.com/itsHabib/workbench/blob/b8aedf4a43470915c3536b260a2f281f89222d70/experiments/work-controller/import-lab/RESULTS.md).

## Let workload pressure choose infrastructure

Both import apps handled 50,000 rows synchronously. Persistence and retry fixes
mattered; another builder, background queue and telemetry service were unnecessary
for that workload. **Build the mechanism that removes an observed bottleneck.**

## Separate reported work from accepted work

Fleet's job-service drill survived an API restart and rejected an expired worker's
late completion. Claims, retries and acceptance belong in dependable code;
decomposition, routing and judging usefulness still require an agent or person.
**A lease fences queue results, not a worker's filesystem or network effects.**
These were deterministic workers on one host, not unattended LLM deployment.
[Job-service evidence](https://github.com/itsHabib/workbench/blob/026b7ea4467a21fd0cf32f7d60bf7347a04b38c4/cmd/fleet/docs/jobs/RESULTS.md).

## Test the experience and the verifier

Python tests passed with broken JavaScript and mobile overflow. Browser checks
caught both. Our oracle also imposed unnecessary UI details; we corrected it and
replayed the original revisions. **Preserve failures, audit the checker and exercise
the path a user actually takes.**

## What comes next

Boost is merged. The [controller](https://github.com/itsHabib/workbench/pull/362)
and [job service](https://github.com/itsHabib/workbench/pull/363) remain proposed
changes at this publication. Next we combine them locally on changing engineering
work, compare against native coordination, and measure accepted output, elapsed
time, cost, repeated work and operator intervention. Rooms follows a useful local
result. [Concrete experiment protocol](experiments/agent-work-loop.md).
