# Inspectable run receipts

Start with [the results](../RESULTS.md) or [summary.json](summary.json).

Each run includes its frozen input, actions, intermediate source snapshots,
model responses, usage, test observations and external grading when present.
Failed and stopped trials are retained. The lesson-call directory contains the
original ad-hoc distillation; `workloop-learn-cli` is the later live CLI export.
Neither demonstrates transfer. `geometry-component` is the code supplied to
Qwen; `workloop-component-cli` exercises export from the final accepted CLI run.

Native request/process files and raw client logs are excluded. Host home and
client temporary paths are replaced with markers; sandbox paths remain useful
for reproducing failures. `manifest.json` records raw and published SHA-256
values. Redacted records are inspection copies, not resumable private runs.
Older negative grading files predate the stronger run/task/evaluator binding;
accepted runs were re-evaluated with the current grader before publication.

To recheck the final code without making model calls, copy the receipt tree
before grading so the published evidence stays unchanged:

```sh
mkdir -p .runs
cp -R experiments/boost_workloop/receipts/workloop-cli-v1 .runs/replay-cli
python3 experiments/boost_workloop/pilot.py grade .runs/replay-cli checked
```

Requires Docker. This grades stored snapshots against the original regression
suite and external exact cases. It does not reproduce model latency or prove
how another task would perform. No private work or production data was used.
