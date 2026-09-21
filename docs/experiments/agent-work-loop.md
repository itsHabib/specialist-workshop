# Next trial: useful work under interruption

**Proposed, not run.** Test whether durable jobs reduce recovery effort, then whether
targeted boost improves completion. Keep those questions separate.

## Workload and baseline

Use a public synthetic document-import application: upload, validation, durable
records, retry-safe requests and a browser result view. During implementation,
release three prewritten requests at fixed times: a new input format, an independent
export feature and a persistence failure report. Publish the schedule and acceptance
checks before launch; reserve unseen input fixtures for final evaluation. No private
work material. Pin the initial repository, harness, prompts and model versions.

Run two arms from identical snapshots:

- Native: one LLM coordinator, up to two cheap workers, native messages and a task file.
- Jobs: the same models and role prompts, using Fleet jobs for claims and acceptance.

Allow both arms tests, retries, handoffs and the same targeted boost tools. The
coordinator may reuse a worker or start the second one, recording why. It owns
judgment; the queue owns durable claims. Use separate worktrees per worker and
stop an expired worker before giving a replacement access to its checkout.

Before starting, record a shared model-spend ceiling and generous wall-time ceiling
in the run manifest. Include coordinator, helpers, retries and verification in cost.
A limit hit is incomplete execution, not proof of inability. Do not extend only one arm.

## Faults and receipts

1. Interrupt one worker after its first uncommitted source edit, before a handoff.
2. After recovery, stop the coordinator and replace it using only durable records.
3. In the jobs arm, restart the queue service and inject a late old-token completion.
   This is an additional infrastructure check, not a matched baseline failure.

Record the fault triggers and timestamps, messages, decisions, claims, source hashes,
checks, model usage and operator interventions. Capture accepted tasks, end-to-end
elapsed time, recovery time, lost/repeated work and rejected stale results. Report
missing usage as unknown. Validate API behavior and the actual browser flow.

Store each run under `experiments/agent-work-loop/runs/<run-id>/` in the execution
repository: `manifest.json`, `events.jsonl`, `usage.json`, `checks/`, source references
and a short `RESULTS.md`. Preserve failed runs. Scrub credentials and machine paths
before publishing; retain exact code/model identities and reproducible commands.

## Decisions fixed before execution

- **Correctness stop:** any accepted wrong result, lost accepted job or accepted stale
  completion blocks unattended use until fixed and replayed.
- **Operational value:** run three paired trials, alternating order. Keep the queue
  integration if it reduces operator interventions or recovery time in at least two
  pairs without reducing accepted output. Report all costs and latency; three pairs
  remain pilot evidence, not statistical proof. Otherwise keep native coordination
  and document the queue's unearned complexity.
- **Boost value:** separately replay three captured stuck states from unrelated repairs,
  ordinary continuation versus targeted boost, with equal resource allowances and
  independent acceptance. Retain an approach if it produces an additional accepted
  repair or reduces total time/cost on at least two cases without correctness loss.
  Otherwise keep the useful checks and drop the extra helper ceremony.

## Local, Rooms, then cloud

First run locally. Resolve review findings on the existing job-service PR rather
than creating another scheduler. Read its current API documentation before wiring
workers; submission alone does not launch an agent.

Only after a useful local result, repeat one accepted workload in local Rooms.
Verify admission first. Give restored clones fresh identities and private workspaces;
test peer loss and stale effects. The current single-host file store is not a shared
multi-VM queue: use a host service through a deliberately secured boundary, or design
a remote store only if that trial requires it. Do not expose the loopback API publicly.

Cloud comes after the Rooms replay: explicit resource/spend limits, authenticated
transport, teardown and the same receipts. No autoscaler until queued eligible work
and measured worker saturation show a need. Each stage must preserve the previous
stage's acceptance checks; deployment alone is not progress.
