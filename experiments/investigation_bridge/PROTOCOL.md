# Can independent investigators help Opus solve a hard repair?

This is an independently authored synthetic experiment. No work-repository
code, data, identifiers, or derived fixtures enter any trial. The workload is
exact rational affine-frame conversion, with an independent matrix reference.

## Freeze before model calls

Compare Opus solo, two independent Opus investigators followed by an Opus
integrator, and Astra solo. The two investigators see the same initial evidence
without each other's explanations. The integrator receives their hypotheses
and executed probe results. No stronger model advises the Opus arms. All arms
receive the same generic investigation guidance and development evaluator.

Each arm shares a **six-call total budget**, 150 seconds per native invocation,
high effort, and at most four custom probe inputs per invocation. The team
spends two calls on its initial investigations, leaving four for integration.
Opus calls have an additional $0.80 native CLI budget each. Count auxiliary
models in Claude's usage receipt. Astra's native CLI has no invoice cost, so
this is a fixed-call feasibility pilot, **not equal-dollar efficiency**. Different
native clients supply different overhead; retain their real token usage.

Each edit is checked on development inputs. The first edit passing development,
an explicit finish, or the last permitted call seals the candidate. Run final
inputs once after sealing; never return their outcomes for repair in that trial.
Acceptance requires both development and final to pass, with matching source.
Record unsuccessful, interrupted and unstarted arms as well as successful ones.
An uncertain paid invocation is never silently retried. A STOP file stops new
calls and requests termination of running calls/candidate execution.

## Calibration and decision

First run one **calibration** task (`hard`, seed 1729), combining the six synthetic
fault classes. Calibration is not scored evidence of generalization. If all
arms pass, report a ceiling and do not launch the proposed large comparison:
this task does not demonstrate an intelligence gap. If all fail, investigate
the task and instruments before buying more attempts.

If calibration separates outcomes, freeze three new seeds (2718, 3141, 5772)
and compare the same policies. Retain all three task outcomes separately; do
not count 52 correlated final inputs as 48 independent solved tasks. A team
benefit is provisional only if it solves at least two tasks solo Opus misses
without extra human guidance. If solo equals/exceeds team correctness, keep
the cheaper adequate policy and its useful instruments, not permanent roles.
The thresholds screen this small experiment; they are not statistical proof.

Before any model trial, the known-good implementation must pass, all seven mutant
controls must fail, and hand-calculated oracle checks must pass. Candidate code
receives source and inputs only, never the reference or expected answers.

## Execution boundary

Prefer the existing local Rooms VM: a fresh repo-free neutral snapshot, private
guest state, no credentials, egress none, witness evidence, CPU/memory/wall
limits, artifact collection and teardown. The models run through authenticated
host clients with execution tools disabled; Rooms runs their proposed programs.
This does **not** claim that model sessions themselves live in Firecracker.

Docker is an explicit fallback: pinned Python image, no host mounts, no network,
read-only root, nobody UID, no capabilities, process/memory/CPU/time bounds.
It is not Firecracker. Neither route shares candidate writable state across
trials. Missing or incomplete execution evidence cannot become a passing score.

All native prompts and logs remain in ignored `.runs/`. Publish only checked
synthetic code, sanitized receipts, exact identities, lessons, and reproduction
commands. No cloud deployment, model training, or merge is part of this trial.
