# Independent investigators, measured

Can several Opus investigators find a repair that a solo Opus session misses?
This experiment compares them with each other and with Astra on independently
authored synthetic programs. No work code, data, or derived fixtures are used.

Two investigators start independently. One traces the computation; the other
questions representation and assumptions. An Opus integrator receives their
hypotheses and measured probes, then repairs the program. The solo arms receive
the same task, generic investigation guidance and checker access. All share a
six-call budget; the team's advisory calls count against it.

The [protocol](PROTOCOL.md) is fixed before calls. The [task](TASK.md) converts
points between exact rational affine frames. An independently implemented
matrix oracle checks the candidate; agreement between agents is not a grade.

## Execution

Models run through the existing authenticated Claude/Codex clients on the host,
with execution tools disabled. Only candidate source and synthetic inputs enter
the sandbox. Reference code and expected answers stay outside. This does not
place whole model sessions inside Firecracker.

- **Rooms:** a fresh repo-free neutral snapshot in the existing local Lima VM,
  1 vCPU/512 MiB, no egress, witness capture, bounded execution and teardown.
- **Docker fallback:** pinned Python image, no host mounts, no network,
  read-only root, unprivileged UID, no capabilities, resource/time limits.

Neither route provisions cloud infrastructure. These are candidate-execution
experiments, not a new persistent agent platform or an arbitrary-code web API.

## Run locally

Requires Python 3.12+, signed-in `claude` and `codex`, and either the configured
local Rooms host or Docker. From this directory:

```sh
# Use Docker for a portable first run. This makes real native model calls.
python3 run.py plan .runs/calibration --backend docker --variant hard --seed 1729
python3 run.py run .runs/calibration
```

For Rooms, set `BRIDGE_ROOMS_CONFIG` to your private local JSON configuration
and select `--backend rooms` when planning. Keep it outside Git. The plan binds
its hash; changing the runtime configuration requires a new plan. Start from
[config.example.json](config.example.json), pointing to a verified repo-free
Python snapshot and immutable toolstore. Verify artifact hashes before recording
them: the adapter records configured identities, it does not rehash VM memory
on every invocation.

```sh
export BRIDGE_ROOMS_CONFIG=/absolute/path/to/private-rooms-config.json
python3 run.py plan .runs/rooms-calibration --backend rooms --variant hard --seed 1729
python3 run.py run .runs/rooms-calibration
```

The native Opus identity is `claude-opus-5`; Astra is `gpt-6-astra`. Both use high
effort. Claude's receipt includes ancillary model usage; Codex reports tokens
but no invoice price. Call counts and token totals are not equal-dollar budgets.

Create `RUN/STOP` to request termination and prevent further calls. Do not alter
frozen code, tasks, protocol or configuration mid-run. Recorded model responses
can be reused after interruption; an uncertain call without a receipt is never
silently retried. Keep the failed run and create a separately labeled new trial.

## Inspect

- `plan.json`: identities, hashes, randomized arm order and limits.
- `ARM/task.json`: frozen synthetic task, development and final inputs.
- `ARM/calls/N/`: private prompt, raw native outputs and usage receipt.
- `ARM/history.json`: hypotheses, candidate changes and development probes.
- `ARM/final.py`, `result.json`: sealed candidate and once-only final result.
- `summary.json`: all arms, including failures and not-started rows.

The first calibration may be too easy for every model. That is a reason to
improve the task, not declare the team as intelligent as Astra. No holdout
failure feeds repair within the same scored trial. See the protocol before
running fresh seeds or interpreting results.

## Test without paid model calls

From the repository root:

```sh
python3 -m unittest discover -s experiments/investigation_bridge -p 'test_*.py' -v
BRIDGE_DOCKER_TESTS=1 python3 -m unittest discover -s experiments/investigation_bridge -p 'test_*.py' -v
```

The first command uses hand-calculated oracle checks and scripted mechanism
controls. The second adds live isolated execution, mutant rejection, blocked
network/host access and timeout cleanup. Synthetic archived results belong in
`receipts/`; raw native events remain ignored under `.runs/`.
