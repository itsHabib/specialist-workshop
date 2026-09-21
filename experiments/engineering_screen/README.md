# Find a real engineering failure before adding more agents

**Result:** no reproducible model gap; a full-size geometry probe exposed a
serialization failure in both models. [Results and lessons](RESULTS.md).

Three independently synthetic tasks test exact geometry, concurrent lease
histories, and constrained network flow. The question is whether solo Opus
actually struggles where Astra succeeds. This is problem discovery, not a claim
that orchestration fixes the gap.

| Task | What makes it useful | How answers are checked |
|---|---|---|
| Geometry | Self-intersections, retracing, cancellation, near-degenerate coordinates | Exact rational integration; hand examples, independent rectangle oracle, invariants |
| Lease history | Pending calls, expiry boundaries, reset fencing, overlapping operations | Legal sequential histories; independent permutation enumeration |
| Network flow | Lower bounds, infeasibility, negative cycles, disconnected profitable circulation | Residual circulation solver; exhaustive bounded integer assignments |

[Protocol](PROTOCOL.md) fixes the tasks, budget and selection rule before calls.
Contracts and reproducible generators are in [tasks.py](tasks.py). No company
code, data, sheet-metal example, or derived fixture is used. These are small
algorithmic specimens of engineering concerns, not full repository integration
or production design exercises.

## Run

From the repository root, with Python 3.12+, Docker and authenticated native
`claude` / `codex` clients:

```sh
python3 -m unittest discover -s experiments/engineering_screen -p 'test_*.py' -v
python3 experiments/engineering_screen/screen.py freeze experiments/engineering_screen/.runs/my-screen
python3 experiments/engineering_screen/screen.py controls experiments/engineering_screen/.runs/my-screen
# Real model calls: maximum 12, two concurrently; Opus cap $0.80 per call.
python3 experiments/engineering_screen/screen.py run experiments/engineering_screen/.runs/my-screen
```

This reuses [the bridge runner](../investigation_bridge/README.md). Candidate
execution is Docker with no network or host mounts; this screen does not use
Rooms. Native model clients run on the host with tools disabled. Their private
raw logs live under ignored `.runs/`. Create `.runs/my-screen/STOP` to stop.
Never retry a reserved call without a receipt; retain the interrupted run.

Development feedback can trigger one repair. Final evaluation happens once,
after sealing, and never becomes repair feedback. A timeout or malformed
response is reported separately from an incorrect program. Task inputs and execution code are
hashed in the plan; changing the frozen inputs requires another named run.

## What an intake should decide

The task's topic alone does not determine its helper. First record the observed
failure, the uncertain assumption, the available validator, and a specific
intervention that might change the result.

- Wrong geometric interpretation: ask for an independent representation and a
  distinguishing example. Check it exactly before requesting another rewrite.
- Incorrect concurrency reasoning: extract the smallest violating history and
  ask a critic to challenge the claimed invariant; replay the proposed repair.
- Incorrect optimization: distinguish feasibility, optimality and scalability;
  use a small exhaustive oracle to expose the missing case.
- No observed failure: keep the solo worker. Do not appoint a committee by default.

These are prospective strategies, not a validated automatic router. A follow-up
must compare solo retry against assistance with fresh evaluation cases and
comparable budgets. No Astra answer should enter the Opus assistance arm.

## Reproduce the deadline diagnostic

[Its separate protocol](DEADLINE-PROTOCOL.md) allows one 300-second call per model
on geometry, with fresh final cases. It does not change the first screen.

```sh
python3 experiments/engineering_screen/screen.py freeze-deadline experiments/engineering_screen/.runs/my-deadline-check
python3 experiments/engineering_screen/screen.py run experiments/engineering_screen/.runs/my-deadline-check
```

The original discovery generator included two shared geometry examples, two
shared flow examples, and one shared lease example in both splits. Archived
results retain that fact and report unseen counts separately. New plans remove
exact input overlap; a regression test protects that boundary.
