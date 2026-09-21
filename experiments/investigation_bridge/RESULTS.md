# The team worked. It did not establish a capability gain.

One synthetic task, three policies, all accepted. **Keep solo execution as the
baseline; this run does not justify permanent investigator roles.** The frozen
protocol called for stopping when all arms passed, so no larger comparison ran.

| Policy | Calls | Seconds* | Estimated cost | Development / final |
|---|---:|---:|---:|---|
| Opus solo | 2 | 99.6 | $0.5204 | 13/13 · 52/52 |
| Two Opus investigators + Opus integrator | 3 | 87.6 | $0.6618 | 13/13 · 52/52 |
| Astra solo | 2 | 70.2 | unavailable | 13/13 · 52/52 |

*Wall time excludes each arm's baseline check; includes model calls and later
sandbox execution. Investigators ran concurrently. One observation per policy
cannot establish a speed advantage. Claude costs are CLI estimates, including
ancillary Haiku usage; Codex supplied tokens but no invoice cost. Unknown is not
zero. This was a six-call limit per arm, not an equal-dollar comparison.

## What actually happened

- **Solo Opus:** first repair still failed six development checks. It used the
  feedback, corrected the repair, and passed the once-only final evaluation.
- **Opus team:** both investigators supplied full repairs despite a diagnosis-only
  instruction. The harness flagged these as invalid advisory responses, but
  forwarded their contents to the integrator. The integrator's first repair passed.
  This is closer to solution pooling than a clean test of diagnosis-only roles.
- **Astra:** requested probes, then supplied a repair that passed.

The shared prompt also described edit actions to investigators. That ambiguity
is a harness lesson, not evidence of a general model obedience problem. A future
run should give advisers a separate response schema and omit edit guidance.
Do not reinterpret this frozen run as if that fix had already existed.

## Lessons worth keeping

1. **A checker rescued the solo run.** Executable feedback is the first thing to
   add when an agent struggles; this run did not require a management layer.
2. **Parallel opinions cost something.** The team cost 27% more than solo Opus.
   Its roughly 12-second advantage here is only an observation, not a result to
   extrapolate.
3. **The sandbox made the experiment inspectable.** Every candidate batch ran in
   a fresh local Firecracker guest with complete no-egress witness and teardown
   evidence. Models themselves ran through text-only native host clients.
4. **Test the experiment before testing intelligence.** Pre-pilot review caught
   a missing-output false pass, an oracle coverage gap, cancellation leaks, and
   failure-path cleanup gaps. Their regressions are retained.
5. **A ceiling is an answer.** We found no task solo Opus could not solve. More
   seeds of the same easy family would not demonstrate the desired capability gap.

## Evidence and limits

[Protocol](PROTOCOL.md) · [Frozen plan](receipts/calibration-01/plan.json) ·
[Summary](receipts/calibration-01/summary.json) ·
[Sandbox controls](receipts/rooms-controls.json)

Each arm directory contains its frozen task, candidate, hypotheses/history,
usage receipts, exact outputs, and Rooms witness/teardown summaries:
[Opus solo](receipts/calibration-01/01-opus-solo/),
[Opus team](receipts/calibration-01/01-opus-team/),
[Astra](receipts/calibration-01/01-astra-solo/).
Raw native logs stay private because clients may print unrelated local metadata.

The reference uses exact rational matrix arithmetic, independently of the
candidate's path-walking implementation. Known-good passed all 65 checks in
Rooms; seven broken controls were rejected. Those 65 correlated checks are
**one task**, not 65 independent successes. No work code, data, or derived fixture
was used. No cloud infrastructure was provisioned.

This demonstrates bounded orchestration and isolated candidate execution. It
proves neither general intelligence amplification nor unattended production
readiness. Rooms runtime identities were verified during setup and recorded;
the adapter does not rehash full snapshot memory before every batch. The guest
runs as root, so its process-count ulimit is not a reliable fork-bomb boundary;
VM resource limits and host wall-time teardown are the backstop.

## Next experiment that would matter

Find several independently sourced **public** failures that solo Opus reproduces
under a fixed budget before designing assistance. Keep their final validators
hidden. Compare solo retry, an independent second solver, and a targeted critic
at matched token/cost budgets; keep stronger-model answers unavailable to all
Opus arms. Freeze selection, prompts and acceptance criteria first. Kill the
team policy if it cannot improve held-out task success enough to cover its cost.
Do this before building a persistent fleet or adding cloud distribution.
