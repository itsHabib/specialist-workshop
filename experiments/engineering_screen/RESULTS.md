# Better boundary tests, not a proven intelligence gap

**We did not find a reproducible Opus/Astra correctness gap.** Both solved the
small geometry, lease-history and network-flow sets. A later full-size geometry
probe broke two returned solutions, one from each model. This gives us a useful
engineering failure to study, not grounds for declaring a stronger team policy.

## Solo screen

| Task | Opus | Astra |
|---|---|---|
| Concurrent lease histories | Passed, 1 call, 67.5s | Passed, 1 call, 43.8s |
| Lower-bounded min-cost flow | Passed, 1 call, 137.8s | Passed, 1 call, 82.1s |
| Exact winding-fill geometry | No program returned by 150s | Passed, 1 call, 58.7s |
| Separate geometry diagnostic, 300s allowance | Passed, 1 call, 161.1s | Passed, 1 call, 41.1s |

“Passed” means the recorded cases, not every input allowed by the contract.
Times include candidate execution. The diagnostic used independent model
starts and fresh final cases; it does not prove the deadline alone caused the
first failure. The timeout remains recorded, with unknown usage/cost.

Eight native invocations total, no assistance arms or Sol arm. Completed Opus
calls report $0.855205 combined, including ancillary Haiku; the timed-out call
is unpriced. Astra receipts report tokens but no invoice cost. These are not
matched-dollar or matched-token comparisons. One observation per condition
cannot establish a repeatable speed advantage.

## The concrete failure we found

Post-run inspection prompted a full-bound input: one 60-vertex self-intersecting
walk with coordinates inside the stated ±1 billion range. Its exact area needs
**12,147 characters** as a canonical fraction.

| Sealed candidate | Result on that input |
|---|---|
| Fresh Opus geometry solution | `ValueError` |
| Fresh Astra geometry solution | `ValueError` |
| Earlier Astra geometry solution | Correct |
| Each failed solution with a serialization-only control | Correct |

Python's default integer-to-string conversion limit was the problem. Both
failed programs computed rational geometry but did not handle conversion of a
very large exact answer. The earlier Astra program explicitly disabled the
limit. Prepending that one runtime setting made both failed programs produce
the exact reference answer, without changing their geometry algorithms.
Those patched controls are **not model successes**. No extra model calls were
made. The [boundary probe](receipts/boundary-01/PROTOCOL.md) was post-hoc, not a
precommitted holdout. It is discovery evidence and must not be used as unseen
evaluation for a future helper trained on this finding.

The reference had the same serialization limit. The initial probe disabled it
in the isolated reference process; the reference now formats large integers
without changing the process-wide limit. Arithmetic was unchanged. A second
replay reproduced all five outcomes using the corrected reference.

## Lessons for the intake

1. **Ask what failed before picking a helper.** “Geometry is hard” did not tell
   us that the eventual defect was runtime serialization. The exact exception
   and a contract-boundary input did.
2. **Check the full contract, including output size.** Small random examples
   missed a valid maximum-size case. Reference implementations need these
   boundary checks too.
3. **More time and more agents are different interventions.** The first apparent
   separation was a timeout. A fresh solo attempt passed before any team existed.
4. **Prefer a cheap reproducer to a committee.** The concrete failure reduced to
   one runtime setting. An adversarial boundary tester is a plausible helper;
   we have not shown an LLM coordinator chooses it reliably.
5. **Treat model identity as evidence, not destiny.** Two Astra attempts differed
   on this engineering detail. One successful stronger-model answer does not
   establish a stable capability boundary.

Next: use this failure as a *development* example for an intake that distinguishes
algorithm errors, environmental limits and weak validation. Test that intake on
fresh public repository failures and unseen boundary cases. Compare a solo
worker with the same test feedback against any proposed helper; otherwise we
would confuse better evidence with better orchestration. Do not build a general
router or declare a skill validated from this screen.

## Evidence and reproduction

[Discovery plan](receipts/discovery-01/plan.json) ·
[Discovery results](receipts/discovery-01/summary.json) ·
[Deadline plan](receipts/deadline-01/plan.json) ·
[Deadline results](receipts/deadline-01/summary.json) ·
[Boundary replay](receipts/boundary-01/replay.json)

All returned candidates, hypotheses, usage receipts and output checks are in
those receipt directories. Native raw logs remain private. This was independently
synthetic work with candidate execution in Docker, **not Rooms**. No work code,
company fixtures, cloud resources or stronger-model answers in Opus prompts.

The first generator accidentally repeated fixed development examples in the
final set: two geometry, one lease, two flow inputs. Original results retain
those overlaps. Unseen counts are 18 geometry, 30 lease and 42 flow inputs;
all completed candidates passed those subsets. New plans remove exact overlap,
and the deadline diagnostic used 18 disjoint final inputs. These are correlated
cases from three tasks, not independent benchmark wins. Full contract bounds
are not exhaustively verified.

Run the boundary reproduction without model calls:

```sh
python3 experiments/engineering_screen/boundary.py experiments/engineering_screen/.runs/my-boundary-replay
```

Frozen execution commits are recorded in each plan. Check out the corresponding
commit to replay its exact original runner; current code includes the disclosed
split and serialization fixes. [README](README.md) gives new-run commands and
prospective intake strategies.
