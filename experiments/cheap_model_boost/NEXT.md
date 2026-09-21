# Next: real work loops, then learning across tasks

**Current evidence:** no repeatable boost demonstrated. The first pilot mostly
measured generation/interface failures under short deadlines. Keep its receipts;
do not treat that as a verdict on iterative agents or small models generally.

## Give the baseline a fair chance

Solo means one capable working session, not one answer. Both conditions get the
same starting repository, tools, visible tests, ability to inspect files, edit,
run commands, diagnose failures, retry and keep context. Assistance adds useful
help to that workflow; it must beat a baseline allowed to do ordinary engineering.
Count the assistance, consolidation and verification costs too.

Start with a repository task: take the synthetic geometry implementation through
a bug report, reproduction, patch and regression tests while preserving its API.
Then test a feature spanning parser, geometry and CLI behavior. Acceptance needs
independent hidden behavior checks AND the existing regression suite. Keep work
material out. These are controlled repo exercises; validate on actual public
backlog work before claiming real-world productivity.

Let agents create tests and small tools, inspect runtime errors, change approach,
and ask for a helper when evidence warrants it. Do not force every task through
a critic. A controller can decide to decompose, derive independently, consult,
repair a representation or continue solo. Record what it actually did.

## Let worthwhile work take time

Remove the four-minute model-call deadline from the next comparison. Declare a
generous total run/spend budget before starting, and allow long reasoning calls.
Observe progress where available; silence from a non-streaming client is not
proof of a hang. Check transport/liveness separately from task difficulty.

Save work and receipts on interruption. Budget exhaustion or a transport failure
is an incomplete run, not proof of mathematical inability. Keep slow completions
in the results. Speed is itself an outcome: assistance can win by reaching the
same accepted quality sooner, even if solo eventually succeeds. Record time to
first valid solution and quality-versus-time checkpoints; report parallel wall
time separately from summed worker time and total spend. Preselect those
checkpoints rather than choosing the one where a treatment looks best. A slow
success demonstrates eventual capability, not necessarily practical usefulness. Compare success versus cumulative time/tokens/spend, including
an equal-spend solo continuation; equal call count alone is not a fair budget.
Candidate-program CPU/memory limits still protect the host and should be sized
to the workload. They are separate from time allowed to think and repair.

## Test whether anything is learned

Use three conditions on matched starting states:

| Condition | What it tests |
|---|---|
| Ordinary iterative agent | What the model already accomplishes with normal tools and retries |
| Same agent with adaptive assistance | Whether assistance improves completion beyond more solo effort |
| Assisted agent with lessons from earlier tasks | Whether retained knowledge improves later, unseen work |

A useful lesson is small: failure pattern, intervention, supporting example,
when it applies and when it does not. It may become a prompt, skill, executable
helper or test technique. Version it; do not assume every retrospective is true.
First learn these artifacts; weight training can follow if this earns its keep.

Freeze the lesson set before evaluation on new task variants. Compare with
lessons disabled, using the same model/tools, and repeat with varied task order.
Improvement on the task that produced the lesson is repair; improvement on new
tasks is evidence of transfer. Keep evaluators outside the agent's editable
workspace so it cannot improve its score by weakening acceptance.

Start with a small pilot, then replicate any win across several independent
problems and runs. Report completed tasks, regressions, human intervention,
latency and total spend. A repeatable boost means better accepted outcomes or
less total effort at comparable quality—not more roles, tokens or activity.

**Implementation status:** this is the revised next-run design. The archived
pilot runner still implements the old bounded text-only protocol; it has not
been converted into the tool-using, persistent work loop described here.

## Candidate investigations

| Experiment | What changes | Evidence worth keeping | Stop / reject condition |
|---|---|---|---|
| Replicate a rescue | New seeds and independent initial attempts; counterbalance solo/boost order | Full-case successes, paired differences, total tokens, dollars, time | No repeatable advantage over equal-budget solo retries: do not sell the critic as an upgrade |
| 1B capability staircase | Rectangle union → simple polygons → self-intersections → winding across multiple walks | First level where the baseline fails, exact error categories | Transport/format failures dominate: fix the interface before claiming a reasoning limit |
| Representation help | Same model gets a generic exact-arithmetic/sweep-line checklist versus an independent critic | Whether reusable structure beats spending tokens on another persona | Handing over a task-specific implementation does not count as model rescue |
| Search, not consensus | Generate independent candidate algorithms; run development tests; choose by evidence | Candidate diversity, selection reliability, unseen success | Selector never beats best fixed policy at matched spend: drop it |
| Hard algorithm transfer | Lower-bound min-cost flow with negative cycles; exhaustive small-graph oracle | No hidden-case feedback; include disconnected cycles, parallel edges, self-loops | Geometry-only benefit does not transfer: scope the recipe narrowly |
| Engineering transfer | Leased-queue linearizability; bounded-history enumeration oracle | Concrete illegal histories found and repaired | “Looks correct” without matching the independent oracle is a failure |

For a serious comparison, preselect several task instances and repeat each
policy, count intake/helper/repair/verifier/selection overhead, and retain the
stronger model only as a separately priced reference. Don't silently promote a
cheap model's failed call to a stronger provider.

**Useful output:** a short recipe saying when to try it, the exact helper prompt,
what artifact changed the result, and when to stop spending. More roles are only
worth keeping if they produce better verified answers.

**Rooms comes later:** first get a repeatable local advantage. Then replay the
same sealed candidate artifacts in Rooms to test isolation and lifecycle. A
cloud run tests deployment behavior; it does not magically strengthen evidence
that the model reasons better.
