# Next experiments that can change our decisions

Start with the current geometry pilot. Keep failures; do not shop for a winning
prompt after reading final answers. Each new comparison needs fresh cases.

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
