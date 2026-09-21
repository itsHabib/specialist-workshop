# Cheap models need useful assistance, not just more agents

The first critic pilot did **not** demonstrate a rescue. It exposed timeouts,
invalid code and an unhelpful critic. One simple formatting fix recovered a
correct Sonnet answer. The simpler-format/algorithm-guide follow-up also failed; details below.

## Critic pilot

Exact area of self-intersecting polygonal walks, including overlapping rings and
two winding rules. Four development cases; 18 disjoint final cases. Each policy
gets three native invocations, including a shared initial candidate. Solo spends
two on repairs; boost spends one on a same-model critic and one on repair.

| Model | What actually happened | What it tells us |
|---|---|---|
| Sonnet 5 | Initial and boosted calls exceeded 240 seconds. Both solo repairs returned code with stray closing XML tags. Raw final programs passed 0/18. | Incomplete comparison; no reasoning-level boost verdict. |
| Haiku 4.5 | Simple availability probe succeeded in 2.5 seconds. Geometry baseline exceeded 240 seconds; the next call was stopped and remaining calls skipped. | Task call timed out; cause not established. No boost comparison completed. |
| Llama 3.2 1B | All five local calls completed in about 12 seconds combined. Initial/solo code was invalid; boosted source was empty. All raw finals passed 0/18. | The interface and proposed critic failed. This does not establish a mathematical capability ceiling. |

**Important diagnostic:** removing exactly two trailing XML closing tags from
Sonnet's last solo candidate yielded **18/18**. No algorithm was changed. This
was post-hoc on the spent final set, so it is not a pre-registered policy win.
It points to output validation/format handling as the next cheap intervention.

The local critic simply repeated the requested winding semantics. Sonnet's
critic speculated about an implementation bug even though the initial call had
produced no code. Neither observation supports treating another agent's prose
as evidence.

## Simpler format and algorithm guide

A four-call adaptive follow-up used a source-only JSON schema and new random
cases. Canonical controls repeat: 2/4 development and 2/18 final cases overlap
with the original run; 16/18 final cases are new. Each local model got one unassisted call and one call with
an exact sweep-line algorithm outline. The driving frontier agent wrote that
outline: this is a test of supplied guidance, not same-model self-improvement.

| Model | Source-only baseline | With algorithm guide |
|---|---|---|
| Llama 3.2 1B | 0/18; incomplete function, 1.7 seconds | Hit the 6,000-output-token cap in 75.7 seconds; execution-invalid, no candidate scored |
| Qwen 2.5 7B | 0/18; indexing exception, 21.3 seconds | 0/18; undefined helper/scope errors, 42.4 seconds |

No rescue here either. The guide changed Qwen's approach but did not produce a
working implementation. A longer answer from 1B mostly spent the output budget.
[Follow-up receipts](receipts/representation-01/) retain the guide, identities,
source and failures. This follow-up did not give execution feedback or retries;
it tests one-shot guide following, not whether repeated component repair works.

The next useful test is **small separately verified components**—exact line
intersection, crossing grouping, slab integration—then composition. Start with
rectangle union if 1B cannot produce these components. Do not spend another
large matrix on generic critics until a narrower mechanism works.

## Receipts and limits

- [Original protocol](PROTOCOL.md), [hosted receipts](receipts/hosted-01/),
  [local receipts](receipts/local-01/), [next experiments](NEXT.md).
- Original runner revision: `c22a3a8`. Its source hashes and exact cases are in
  each plan. The original plan SHA-256 was
  `331b91480ff61af0a32cf6e20c11adf5ec8ebb3a2316850d9549f45dfb31761e`.
- Hosted pilot was deliberately stopped after repeated timeouts; local work
  continued in a separately labeled root using identical cases. The run did not
  silently retry uncertain calls. See `hosted-01/adaptation.json`.
- Haiku native receipts: [baseline timeout](receipts/hosted-01/haiku/initial/receipt.json),
  [stopped retry](receipts/hosted-01/haiku/solo-2/receipt.json),
  [availability probe](receipts/availability-01/receipt.json).
- Known Claude CLI estimates total **$0.230165**, including the availability
  probe. Four interrupted/timed-out calls have unknown usage: this is a lower
  bound, not the total bill. Local electricity/hardware and the driving/review
  agents are not priced here. Policy costs include the shared initial call in
  both arms. An invocation can include native-client internal model calls;
  aggregate usage receipts are retained.
- One trial/model, fixed treatment order, uncontrolled host load. No statistical
  claim. Cases are small, not full-bound coverage. The oracle has hand-known
  cases, an independent grid oracle for rectangle ensembles, and geometric
  invariants; arbitrary self-intersections are not independently enumerated.
- Local model digest was recorded before and checked after the critic run.
  Requests use the tag, not immutable digest addressing. Plans are inspectable
  procedural freezes, not tamper-proof attestations.
- Python candidates ran networkless with resource limits in Docker; all retained
  candidate executions confirmed cleanup. Model clients ran on the host, not
  inside Rooms. Native logs remain private; published cases are now spent.

## Lessons worth using

1. **Check the artifact before recruiting a team.** Valid JSON can still contain
   invalid Python, empty source, or prose. Separate these from wrong answers.
2. **A critic needs something real to inspect.** An absent candidate is a failed
   generation, not an invitation to invent a diagnosis.
3. **Match the help to the bottleneck.** Format repair, an algorithm outline,
   narrower subtasks and an exact verifier are distinct interventions. “More
   agents” is not a substitute for choosing one.
4. **Count failures and overhead.** Unknown timeout costs are not zero. Faster
   empty answers are not better answers. A stronger author's supplied algorithm
   is assistance, not evidence of unaided small-model discovery.
5. **Repeat only a promising result.** Use fresh cases and counterbalanced order
   before claiming a reliable upgrade. Keep the failed runs visible.

Validation: **149 tests passed, four explicit live skips**; JavaScript syntax
check passed. An independent Sol review checked receipts, privacy and claims;
its fixes clarified repeated canonical cases and execution-invalid scores. No
remaining material findings. This does not turn the small pilot into a general
benchmark.
