# Iterative repair pilot

This is a controlled repository repair exercise, not a claim about an entire
production codebase. Both policies get identical files, goals, test commands,
read/edit/run tools, visible test feedback and persistent conversation state.
They may write tests and retry. Candidate code runs in networkless Docker.

Start with Qwen 2.5 7B on rectangle-union variant 0, one solo and one assisted
run. The same model supplies assistance: the worker may request an investigator;
two consecutive official test failures also trigger one investigation. No
mandatory end-of-task critic. An investigator returns advice and optionally an
executable probe; its claims remain untrusted until tested. All helper calls
count toward the run's budget. Initial call allowance12, max2hr active time,
max30min/call; hosted CLI cost ceiling$4/run if used. Local cost is unpriced.
These are generous ceilings, not targets. Time and cost exhaustion are incomplete.

Use a frozen final seed90137, external exact coordinate-cell oracle and original
regression tests. Grade saved edit snapshots after generation to measure when
an eventually accepted candidate first appeared. Hidden answers never enter a
worker prompt. Native clients have tools disabled; only the loop can execute
commands, inside its disposable Docker environment. Shell-generated filesystem
changes do not persist: explicit file writes do.

Report full-task success, first hidden-valid candidate time, actual finish time,
model/token/known-cost receipts, investigator use and false diagnoses. Equal call
caps are only an initial allowance, not an equal-spend result. If one policy has
an apparent win, give the losing solo the same cumulative token/spend opportunity
(or explicitly report the unmatched budget), then repeat with new independent
runs. Preset time checkpoints:2,5,10,20min. Local token totals include input and
output; compare wall time separately from summed model/tool time.

Before any learning claim: distill a short lesson from practice traces, freeze
it, and test on variant1 with original tests and new final seed90149. Compare
lessons enabled/disabled with the same model and tools. The two variants share
geometry and canonical controls, so this is near transfer, not generalization
to a different domain. Do not put final failures or reference solution into the
lesson. Retain negative runs. Add genuinely distinct tasks before broader claims.

Stop a matrix if its setup is invalid; repair the harness, label a new run, and
retain affected receipts. STOP files request cancellation. Resume reconciles a
reserved call's receipt; missing receipts require inspection rather than a
silent duplicate. Unknown usage remains unknown. No cloud resources required.

## Adaptation before assisted results

The first solo run repeatedly wrote files without running tests. Preserve it;
the tools were available, so this is an observed tool-selection failure. Add an
explicit **feedback** policy: same solo model and tools, automatically execute
visible tests after edits. **assist** now adds a same-model investigator to that
feedback loop. Thus comparisons distinguish automatic feedback from another
agent. Both use the same tests, never the hidden oracle. Tests/latency and all
calls count toward cost. This amendment follows observed solo behavior and
precedes any assisted results; repeat any gain with the new policy fixed.

## What actually ran

The original solo/assist and lesson/no-lesson matrix was not completed. This
became an adaptive build session: solo, automatic feedback, Haiku control,
frozen prose lesson, verified source reuse, fresh-worker continuations and
external development feedback. Negative trials remain in the receipts. No
statistical, same-model helper, or general learning advantage is claimed.

The v1 rescue missed compact JSON in final seed90149. That evaluation was spent.
A subsequent repair received a separate development probe (seed73021), then was
checked with final seed771239. A later full CLI run started from the original
v1 source with public development checks and final seed771251. These are
adaptive engineering trials; new seeds do not turn the discovery into a
preregistered comparison or remove shared canonical cases. See RESULTS.md.
