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
