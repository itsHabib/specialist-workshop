# Engineering failure discovery

This is a discovery screen, not a benchmark win or an assistance trial. All
problems are independently synthetic: no work examples, derived fixtures, or
sheet-metal artifacts. No writing tasks. No claim of benchmark novelty.

Screen exact winding-fill geometry, linearizability of a leased job, and
lower-bounded minimum-cost circulation. References are authored first and
checked against hand calculations, independent enumeration, and invariants.
Run known-good and invalid-output controls in the same candidate sandbox.

Freeze three contracts and their development/final inputs before native calls.
Models: claude-opus-5 and gpt-6-astra, high effort, at most two calls per task,
150 seconds per call. Native Opus calls have the existing $0.80 limit; auxiliary
usage counts. Maximum twelve calls; no additional retries or task tweaking to
manufacture separation. Same contracts, examples and feedback for both models.
This is not an equal-token or equal-dollar comparison. No Sol arm this round.

Each task gives four development examples. The first development-passing edit,
or the second attempt, seals the program. Final results are never fed back in
this run. Missing/invalid output, timeout, format failure and model failure are
recorded separately from a valid program's wrong answer. All attempts retained.

Candidate programs run in the existing pinned Docker sandbox: no host mounts,
network disabled, read-only root, unprivileged user and resource limits. Models
run text-only through authenticated host clients. This run is NOT a Rooms run,
not a privacy boundary around model prompts, and not a hosted arbitrary-code API.

Selection: prioritize a task only when Astra passes and Opus produces a valid
but incorrect program within this budget. Transport/format failures do not
establish a reasoning gap. Before claiming a reproducible gap, repeat the task
with independent model starts and genuinely fresh cases. These screened cases
cannot later be called untouched evaluation data.

If all pass, report a ceiling and stop. If both fail, inspect the task/checker
before assistance. If separation exists, save the specific failure and a
prospective assistance experiment: solo retry versus independent critic or
counterexample seeker, no Astra solution supplied, fresh sealed final inputs.
Do not run that follow-up until its new protocol is fixed.

Intake remains an explicit judgment: executable contract, observed failure,
uncertainty, useful helper, and acceptance evidence. No general router or skill
is claimed to be validated by three hand-selected tasks.
