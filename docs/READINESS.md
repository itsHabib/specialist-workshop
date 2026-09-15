# Local substrate readiness

The reusable path is **/packages**. Follow [MONDAY.md](MONDAY.md) to bring another
bounded JSON-output task through import, baseline, supervised training, checkpoint
inference, evaluation, comparison and correction. This is a local experiment
substrate. None of the measured specialist models earned production rollout.

Fresh economic qualification under a retry budget is **unmeasured**. A
[diagnostic comparison](RETRY-EXPERIMENT.md) now demonstrates a cheaper API model
recovering one task with validation feedback; the local adapter did not improve. A specialist may
be worthwhile with multiple attempts if complete task cost is lower at comparable
quality and acceptable latency. Existing one-attempt scores do not answer that
question. The [economics protocol](ECONOMICS.md) defines the comparison and the
current runner's limits; historical outputs and scores remain unchanged.

## Execution evidence

The compact [validation packet](../artifacts/package-validation/summary.json)
names exact packages, runs, artifact hashes and caveats. It includes original raw
records rather than replacing failed outputs with cleaned success examples.

- Three task packages imported: support intake (6/2/4 cases), analytics planning
  (160/6/24) and CI evidence (240/20/40 plus 16 final episodes).
- The support package completed real 20-step MLX LoRA training and loaded those
  weights for inference and evaluation. Its tiny dataset is a plumbing specimen:
  base scored 1/4; adapter scored 0/4 and produced malformed tool-call prefixes.
  That failure remains visible. It is not a useful specialist result.
- The CI package's deterministic policy completed 16/16 resettable episodes,
  with final filesystem checks and no protected-file changes. This validates
  the installed environment adapter, not general agent success or live CI repair.
- The analytics adapter trained for 120 steps, then was adopted into the generic
  package path. Adoption checked exact serialized train/development streams and
  bound the current complete adapter manifest. Original training records lacked
  configuration digests; adoption does not retroactively attest that history.
- On the reviewed code, browser inference `pkg-dbeb81ac83e2` loaded adopted
  checkpoint `pkg-f0a24974e753` and preserved raw/content scores separately:
  `raw_valid=false`, `valid=true`, `correct=null`. Manual inspection finds an
  intent error (both events selected for a Summit-only question). A fresh input
  correctly has no automatic claim of semantic success.
- Browser import and practice correction produced a second immutable support
  version: six practice examples became seven; development and final data stayed
  byte-equivalent under canonical serialization. The old version remains readable.
- A later deterministic majority evaluation records four explicit raw-valid rows.
  Earlier support model records omitted that field; their stored aggregate zero
  is unmeasured telemetry. Those records are retained with this annotation.

See [analytics outcomes](ANALYTICS-EXPERIMENT.md), [workflow outcomes](WORKFLOW-EXPERIMENTS.md)
and the [original classifier study](EXPERIMENT.md) for model-level results.

## Checks and independent review

Core revision: **d43442d0372babdcf434df033432e1ccd03c7532**. Subsequent delivery
changes contain documentation and captured evidence only.

The implementing task ran the full suite: **85 passed**, with two dependency
deprecation warnings, plus JavaScript syntax checks for all four pages and
`git diff --check`. Tests cover split isolation, immutable corrections, contract
and schema validation, bound adapter files, mismatched holdouts, strict versus
decoded outputs, API contracts and environment outcomes. The process-lock test
uses a real child process surviving a killed parent; it does not need ML compute.

The independent coordinating task first reviewed the package/worker/environment
core and found four issues: checkpoint reuse after holdout rotation, missing
raw-valid evaluation fields, unbound adapter configuration, and an accelerator
lock that could disappear while the numerical child survived. All four were
fixed in the core revision above. Its focused exact-head recheck reported **PASS**
and independently ran `tests/test_packages.py` (**14 passed**), with no further
blockers in that scope. It did not independently repeat the full suite, browser
flows or training. This is a bounded source review, not production certification.

The restarted localhost app was exercised against this revision. Desktop
(1440 px) and mobile (390 px) screenshots were inspected; neither has horizontal
page overflow. The result history is horizontally scrollable on narrow screens.
[Desktop capture](../artifacts/packages-desktop.png) ·
[Mobile capture](../artifacts/packages-mobile.png).

## Boundaries for Monday

Changing an ordinary JSON-output task is mainly package/data work. New semantic
equivalence rules, tools, environments, models or decoding recipes require a
trusted code extension and checks. The current numerical backend is pinned
Qwen3-4B on Apple Silicon; there is no universal backend plugin system.

Schema-valid output and a match to supplied labels are distinct from trustworthy
labels, live approval and useful outcomes. Family separation relies partly on
honest dataset design. Keep the next task's final evaluation independently
adjudicated and reserved; do not optimize against exposed final failures and
then call them an untouched test.

No RL/GRPO, hosted tenancy, automatic promotion or production integration was
added. Company examples belong in a separately approved work environment. The
Monday deliverable is a reusable way to find and measure the right task.
