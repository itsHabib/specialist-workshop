# From a clean checkout to a decision in another application

An adopting engineer should be able to point an agent at a private program, train a local model for one existing decision, and see whether the program does better with it. The first useful deliverable is a model running in offline replay alongside the current implementation. A result in favor of the current implementation is a successful evaluation.

## What is available

On November 21, 2026 a fresh public clone of `itsHabib/specialist-workshop` at `d85c436` exposed the original task-package workflow. Running `python3 -E -S -m decision_workshop --help` there reported a missing module. A clean local clone of `codex/decision-maker` at `dacf95b` exposed the decision trainer. The standalone sequence kit includes this source, example models and synthetic measurements.

| What the application can supply | What an adopting agent can build |
| --- | --- |
| A fixed set of actions, inputs recorded before the choice, and reviewed selections | A local tree that learns those selections. The `train`, `evaluate` and `predict` commands use standard Python. Use independent workload groups to check whether the choices transfer. |
| A fresh list of candidates per request and comparable evidence about the candidates | A model that ranks numeric candidate features. The application supplies the handles, checks eligibility, executes the selected candidate and grades the result under a fixed budget. The documented `fit_rankers` convenience function currently fits both a tree and Google BERT and requires the local Google base and its dependencies. |
| A repeatable operation, alternative actions and a way to check the final result | A sequence policy trained by running alternatives from the same saved starting point. The application supplies the isolated copies, continuation, time budget and verifier. |
| Final outputs alone | An inventory of recoverable decisions. Supervised training becomes possible when the original inputs and reviewed choice can be reconstructed. Judging alternatives additionally requires measured outcomes, comparisons or faithful replay. |

The tree trains from scratch. The neural decision path uses the pinned Google BERT-Tiny base. The public examples use synthetic inputs. Store the private adapter, data, evaluation reports and checkpoints in the adopting project's own location. A checkpoint can include category values and the ranker run directory contains training rows. [Decision trainer](DECISION-MAKER.md) · [Candidate ranking](DECISION-STEERING.md) · [Sequence evaluation](DECISION-SEQUENCES.md)

## Two cold adoption trials

Two agents with fresh contexts received the candidate checkout and separate, unfamiliar applications. Both applications had complete synthetic histories with measured outcomes for the actions offered. They ran on one host using Python 3.14.6. Neither agent received the independent grading workloads. Each wrote an adapter, trained a checkpoint outside the workshop, and returned a callable. A separate interpreter with site packages disabled then loaded each delivery from a relocated directory and scored it on those grading workloads.

| Task | Integration cost observed | Independent application result |
| --- | --- | --- |
| Pick one of two storage formats | The README led the agent to unavailable neural dependencies. It needed one author hint that a from-scratch tree was acceptable. It wrote the features, label conversion, split and fallback, and copied the runtime into its application. | All 150 blocks from 30 new groups decoded exactly and used allowed formats. The model used 7,820 bytes, the existing policy 10,131, and an exact run-count rule 7,755. |
| Rank new queue entries under a one-attempt budget | The documented two-model fitter required missing neural dependencies. The agent found the offline tree route by reading the fitter's implementation. It needed zero author hints, wrote the adapter and brought the required runtime source into its application. | All 96 queues from 32 new groups returned legal permutations and preserved their ranking after their IDs were renamed. Total penalty fell from 9,165 to 8,558. The model won 23, tied 53 and lost 20. Its own earlier ten-family test was inconclusive. |

These trials establish that a fresh agent can adapt the local code when an application supplies measurable outcomes. The fixtures arrived with a known decision and a callable incumbent. Discovering a decision in a much larger codebase, reconstructing incomplete logs, installing on another platform and verifying results in an external team's environment remain open. The immediate usability work is a tree-first handoff from a separate project and a documented offline path for fitting a candidate ranker. Repeat with a fresh agent and only historical final choices, and require an honest inventory of missing evidence.

## Milestones

| Milestone | Evidence that earns it |
| --- | --- |
| A1. Obtain the toolkit | Starting at the exact URL or archive an adopting team will receive, a fresh agent can discover the offline path, identify the model's origin and run the tool from another project's process. Repeat on the intended operating system and runtime. For adoption by GitHub clone, repeat against the published revision after a separately authorized release. |
| A2. Train one genuinely new decision | The agent finds one call site, records its incumbent and available actions, builds an adapter outside the toolkit, trains a checkpoint and reloads it in a fresh process. It documents what it had to infer and any author help. Include an unknown or unavailable action and exercise fallback. |
| A3. Decide whether the target application benefits | Freeze the model and threshold on development families. On untouched families, run the incumbent and model with the same candidates, validation and time budget. Report paired wins and losses, invalid results, reviews or overrides, and full elapsed cost including feature collection. For ranked decisions also report how often a usable candidate was present. |
| A4. Earn a live pilot | In the private application, log model suggestions alongside normal decisions. Check that the replay inputs and latency match the live call site. Enable limited steering only after the application's own validator, fallback and rollback have been exercised and its owner accepts the tradeoff. |

### First handoff

1. **Choose one decision.** Find the function that already picks an action or orders candidates. Identify what was known at that instant, how related jobs can be grouped and how the user recognizes a good result.
2. **Inspect one good case and one correction.** Reconstruct the choices available in each. Identify missing fields, observations made after the decision, and historical versions that prevent faithful replay. If a decisive field is missing, the first deliverable is a recorder or importer for that field.
3. **Build the private adapter.** Map raw inputs to features, load the model once, and submit suggestions to the existing program. Keep option generation, validation and execution under application control. Use the original behavior when the model defers.
4. **Return an inspectable verdict.** Provide the checkpoint identity, a command from the application's directory, a paired replay report, a concrete failed case and the choice between keeping the incumbent, collecting better evidence or running a shadow pilot.

An agent can start the inventory immediately. A first model needs evidence for one identifiable choice. A model earns authority to steer a particular application only after that application's outcome test. These milestones depend on the evidence available for a decision, so record the number of independent families and their coverage as well as the raw number of rows.

## Questions for the next depth experiments

| Question | Bounded test | What the result changes |
| --- | --- | --- |
| Can a fresh agent adopt the same method without help? | Give two fresh agents separate repositories, a short assignment and a read-only toolkit. Record setup obstacles, code written by the adopter, fallback behavior, and results on privately held jobs. Repeat on a different environment with ordinary historical logs. | Fix the first repeated onboarding obstacle. Retest with a fresh agent before expanding the API. |
| How much can this decision possibly help? | On small held-out cases, run all offered candidates through the checker to compute a hindsight reference. Compare the incumbent and learned order under the same execution budget. Count cases in which every offered candidate fails. | A small gap suggests improving the generator or executor. A large gap justifies more work on the model. |
| Can a model learn a stable preference when choices tie? | Hold features, training cases and recipe fixed. Change only the priority order used to label tied outcomes. Then compare credit for all best actions and a measured secondary cost. Validate on fresh cases and charge for feature time. | Pursue a training change if the measured end result improves across workloads. |
| Does a gain survive a changed program? | Hold out workload families and a later program version. Rename and reorder options, remove a useful option and omit a decisive observation in separate runs. Use the actual integration with its fallback and verifier. | Expose a missing signal, narrow the supported decisions or keep the existing rule where performance drops. |

### Starter prompt for an adopting agent

> Use this toolkit locally for one decision in our application. Find where our code chooses between allowed actions or candidates. Trace one successful case and one correction. Tell me which facts were available at decision time, which outcomes can be verified and what is missing. If the evidence supports training, build the smallest local model in our private project, reload it from our application and compare it with the current implementation on unseen workload families under the same budget. Return the reproduction command, checkpoint, paired results and one case worth inspecting. Run the existing checker on each result and keep live execution on its current policy during the trial.
