# Decision Model Maker roadmap and depth experiments

**First application milestone.** Give a fresh adopter a real software test suite. Compare the order its host actually executes under the current rule and under a frozen model. Record whether any changes produce earlier verified failure feedback. Two public CI jobs have been imported and reconciled. The next step is to reproduce their tested revisions from the host. The [real test-history protocol](DECISION-TEST-HISTORY.md) owns this trial and its stopping rules. The event-driven and exact-cover studies below are conditional follow-ups.

The workshop has tested candidate ranking, early stopping and regeneration. The scheduling experiments found cases where a fixed budget beat learned stopping. The next question is whether decisions at different stages can improve one another, and where several models compound mistakes. [Earlier steering results](DECISION-STEERING.md) · [Adoption plan](DECISION-ADOPTION.md)

## What the maker must deliver

An engineer should be able to give a fresh agent the toolkit and a separate application, point it at one decision, and get back a runnable comparison and a defensible recommendation. The host supplies the legal actions, verifier and fallback. A learned policy earns its place by improving the host's result. A reproducible reason to keep the existing code is a complete outcome.

### Where the evidence stands

| Capability | Evidence in this snapshot | What still needs to be demonstrated |
| --- | --- | --- |
| Train and reload | A from-scratch structured tree and a pinned Google BERT-Tiny encoder accept labeled records. The tree runs with standard Python. Clef-Flash is an optional Qwen-derived comparison. | A fresh adopter picks a suitable model from the actual handoff and identifies its training origin, feature requirements and runtime dependencies. |
| Steer executable code | Candidate ranking, bounded sequences, a SQLite recovery worker and two cold adoption trials have executable evidence. In the ordinary recovery fixture, the tree verified 225/384 jobs by deadline, the tuned rule 271, and a repaired executor 337. | One external application with imperfect records runs the complete selected policy through its own verifier. The incumbent, a simple code repair and the model get the same observations and budget. |
| Score the whole application | The evaluator distinguishes unknown candidate lists from known empty lists. It protects against reusing training IDs or groups and reports paired comparisons where baseline choices exist. | Every incoming job gets an accounting entry, including missing inputs, silence and fallback. Report separately which outcomes could be graded. |
| Ship a decision | Reloadable checkpoints and caller-side prediction exist. One-host trials relocated their runtimes and exercised unseen groups. | The exact runtime, model, feature contract, threshold and fallback travel together. A new process reproduces the accepted behavior, and the previous version can be restored. |
| Learn when to spend more compute | The flagship below specifies a router, selector and final policy with strong coded controls. | One real application records receipt and response times and can grade paired decisions under a shared deadline. The incremental value of the stack is an open question. |

[Model guide](DECISION-MAKER.md) · [Cold adoption evidence](DECISION-ADOPTION.md) · [Recovery comparison](DECISION-RECOVERY.md) · [Limits under shift](DECISION-LIMITS.md)

### Three gaps that can change a decision today

These observations were reproduced on implementation revision `f25be5f`.

1. **A job can disappear from headline coverage.** With two labeled held-out jobs, one correct suggestion and one unrecorded candidate list, the top-level evaluator counted two jobs. Its `model` section counted one, reported zero reviews and showed 100% coverage. To an operator there were two jobs and one suggestion, or 50% coverage. Preserve the conditional statistic and show its denominator alongside the total and the actual fallback result. [Evaluator](../decision_workshop/evaluation.py)
2. **A reloaded checkpoint can make a different operational decision.** A tree with two votes for action A and one for B produced a review at a cutoff of 0.8. Calling the reloaded checkpoint with the default cutoff of zero produced a suggestion on the same request. The final-test and integration examples currently rely on the caller to carry the development setting forward. Treat the threshold and fallback as part of the deployed policy. [Runtime](../decision_workshop/core.py) · [Integration examples](DECISION-MAKER.md)
3. **The default starter can silently teach a tree only the majority action.** The scaffold starts with one text field. The tree consumes structured features and ignores text. Training on three cases whose text explicitly identified A, A and B produced the same A preference on both input strings. A tree-first handoff needs a structured example and a way for the adopter to see which features the learner consumed. A candidate ranker also needs an explicit host tie and fallback policy. [Task scaffold](../decision_workshop/scaffold.py) · [Tree training](../decision_workshop/core.py) · [Ranker](../decision_workshop/steering.py)

### Next work, in dependency order

| Work | Deliverable | Acceptance evidence |
| --- | --- | --- |
| A1–A2, adoption and shipping | From the actual handoff artifact, have a fresh agent discover one decision in an unfamiliar application, use reviewed examples, train a structured tree offline and reload the selected policy from that application. Record Python/runtime version, source and model hashes, feature contract, threshold, allowed actions and fallback. | The training directory can be moved away. In a fresh process and on the adopter's target platform, an ordinary case, a low-score case, a missing list, a removed action and an unreadable checkpoint each produce the declared host behavior. Record command, friction, author hints and time to first checked result. |
| F1, evidence for a real decision, in parallel with A1–A2 | The adopting team identifies the choice, unit of opportunity, time of first receipt, original policy or explicit skip, deadline and later measurable outcome. Preserve input versions and distinguish unknowns from observed events. | Replay a completed session. Change future data and insert a late correction. Each prior decision must stay identical. A missing original response stays unknown. Count observed, reconstructable and scorable opportunities separately. |
| A3 and F2, headroom and fair comparison | Once there is a caller and a usable tape, freeze the objective, review costs, models, cutoff and minimum worthwhile improvement on development groups. Compare the incumbent, a repaired coded baseline, and the trained policy plus fallback. Use a bounded hindsight reference where alternatives can be verified. | Report paired wins, ties and losses on later sessions or untouched families, group-level uncertainty, wrong suggestions, review load, verified outcomes and full latency/cost. State which counterfactuals could be measured. If a simple rule wins or the interval is inconclusive, document the result and keep its reproducer. |
| A4, shadow and recovery | Run the selected policy in shadow, then have the host owner decide whether its measured tradeoff warrants a limited pilot. Track overrides, missing features, changed input distributions and delayed outcomes by policy version. | Exercise a timeout, unavailable model, changed input schema and rollback on the host. The coded fallback returns the expected result and retry cannot duplicate the host's effect. Identify a concrete stop condition and reproduce the previous policy after rollback. |
| F3–F4 and D1–D3, conditional research | After measurable headroom, compare one learned stage with a generalist and combinations of learned and coded stages. Use exact cover for solvable controls and a second application to test transfer. | Match end-to-end budget, including feature extraction, queue delay and verification. Measure missed candidates, ordering mistakes, premature stopping and downstream effects. Keep the simplest configuration that earns the declared gain. Deliver a minimized failure even when every learned variant loses. |

### First deliverable and priorities

**1. Make the evaluated policy travel with the model (A1–A2).** Start with one host function that already dispatches an allowed action or uses its own fallback. Train a structured tree on synthetic examples. Record the model identity, input fields, nonzero cutoff and which host fallback is expected. Load it in a fresh process outside the training checkout. Use the following two-job regression to check the caller and evaluator together.

| Labeled job | Expected observation in the host |
| --- | --- |
| The frozen model prefers an available, verifiably correct action different from the incumbent's choice. | Observe the action the host actually dispatched and the verifier's result. Deliberately route this job to the incumbent and confirm the dispatch check fails, even if both actions produce valid results. |
| The candidate list was never recorded. | The model refers the job for review. The host executes its declared fallback. Record the fallback outcome independently of the model's decision. |

The report must reconcile two arrivals, one correct suggestion, zero wrong suggestions and one review. Overall suggestion coverage is 1/2. Coverage among jobs with recorded candidates is 1/1. The host's final verified outcomes get their own counts. Repeat after restart and run one request whose score is below the frozen cutoff. A changed decision after reload fails the trial.

**2. Make the unfamiliar application the real trial (A1–A3).** In the Click test-history study, capture the candidate order the host actually ran, changes versus the incumbent after fallback, and their verified outcomes. A fresh agent should reproduce the result from the host directory using the offline structured-tree path. Capture author hints and packaging friction. A precise inventory of missing labels is a useful feasibility result. Training and adoption remain open until the model runs from that caller. [Work sequence and evidence](DECISION-TEST-HISTORY.md)

**3. Measure whether the real decision benefits from learning (F1–F2).** Collect the Click evidence in parallel with packaging. Freeze the opportunity list and budget. Compare the best simple ordering rule with the frozen tree on later changes and account for all arrivals. Where a private event-driven host also records the incumbent, skips, response readiness and eventual outcome, the same comparison can transfer there. Each host retains its application records.

**4. Test depth where the application shows room to improve (F3–F4).** Start with the cheap rule, then one model deciding whether to request extra analysis. Add a second model to choose an action only if the extra evidence changes useful decisions often enough to pay for its cost. Remove each model in turn, keep the same deadline and compute budget, and count verified outcomes across full jobs. If a coded repair wins, keep the repair and preserve the smallest case that explains the result.

For the provisional 80% coverage and 95% accuracy example, 100 eligible cases could yield 76 correct suggestions, four wrong suggestions and 20 reviews. That meets both percentages exactly. What the user ultimately received on those 20 cases depends on the fallback. Record its outcome, latency and cost. Set adoption thresholds per task, including a cost for harmful actions and a floor for performance on important rare families. The simulator already has a case where the cost-minimizing policy sensibly reviews more often. [Costed release experiment](DECISION-RELEASE-SPEC.md)

The release decision and the research decision are separate. Access to the local review artifact is enough for an independent technical review. GitHub-clone adoption is earned when a fresh clone from the intended distribution channel passes A1. The adopting application retains its data, checkpoints and operational traces. [Independent review prompt](DECISION-REVIEW.md)

## Flagship. Decide when more computation is worth buying

Start with one real application, one time-sensitive decision, its current implementation and an outcome that becomes observable later. All application-specific records, models, evaluation choices and findings belong to that application. The reusable deliverable is an adapter that plays back received events, decision opportunities, completed computations and subsequent outcomes in their original order.

| Stage | Experiment | Evidence and stopping condition |
| --- | --- | --- |
| F1. Reproduce one session | Record source event time, first local receipt, revisions, decision cutoff, input versions, processing completion and fallback. Mutate later records and inject a delayed update. | Earlier inputs and decisions replay identically from the same prefix and frozen responses. Count unreconstructable decisions. If key evidence is missing, deliver the recorder before interpreting model scores. |
| F2. Measure headroom | Run the cheap path and a more expensive analysis on the same representative replay cases. Include every eligible opportunity in the denominator. Account for cases where either answer would arrive too late. | Compare always-cheap, budgeted periodic or random analysis, the incumbent, and a retrospective allocator with the same budget and realized outcomes. If even the retrospective reference buys little, change the decision under study. |
| F3. Test a stack | A router allocates analysis effort; a selector combines the evidence that has actually arrived; a final policy can accept, defer or use the existing choice. Fixed application rules validate anything executable. Compare one generalist, each single learned stage and all eight combinations of coded or learned stages. | Freeze on earlier sessions. Replay later sessions with the same incoming events and each arm's own queues, outstanding work, cache and prior actions. Score full outcomes, coverage, missed deadlines and end-to-end cost. Retain a stage only when its incremental contribution survives comparison with simpler controls. |
| F4. Discover a failure that transfers | Find a session where extra analysis helped one decision but delayed the next. Shrink the sequence until the loss is reproducible. Try one repair chosen on development data, then replay new sessions. | Deliver the smallest failure, the changed decision, and a fresh paired result. If gains vanish, retain the replayer, measured cost curve and regression case. |

The first artifact should be a timeline for a real session that the application's owner recognizes. It should show the original decision, the data available at that instant, the work in progress, and the outcome. Full fidelity on one session is an integration milestone. Operational claims require independent future sessions. Use contiguous held-out time blocks, account for delays before outcome labels mature, and preserve the first sealed score after model selection.

Two ambitious extensions depend on what fails. If the system repeatedly pays for an expensive analysis that changes nothing, train a model to predict its incremental value from cheap observations. If local wins create downstream queues, give the controller a shared compute budget across concurrent requests and charge for the opportunities delayed by each choice. Both require executable comparisons with a simple priority queue under the same resource ceiling.

## D1. Prove the test can distinguish a good combination

Start with exact cover. Select disjoint sets of items so each required item appears once. A reserve inventory costs one query to reveal. Generate one concrete certificate for a solution so a separate checker can validate every selected set, access charge and covered item.

The six-item witness uses active tiles A = abc, B = ade, C = df and reserve tile R = def. Allow two placements. The first generator can focus on item a or item d. The ranker chooses the cheapest tile or the tile covering the most items unique to one visible tile. Give A and R cost two, B and C cost one, and resolve equal scores by tile name. At the root, compute scarcity against the active inventory. After the first placement, only disjoint tiles are eligible. The controller either spends a query on the reserve or continues with what it has.

Enumeration found one valid certificate, A followed by R in this search. Exactly one of the eight fixed generator, ranking and reserve choices found it. It focused on a, picked A by scarcity, then opened the reserve. Committing the legal C after A strands item e. This is a checked example of interaction. A single example can be solved by memorizing the choices.

The first implementation deliverable is a family generator with an independent exhaustive solver on small cases. Include ordinary cases that require no reserve, cases where the cheap rule wins, cases with several equally good certificates, and cases with no affordable solution. Vary the names, topology and placement of decoys. Check that the true optimum, best reachable result at the stated budget and information available to a policy are distinguishable. The policy never sees reserve contents before paying.

## D2. Find out whether the stack earns its cost

1. Build a strong coded search that chooses the least-supported uncovered item, ranks legal moves and can backtrack. Give it the same permitted operations as the learned variants. Compare equal numbers of candidate evaluations and separately compare under equal elapsed-time limits, including feature and model cost.
2. Train a generalist serving all three roles with the role and available observations in its input. Then train three specialists. Train later stages on the states produced by earlier policies on held-out training folds. This exposes downstream models to upstream mistakes before final evaluation.
3. Freeze all policies on development families. Run the eight combinations of coded or learned generator, ranker and budget controller on matched hidden inventories. Include the single generalist as a separate control. Report verified solutions before deadline, invalid claims, queries, backtracks and time to the first verified solution. Report shortfall from optimum only where an exact answer or valid bound is known.
4. Find the smallest case where the stack loses and replay it with one stage replaced at a time. Calculate how often the successful branch was absent, ranked too late, or abandoned early. If two cheaper stages perform as well as three, retain the simpler system. Advance a stack when it improves the declared objective against the strongest simpler control on fresh families and respects the verifier and budget.

## D3. Transfer to a harder planner

Use jobs that compete for multiple resources and have downstream dependencies. In a hand-checked example, X and Y each need five ticks on resource one. X later needs nine ticks on resource two and Y needs one. Starting X gives a best finish of 15 ticks. Starting Y gives 19. Swapping their downstream durations reverses the winning first move. A rule that looks ahead one resource solves this pair and belongs in the baseline suite.

Start with small instances whose legal schedules can be enumerated, then add fork-and-join dependencies, unavailable resources and uncertain job times one at a time. Reserve entire dependency patterns for final evaluation. The transfer test reuses the decision interfaces and training procedure with a new application adapter and newly trained weights. Compare against the best deterministic heuristic under the same budget.

## Follow the first measured limit

| If the experiment fails here | Next depth experiment |
| --- | --- |
| Indistinguishable states need different moves | Let the controller buy one diagnostic observation. Measure whether information gained justifies its cost on unseen cases. |
| Tied labels change which branches survive | Vary only tie priority, then train with credit for all equally good moves or a secondary cost. Check whether full-job results become more stable. |
| The model is confident on novel failure families | Generate legal variations that break the policy, shrink each failure to a small reproducible input, and test the correction on fresh families. |
| Correct results are known for too few cases | Allocate a fixed expert-review budget to the cases where policies disagree. Compare improvement per reviewed case with random selection on untouched workloads. |

## Calibration before expanding the model

**A ceiling with identical observations.** In two equally likely worlds the visible tiles are abc and ade. Before selecting either one, the solver can pay for a clue. The hidden last tile is def in one world and bcf in the other. The correct first move flips. Without a clue, even a randomized policy succeeds on half of trials in expectation. A clue with known symmetric 75% accuracy and cost 1/8 of a successful solve gives an optimal expected net score of 5/8. An independent random clue at the same price makes skipping it optimal. A single coded rule attains both optima when the clue reliability is declared. Use these controls to detect future information leaking into training and scoring.

**A proposed shortcut that cost more.** In a separate synthetic permutation search, six recurring short move sequences were mined from shortest solutions on boards of width five. On 128 width-seven starts both the primitive greedy control and the expanded search solved every case within 1,500 primitive transitions. Adding the moves at every explored node raised median work from 129 to 261.5 transitions. It helped 18 cases and cost more on 110. Each of five frequency-blind catalogs matched on length and multiplicity had a lower median cost than the mined catalog. This motivates a selective-use test where promising routines can be skipped and their internal moves are fully charged. The stacked application and larger solver trials remain proposed.
