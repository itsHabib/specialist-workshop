# Depth plan for models that steer deterministic search

**First application milestone.** Pick an existing event-driven application that already spends time evaluating candidates. Reconstruct what its current decision loop could observe and whether its output arrived in time. Test whether selectively doing more work improves the user's result per unit of compute. The exact-cover and scheduling studies below are small controls where the right answer can be checked exhaustively.

The workshop has tested candidate ranking, early stopping and regeneration. The scheduling experiments found cases where a fixed budget beat learned stopping. The next question is whether decisions at different stages can improve one another, and where several models compound mistakes. [Earlier steering results](DECISION-STEERING.md) · [Adoption plan](DECISION-ADOPTION.md)

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
