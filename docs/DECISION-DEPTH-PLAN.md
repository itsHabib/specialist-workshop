# Depth plan for models that steer deterministic search

**First experiment.** Test a team of three small policies on one budgeted search. One chooses which deterministic generator to call. Another ranks the legal candidates that generator produces. The third decides whether to spend the remaining budget exploring another branch, requesting more candidates, or returning the best verified result. Train each role from scratch with local data. The search program owns the constraints, executes every move and checks the final answer.

The workshop has tested candidate ranking, early stopping and regeneration. The scheduling experiments found cases where a fixed budget beat learned stopping. This experiment asks whether decisions at different stages can improve one another, and where training several models simply compounds mistakes. [Earlier steering results](DECISION-STEERING.md) · [Adoption plan](DECISION-ADOPTION.md)

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

The stack, larger solver and transfer trials are proposed. The two tiny witnesses were enumerated during planning. Neither provides evidence of a learned advantage.
