# Steering deterministic programs

The measurements below were recorded before the scoring and Python scope fixes, with results available at commit `c7fd312`.


A program supplies its current candidates and a numeric feature vector for each. The ranker scores the candidates and returns their handles in priority order. The program owns legality, execution, success checks and its search budget. This interface can rank candidates that have never appeared in training.

## Use the ranker from code

```python
from pathlib import Path
from decision_workshop.core import load_model
from decision_workshop.steering import fit_rankers, rank_candidates

# Illustrative labels. In a benchmark, derive these from training outcomes.
examples = [
    {"group": "batch-1", "features": {"duration": 2, "due": 3}, "preferred": True},
    {"group": "batch-1", "features": {"duration": 8, "due": 12}, "preferred": False},
    {"group": "batch-2", "features": {"duration": 3, "due": 4}, "preferred": True},
    {"group": "batch-2", "features": {"duration": 7, "due": 14}, "preferred": False},
]
fit_rankers("scheduling", examples, Path(".state/rankers"), Path(".state/base"))
tree = load_model(".state/rankers/tree.json")
ranked = rank_candidates(tree, [
    ("new-job-A", {"duration": 3, "due": 5}),
    ("new-job-B", {"duration": 7, "due": 12}),
])
next_job = ranked[0][0] if ranked else None
```

Supply only train-split examples to `fit_rankers`. Keep an entire queue, patch or other related group in one split. Numeric features must be finite and available at decision time. The software can compute candidate features relative to the other choices, such as slack compared with the rest of the queue. Each task trains its own decision tree and Google BERT-Tiny checkpoint. Each score estimates preference for that individual candidate on the training task. Scores across the offered candidates need not sum to one.

Renaming a candidate cannot change its features or score. On equal scores the ranker orders by the sorted feature vector. Identical vectors retain their incoming order with equal scores. If an executor needs to distinguish two such candidates it must supply a meaningful feature or make its own tie decision.

A deployed ranking policy can freeze native order for every score tie. Save the checkpoint with `save_policy(..., kind="ranking", adapter_revision=..., fallback_revision=...)`, then load the policy with the revisions and digest expected by the host. Pass the host's collected order to `policy.rank`. The result is a list of identifier and preference-score pairs. Capture the first items the executor actually starts so a bypass or fallback remains visible. [Deployment example](DECISION-MAKER.md#ship-a-trained-tree-with-its-decision-settings)

The Click example uses the host's pytest collection hook. Here `prior_failures` contains a count for every previously observed node, including zero. The host supplies changed filenames from the tested diff and gets the policy digest from its deployment configuration.

```python
from experiments.steering.pytest_history import order_tests

def pytest_collection_modifyitems(items):
    decision = order_tests(items, bundle, policy_sha256=deployed_pin,
                           changed_paths=changed_paths, prior_failures=prior_failures)
    record_collection_decision(decision)

def pytest_runtest_logstart(nodeid, location):
    record_actual_start(nodeid)
```

The two recording functions belong to the application. Comparing their records distinguishes an executed reordering, a fallback to native order and a caller that silently ignores a valid decision. [Pytest adapter](../experiments/steering/pytest_history.py)

## What counts as improvement

- **Candidate ordering.** Hold the executor and budget fixed. Compare the resulting solution or first detected regression against a deterministic control selected on development cases. Charge for scoring as well as execution. Report shifted workloads separately.
- **Strategy choice.** Let a tree choose among existing deterministic scheduling rules using the visible queue. Grade the schedule it actually produces.
- **Stopping.** Stop after 1, 5 or 20 completed attempts. Before running the comparison we priced each attempt at 0.5 weighted-tardiness units. Compare fixed budgets and a deterministic stopping rule with a learned policy that sees only the observed cost prefix. Report actual attempts and host latency separately from priced loss.
- **Candidate regeneration.** Initially allow only the two shortest jobs as first moves. After five attempts a controller may pay two weighted-tardiness units to request a broader first-action list. All controllers receive 20 attempts. Re-examining a schedule consumes an attempt. Compare with always expanding, never expanding and expanding when the possible gain exceeds the fee. Evaluate the opportunity to expand using paired executions of both paths.

The two prices are synthetic experimental assumptions. They express cost preferences independently of the machine's timing. Raw reports include the information needed to change the comparison for other prices, with a new development selection and fresh validation.

## Choosing the rule and search budget

We first compared candidate orderings through the same deterministic search at three budgets. Regret is the selected schedule's weighted tardiness minus the independently checked optimum for that queue. These are means over 24 untouched queues. The times are local wall time, including ranking and execution with already-loaded models.

| Ordering | Regret at 1 attempt | Regret at 5 | Regret at 20 | Time at 20 |
| --- | ---: | ---: | ---: | ---: |
| Earliest deadline first (development winner on normalized regret) | 43.92 | 31.62 | 19.17 | 0.19 ms |
| Apparent tardiness cost heuristic (ATC) | 12.54 | 4.79 | 1.29 | 0.20 ms |
| Trained tree | 22.96 | 10.67 | 6.83 | 0.76 ms |
| Google BERT-Tiny | 43.17 | 27.42 | 16.21 | 24.83 ms |

The tree found a lower-cost schedule in five attempts than the development-selected rule found in twenty, but those five attempts took about 0.43 ms. ATC gave better mean raw regret and lower latency on this cohort. Which control looks strongest depends on the queue distribution and whether the evaluation weights absolute or normalized loss. [Scheduling protocol and all controls](STEERING-SCHEDULING.md)

**A wrong cost estimate can defeat more search.** With noisy durations, increasing the ATC budget from five to twenty raised mean realized regret from 33.21 to 42.21 on the original 24 queues. We tested a correction on 64 fresh queues. For each visited schedule we averaged its predicted cost over 32 independent duration scenarios and kept the lowest. The scenarios were shared across candidate schedules and sampled independently of the real outcome. With matched noise and 20 attempts, this reduced realized cost by 3.53 units on average, but the paired 95% bootstrap interval for the cost difference ranged from −10.41 to +1.64. With exact durations it increased cost by 6.08, and with faster-than-expected jobs by 3.05. For this cohort, paying for a more elaborate evaluator did not establish a general win.

The scheduler minimizes weighted tardiness on a single worker. For every completed job, charge its weight times the number of time units it finished after its due time. A hand check uses two jobs that both take two units. A has weight three and deadline two, and B has weight one and deadline three. Running A first costs 1, while B first costs 6. Moving only A's deadline to four flips the best first move to B, with costs 1 and 0. All six policies initially chose A. After the deadline moved, BERT and earliest-deadline-first switched to B, while the tree and ATC kept A. An independent grader enumerated small queues and checked the scheduler's optimal continuation costs.

For adaptive control we generated 96 training queues, 40 development queues, 64 untouched queues of four to eight jobs and 32 larger queues of nine to ten. At each stopping opportunity the deterministic control calculates a lower bound by allowing every job to start immediately. If even reaching that bound cannot pay for the attempts until the next stopping point, it stops. The learned controller can also stop based on the queue and the cost history it has observed. A separate tree can select the deterministic ordering rule. Here the development winner was the apparent tardiness cost heuristic (ATC) with a fixed budget of five.

| Policy on the 64 untouched queues | Mean attempts | Mean priced loss, lower is better |
| --- | ---: | ---: |
| Development winner, ATC with five attempts | 5.00 | 49.98 |
| ATC with twenty attempts | 20.00 | 54.50 |
| ATC with certified early stopping | 16.42 | 52.79 |
| ATC with learned early stopping | 4.92 | 51.80 |
| Learned rule choice with twenty attempts | 20.00 | 60.75 |
| Learned rule choice and learned stopping | 11.36 | 57.60 |

Learning when to stop saved attempts against a twenty-attempt budget. A fixed five-attempt budget was cheaper overall. On larger queues the learned stopper averaged 5.88 attempts and 91.75 priced loss, compared with 90.44 for the development winner. This comparison executes the stopping callback during the search. Unvisited schedules incur neither an evaluation nor an attempt charge.

## Recovering candidates

The first-action shortlist initially included only the two shortest jobs. The scheduler could still place any other job later. A broader deterministic generator could restore all first moves after attempt five. On 64 fresh ordinary queues, paying for expansion was worthwhile on 18. The trained controller requested it 14 times and found four of those 18 opportunities. Mean priced loss was 63.78 versus 64.39 for never expanding, the development-selected control, and 63.09 for always expanding. It improved four queues, tied on fifty, and lost on ten against never expanding. On larger queues it caught one of five opportunities. This is weak evidence for deploying a learned recovery decision.

On 60 of the 64 ordinary queues, broadening the search revisited at least one previously evaluated schedule. The measured policy charged every repeated evaluation to the budget. Persisting a cross-pass cache is a concrete follow-up to the regeneration experiment.

## Ranking executable regression tests

For each patch, the second deterministic program could run up to 28 actual regression tests. Six synthetic utility families were reserved for final evaluation. Each received two faulty edits and one behavior-preserving edit, giving 12 reachable faults in 18 patches. Training and development used different changed utility families. Inputs available to the ranker included pre-change line coverage, assertion precision and estimated fixture cost. The scorer executed the ranked tests until one failed or the budget was exhausted.

[Test-selection protocol, rare faults and timing method](STEERING-TEST-SELECTION.md)

| Execution order | Faults detected at 1 test | At 5 tests | At 20 tests |
| --- | ---: | ---: | ---: |
| Discovery order | 5 | 11 | 12 |
| Line coverage and assertion precision | 12 | 12 | 12 |
| Learned tree | 5 | 10 | 12 |
| Google BERT-Tiny | 7 | 12 | 12 |

At a five-test budget, the coverage rule and BERT both caught all 12 faults. BERT spent 38.8 modeled execution units per patch compared with 26.2 for the rule. Mean ranking CPU was about 18.29 ms for BERT and 0.018 ms for the rule. The real synthetic assertions ran in microseconds. These CPU measurements use repeated batches to resolve such short operations. They exclude the initial collection of shared coverage features and model training. Removing all failing tests left zero detections on the same twelve faulty patches. A ranker can only order the tests the program actually supplies.

**When 28 tests look identical to the ranker, change what it can observe.** We generated a new utility with rare boundary faults. Each faulty patch had exactly one failing test and every test covered the changed line with one precise assertion. In the first four faulty revisions all three informed policies were tied with discovery order, detecting one fault at five tests. On four further offsets with both directions of error, the frozen rule, tree and BERT each detected zero of eight faults within five tests. A deterministic rule then probed the *unchanged* implementation at each test input's two neighbors. Putting tests next to a pre-existing output transition first caught four of eight with one test and all eight within five tests, without consulting the mutant. This required 56 extra clean calls per 28-test suite, averaging about 3.2 µs CPU on this fixture. The construction deliberately puts faults at output transitions. It shows the value of that particular observable for this fault family.

## Reproduce

Set up the CPU trainer and local Google checkpoint using the [maker guide](DECISION-MAKER.md). Run from the repository or kit root with fresh output directories.

```sh
python -m experiments.steering.scheduling --base .state/base --out .state/steering-schedule
python -m experiments.steering.search_control --out .state/steering-control
python -m experiments.steering.regenerate --out .state/steering-regeneration
python -m experiments.steering.test_selection --base .state/base --out .state/steering-tests
python -m experiments.steering.rare_tests --prior .state/steering-tests --out .state/steering-rare
python -m experiments.steering.noisy_scoring --out .state/steering-noisy-scoring
```

Each steering study keeps the generated inputs or their generator parameters, the final decisions, and the measured outcomes locally. The transfer kit includes recorded synthetic measurements. Use an ignored directory or an external task directory for your own records.
