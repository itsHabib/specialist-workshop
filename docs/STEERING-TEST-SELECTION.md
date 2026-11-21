# Which regression test should run first?

A hand-written rule detected all 12 faulty patches with its first test in the initial study. Six further patches preserved behavior, so the result across all patches was 12/18. The learned rankers caught up with larger budgets. A second study made a single test decisive while giving every test identical coverage features. There the same rule lost its advantage.

## New utility families

Each synthetic software change affected a short integer utility. Each suite contained 28 executable checks. Some compared exact values, others compared buckets of two or four possible values. The checks captured outputs from the original implementation, which was then replaced by either of two faulty implementations or a behavior-preserving edit. Every check ran against the edited program to establish the exhaustive answer key. Executing the ranked prefix stopped at the first failed assertion. The oracle and the test bodies stayed outside the ranker interface.

Six utility families supplied 504 training candidates of which 137 failed. The three development families selected the hand-written rule before evaluation on six further families. The winning rule first prefers tests that previously executed the edited line, then more precise assertions, more distinct inputs at that line, and lower declared test cost. Two competing rules explicitly favored coverage per unit cost, one also prioritizing boundary inputs. Both had fewer development detections.

| Frozen ordering | Faults detected in 1 test | In 5 tests | In 20 tests | Mean modeled units to stop at 5 | Ranking CPU per patch |
| --- | ---: | ---: | ---: | ---: | ---: |
| Supplied discovery order | 5 | 11 | 12 | 33.9 | 0.0003 ms |
| Selected coverage rule | 12 | 12 | 12 | 26.2 | 0.0175 ms |
| Decision tree | 5 | 10 | 12 | 31.7 | 0.2069 ms |
| BERT-Tiny | 7 | 12 | 12 | 38.8 | 18.2898 ms |

All columns include the six equivalent edits. Their tests ran to the budget without failing. Mean measured CPU to execute the selected prefix at budget five was 0.84, 0.64, 0.73 and 0.83 microseconds respectively. Ranking CPU was measured separately. Each per-patch timing used the median of three batch means, with repeated real executions consuming at least 0.5 milliseconds of process CPU per batch. Modeled units represent a declared fixture cost. These tiny, in-memory assertions do not reproduce CI wall time, setup, or parallel execution.

Opaque IDs were new for every held-out patch. Renaming IDs changed no detection totals. Shuffling the suite reduced discovery-order first-test detection from five patches to two. The three feature-based orderings retained their totals. In a separate paired condition we removed every test that actually failed from each of the 12 faulty suites. Every policy then detected 0/12 at every budget.

## One revealing test among 28

Four new phase counters had starting offsets of 26, 19, 15 and 4 with a period of 32. Each test checked an independently computed count of scheduled events for one input from 3 through 30. Adding one to the counter numerator produced exactly one regression per offset. Every candidate hit the same edited line, used an exact assertion, and had identical values for every feature visible to the frozen rankers. We paired each faulty edit with an equivalent edit.

All four original orderings retained caller order within the tie. They detected 1, 1 and 4 of the four reachable faults at budgets 1, 5 and 20 on the supplied queues, across eight total patches. Renaming ID strings had no effect. Which fault an ordering finds then depends on where the caller placed that one test.

On four further offsets (24, 20, 13, 7) we tried a new feature. Before running the patch, we checked whether the clean function's result changed at either neighboring input. A deterministic rule put both tests bordering a transition first, retaining queue order within ties. We independently tested both positive and negative numerator shifts on the same queue. The selector saw neither the direction nor the answer key.

| On the same 12 patches, including four equivalent edits | In 1 test | In 5 tests | In 20 tests |
| --- | ---: | ---: | ---: |
| Any frozen original ordering | 0 | 0 | 5 |
| Prefer a clean-output transition | 4 | 8 | 8 |

There were eight reachable faults. Each shift was exposed by a different one of the two neighboring inputs. The probe cost 56 additional clean function calls per offset, reusable across all three edits, at a measured mean of 3.18 microseconds of host CPU per offset. Removing the sole failing test again produced 0/8 detections at every cap. This feature is particularly well matched to off-by-one mutation semantics. The experiment establishes a way to recover information on these counters. Other defect classes, wider input domains, expensive clean probes and real CI queues require separate measurements.

## Reproduce

Run from the repository root with the decision dependencies and verified base.

```sh
python -m experiments.steering.test_selection --base .state/base --out .state/steering-tests
python -m experiments.steering.rare_tests --prior .state/steering-tests --out .state/steering-rare
python -m unittest discover -s tests/decisions -p test_steering_test_selection.py
```

The run directories contain the preregistered design, model hashes, exhaustive per-patch test outcomes, ranked prefixes, first failing IDs, measured CPU and modeled cost. The second command reloads the original checkpoints. [Main experiment](../experiments/steering/test_selection.py) · [Rare and clean-probe experiments](../experiments/steering/rare_tests.py)
