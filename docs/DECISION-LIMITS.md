# How the decision models failed

Three local experiments tested numeric routing, support requests written in prose, and sequential release decisions. Google BERT-Tiny and the decision tree trained on synthetic records. The optional Clef-Flash comparison used a pinned 4-bit MLX conversion with its decision head and zero task-specific training. All results below are measurements on these tasks.

## Numeric routing

The input is an integer from 0 to 1023. The correct route is its equal-width interval. Options have arbitrary names. Each run used 32 or 128 labeled examples, 128 development examples and 128 test examples. The training budgets were nested, and the 2-, 4-, 8- and 16-action tasks shared input values. Two seeds changed the data and BERT initialization. Values were distinct across splits within a run. Each split was balanced by interval.

To qualify, a model had to answer at least 80% of development cases and get at least 95% of those answers right. The chosen cutoff maximized coverage among qualifying development cutoffs. Each row below gives the range across both training budgets and both seeds. The last two columns describe the default zero cutoff, including runs that did not qualify on development.

| Actions | Default tree passed on dev and test | BERT passed on dev and test | Tree answered / 128 test | BERT correct / 128 test |
| ---: | ---: | ---: | ---: | ---: |
| 2 | 4 of 4 | 0 of 4 | 128 | 107–119 |
| 4 | 4 of 4 | 0 of 4 | 128 | 32–63 |
| 8 | 0 of 4 | 0 of 4 | 80 | 16–39 |
| 16 | 0 of 4 | 0 of 4 | 38–40 | 8–15 |

At eight actions, the depth-five tree found useful boundaries but left several routes tied in a leaf. For example, with 128 training records and seed 17 it separated the first five routes one by one. The last leaf contained 16 examples apiece from the three remaining routes. Every test value that reached it went to review. BERT rendered the integer as text, scored each state/action pair and answered every test case at zero cutoff. None of its 16 fits had a development cutoff that met both targets.

**Changing depth recovered eight-way routing.** We tried maximum depths 5, 8, 12 and 16, selected depth and cutoff on development data, and evaluated on 256 additional values absent from every original split for that seed. With 128 training examples, depth eight got 250/256 and 245/256 correct with zero reviews. Depth sixteen got 245/256 and 241/256 on the sixteen-way task. Only the first cleared 95%. With 32 training examples there was no qualifying development configuration for either eight or sixteen routes. Two seeds establish the observed spread, not a universal action limit.

**A smaller change also worked.** When two training splits have exactly equal Gini scores, favor the more evenly sized children. This lets the tree separate several routes on each side of a boundary. We held the maximum depth at five, kept the same examples, and used a further 256 values per seed excluded from both previous evaluations and all training and development splits for that seed. With 128 training examples and eight actions, the new rule answered every case and got **254/256 and 251/256** right. With sixteen actions it got **242/256 and 247/256**, passing on one seed. With 32 training examples, neither tree qualified on development for eight or sixteen actions. The opt-in tree trainer flag is `--balanced-ties`.

## Decisions under uncertainty

In the release simulator, publishing a faulty release costs 30 units, rolling back a healthy one costs 10, and human review costs 3. A policy may first pay for up to two noisy diagnostics. We checked the grader against hand-calculated examples and against exhaustive enumeration of possible follow-up policies on all 744 generated observation prefixes. The grader minimizes expected cost using only the available information. Realized loss also depends on the hidden fault and the diagnostic outcomes drawn for each episode. [Simulator and hand calculations](DECISION-RELEASE-SPEC.md)

On 80 ordinary held-out episodes, final suggestions were graded against the best moves available in the state each policy reached. The episodes were independently generated. Of their 240 possible observation prefixes, 21 had input values also seen during training. All 48 shifted episodes used new diagnostic reliability values.

| Policy | Correct | Wrong | Review | Justified reviews | Harmful publishes | Mean realized cost | Mean excess expected cost¹ |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Expected-cost optimum | 43 | 0 | 37 | 37 | 1 | 2.65 | 0 |
| Rule that always checks once | 53 | 27 | 0 | 0 | 6 | 4.79 | 1.57 |
| Decision tree | 42 | 6 | 32 | 26 | 1 | 2.61 | 0.13 |
| BERT-Tiny | 18 | 62 | 0 | 0 | 0 | 5.25 | 2.78 |

¹ For each reached state, subtract the minimum expected cost from the chosen action's expected cost under optimal continuation. Sum these differences per episode and average across episodes. The tree incurred slightly less realized cost than the optimum on this draw while making choices with higher expected loss.

When diagnostics became less reliable, the optimum bought zero checks on 48 episodes. The rule bought 48 and published 12 faulty releases. The tree bought 11 checks, all unnecessary, and sent 36 cases to review. BERT bought none and rolled back all 48 releases. In a separate paired replay, the caller omitted the unique best initial action on 44 of the 80 ordinary episodes. The tree sent 64 to review across that entire cohort, compared with 32 when all actions were offered.

The cost-minimizing policy answered 43/80 ordinary cases and 13/48 shifted cases autonomously. A blanket requirement for 80% coverage conflicts with the economics specified for this task. Counting justified reviews separately makes that conflict visible.

**Clef-Flash received the same state, costs and horizon.** We froze a smaller comparison cohort before inference. On the first 16 ordinary test episodes, it issued 16 terminal suggestions, none optimal at the state reached, and published nine faulty releases. The tree issued eight correct suggestions, two wrong and six reviews on those exact episodes. On the first eight shifted episodes, Clef-Flash issued five wrong suggestions and three reviews. One concrete failure occurred at a stated 79% chance of fault. Publishing cost 23.7 in expectation and rolling back cost 2.1. Clef-Flash preferred publishing with a score of 0.419. The scores came from the 9B 4-bit model at zero cutoff.

## Text routing

Four actions were available. We held out entire phrasing families and paired each of the 24 clean test requests with three variants. At the default cutoff, BERT answered 14/24 clean requests correctly. Adding a resolved but irrelevant issue cut that to 7/24 and changed 15 preferred actions. Hiding the decisive clue left identical requests with four equally represented correct labels. BERT answered all of them, scored 6/24 and assigned its top choice an average probability of 73%. With the correct action omitted, the candidate guard caught 14 cases. The remaining 10 became wrong suggestions. The current tree uses structured fields only, so it saw tied class counts on this text-only task and reviewed all 24 in every condition.

On one phrasing family selected before inference, Clef-Flash got all four clean requests and all four distracted requests right. Its four decisions on identical hidden-clue inputs were all the same, yielding one correct and three wrong. It referred all four cases with the correct action omitted. This small paired comparison illustrates each perturbation in one family. [Text study and per-condition results](DECISION-TEXT-RESULTS.md)

We then trained BERT with 48 original requests plus 48 distractor variants. A control duplicated the clean rows to give it the same number of updates. On four additional held-out families with new distractors, the duplicated-clean control changed its answer on 6/16 paired requests. The augmented model changed on 1/16. Correct distracted answers rose from 7/16 to 9/16, and both models got 9/16 clean requests right. Stability improved more than correctness.

## Repeat the measurements

Set up the local maker and verify the Google checkpoint as described in the [maker guide](DECISION-MAKER.md). Use fresh output directories from the repository or extracted kit root.

```sh
python -m experiments.decision_limits.sweep --base .state/base --out .state/limits-sweep
python -m experiments.decision_limits.depth --out .state/limits-depth
python -m experiments.decision_limits.balanced_tree --out .state/limits-balanced
python -m experiments.decision_limits.text_routing --base .state/base --out .state/limits-text
python -m experiments.decision_limits.text_repair --base .state/base --prior .state/limits-text --out .state/limits-text-repair
python -m experiments.decision_limits.release_run generate --out .state/limits-release
python -m experiments.decision_limits.release_run run --base .state/base --out .state/limits-release
python -m experiments.decision_limits.release_report --out .state/limits-release
```

The numeric sweep saves each checkpoint, labeled input, raw score and cutoff curve. The release study saves each action taken, what was visible at the time, and the hidden outcome separately in the episode trace. The text study saves matched case IDs so a changed decision can be traced to the edited sentence.

For fresh reference inference on Apple Silicon, prepare the pinned MLX files and separate environment described in the maker guide, then run these commands from the same root.

```sh
python -m experiments.decision_limits.reference_release prepare --source .state/limits-release/task --packet .state/limits-release-clef-task
.state/clef-env/bin/python -m decision_workshop.clef_mlx_replay --task .state/limits-release-clef-task --model-dir .state/clef-flash --out .state/limits-release-clef-receipt.json
python -m experiments.decision_limits.reference_release score --packet .state/limits-release-clef-task --receipt .state/limits-release-clef-receipt.json --out .state/limits-release-clef-results.json
.state/clef-env/bin/python -m decision_workshop.clef_mlx_replay --task .state/limits-text/clef-task --model-dir .state/clef-flash --out .state/limits-text/clef-receipt.json
python -m decision_workshop.clef_reference --task .state/limits-text/clef-task --receipt .state/limits-text/clef-receipt.json --maker .state/limits-text/maker --out .state/limits-text/clef-comparison.json
```
