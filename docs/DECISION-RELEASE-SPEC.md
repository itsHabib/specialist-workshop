# Choosing the next release action

This synthetic experiment asks whether a learned policy can decide when another noisy check is worth its cost. All rates and costs below are assumptions of the simulator.

## Episode and costs

One release is either faulty or healthy throughout an episode. Before any check, the caller knows the fault probability of its risk class. The feature is that prior probability, expressed as a whole percent. Publishing costs 30 units on a faulty release and zero on a healthy one. Rolling back costs 10 on a healthy release and zero on a faulty one. Human review ends the episode for 3 units and resolves the fault correctly. A diagnostic costs 1 or 2 units and reveals a positive or negative answer. At most two checks may be bought. At a given stated reliability q, each check reports the hidden condition correctly with probability q; checks are conditionally independent given the fault. A deterministic incumbent requests one check on every episode, then rolls back on positive or publishes on negative.

The visible state has five integer fields: prior fault percent, check accuracy percent, check cost, positive checks to date, and negative checks to date. The model never sees the hidden incident, future results, episode ID, split, or any posterior or optimal value. Every policy receives the same episode and pre-sampled sequence of results.

## Expected-cost answer key

Given the posterior fault probability p, publishing costs 30p in expectation, rollback costs 10(1-p), and human review costs 3. A positive check has probability w = pq + (1-p)(1-q) and produces posterior pq/w. A negative check produces posterior p(1-q)/(1-w). With another check available, its expected cost is its price plus the weighted minimum achievable cost in the two child states. Compute backward from zero checks remaining. Initially all four moves are available. After two checks, only the three terminal moves are available.

Every move within 1e-9 of the minimum is correct. For a single training label, ties prefer publish, rollback, review, then diagnose. The simulator also scores expected regret and realized cost, so equal-value choices receive equal decision credit.

## Hand-calculated checks

| Prior and history | Reliability | Checks left | Result at diagnostic price 1 |
| --- | --- | --- | --- |
| 25% faulty, no check yet | 90% | 1 | Publish and rollback each cost 7.5; review costs 3. After a positive (probability .30), roll back for expected loss 2.5. After a negative, publish for expected loss 1.071429. Diagnostic costs 1 + .30 × 2.5 + .70 × 1.071429 = 2.5, so buy it. |
| Same prior and history | 60% | 1 | The positive and negative posteriors are 1/3 and 2/11. Both favor review. Diagnostic costs 4 against immediate review at 3. |
| 25% initially, one positive | 90% | 1 | Posterior .75. Immediate rollback costs 2.5. A second positive occurs with probability .70, then rollback costs 5/14. A second negative restores the .25 prior, then review costs 3. A repeat check costs 1 + .70 × 5/14 + .30 × 3 = 2.15, so buy it. |
| Same prior, one negative | 90% | 1 | Posterior 1/28. Publishing costs 15/14. A second check costs about 1.493. Publish. |
| 10% initially, two opposite checks | Any stated reliability | 0 | Opposite checks cancel. Publish and review tie at 3; rollback costs 9. |

The one-check horizon in the first two hand examples tests the recursion directly. The benchmark starts every episode with two checks available.

## Frozen trial

Use independent pseudorandom streams keyed by release family and index. Draw prior percentages uniformly from integers 4–94, diagnostic prices uniformly from 1 and 2, and reliability uniformly from 72%, 84%, and 93%. The separate shifted test samples reliability from 54%, 62%, and 68%; its lower reliability is explicitly visible to the policy. Generate 96 training episodes, 24 development episodes, 80 ordinary test episodes and 48 shifted test episodes. Draw the fault once per episode, and pre-sample both diagnostic outcomes. Training exposes the three feasible observation prefixes of each training episode for supervision; related prefix rows always share one split. A simulator run scores each policy on only the prefix states it actually visits.

Run one additional paired caller-failure replay on ordinary test episodes. Hide the initial unique optimum when it is publish, rollback, or diagnose. Review remains available. Later states retain all ordinarily legal actions. Compute the oracle for the restricted action set and report how often the unrestricted optimum was omitted. This uses the same 80 episodes; it is not a separate independent sample.

Train the workbench BERT and decision tree with one fixed recipe. BERT uses seed 17, four epochs, batch size 16 and AdamW at 0.0003. The tree uses the maker defaults. Freeze this before inspecting numerical results. A policy with an unavailable preference, tie, or unresolved recommendation ends in human review, incurring its actual cost. Report dev, ordinary test, shifted test and paired caller-failure separately. For each report total and mean realized loss, harmful publishes, unnecessary healthy rollbacks, diagnostic count, diagnostics with positive expected regret, human reviews, correct decisions by step, and mean expected regret per episode. Terminal correctness counts the selected terminal move among the expected-cost optima at its observed state, with review split out so accuracy among suggestions and coverage are visible. Full episode traces retain the private latent outcome only in the simulator analysis output.

## Measured replay

One Google BERT-Tiny run updated 4,369,536 parameters in 72 steps on 288 prefix rows. Training cross-entropy decreased from 1.38725 to 1.30547. Its first move on every ordinary held-out episode was rollback. The training prefixes contained 105 rollback, 89 review, 62 publish and 32 diagnose labels. The decision tree used the same episodes, labels and visible fields.

Coverage is the fraction of episodes ending in a publish or rollback suggestion. Accuracy is how often that terminal suggestion minimizes expected cost given the evidence at the time. A policy can make an unnecessary intermediate check and still make an optimal terminal suggestion. Total cost and per-step optimality capture that difference.

| Held-out episodes | Policy | Coverage | Accuracy among suggestions | Mean realized cost | Mean expected regret | Harmful publishes | Unnecessary / total diagnostics | Reviews |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Ordinary, 80 | Oracle | 53.8% | 100% | 2.650 | 0 | 1 | 0 / 29 | 37 |
| | One-check incumbent | 100% | 66.3% | 4.788 | 1.572 | 6 | 56 / 80 | 0 |
| | Tree | 60.0% | 87.5% | 2.613 | 0.127 | 1 | 14 / 33 | 32 |
| | BERT | 100% | 22.5% | 5.250 | 2.777 | 0 | 0 / 0 | 0 |
| Lower reliability, 48 | Oracle | 27.1% | 100% | 2.396 | 0 | 0 | 0 / 0 | 35 |
| | One-check incumbent | 100% | 27.1% | 11.458 | 5.745 | 12 | 48 / 48 | 0 |
| | Tree | 25.0% | 91.7% | 2.688 | 0.236 | 0 | 11 / 11 | 36 |
| | BERT | 100% | 22.9% | 4.375 | 2.429 | 0 | 0 / 0 | 0 |

On ordinary episodes the tree reduced mean realized cost by 2.175 against the incumbent. A paired, 2,000-resample episode bootstrap gave a 95% percentile interval of [0.313, 4.238] units saved. BERT increased the mean by 0.463; its interval for *extra* cost was [−1.863, 2.738]. Independent episodes can produce identical visible states: 21 of the 240 ordinary test prefixes matched a training state exactly. None of the 144 lower-reliability test prefixes matched. Intervals measure the uncertainty across these episodes at one training seed.

The tree made an optimal first decision on 61/80 ordinary episodes, then 25/32 and 1/1 on the later decisions it reached. The incumbent scored 24/80 then 53/80. BERT scored 18/80 and never requested another observation. Of the tree's 32 ordinary reviews, 26 minimized expected cost. Under the shift, the tree made 11 unnecessary checks while the oracle bought none. Lower realized cost than the oracle on one sample is possible because the oracle minimizes expected loss before observing the hidden condition. Every oracle choice has zero expected regret by construction.

Removing a unique best initial option changed 44 of the same 80 ordinary episodes. Tree coverage fell from 60% to 20% and accepted accuracy from 87.5% to 81.25%. Its 64 reviews included 35 automatic deferrals and 51 cases in which review was optimal among the remaining moves; those categories overlap. The oracle itself reviewed 64 episodes under the caller restriction. A blanket 80% coverage target conflicts with frequent human review being the cost-optimal action here.

On the two-check hand pairs, the tree switched from diagnose to review when reliability fell from 90% to 60%. At 90% reliability it deferred on the repeated-positive-check decision and correctly published after one negative check. BERT chose rollback in all four cases. These results describe one small supervised training run and this explicit cost model.

## Reproduce

Run from the repository root with the decision-maker dependencies installed and the verified Google base in a local directory. Choose a new output directory; the run keeps both checkpoints, per-episode traces and summary counts there.

```sh
python -m experiments.decision_limits.release_run generate --out .state/limits-release
python -m experiments.decision_limits.release_run run --out .state/limits-release --base .state/base
python -m experiments.decision_limits.release_report --out .state/limits-release
python -m unittest discover -s tests/decisions -p test_release_limits.py
```

For the original experiment’s retained checkpoints, add `--min-score 0` to `release_report` to reproduce the hand checks. New runs record their cutoff.

The task JSONL contains visible features and supervised targets. Only the offline episode traces contain realized fault status. The scoring interface accepts a policy called with a row ID, visible state and legal candidates. For an externally scored model, return an action plus its original preference and use `rollout(episode, policy)` and `summarize(traces)` from `experiments.decision_limits.release`. The test processes are reproducible with `episodes("iid")` and `episodes("shift")`.
