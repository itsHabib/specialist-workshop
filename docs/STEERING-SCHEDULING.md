# Preference-guided scheduling

A deterministic program searches complete job orders on one worker. All jobs are available at time zero, run once, and cannot be interrupted. The cost of a schedule is the sum of each job's weight times the amount by which its completion time exceeds its deadline. Finishing exactly on a deadline has zero penalty.

## Frozen comparison

The queue generator uses a SHA-256-derived pseudorandom stream for each family and index. Training uses 64 separate queues, with one root and one randomly chosen nonterminal prefix per queue. An exact dynamic program labels every eligible next job whose choice permits an optimal completion as preferred. The model receives estimated duration, weight, time until due, slack, remaining work, remaining job count, relative due order and the immediate penalty of choosing this job. Identifiers, hidden durations, and the downstream optimum are absent from these features. The same training groups fit the tree and Google BERT-Tiny using the shared training recipe.

Compare shortest processing time, earliest due date, weighted shortest processing time, and apparent tardiness cost. The last prioritizes weight / duration × exp(−max(0, slack) / (2 × mean remaining duration)). Select the strongest deterministic control on 16 development queues by mean normalized regret across the three budgets. Scale by max(1, optimal cost) per queue, which also counts mistakes on zero-optimum queues. Break exact ties by the published control order.

The primary held-out comparison uses the first 24 fresh queues with four through eight jobs. Also run 12 queues with nine or ten jobs, the same 24 primary queues with realized durations multiplied by an unobserved factor of 0.5, 0.75, 1, 1.5 or 2 (rounded, with a one-unit floor), and the primary queues with all optimal first jobs withheld when an alternative exists. The last condition removes only first-position choices. Every job still runs. The larger queues, noise, and branch removal are separate tests. Paired primary and noisy queues share observations; compare them as matched cases.

The common search starts with the greedy complete order and enumerates alternatives in order of total rank displacement, breaking ties by trying deviations at earlier positions first. Candidate scores are computed only at visited job subsets. Within a search, identical remaining subsets reuse their ranking. At caps of 1, 5 and 20 attempted orders, the program selects the complete order with lowest *estimated* cost, retaining the first in a tie. It never consults realized durations to choose an order. When fewer legal orders exist than the cap, the search exhausts them. The raw traces retain the hindsight-best realized cost among visited orders separately. Timings start before constructing the first move and include policy inference and all attempted schedules. They exclude one-time training and checkpoint loading.

The independent grader exhaustively enumerates small queues. On jobs A(2 time, deadline 2, weight 3) and B(2 time, deadline 3, weight 1), A then B costs 1 and B then A costs 6. Changing only A’s deadline to 4 makes B then A cost 0, with A then B still costing 1.

Run from the repository root:

```sh
python -m experiments.steering.scheduling --out .state/steering-schedule --base .state/base
python -m unittest discover -s tests/decisions -p test_steering_scheduling.py
```

The output contains the learned checkpoints, training examples, original jobs, every attempted ordering, the selected estimated and realized costs, cumulative elapsed times, per-queue exact optima and per-budget summaries. Raw group IDs identify matched noisy and omitted variants. Every job triple (duration, due, weight) is distinct within a generated queue. When inputs have identical features, swapping such jobs is an equivalent choice in the visible problem.

## Measured results

The rankers trained on 612 candidate examples from 64 queues. The development control was earliest due date (EDD). On the 24 fresh ordinary queues, apparent tardiness cost (ATC) actually had lower mean loss than EDD and both learned orderings.

| Ordinary queues | First complete order | Best after 5 attempts | Best after 20 attempts | Exact optima at 20 | Mean wall time at 20 |
| --- | ---: | ---: | ---: | ---: | ---: |
| EDD | 100.33 | 88.04 | 75.58 | 14/24 | 0.19 ms |
| Tree | 79.38 | 67.08 | 63.25 | 15/24 | 0.76 ms |
| BERT | 99.58 | 83.83 | 72.63 | 14/24 | 24.83 ms |
| ATC | 68.96 | 61.21 | 57.71 | 18/24 | 0.20 ms |

The tree beat EDD on eight queues, tied on thirteen and lost on three after 20 attempts. Against ATC the counts reversed: three wins, thirteen ties, eight losses. EDD's development result used normalized regret across budgets; the held-out cost column gives unnormalized mean loss. With 9–10 jobs, ATC cost 79.92, tree 98.08, EDD 113.75 and BERT 114.42 at 20 attempts. On four fixed queues we renamed every job and reversed caller order; all 480 checked trial orderings, selected attempts and both cost calculations were preserved across the four rules and both learned models.

Every omitted-first cohort queue lost all of its globally optimal first branches. Even an exact legal solver then had a mean excess cost of 12.13. At cap 20 tree's excess above the unrestricted optimum was 16.21, and ATC's was 15.71. Five of the smallest restricted queues exhausted their legal permutations before spending all 20 attempts.

When execution durations diverged from estimates, more search could hurt. ATC's realized mean loss rose from 167.67 at five candidates to 176.67 at twenty, even though the scheduler always retained the lowest estimated loss. Weighted shortest processing time reached 172.00 at twenty. These are paired outcomes from the same schedules the policies considered in the ordinary regime, with the true durations hidden from the selector.

## Can sampled duration uncertainty repair the selection?

A follow-up fixed 64 new queues before inspecting their outcomes. It kept ATC's 20 visited schedules exactly the same. The robust selector ranks candidates by mean loss under 32 common sampled duration profiles, choosing at both 5 and 20 attempts. The actual matched-noise, exact-duration, and deliberately faster-duration worlds are generated after both selectors have chosen, using separate seeds. All scenario draws, candidate costs, choices and latencies are saved for replay.

| 64 queues | Attempts | Nominal mean cost | Robust mean cost | Robust minus nominal, paired 95% interval |
| --- | ---: | ---: | ---: | ---: |
| Matched noise | 5 | 71.94 | 71.98 | +0.05 [−3.84, +3.94] |
| Matched noise | 20 | 74.92 | 71.39 | −3.53 [−10.41, +1.64] |
| Exact durations | 20 | 42.17 | 48.25 | +6.08 [+4.38, +7.97] |
| Faster durations | 20 | 11.88 | 14.92 | +3.05 [+1.59, +4.61] |

At 20 attempts the robust selector changed 48 of 64 schedules. Under matched noise it won 22 times, tied 23 and lost 19. Its selector cost averaged 0.276 ms, including scenario construction, versus 0.013 ms for nominal selection, in addition to their common ATC search. The matched-noise improvement remains uncertain. Simulating too much uncertainty measurably worsened both control regimes. Intervals are paired percentile bootstraps with 2,000 whole-queue draws; they describe this synthetic generator.

```sh
python -m experiments.steering.noisy_scoring --out .state/steering-noisy-scoring
```
