# Which question should a decision model ask next?

A batch arrives with eight date feeds and a budget of two reference lookups. Both April 7 and July 4 are valid parses of `04/07/2025`. The host can buy a trusted ISO date for one record and use it to resolve that feed. It has to decide **where a lookup is worth spending**, then decide again after seeing the answer. The objective is the number of dates returned correctly across the entire batch.

With two lookups on each of 64 correlated batches, the adaptive tree corrected **502 dates**, Google BERT-Tiny corrected **497**, and simply checking the largest ambiguous feeds corrected **420**. A small table of observed training frequencies matched the tree at 502. On a held-out regime where feeds changed independently, the tree corrected 427 and the size rule corrected 414. The advantage of predicting which feeds to check depended on the data-generating process.

## Try two batches before training anything

Run from the repository root with Python's standard library.

```sh
python3 -m experiments.date_probe.explain test-037
python3 -m experiments.date_probe.explain test-112
```

In `test-037`, the greedy policy spent the first lookup on a large feed from channel Beta. Its date hint checked out and zero rows changed. The second lookup on Alpha fixed six dates. Planning both lookups ahead changed the first choice to Alpha. Its hint was wrong, which made checking another Alpha feed attractive. That second lookup fixed five more dates. Greedy returned **26 of 35** dates correctly and lookahead returned **31 of 35**.

In `test-112`, following the first success with another lookup on the same channel fixed nothing. Greedy corrected **14** dates and lookahead corrected **8**. These cases were selected after evaluation to illustrate one win and one loss. Change `--budget` to 1 or 3, or select `--policy tree_fixed tree_adaptive bert_adaptive oracle` to inspect another comparison.

## What the models can see

Each channel carries a public format-change notice. Some feeds resolve themselves when a date such as `16/07/2025` makes one convention impossible. That exposes the correctness of the feed's hint for free. Other feeds need a paid lookup. Buying one reveals whether that feed's hint was right and exactly which dates can change. The host changes only the verified feed.

During fitting, each unresolved target contributes an example before any paid lookups and after up to two observable same-channel peer outcomes. Its own witness supplies the label. The state contains only the notice and a summary of peer evidence: unseen, confirmed hints, contradicted hints, or mixed. There were **1,834 labeled observations from 96 independent training batches**. They share episodes and are analyzed as grouped data. An additional 32 coupled batches were reserved for development. All model and policy choices were fixed before grading either evaluation partition.

The empirical method estimates probabilities from eight observable states using add-one smoothing. The tree and Google's pretrained BERT-Tiny receive the same labeled records. The BERT fit updated the encoder and decision head over 460 optimizer steps. The saved models score all eight possible states. The host ranks a feed by its estimated probability of a wrong hint multiplied by the number of date rows on which the two parses disagree. A high probability alone is insufficient if checking the feed could correct only one row. [Training code and task definition](learn.py) · [Checkpoint identities and all eight scores](evidence/fits.json)

## Three decisions to test

**1. Where should the first lookup go?** Compare largest-first with a score that accounts for both likely error and the number of affected rows. The fixed policies keep the evidence available before the first paid lookup. Adaptive policies update that evidence after each purchase.

**2. Should an answer change the next choice?** The training batches have correlated hints within each channel. The held-out batches include 64 with that structure and 64 where each feed's hint correctness is drawn independently conditional on its notice. The individual mistake probabilities are the same by design. The information one feed conveys about another changes.

**3. Is planning ahead worth the extra computation?** The greedy policy selects the largest expected immediate gain. The lookahead policy also evaluates the remaining choices under two hypothetical outcomes, weights those futures by the estimated probabilities, and selects the best expected total. It sees the actual witness only after committing a lookup. An independently checked three-feed example has a greedy first choice worth 3 expected dates and an exploratory choice worth 2.5 immediately. With two calls, their expected totals are 5.5 and 6 respectively. [Planner](policy.py) · [Executable control](../../tests/test_date_probe.py)

## Held-out results

Cells count **additional exact date rows beyond the native import** at two lookups per batch. Both families have 64 batches and every budgeted policy spent 128 lookups per family. Native import got 1,567 of 2,524 rows right in the coupled family and 1,576 of 2,545 in the independent family.

| Policy | Coupled channels | Independent channels |
| --- | ---: | ---: |
| Largest ambiguous feeds | 420 | 414 |
| Empirical rates, fixed | 486 | 418 |
| Empirical rates, updated after each lookup | 502 | 427 |
| Empirical rates with two-step lookahead | 507 | 421 |
| Tree, fixed | 486 | 418 |
| Tree, updated | 502 | 427 |
| BERT, fixed | 487 | 404 |
| BERT, updated | 497 | 430 |
| Best possible with the same lookup budget, scored in hindsight | 626 | 663 |

At two calls, updating the empirical order added 16 exact rows over the fixed order in coupled batches. Four batches improved, one lost, and 59 tied. The mean gain was 0.25 rows per batch, with a paired 95% bootstrap interval of −0.14 to 0.69. On independent batches the gain was nine rows, with five wins, six losses and 53 ties. The interval was −0.38 to 0.67 rows per batch. Both intervals include zero.

Explicit lookahead added five rows over greedy empirical selection on the coupled batches and lost six on the independent batches. One batch in each family accounted for the entire difference. At three calls, greedy empirical selection also recovered more dates than lookahead in both families. These results support starting with empirical risk × impact and measuring whether richer planning adds repeatable value. The full budget sweep retains remaining errors, unused opportunity and per-batch decisions. [All budgets and aggregate results](evidence/summary.json) · [Per-batch traces](evidence/cases.jsonl.gz)

## Audit or train

Check the archived run with standard Python. The auditor parses the dates independently, reconstructs the training-state frequencies, replays each policy from the saved score table and checks the paired results.

```sh
python3 -m experiments.date_probe.audit --run experiments/date_probe/evidence
```

To train fresh checkpoints, use a new output directory. The setup downloads the project's pinned Google base on first use. Training and dispatch run locally.

```sh
python3 -m venv .venv-decisions
. .venv-decisions/bin/activate
python -m pip install -r requirements-decisions.txt
python -m decision_workshop prepare-base --out .state/base
python -m experiments.date_probe.run --base .state/base --out .state/date-probe
python -m experiments.date_probe.audit --run .state/date-probe
python -m experiments.date_probe.explain test-037 --run .state/date-probe
```

The experiment uses one learned-model seed, two channels, a reliable lookup, equal lookup costs and a single format per feed. No elapsed-time benefit or suitability for a real importer was measured. [Protocol and budgets](PROTOCOL.md) · [Source and runtime fingerprints](evidence/freeze.json)

## Take the idea to another host

Pick a resource the application can actually spend, such as a verification call, a diagnostic query or a reviewer minute. Record what information is free, what each purchase reveals, which actions it permits and how much a wrong decision costs. For a first comparison, replay the existing order, a simple risk-by-impact rule and a learned ranker on the same held-out groups. Keep the original input, chosen inspection, observed result and final host output together. Test a second cohort where the relationship that made past inspections informative is weaker.
