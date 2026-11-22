# How well does the support router handle new wording?

On 24 synthetic support requests written in six held-out styles, the fine-tuned BERT-Tiny routed 14 correctly. Appending one sentence about a resolved, unrelated issue dropped that to 7. It suggested a team on every request in both conditions. A report of 100% coverage alone would conceal the errors.

The task routes the current issue to billing, account access, delivery or software faults. The 48 training requests came from 12 phrasing families, each with one separately written issue per team. The eight development requests used two further families. Every clean text was unique. The input contained only the message. The structured-feature tree had a four-way tie on every case and sent all of them to review.

| BERT input condition | Correct | Wrong | Sent to review | Coverage | Accuracy among suggestions | Raw first choice correct | Mean highest score |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| New wording | 14 | 10 | 0 | 100% | 58% | 14/24 | 0.65 |
| Same request plus a resolved issue from a different team | 7 | 17 | 0 | 100% | 29% | 7/24 | 0.33 |
| Decisive clue removed | 6 | 18 | 0 | 100% | 25% | 6/24 | 0.73 |
| Same clean request, correct team removed from the offered list | 0 | 10 | 14 | 42% | 0% | 14/24 | 0.65 |

The unrelated sentence changed 15 of the 24 first choices. For example, the model routed a statement listing the same purchase twice to billing. Appending a note that an older password issue had been resolved changed its first choice to account access. The perturbation adds both text length and a competing topic, so their individual effects remain unknown.

For the hidden-clue condition, all four teams were equally represented behind the exact same message within each family. Any deterministic system assigning a single team can get at most one of those four right. The model's average highest score rose to 0.73 even as accuracy hit that 25% ceiling. For the missing-team condition, the model requested review in all 14 cases where it preferred the missing team. The other 10 wrong answers remained available and were suggested.

The CPU run used seed 17, 12 epochs, and the maker’s standard training recipe. Before fitting, the threshold rule required at least 80% coverage and 95% accuracy among suggestions on clean development cases. At cutoff zero BERT got 5 of 8 right. At 0.8 it issued four suggestions and only one was correct. No tested cutoff qualified, so the table reports the predeclared fallback of zero. Clean scores ranged from one to three correct among the four cases in each test family. These are small, authored samples from a single training seed.

## Reproduce

Use the decision trainer’s Python environment and prepared base from the repository root.

```sh
python -m experiments.decision_limits.text_routing --base .state/base --out .state/limits-text
```

The output directory must be new. It contains the frozen protocol, generated inputs, trained checkpoints, every per-action score and the summary table. The `clef-task` directory retains the same training and development rows and a preselected set of 16 matched test requests from the first family. The full test population contains 24 distinct clean cases, each replayed in four conditions. [Experiment source](../experiments/decision_limits/text_routing.py)

## Did training on resolved issues help?

In a second run we kept the original 48 training messages and gave each one an extra copy. The control repeated the clean message. The repair appended a resolved issue belonging to a different team. The two fits used identical record IDs, labels, seed, batches and 144 optimizer updates. The augmented messages used more tokens. We fixed the score threshold at zero and tested four newly written families with previously unseen distractor sentences.

| Fit | Clean correct / 16 | Distracted correct / 16 | Hidden correct / 16 | Preferences changed by distractor |
| --- | ---: | ---: | ---: | ---: |
| Original, 48 rows and 72 updates | 8 | 6 | 4 | 5/16 |
| Duplicate clean, 96 rows and 144 updates | 9 | 7 | 4 | 6/16 |
| Add a resolved issue, 96 rows and 144 updates | 9 | 9 | 4 | 1/16 |

All three models suggested an action on all 16 cases in every condition. The repair made the router less sensitive to the additional sentence on this set. Compared with duplicate-clean training, it corrected four distracted requests and newly misrouted two. The augmented model still misrouted seven of the 16 distracted requests. The experiment covers one seed, four test families and distractors in the same trailing-sentence position used in training.

After running the original experiment, reproduce the repair with `python -m experiments.decision_limits.text_repair --base .state/base`. It saves the original and modified datasets, recipe, per-case outputs and summary in `.state/limits-text-repair`. [Repair experiment source](../experiments/decision_limits/text_repair.py)
