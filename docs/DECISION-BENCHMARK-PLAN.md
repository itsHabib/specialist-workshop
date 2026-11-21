# Decision benchmark plan

## Score a decision

Give every case exactly one outcome. A correct suggestion agrees with the independent answer key. A wrong suggestion disagrees. A review means the system defers to a person without issuing a final suggestion. Preserve the reason for review and the model's first preference so an omitted option can be diagnosed separately.

- Coverage = (correct + wrong) / all cases.
- Accuracy among suggestions = correct / (correct + wrong). Leave this undefined if the system makes zero suggestions.

A provisional target is at least 80% coverage and at least 95% accuracy among suggestions. On 100 cases, 76 correct, 4 wrong and 20 reviews meet both thresholds exactly. Select any confidence cutoff on development cases, freeze it, then report all three counts on unseen groups. Show uncertainty and how the counts change across cutoffs. An acceptable error budget depends on the consequences of the decision.

## First harder task

Build a synthetic release simulator with four moves: publish, roll back, run one diagnostic, or request human review. Specify fault frequencies, observable clues, diagnostic reliability and action costs before generating cases. Choose the answer key by calculating expected cost from the visible evidence and available moves. Keep the hidden fault available only to the simulator. Score equally good moves fairly.

Start with paired cases where the diagnostic would change the best final action in one case and add cost without changing it in the other. A diagnostic is an intermediate action. Score the eventual suggestion or review once per episode and separately count harmful releases, unnecessary checks and total cost. Compare the rule-based policy, tree and Google BERT-Tiny on identical cases. Clef-Flash can serve as a separate reference.

## Find the limits

1. Sweep 2, 4, 8 and 16 actions with 32 and 128 training examples. Use matched underlying cases, multiple seeds and unseen test groups. Report how many examples each action received.
2. At the first missed target, vary one factor at a time. Add examples, introduce two similar actions, hide decisive evidence, or omit the best action from the offered list. Record which changes restore performance.
3. Extend the simulator to longer sequences and noisy diagnostics. Change one decisive clue or one irrelevant clue at a time to see whether decisions respond appropriately.
4. Repeat on a text-based task. Report limits separately for each task, model and training budget.

The first deliverable is a reviewed simulator specification, a small set of hand-checkable cases and their expected decisions. Numerical training and evaluation follow that check.
