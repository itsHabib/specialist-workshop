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

## Next experiment — recovering a job after an exception

Planned from local workshop revision `ea067da`. Can a trained decision tree help a caller get the correct result before its deadline when a worker throws an exception? Start with one synthetic batch job. Prior steering experiments found that adding a useful observation could change the outcome for every policy. Give the coded handler and the tree equal access to each new observation. [Prior steering results](DECISION-STEERING.md) · [Recovery experiment](DECISION-RECOVERY.md)

### 1. Make one recovery observable

| Task | Depends on | Deliverable and acceptance evidence |
| --- | --- | --- |
| R1 | Ready | A local worker sums batches of integers, saves progress in SQLite and publishes one completion event with the result. Give the request a stable ID and raise the same `WorkerTimeout` before any work, after the first saved batch, and after the result commits but before the caller receives it. Keep one uninterrupted request as a control. Inspect the caller's response, durable state and event history for each trace. |
| R2 | R1 | A checker computes the expected sum directly from the original input and separately counts completion events. It requires the caller to receive that result for the original request. Give it deliberately wrong totals, duplicate events and a successful-looking reply with no committed result. It must reject each one. |
| R3 | R1, R2 | Put a typed exception handler around the job call. Look up durable status, resume saved work or retry with the original ID, and return a committed result if it already exists. A second failure spends the remaining attempt budget or ends explicitly. Replay the four R1 cases and an unrelated exception. The four callers should receive the correct result once. The unrelated exception should propagate with its cause. |

The first demo is the lost reply. A caller sees a timeout although its result already exists. Show the exception, the handler retrieving the committed result, and the independent check that exactly one completion event exists after all attempts have settled. Save the trace. Include an ordinary same-ID retry as a comparison and count repeated work even when deduplication succeeds.

### 2. Find the decision that benefits from training

R4 depends on R3. Keep the worker and exception fixed. Introduce a status lookup that can be temporarily unavailable, then vary the remaining deadline. A lookup, a wait and redoing a batch each consume a declared amount of simulated time. Count actual operations and measure host latency separately. Include a case where checking status pays off, one where an immediate same-ID retry saves the deadline, and a matched pair whose visible history is identical although one job has already committed and the other has yet to start. Also run a free, always-available status check as a control. A reliable lookup can remove the entire decision advantage.

Give both policies the same caller-visible history, time remaining, permitted actions and budget. Keep the fault location and future lookup results private to the fixture and checker. Compare status-first recovery, a bounded same-ID retry, and a coded rule that uses time remaining and recent lookup failures. Fit the coded rule’s lookup-versus-retry cutoff using training outcomes and select the strongest baseline on development jobs. Train the workshop's from-scratch tree with its existing prediction API. Derive training labels by replaying permitted next actions from training snapshots under a shared continuation rule and grading their eventual outcomes. Average over compatible hidden states and future delays when observations coincide. Preserve ties for evaluation. [Workshop model interface](DECISION-MAKER.md#use-from-python)

Freeze both policies before testing. Keep all variants of a batch workload in one split. Run each policy on independent copies of the same held-out workload and fault schedule. Keep a separate cohort where status outages last longer, and report its results separately. Count every job once as verified automatic recovery by deadline, review, explicit failure or timeout, or incorrect success. Report duplicate effects, repeated work, status lookups and elapsed time separately. The proposed primary measure is verified automatic recovery by deadline across all jobs. Check for late duplicate effects even after the caller has returned. Advance the tree if the paired 95% interval for its advantage over the best coded policy is above zero and it exhibits zero incorrect successes and duplicate effects in those tests. Report the sample size and failure mix alongside that result.

### 3. Use the failures to improve the system

R5 depends on R4. Inspect the first reproducible miss on development cases. If the handler lacks a necessary observation, add it to both policies and rerun the comparison. If the executor loses the checkpoint or repeats an effect, fix the code and add a regression case. If the tree chooses poorly despite sufficient evidence, collect reviewed training examples for that decision and evaluate a new model version on fresh held-out jobs. Keep a trace of the original failure and the corrected outcome.

Then stress the winning handler with a worker restart, followed by two callers recovering the same request concurrently. Introduce one change at a time and repeat the independent outcome checks. Before proposing a product integration, demonstrate loading the frozen policy from disk in a sample caller and recovering through the coded path when the model is unavailable. A coded win is a result worth shipping. A synthetic win qualifies a model for a private shadow trial.

R1 through R5 have now been exercised on the synthetic worker. The reusable sequence evaluator also matched 384 earlier release-policy runs. The original and feature-corrected trees both trailed their coded control. Reusing saved progress in the executor recovered all jobs feasible under the fixture's deadline and minimum-work bound. [Results, limits and reproduction commands](DECISION-RECOVERY.md)

The next milestones cover [adoption from a clean checkout](DECISION-ADOPTION.md) and [stacking learned decisions in a constrained search](DECISION-DEPTH-PLAN.md).
