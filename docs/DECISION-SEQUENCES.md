# Train from the outcome of a sequence of decisions

Some applications only learn whether a choice helped after several calls. The workshop can replay each available first action, let a common policy finish the job, and train on the resulting outcomes. Each application defines what the model can observe, which actions its code can execute, and how to check the result.

## Integrate one operation

Implement four methods on a local episode, representing one request from its current state through completion. Use a disposable copy for training and evaluation.

| Method | Application responsibility |
| --- | --- |
| `observe()` | Return the visible state dictionary and a list of currently available action names. Observing the state must leave it unchanged. |
| `apply(action)` | Execute one permitted step. Handle expected application exceptions at the boundary that knows the original request. |
| `is_terminal()` | Tell the runner when the operation is over. |
| `verify(stop_reason)` | Inspect the caller's result and application state. Return a dictionary with `status` equal to `verified`, `incorrect`, `failed` or `review`, plus the evidence and measurements needed to audit it. |

Load a frozen model once. At each step the policy receives the observation and candidate list. The workshop tree and BERT score actions from the state. The candidate list guards which suggestion may execute. If availability itself changes the right decision, expose that fact in the state too, for example a boolean `can_resume` when a checkpoint is available. The step budget bounds actions. The application owns deadlines, resource cleanup and concurrency. Callback errors propagate to their caller.

```python
from decision_workshop import load_model, predict
from decision_workshop.sequence import run_episode

model = load_model("models/decision.json")
policy = lambda state, candidates: predict(model, state, candidates)
# episode is the application's adapter for this request.
receipt = run_episode(episode, policy, max_steps=4)
print(receipt["outcome"])
```

The receipt includes the observation at each decision, available actions, proposed action, and whether code executed it. A policy can defer by returning `None` or a workshop review decision. An unavailable action stops execution. The verifier receives `terminal`, `review`, `unavailable_action` or `step_limit` and grades what actually happened. An operation may finish immediately with zero model calls.

## Turn local replays into training data

Start from a saved training situation. Supply a context manager that opens an independent copy at the same step on every call. Reuse the same inputs, hidden state and external event schedule. If the continuation itself uses randomness, restore its random state for each branch as well. The harness checks that observations and candidate order agree when each branch starts. The application must provide isolation and reproduce the hidden events. Keep simulations and side effects inside the local test environment.

```python
from decision_workshop.sequence import replay_alternatives

branches = replay_alternatives(
    open_episode, continuation=coded_policy,
    score=lambda trace: int(trace["outcome"]["status"] == "verified"),
    max_steps=4,
)
training_trial = {
    "id": "job-12-step-0", "group": "job-12", "split": "train",
    "state": branches["state"], "candidates": branches["candidates"],
    "values": branches["values"],
}
```

Save the training trials in a JSON array and keep the full branch traces for inspection. Every available action needs its own outcome under the same continuation and budget. When several training trials have identical observations and available actions, the best mean value determines their shared label. An exact tie follows the task label order. A trial corresponds to one equally weighted occurrence. Logged jobs that record only the executed choice can use the workshop's ordinary reviewed-record format. Comparing untried alternatives requires replay or additional evidence.

```sh
python -m decision_workshop train --task task.json --outcomes trials.json --model tree.json
```

`train-bert` accepts the same `--outcomes` input and a verified local Google base. The sequence evaluator accepts either saved model through `predict()`. Freeze a model using development cases, then give both it and the strongest coded policy the same untouched jobs and budgets. Grade the entire operation. Keep future results and fault causes out of the policy input.

One regression case illustrates why to evaluate the learned continuation. The correct sequence is to add once, then finish. When both steps look identical, the trained tree chooses the correct first move, repeats it, and exhausts its budget. Exposing the current count in the observation lets the same trainer produce a tree that finishes correctly. [Executable two-step check](../tests/decisions/test_sequence.py)

Use `compare_verified(model_statuses, coded_statuses)` from `decision_workshop.sequence` to compare policies. Both mappings use `(workload_group, case_id)` keys and the verifier's status values. Run each policy on identical held-out worlds and preserve the receipts. The comparison checks matching keys, counts jobs recovered exclusively by either policy, and estimates a paired 95% interval by resampling whole workload groups. Report harmful outcomes separately even if overall recovery improves. [Executed recovery comparison](DECISION-RECOVERY.md)

## Check the runner in an existing domain

The diagnostic release experiment already records decisions and realized cost. The following check drives it through both runners, comparing all visible states, available actions, selections, terminal actions and costs. The two coded policies agree with their original traces on 128 held-out jobs each. Adding the saved release tree checks another 128 runs.

```sh
python -m experiments.decision_limits.sequence_replay
python -m experiments.decision_limits.sequence_replay --tree .state/limits-release/run/tree.json
```
