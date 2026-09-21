# A boost that can do work

A small agent loop for Python repository repairs: inspect files, edit, execute
commands/tests, read failures, retry and ask for a fresh investigator. Both the
worker and investigator can use the same inexpensive model. The investigator
can supply an executable counterexample; the worker still has to check it.

Use this when an ordinary repair loop is stuck or when you want to compare a
specific assistance mechanism. It is an experimental Python runner, not a
replacement for your editor or a general hosted agent service.

## Try it

Requires Python 3.12+, Docker, and either authenticated Claude Code (`haiku`,
`sonnet`) or Ollama (`qwen2.5:7b`, `llama3.2:1b`). From this repository:

```sh
python3 - <<'PY'
import json
from experiments.boost_workloop.taskpack.taskpack import build_task
open('/tmp/boost-task.json','w').write(json.dumps(build_task(0)))
PY
python3 experiments/boost_workloop/agent.py /tmp/boost-task.json \
  --model haiku --policy solo --auto-test --out .runs/my-boost
```

A task is a JSON object with `goal`, `files` (relative filename → text) and
`test_command`. The supplied taskpack is only an example. Candidate commands
run in a fresh, networkless Python Docker container. Use the agent's `write` or `edit`
action to retain file changes; shell side effects are disposable. Current
files, actions, tests, model receipts and intermediate snapshots remain in the
run directory. The original repository is not edited.

Add `--auto-test` to run the visible suite immediately after each file edit.
This prevents a worker from repeatedly changing code without seeing its current
failures. Use `--helper-model haiku` only when you want to try a different
investigator; the default uses the worker model. Report that model change.

The default allows 12 native invocations, two hours active time and thirty
minutes per call. Claude calls share a $4 known-cost budget; interrupted calls
may have unknown cost. Local electricity/hardware is not priced. Increase
`--calls`, `--seconds` or `--max-dollars` for a new run when the work warrants it;
these are experiment budgets, not measures of a model's ability.

Use `--policy solo` for the same working loop without a helper. Assistance is
requested by the worker or triggered after two consecutive official test
failures. There is no mandatory final critic. Helper calls count toward the
same budget. A green editable test suite is a worker's completion claim,
**not independent acceptance**.

## Supply the check the worker is missing

```sh
python3 experiments/boost_workloop/agent.py task.json --model haiku \
  --policy solo --auto-test --check my_contract_check.py --out .runs/checked
```

The check is a Python assertion script run against the current files in Docker.
The worker cannot replace the caller's script. Its diagnostics **are returned
to the worker**: this is development feedback, never a hidden final evaluation.
For the example variant 1, use [check_rectarea.py](examples/check_rectarea.py).
The original script content must match on resume. Trusted Python callers can
also pass `checker(files) -> {passed: bool, ...}` and a stable `checker_name` to
`agent.run`; diagnostics go into the same feedback loop.

Continue a stalled run with a fresh worker:

```sh
python3 experiments/boost_workloop/rescue.py .runs/stalled \
  --model haiku --check my_contract_check.py --out .runs/rescue
```

This carries the patch, goal and recent observations forward, restores original
`tests/` and `test_*.py` files, and records upstream time/cost. New budgets apply
to the continuation, so assess both runs together. Test layouts outside those
conventions need an explicit caller-owned check. Completion still requires
independent validation. Changing models is escalation, not same-model uplift.

## Stop, inspect, resume

Create `<run>/STOP` to request cancellation. To continue, remove that file and
repeat the command with `--resume`. The original plan is retained. A reserved
call with no receipt requires investigation; the runner refuses to silently
issue it again. Inspect `state.json`, `call-*/receipt.json` and `snapshots.json`.
Do not delete a call directory to force a retry.

For the supplied task, `pilot.py freeze <run> <model> <variant>` saves a comparison
plan and checks the external oracle. `pilot.py run <run> solo`, `... feedback` and `... assist`
compare manual test selection, automatic feedback, and feedback plus assistance;
`pilot.py resume <run> <policy>` resumes the original plan; `pilot.py grade <run> <policy>` evaluates saved snapshots with
original regression tests and hidden exact cases. Do not feed that final grading
back into a repair run. [Protocol](PROTOCOL.md) explains budgets and transfer.

A `--lessons file.json` list can supply short findings from earlier practice.
Keep their scope, evidence and failure conditions. Freeze the list before a
new task; better performance on the task that produced it is repair, not proof
of learning. Results must justify keeping a lesson or a helper.

Native model clients run on the host with their own tools disabled. Candidate
code runs inside Docker; this is not full model-client isolation. No production
credentials or private work material belong in the task snapshot. Existing
Rooms integration remains separate.

## Keep what a successful repair learned

Two complementary artifacts can come from an independently accepted practice
snapshot:

```sh
python3 experiments/boost_workloop/learn.py .runs/practice/feedback \
  --grading .runs/practice/feedback/grading.json --out .runs/new-lesson
python3 experiments/boost_workloop/reuse.py .runs/practice/feedback \
  --grading .runs/practice/feedback/grading.json --file rectarea.py \
  --out .runs/new-component
```

`learn.py` asks a model for a short candidate lesson, excluding final evaluator
cases. `reuse.py` exports the actual accepted source with its digest and prior
goal. `reuse.apply(task, component, 'learned_geometry.py')` adds it as a visible,
inspectable dependency to another task without overwriting existing files.
Acceptance must bind the exact run plan, task, current source and evaluator before either export. These are trusted local records, not tamper-proof attestations.

A component may save a later model from re-deriving a hard algorithm. It is not
proof the model learned new weights, and the old acceptance does not establish
new API/CLI requirements. Re-test it in the new task. Count the original work,
verification and lesson-generation costs when evaluating reuse economics.

[Measured results and lessons](RESULTS.md) retain failed trials alongside working repairs.
