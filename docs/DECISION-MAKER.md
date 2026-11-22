# Decision Model Maker

Train one local specialist to answer one question by choosing among named actions. The maker fine-tunes Google's BERT-Tiny encoder on your labeled decisions and compares it with a decision tree. The included example asks what a developer should do after a CI failure. Its records are synthetic.

## Set up

```sh
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-decisions.txt
python -m decision_workshop prepare-base --out .state/base
```

On Windows, use `.venv\Scripts\activate` in Command Prompt. With internet access, `prepare-base` downloads the pinned Google configuration, vocabulary and weights and verifies their SHA-256 hashes. With the separate base bundle, extract it into the kit root so the files sit in `.state/base`. The same command verifies them locally. Training and prediction then run locally on a CPU.

For offline dependency installation, use a connected computer with the same OS, CPU architecture and Python version. Run `python -m pip download --only-binary=:all: -r requirements-decisions.txt -d wheels`, copy the `wheels` directory to the kit root, and install with `python -m pip install --no-index --find-links wheels -r requirements-decisions.txt`.

## Train and try the example

For measured limits across action counts, missing information and sequential decisions, see the [decision benchmark results](DECISION-LIMITS.md).

For ranking candidates supplied by a program at runtime, see [steering deterministic programs](DECISION-STEERING.md).

For decisions whose value becomes clear after several steps, see [training from verified outcomes](DECISION-SEQUENCES.md).

```sh
python -m decision_workshop make examples/decisions/software --out .state/ci-run
python -m decision_workshop predict --model .state/ci-run/model --input examples/decisions/software/request.json
```

Open `.state/ci-run/report.json` for the development comparison. The run also contains the trained checkpoint in `model/`, the baseline in `tree.json`, and per-case results in `bert-dev.json` and `tree-dev.json`. The run preserves the task definition and the training and development rows in `inputs/`. This directory contains the original case data, so keep it private when sharing a checkpoint. Use a fresh output directory for each run. The example's recorded `planner` and `llm` choices are synthetic.

For a second tree on the same data, try `--balanced-ties`. When several splits are equally predictive on the training records, this version prefers more evenly sized children. Compare it on development groups before choosing a model.

```sh
python -m decision_workshop train --task examples/decisions/software/task.json \
  --data examples/decisions/software/records.jsonl --balanced-ties --model .state/balanced-tree.json
python -m decision_workshop evaluate --model .state/balanced-tree.json \
  --data examples/decisions/software/records.jsonl --split dev
```

The maker fine-tunes the two-layer Google encoder and a shared scoring head on the reviewed action for each training case. At prediction time it scores each action in the task. If the preferred action is missing from the caller's available options, the result includes that preference and sends the case for review. This lets you measure how often the caller omitted a useful option. The decision tree learns from the same records and uses their numeric, categorical and boolean fields. [Google BERT-Tiny checkpoint](https://huggingface.co/google/bert_uncased_L-2_H-128_A-2/tree/30b0a37ccaaa32f332884b96992754e246e48c5f)

## Create your own task

```sh
python -m decision_workshop init .state/tasks/triage \
  --question 'Which team should handle this request?' \
  --action 'billing=Route the request to billing.' \
  --action 'support=Route the request to technical support.'
```

Supply at least two actions. The starter includes free-text `context` and a structured `case_type` category. Replace the example category with an observation your host can make before choosing an action. To define your own schema, supply one `--field name=type` per field. Supported types are `text`, `number`, `boolean`, and `category`.

The tree can split on numbers, categories and booleans. A text-only tree uses the same training label counts for every request. Training warns when the task has no structured fields or when more than half the training rows have a once-seen value for a text field. After `train`, inspect `tree_diagnostics.split_features` in the checkpoint. After `make`, the same list is in `tree.diagnostics.split_features` in the run report. An empty list means every request reaches the same leaf.

The repository ignores `.state/`. You can also pass a task directory outside the checkout. Use `record.example.json` as a template and write one reviewed decision per line to the empty `records.jsonl`. Record what was known at decision time, the reviewer’s chosen action, and the options actually offered. Use `null` for an unknown historical option list and `[]` for a known empty list. Keep all records from a related job or thread in one `group` and one split. Include `train` and `dev` groups, and reserve unseen `test` groups for the final check.

```sh
python -m decision_workshop make .state/tasks/triage --out .state/triage-run
```

The maker evaluates development groups by default. To compare cutoffs, run `evaluate --split dev --min-score VALUE` against the same checkpoint. Choose the model and cutoff on development groups. `train`, `train-bert` and `make` accept `--min-score` to store that choice. Training defaults to a recorded cutoff of zero. The maker records the cutoff in both checkpoints and in `report.json`. If you choose a new cutoff after fitting, use `save_policy` as shown below to bind it to those same weights. For BERT, supply the checkpoint directory in place of the tree JSON file. Load that policy in the host and use the same explicit cutoff when evaluating its bare checkpoint.

Python and CLI `predict` and `evaluate` use the checkpoint’s cutoff when you omit it. An explicit override applies to that call. Older checkpoints without a recorded cutoff require an explicit value. For example, a tree leaf with two votes for A and one for B scores A at 2/3. A saved cutoff of 0.8 sends it to review before and after reload. Use unseen test groups for the final check.

```sh
python -m decision_workshop evaluate --model .state/triage-run/model --data .state/tasks/triage/records.jsonl --split test --report .state/triage-test.json
```

Read `model.suggested / model.rows` for coverage over every held-out case and `model_with_recorded_candidates.suggested / model_with_recorded_candidates.rows` for coverage where the host recorded its options. One correct suggestion and one review for unrecorded options give **1 of 2 overall (50%)** and **1 of 1 with recorded candidates (100%)**. In the headline, `matched + wrong_suggestions + review = rows`. A known empty list counts in both denominators. The training-majority baseline also uses every held-out row. Comparisons against a named incumbent use the paired rows disclosed in each comparison. Compare the host’s actual fallback and verified result separately with `compare-host`. In a live pilot, also measure time to an accepted decision and how often a reviewer changes it.

## Import records from a host

Keep the adapter in the adopting project. Expose a trusted Python function `iter_records(source) -> Iterable[dict]` in an importable module. The CLI passes a `pathlib.Path` for `--source`, or `None` when omitted. Each yielded dictionary follows the same contract as a line in `records.jsonl`. The importer checks state types, action IDs, candidate lists, unique record IDs, and one split per group. Use `None` for unrecorded candidates and `[]` for a recorded empty list.

Try the included synthetic CI event adapter from the checkout. The output path must be new. When using an initialized task, write to a fresh filename and pass it to `train --data`. For `make`, put the task definition and imported `records.jsonl` together in a fresh task directory.

```sh
python -m decision_workshop import --task examples/decisions/software/task.json \
  --adapter examples.decisions.software.host_adapter:iter_records \
  --source examples/decisions/software/host-events.jsonl --out .state/imported/records.jsonl
python -m decision_workshop train --task examples/decisions/software/task.json \
  --data .state/imported/records.jsonl --model .state/imported/tree.json
```

To use your own module, install it in the same environment or add its containing directory to `PYTHONPATH` and pass `--adapter my_module:iter_records`. The summary counts rows, distinct groups per split, recorded candidate sets and reviewed targets missing from those sets. An empty list counts as a recorded set missing its target. With an unknown list, target membership stays unknown. A validation or adapter failure leaves the destination unpublished. [Example adapter](../examples/decisions/software/host_adapter.py) · [Synthetic input](../examples/decisions/software/host-events.jsonl)

## Use from Python

```python
import json
from pathlib import Path
from decision_workshop import load_model, predict
from decision_workshop.maker import make

task = Path("examples/decisions/software")
run = Path(".state/ci-api-run")
make(task, run, Path(".state/base"))
model = load_model(run / "model")
request = json.loads((task / "request.json").read_text(encoding="utf-8"))
decision = predict(model, **request)
print(decision["status"], decision["action"])
```

Load the checkpoint once and reuse it for new requests. A null `action` means the request needs review. The result includes the model’s preferred action and the reason for review.

## Ship a trained tree with its decision settings

The structured tree trains from scratch using standard Python and your labeled records. This example works offline from a checkout. Use a fresh output directory for a new training run.

```sh
python3 -m decision_workshop train --task examples/decisions/software/task.json \
  --data examples/decisions/software/records.jsonl --model .state/portable-ci/tree.json
```

Freeze the cutoff chosen on development records, the revision of the host code that constructs the features, and the revision of the host's fallback. The cutoff below is illustrative. Store the returned digest in the application's deployment configuration alongside its expected revisions.

```python
from pathlib import Path
from decision_workshop.policy import save_policy

bundle = Path(".state/portable-ci")
pin = save_policy(bundle / "policy.json", bundle / "tree.json", kind="fixed",
                  min_score=0.8, adapter_revision="ci-features/1",
                  fallback_revision="ci-default/1")
print(pin)
```

Ship both JSON files and the `decision_workshop` Python package. The application loads the policy once from its deployment directory. It supplies its own pinned digest, adapter and fallback revisions. The model sees the actions available on the current request.

```python
from decision_workshop.policy import load_policy

policy = load_policy(bundle / "policy.json", bundle / "tree.json",
                     expected_sha256=pin, adapter_revision="ci-features/1",
                     fallback_revision="ci-default/1")
decision = policy.decide(request["state"], request["candidates"])
```

Dispatch the suggested action when `decision["status"] == "suggest"`. On review, run the host's declared fallback and record the reason and the action actually taken. Handle a load error in the host as well. To roll back, restore the previous pair of files and its deployed pin. The digest identifies the selected content. The host owns the mapping from revision strings to running code.

For a variable list of candidates, save with `kind="ranking"` and use `policy.rank(candidates)`. Equal scores retain the supplied order. The Click trial includes a pytest collection adapter that keeps newly seen tests in their original slots and records fallback on an incompatible artifact. [Host adapter](../experiments/steering/pytest_history.py) · [Real test-history trial](DECISION-TEST-HISTORY.md)

## Measure what the application delivered

Start with an arrival ID from the application, before either decision path runs. On a replay, give the incumbent and candidate the same request, starting state, budget and verifier. Record the suggestion at the decision point, the action at the executor and the result at the verifier. An application that runs only one path on a request has an unobserved outcome for the other path.

Use `compare-host` offline with those records. The included example uses three illustrative arrivals. One changed action meets the deadline, one referral uses the incumbent fallback successfully, and neither path has a receipt for the third arrival.

```sh
python3 -m decision_workshop compare-host \
  --input examples/decisions/software/host-comparison.json \
  --report .state/host-comparison.json
```

The input declares `arrivals`, `baseline`, `candidate`, a named `metric` and the direction of improvement (`higher` or `lower`). Each available receipt has its arrival `id`. For a binary endpoint, record `value: 1` for a verified success and `value: 0` for a verified miss. Leave the value unknown if the verifier never observed the result. Use `elapsed_seconds` only with a common start point and endpoint on both paths.

For fixed actions, `proposed` and `executed` are action IDs. For rankings, supply the proposed order and the prefix that actually started by the first failure or deadline. Omit a proposal that was never observed. An explicit null proposal records a referral. Set `source` to `fallback` and record the reason when the executor uses its fallback. Comparing executed prefixes prevents changes in an unexecuted tail from appearing as changes to delivered behavior.

The example reports three arrivals, two paired outcomes, one improvement and one unchanged outcome. The overall mean remains unknown. The per-case records and `summary.by_dispatch` distinguish changed actions that helped, changed actions that hurt, and identical actions with different observed results. The comparison is also available from Python as `decision_workshop.outcomes.compare_host_outcomes`. [Example input](../examples/decisions/software/host-comparison.json) · [Evaluator](../decision_workshop/outcomes.py)

## Compare with Cloudflare Clef-Flash

The optional comparison runs a 4-bit Apple MLX conversion of Cloudflare's 9B Clef-Flash. It scores every action from the same question and state that the maker sees. The comparison applies the same availability and review rules to all three models. Clef-Flash uses the Qwen3.5-9B backbone. The maker's Google and tree training paths run independently of it. [Cloudflare model card](https://huggingface.co/Cloudflare/clef-flash) · [Pinned MLX conversion and hardware requirements](https://huggingface.co/mlx-community/clef-flash-4bit/tree/b5adbe2c3a95200d4bf04a0727c8ec457157d5e1)

After running `make` on the included example, replay its measured Clef-Flash outputs alongside your local checkpoints. This step works on the same CPU setup as the maker. The comparison uses the shared cutoff in the maker checkpoints. For an archived run that used zero, pass `--min-score 0` explicitly if the checkpoints predate stored cutoffs.

```sh
python -m decision_workshop.clef_reference --task examples/decisions/software \
  --receipt examples/decisions/software/clef-flash-receipt.json \
  --maker .state/ci-run --out .state/ci-comparison.json
```

The included run used 48 training records, 12 development records and 12 final test records. Related cases stayed together in one split. On the final test, the recorded options included the expected action in 11 of the 12 cases. The default review policy produced these results.

| Model | Correct suggestion | Wrong suggestion | Sent to review |
| --- | ---: | ---: | ---: |
| Decision tree | 11 | 0 | 1 |
| Google BERT-Tiny | 8 | 1 | 3 |
| Cloudflare Clef-Flash, MLX 4-bit | 7 | 4 | 1 |

The tree preferred the expected action on all 12 cases before checking availability. In one case the caller omitted that action, which triggered review. Each row in the comparison JSON shows the offered actions, the expected action and each model's scores and decision.

To generate fresh Clef-Flash predictions on Apple Silicon, install the MLX backend in a separate environment and download the pinned checkpoint. Allow about 6.2 GB for the download. The replay verifies the checkpoint files, scores the development and test records locally, and writes each model input and output to a new receipt.

```sh
python3 -m venv .state/clef-env
.state/clef-env/bin/python -m pip install 'mlx==0.32.3' 'mlx-lm==0.32.0' 'mlx-vlm==0.7.4' \
  'transformers==5.24.0' 'huggingface_hub==2.4.0' 'torch==2.10.0' 'torchvision==0.25.0'
.state/clef-env/bin/hf download mlx-community/clef-flash-4bit \
  --revision b5adbe2c3a95200d4bf04a0727c8ec457157d5e1 --local-dir .state/clef-flash
.state/clef-env/bin/python -m decision_workshop.clef_mlx_replay \
  --task examples/decisions/software --model-dir .state/clef-flash --out .state/clef-fresh.json
python -m decision_workshop.clef_reference --task examples/decisions/software \
  --receipt .state/clef-fresh.json --maker .state/ci-run --out .state/ci-fresh-comparison.json
```
