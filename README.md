# Specialist Workshop

**Find out whether a small specialist is worth training for your task.**

Bring examples, compare a baseline, train real LoRA weights, and inspect where the result fails. Keep useful corrections as a new version and repeat. The outcome can be a trained specialist, better task data, or evidence that ordinary code or an existing model is enough.

![Task packages in Specialist Workshop](artifacts/public-packages-desktop.png)

## Start locally

The runnable preview is on [`feat/workshop-poc`](https://github.com/itsHabib/specialist-workshop/tree/feat/workshop-poc); the initial implementation is tracked in [PR #1](https://github.com/itsHabib/specialist-workshop/pull/1).

Python 3.12+ is enough to explore packages and deterministic controls. **Apple Silicon is required for local model inference and training.** No API key is needed for the local path. First model use downloads pinned weights from Hugging Face: about 2.5 GB for task packages.

```sh
git clone --branch feat/workshop-poc https://github.com/itsHabib/specialist-workshop.git
cd specialist-workshop
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/uvicorn workshop.app:app --host 127.0.0.1 --port 8787
```

Open **http://127.0.0.1:8787/packages**. Download the starter package from the page and import it, or use `examples/support-intake.package.json`.

1. **Inspect the contract.** Inputs, output schema, provenance, and separate practice/development/final cases travel together.
2. **Establish a baseline.** Try the base model before deciding training is worthwhile.
3. **Train and compare.** Load an actual checkpoint and evaluate it against the same reserved cases.
4. **Inspect mistakes.** See raw responses alongside format validity and semantic scores.
5. **Correct the next version.** Add reviewed practice examples without changing past runs or evaluation data.

For a first run without downloading a model, open another terminal in this directory:

```sh
TASK_REF=$(.venv/bin/python -m workshop.experiment import examples/support-intake.package.json | .venv/bin/python -c 'import json,sys; print(json.load(sys.stdin)["reference"])')
.venv/bin/python -m workshop.experiment run "$TASK_REF" evaluate --policy majority
```

Reload the packages page and open the recorded run. This is a real deterministic control that always emits the most common training output; it is explicitly labeled `majority` and does not pretend to be model inference.

## Three examples to build on

| Package | What it exercises |
| --- | --- |
| [Support intake](examples/support-intake.package.json) | A tiny JSON-output task and immutable corrections |
| [Analytics planning](examples/roxiq-planning.package.json) | Interpret a request, then calculate with deterministic code |
| [CI evidence](examples/ci-practice.package.json) | A resettable environment with observable final outcomes |

[Bring your own task](docs/MONDAY.md) explains data design, baselines, training, comparison, and the limits of the current backend. The [original classifier and API](docs/CLASSIFIER.md) are also available.

## What the experiments found

The loop trains and loads real adapters. **None of the measured specialists qualified for production.** The tiny support adapter scored 0/4; the analytics adapter scored 0/24 under its original strict JSON evaluation. Failed outputs remain inspectable. [Evidence and review](docs/READINESS.md) · [Analytics results](docs/ANALYTICS-EXPERIMENT.md).

That is the point of the workshop: make the decision from evidence. Valid JSON, a matching quote, and a useful answer are different outcomes. New unlabeled inputs report correctness as unknown.

## Development and scope

```sh
.venv/bin/python -m pytest -q
node --check static/app.js
node --check static/packages.js
node --check static/practice.js
node --check static/analytics.js
```

This is a single-user localhost application. Keep `.state/`, credentials, model weights, and your imported data out of Git. Run one app worker. Hosted access, arbitrary uploaded tools, automatic promotion, and RL training are outside the current scope.

[Design](docs/DESIGN.md) · [Publication notes](docs/PUBLICATION.md) · [MIT license](LICENSE)
