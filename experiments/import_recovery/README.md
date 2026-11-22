# When a better router breaks the recovery model's assumptions

A trained tree chose the right CSV delimiter on all 256 held-out files. The supplied hint got 192 delimiters right. Of the 64 corrected choices, 50 yielded an exact first-pass import, eight still hit a decoding error and six exposed a bad value. First-pass imports rose from **143 to 193**. With the original recovery tree still in place, both pipelines finished **the same 193 files**. Training recovery on failures from the new router raised the total to **240**. So did a small coded error handler.

This is a runnable example of training two decision models around deterministic software. It also shows how to tell whether a disappointing result comes from the training examples, the observation a model receives, or a limit on what the host can do.

## Follow one file

The router reads a supplier's delimiter hint and counts commas and semicolons in the header. It chooses a delimiter and calls the real CSV parser with UTF-8. After an error, the recovery specialist can change the delimiter or switch to Windows-1252 and retry the original bytes once. The final grade compares every decoded field to the generated source record. The expected record is available to the grader.

File `test-44130` has a semicolon header and a quoted, multiline description beginning with `Zoë`. Both routers selected semicolon. The byte for `ë` in this file is `eb` in hexadecimal, so the UTF-8 attempt raised a decode error.

| What the repair tree learned from | What it saw | Its one retry | Outcome |
| --- | --- | --- | --- |
| Failures from the original hint router | Generic “failed” | Switch delimiter | Decode error again |
| Failures from the new tree router | Generic “failed” | Switch encoding | Exact original records |
| Failures from the original hint router | Parser status `decode_error` | Switch encoding | Exact original records |

Those are three recorded outcomes for the same bytes. The complete [host](host.py), [training calls](fit.py) and [dispatch loop](run.py) are short enough to read in that order.

## Experiment 1 · Change which failures reach stage two

Train a router on 128 files. Run the saved router on a separate collection of 256 files. Whenever an import fails, execute both legal repairs against the original bytes. A unique content-correct repair supplies a training label. Match the training budget at **48 recovery examples per source**. After the hint router, 30 labels called for switching delimiter and 18 for switching encoding. After the tree router, all 48 called for switching encoding.

Give the recovery tree only the word “failed”, then change one part of the pipeline at a time.

| Deployed router | Recovery trained on failures from | First-pass exact | Final exact | Retries |
| --- | --- | ---: | ---: | ---: |
| Supplied hint | Supplied hint | 143 | 193 | 113 |
| Tree | Supplied hint | 193 | 193 | 63 |
| Tree | Tree | 193 | 240 | 63 |

With one indistinguishable input, the recovery tree always picks the most frequent training action. Recollecting changed that constant from “switch delimiter” to “switch encoding”. The paired improvement was **47 wins, zero losses and 209 ties**. It added 23 exact imports in the plain family and 24 in the held-out quoted family, at the same retry count. The gain was 18.36 percentage points. The paired 95% bootstrap interval on these generated files was 13.67 to 23.05 points. The trees and sample order are deterministic. All three runs use the same files, and only BERT training varies with the seed.

## Experiment 2 · Measure the value of an error message

For each observable error, count the test files that each single repair would fix and take the larger count. Add the files that already passed. This gives an upper bound on what a deterministic repair specialist with that observation could achieve on these files, even with excellent training. Compute a second upper bound by letting each file have its own best repair.

| Deployed router | First-pass exact | Best with generic “failed” | Best with the error class | Best per file |
| --- | ---: | ---: | ---: | ---: |
| Supplied hint or fitted BERT | 143 | 193 | 233 | 233 |
| Fitted tree or coded header rule | 193 | 240 | 240 | 240 |

After the hint router, an error class can distinguish two useful repairs. The diagnostic tree reached 233. After the tree router, encoding was the sole successful repair on this test set. A constant encoding choice reached 240, spending 63 retries. The diagnostic tree reached the same total in 50 retries by referring unknown value errors. Sixteen files contained a corrupted numeric value outside either repair's ability. On the hint route, a further seven needed both a delimiter change and an encoding change.

## Experiment 3 · Exchange the architectures at each stage

These pairings expose the parser error class and train recovery on the deployed router's own failures. Both architectures consume identical labeled rows within each fit. BERT fine-tunes Google's pretrained two-layer encoder and its decision head on CPU. Each tree fits from scratch. Across all three seeds, the BERT router followed the hint on every measured file and the tree router agreed with the header rule.

| Router → recovery | Exact imports, seed 17 | Seed 29 | Seed 43 |
| --- | ---: | ---: | ---: |
| Tree → tree | 240 | 240 | 240 |
| Tree → BERT | 240 | 240 | 240 |
| BERT → tree | 233 | 233 | 233 |
| BERT → BERT | 193 | 233 | 233 |
| Header rule → coded error handler | 240 | 240 | 240 |

Here the fully coded pipeline reaches the ceiling with 50 retries. That is the implementation to keep for this importer. The structured header makes this a favorable case for code. The quoted holdout tests escaped quotes, delimiters inside fields and embedded newlines. Its header follows the same convention as the training headers. The experiment uses synthetic fault rates, one parser and two repair actions. A different error distribution or host warrants a new comparison.

## Reproduce it from a fresh clone

**Audit the published run** with the Python standard library. From the repository root, run:

```sh
python3 -m experiments.import_recovery.audit --run experiments/import_recovery/evidence
```

The auditor rebuilds the inputs, reads the compressed trace, parses with `csv.DictReader`, checks the recorded actions and recomputes the totals, paired contrast and information ceilings. The published run covers **50,688 file–policy receipts and 720 aggregate rows** across development and test. The audit grades the decisions in the archive. The training command below recreates the checkpoints and invokes them. One regression test deliberately decodes valid UTF-8 text as Windows-1252. Both parses return success, but the wrong one returns `CafÃ©` and fails the exact-content grade.

**Train your own copies** with the decision dependencies and Google base. Use a new output directory for each run. The first download fetches the pinned weights. Subsequent runs use the verified local copy.

```sh
python3 -m venv .venv-decisions
. .venv-decisions/bin/activate
python -m pip install -r requirements-decisions.txt
python -m decision_workshop prepare-base --out .state/base
python -m experiments.import_recovery.run --base .state/base --out .state/import-recovery
python -m experiments.import_recovery.audit --run .state/import-recovery
```

For a quicker wiring check, append `--seeds 17` to the training command. Open `fits/receipt.json` in your output to locate a trained checkpoint, its task, the actual training rows and its digest. Each neural entry records the pinned base, encoder hashes before and after fitting, and optimizer steps. The full three-seed run retained about 265 MiB of local fit artifacts on the measured machine. The published evidence includes the [training receipt](evidence/fits.json), [code and runtime fingerprints](evidence/freeze.json), [per-seed results](evidence/summary.json), [compressed per-file decisions](evidence/cases.jsonl.gz) and [protocol frozen with the run](PROTOCOL.md).

## Adapt it to your own software

1. Pick one host operation with an observable success condition and a bounded set of recovery actions. Capture what the caller received. Parser success alone missed a deliberately corrupted-text control here.
2. Create separate datasets for the initial decision and the recovery decision. Record which router produced each failure, which action was attempted, and how the host verified the result. When untried actions have unknown outcomes, leave those labels unknown.
3. Split by independent job, customer or revision before collecting labels. Fit both architectures within each stage on identical data. Reload the resulting checkpoints, switch one stage at a time and retain the existing coded policy as a baseline.
4. Choose using whole-job successes at an agreed budget. Count review, wasted retries and successful parser returns that fail the downstream contract. If the per-file ceiling is high but the ceiling for the visible error is low, test a richer error signal. If even the per-file ceiling falls short of the target, revisit the permitted actions or the retry budget.
