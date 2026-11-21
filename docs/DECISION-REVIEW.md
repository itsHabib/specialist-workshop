# Independent review of the Decision Model Maker

Use this text as the assignment for a human reviewer or a coding agent. The review packet contains the decision-maker source, synthetic fixtures and earlier measurement receipts. Start at the packet root. Identify the supplied revision in `REVIEW-SNAPSHOT.json`, or record `git rev-parse HEAD` if reviewing an equivalent checkout.

> Review the Decision Model Maker as an engineer deciding whether to embed it in an unfamiliar application. The intended user has local historical records, deterministic code that chooses among actions, and an existing fallback. They want to train on their own data, run locally, verify the outcome, and retain the current rule if it wins. Supported training paths should meet the owner's non-Chinese-base requirement. Verify the provenance of the from-scratch tree and pinned Google encoder and the separate role of the optional Clef comparison.
>
> Read the current roadmap in `docs/DECISION-DEPTH-PLAN.md`, the adoption and maker guides, and the recovery and steering results. Follow one claim through the implementation, tests and recorded measurements. Label each material claim reproduced, documented, proposed or unverified. Identify exactly which revision and transfer artifact you used.
>
> Work from a clean extracted copy with the supplied synthetic material. Train a stdlib tree, reload it from a separate caller, and compare its actual behavior with a simple coded incumbent on held-out groups. Run one check that could change your adoption verdict. Examples include unrecorded candidates, a missing action, a lost threshold after reload, an all-text tree or a timeout requiring fallback. Try one failure that the existing roadmap overlooks. If another backend needs unavailable weights or dependencies, record that limit and keep going with the offline path.
>
> Audit the measurements. Reconcile the counts for all arrivals, eligible cases, suggestions, correct and wrong suggestions, reviews, fallback outcomes and missing labels. Determine who selected historical actions, whether rejected alternatives can be scored, what was known before each decision, and how group or time leakage is prevented. Check whether a code change would solve the same problem more cheaply. For a proposed stack, name the simplest baseline, the shared resource budget, and the result that would make you remove a stage.
>
> Return an adoption verdict for offline evaluation, shadow use and live steering separately. Give the three highest-impact gaps with file and line, reproduction command or missing evidence, user consequence and smallest corrective action. Include one strength worth preserving, one claim you would narrow, the hardest useful next experiment with a stopping rule, and the single next milestone you would fund. A negative result is welcome. Use temporary working files for probes and return the review as text. Keep source, checkpoints, logs and results local.
>
> If the owner separately provides an approved private application environment, evaluate its evidence there and return a separate local appendix. Use the supplied snapshot as the full scope of an external review.

## Offline starting check

From the packet or checkout root, use a new temporary directory. The following commands exercise the structured tree with standard Python.

```sh
run_dir=$(mktemp -d)
task_dir=examples/decisions/software
python3 -E -S -m decision_workshop train --task "$task_dir/task.json" \
  --data "$task_dir/records.jsonl" --model "$run_dir/tree.json"
python3 -E -S -m decision_workshop evaluate --model "$run_dir/tree.json" \
  --data "$task_dir/records.jsonl" --split dev --min-score 0.8 --report "$run_dir/dev.json"
python3 -E -S -m decision_workshop predict --model "$run_dir/tree.json" \
  --input "$task_dir/request.json" --min-score 0.8
```

The cutoff in this smoke test exercises parameter passing. Select an operating cutoff on development data for the reviewer's own task. In the review packet, the historical kit under `evidence/` contains its own revision and manifest. Its synthetic receipts can substantiate older experiments. The source and plans at the packet root are the review target.
