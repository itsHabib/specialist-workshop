"""Test one observable suggested by a development miss on a separate fresh cohort."""

import argparse
from collections import defaultdict
from dataclasses import asdict
import hashlib
import json
from pathlib import Path

from decision_workshop import load_model, predict
from decision_workshop.contracts import read_json
from decision_workshop.core import decision_min_score, digest, train, write_new
from decision_workshop.sequence import compare_verified, outcome_labels
from .benchmark import RULE_GRID, TREE_GRID, cutoff_policy, episode, same_id_retry, select, totals, worlds
from .features import SLACK_TASK, with_slack
from .policy import status_first


SEED = 735269


def slack_policy(model):
    return lambda state, candidates: predict(model, with_slack(state), candidates)


def run(baseline_run, output, *, min_score=None):
    selection = read_json(baseline_run / "selection.json")
    original = load_model(baseline_run / "tree.json")
    cutoff = decision_min_score(original, min_score)
    trials = read_json(baseline_run / "trials.json")
    if original["sha256"] != selection["model_sha256"] or digest(trials) != selection["train_trials_sha256"]:
        raise ValueError("training evidence or original checkpoint differs from the frozen trial")
    output.mkdir(parents=True, exist_ok=False)
    adjusted = [{**t, "state": with_slack(t["state"])} for t in trials]
    rows = outcome_labels(SLACK_TASK, adjusted)
    models = {f"depth={d},leaf={leaf}": train(SLACK_TASK, rows, max_depth=d, min_leaf=leaf, min_score=cutoff)
              for d, leaf in TREE_GRID}
    winner, dev = select({name: slack_policy(model) for name, model in models.items()}, list(worlds("dev")))
    write_new(output / "tree.json", models[winner])
    fixed = load_model(output / "tree.json")
    freeze = {"fresh_seed": SEED, "development": dev, "selected": winner,
              "original_checkpoint": original["sha256"], "corrected_checkpoint": fixed["sha256"],
              "min_score": cutoff,
              "control": selection["selected_baseline"], "original_selection_sha256": digest(selection),
              "input_trials_sha256": digest(trials), "adjusted_trials_sha256": digest(adjusted),
              "source_sha256": {f: hashlib.sha256(Path(__file__).with_name(f).read_bytes()).hexdigest()
                                for f in ("features.py", "slack.py")}}
    write_new(output / "selection.json", freeze)
    fitted = {f"margin={m},failures={f}": cutoff_policy(m, f) for m, f in RULE_GRID}
    control = {"status_first": status_first, "same_id_retry": same_id_retry,
               "fitted_rule": fitted[selection["cutoff_fit"]]}[selection["selected_baseline"]]
    policies = {"original_tree": lambda s, a: predict(original, s, a, min_score=cutoff), "slack_tree": slack_policy(fixed),
                "coded": control}
    statuses = defaultdict(dict)
    outcomes = defaultdict(list)
    with (output / "fresh.jsonl").open("x", encoding="utf-8") as stream:
        for group, case_id, scenario in worlds("test", seed=SEED):
            for name, policy in policies.items():
                trace, host_ns = episode(policy, scenario)
                outcome = trace["outcome"]
                if (group, case_id) in statuses[name]:
                    raise ValueError("duplicate fresh evaluation case")
                statuses[name][group, case_id] = outcome["status"]
                outcomes[name].append(outcome)
                json.dump({"group": group, "case_id": case_id, "policy": name, "scenario": asdict(scenario),
                           "host_ns": host_ns, **trace}, stream, separators=(",", ":"))
                stream.write("\n")
    summary = {"selection_sha256": digest(freeze), "policies": {n: totals(o) for n, o in outcomes.items()},
               "versus_original_tree": compare_verified(statuses["slack_tree"], statuses["original_tree"]),
               "versus_coded": compare_verified(statuses["slack_tree"], statuses["coded"])}
    write_new(output / "summary.json", summary)
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-run", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--min-score", type=float, help="cutoff for both trees (defaults to the baseline checkpoint)")
    args = parser.parse_args()
    try:
        print(json.dumps(run(args.baseline_run, args.out, min_score=args.min_score), indent=2))
    except (ValueError, OSError) as exc:
        parser.error(str(exc))
