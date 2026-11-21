"""Generate a release task or train and replay it locally."""

import argparse
import json
from pathlib import Path

from decision_workshop import load_model, predict
from decision_workshop.maker import make
from .release import TASK, episodes, incumbent, oracle, records, rollout, summarize


def generate(root):
    task = root / "task"
    task.mkdir(parents=True, exist_ok=False)
    (task / "task.json").write_text(json.dumps(TASK, indent=2) + "\n", encoding="utf-8")
    with (task / "records.jsonl").open("x", encoding="utf-8") as f:
        for row in records():
            print(json.dumps(row), file=f)


def learned(model):
    def policy(row_id, state, candidates):
        return predict(model, state, candidates)
    return policy


def run(root, base):
    run_dir = root / "run"
    make(root / "task", run_dir, base, epochs=4, batch_size=16, learning_rate=0.0003, seed=17)
    policies = {"oracle": oracle, "incumbent": incumbent, "tree": learned(load_model(run_dir / "tree.json")),
                "bert": learned(load_model(run_dir / "model"))}
    summary = {}
    with (root / "episodes.jsonl").open("x", encoding="utf-8") as f:
        for cohort in ("dev", "iid", "shift", "omission"):
            actual = "iid" if cohort == "omission" else cohort
            summary[cohort] = {}
            for name, policy in policies.items():
                traces = [rollout(ep, policy, omit=cohort == "omission") for ep in episodes(actual)]
                summary[cohort][name] = summarize(traces)
                for trace in traces:
                    print(json.dumps({"cohort": cohort, "policy": name, **trace}), file=f)
    with (root / "summary.json").open("x", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
        print(file=f)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("command", choices=("generate", "run"))
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--base", type=Path)
    args = p.parse_args()
    if args.command == "generate":
        generate(args.out)
    else:
        if args.base is None:
            p.error("run requires --base")
        run(args.out, args.base)
