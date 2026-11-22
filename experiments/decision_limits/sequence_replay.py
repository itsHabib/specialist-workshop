"""Check the sequence evaluator against the existing two-diagnostic release experiment."""

import argparse
import json
from pathlib import Path

from decision_workshop import load_model, predict
from decision_workshop.core import decision_min_score
from decision_workshop.sequence import run_episode
from .release import HORIZON, episodes, incumbent, oracle, rollout


class ReleaseTrial:
    def __init__(self, case):
        self.case = case
        self.checks = 0
        self.final = None

    def observe(self):
        actions = ["publish", "rollback", "review"]
        if self.checks < HORIZON:
            actions.append("diagnose")
        return self.case.state(self.checks), actions

    def apply(self, action):
        if action == "diagnose":
            self.checks += 1
        else:
            self.final = action

    def is_terminal(self):
        return self.final is not None

    def verify(self, stop_reason):
        if stop_reason == "review":
            self.final = "review"
        terminal_cost = {"publish": 30 * self.case.faulty, "rollback": 10 * (not self.case.faulty), "review": 3}
        if self.final is None:
            return {"status": "failed"}
        if self.final == "review":
            status = "review"
        elif terminal_cost[self.final]:
            status = "incorrect"
        else:
            status = "verified"
        return {"status": status, "terminal_action": self.final,
                "total_cost": terminal_cost[self.final] + self.checks * self.case.price}


def compare(tree_path=None, *, min_score=None):
    policies = {"coded_rule": incumbent, "expected_cost": oracle}
    if tree_path is not None:
        model = load_model(tree_path)
        cutoff = decision_min_score(model, min_score)
        policies["tree"] = lambda row_id, state, candidates: predict(model, state, candidates, min_score=cutoff)
    counts = {}
    for name, policy in policies.items():
        checked = 0
        for cohort in ("iid", "shift"):
            for case in episodes(cohort):
                old = rollout(case, policy)
                trial = ReleaseTrial(case)
                new = run_episode(trial, lambda s, a: policy(f"{case.id}-s{trial.checks}", s, a),
                                  max_steps=HORIZON + 1)
                expected = [(s["state"], s["candidates"], s["choice"]) for s in old["steps"]]
                actual = [(s["state"], s["candidates"], s["action"]) for s in new["steps"]]
                if (expected != actual or new["outcome"]["terminal_action"] != old["terminal_action"]
                        or new["outcome"]["total_cost"] != old["total_cost"]):
                    raise AssertionError(f"release replay differs for {name} on {case.id}")
                checked += 1
        counts[name] = checked
    return counts


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tree", type=Path, help="optional trained release tree.json")
    parser.add_argument("--min-score", type=float, help="override the decision cutoff for this replay")
    args = parser.parse_args()
    try:
        print(json.dumps(compare(args.tree, min_score=args.min_score)))
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
