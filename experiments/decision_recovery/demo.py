"""Load a frozen policy at startup and use it in a synthetic caller's recovery path."""

import argparse
from contextlib import closing
import json
from pathlib import Path

from decision_workshop import load_model, predict
from decision_workshop.sequence import run_episode
from .features import SLACK_TASK, with_slack
from .policy import TASK, status_first
from .worker import Episode, Scenario


def startup(path):
    if path is None or not Path(path).exists():
        return status_first, {"policy": "status_first", "reason": "model_unavailable"}
    try:
        model = load_model(path)
    except (ValueError, OSError, ModuleNotFoundError) as exc:
        return status_first, {"policy": "status_first", "reason": "model_load_error",
                              "error_type": type(exc).__name__}
    version = model["task"]["version"]
    contract = {"1": TASK, "2": SLACK_TASK}.get(version)
    if contract is None or any(model["task"][k] != contract[k] for k in ("name", "labels", "features")):
        return status_first, {"policy": "status_first", "reason": "incompatible_task"}

    def policy(state, candidates):
        return predict(model, with_slack(state) if version == "2" else state, candidates)
    return policy, {"policy": "model", "version": version, "model_sha256": model["sha256"]}


def demo(path=None):
    policy, identity = startup(path)
    jobs = []
    for fault in ("before", "checkpoint", "lost_reply", "none"):
        case = Scenario(f"demo-{fault}", ((2, 3), (-1, 9)), fault, deadline=10)
        with closing(Episode(case)) as episode:
            trace = run_episode(episode, policy, max_steps=6)
        jobs.append({"fault": fault, **trace})
    return {**identity, "jobs": jobs}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, help="optional local tree checkpoint")
    print(json.dumps(demo(parser.parse_args().model), indent=2, allow_nan=False))
