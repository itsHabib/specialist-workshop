"""Caller-visible input for the second recovery model."""

from .policy import TASK

SLACK_TASK = {**TASK, "version": "2", "features": {**TASK["features"], "retry_slack": "number"}}


def with_slack(state):
    return {**state, "retry_slack": state["remaining"] - state["batches"] * state["batch_cost"]}
