"""Decision contract and coded fallback for a caller recovering a batch job."""

ACTIONS = ["return_result", "resume", "retry", "lookup", "wait"]

TASK = {
    "name": "local_batch_recovery", "version": "1", "labels": ACTIONS,
    "features": {
        "remaining": "number", "batches": "number", "batch_cost": "number",
        "lookup_cost": "number", "wait_cost": "number", "lookup_failures": "number",
        "recovery_attempts": "number", "elapsed": "number", "checkpoint_batches": "number",
        "known": "category",
    },
}


def known_action(state):
    return {"committed": "return_result", "checkpoint": "resume", "missing": "retry"}.get(state["known"])


def status_first(state, candidates):
    """Look up progress once, and retry the original request if the lookup failed."""
    known = known_action(state)
    if known in candidates:
        return known
    if state["lookup_failures"] == 0:
        return "lookup"
    return "retry"
