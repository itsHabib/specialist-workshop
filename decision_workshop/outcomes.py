"""Pair independently observed application receipts against a declared arrival list."""

import math
from collections.abc import Mapping
from copy import deepcopy
from fractions import Fraction
from statistics import mean

from .contracts import _fields, _string


_FIELDS = ("source", "proposed", "executed", "value", "elapsed_seconds",
           "fallback_reason", "missing_reason")
_OUTCOMES = ("better", "equal", "worse", "unknown")
_DISPATCH = ("changed", "same", "unknown")
_CHECKS = ("matched_prefix", "mismatch", "fallback", "no_starts", "no_suggestion", "unknown")


def _number(value, field):
    if type(value) not in (int, float) or (type(value) is float and not math.isfinite(value)):
        raise ValueError(f"{field} must be a finite number")


def _prefix(value):
    if value is None:
        return None
    return value if isinstance(value, list) else [value]


def _rows(values, field):
    if isinstance(values, (str, bytes, Mapping)):
        raise ValueError(f"{field} must contain separate records or IDs")
    try:
        return iter(values)
    except TypeError as exc:
        raise ValueError(f"{field} must be iterable") from exc


def _index(receipts, arrivals, side):
    index = {}
    for receipt in _rows(receipts, side):
        _fields(receipt, ("id",), side, optional=_FIELDS)
        job = receipt["id"]
        _string(job, "receipt id")
        if job not in arrivals or job in index:
            raise ValueError(f"unexpected or duplicate {side} receipt for {job!r}")
        if "source" in receipt and receipt["source"] not in ("native", "policy", "fallback"):
            raise ValueError("receipt source must be native, policy or fallback")
        for field in ("proposed", "executed"):
            prefix = _prefix(receipt.get(field))
            if prefix is not None:
                for action in prefix:
                    _string(action, field)
        for field in ("value", "elapsed_seconds"):
            if receipt.get(field) is not None:
                _number(receipt[field], field)
        if receipt.get("elapsed_seconds") is not None and receipt["elapsed_seconds"] < 0:
            raise ValueError("elapsed_seconds must be nonnegative")
        for field in ("fallback_reason", "missing_reason"):
            if field in receipt:
                _string(receipt[field], field)
        index[job] = deepcopy(receipt)
    return index


def _check(receipt):
    if receipt is None:
        return "unknown"
    actual = _prefix(receipt.get("executed"))
    if actual is None:
        return "unknown"
    if not actual:
        return "no_starts"
    if receipt.get("source") == "fallback":
        return "fallback"
    if "proposed" not in receipt:
        return "unknown"
    proposal = _prefix(receipt["proposed"])
    if not proposal:
        return "no_suggestion"
    return "matched_prefix" if actual == proposal[:len(actual)] else "mismatch"


def _difference(baseline, candidate, field, direction):
    if baseline.get(field) is None or candidate.get(field) is None:
        return None
    improvement = Fraction(candidate[field]) - Fraction(baseline[field])
    return improvement if direction == "higher" else -improvement


def _report_number(value, field):
    """Preserve integral results exactly, rounding a fractional result only for JSON."""
    if value is None:
        return None
    if value.denominator == 1:
        return value.numerator
    try:
        rounded = float(value)
    except OverflowError as exc:
        raise ValueError(f"{field} is outside the finite floating-point range") from exc
    if rounded == 0:
        raise ValueError(f"{field} is too small to represent as a nonzero float")
    return rounded


def _arm_summary(cases, side):
    observed = [case[side] for case in cases if case[side] is not None]
    with_proposal = [receipt for receipt in observed if "proposed" in receipt]
    suggested = sum(bool(receipt["proposed"]) for receipt in with_proposal)
    return {
        "receipts": len(observed), "missing_receipts": len(cases) - len(observed),
        "observed_suggestions": suggested,
        "explicit_no_suggestion": len(with_proposal) - suggested,
        "unknown_proposals": len(cases) - len(with_proposal),
        "observed_suggestions_per_arrival": suggested / len(cases) if cases else None,
        "fallbacks_reported": sum(r.get("source") == "fallback" for r in observed),
        "dispatch_check": {status: sum(case[side + "_check"] == status for case in cases)
                           for status in _CHECKS},
    }


def compare_host_outcomes(arrivals, baseline, candidate, *, metric, direction):
    """Compare both policies on the same independently declared opportunity IDs.

    Each receipt needs an id. Optional fields are source (native/policy/fallback),
    proposed and executed (an action ID or ordered IDs), value (a verified endpoint
    in the named metric), elapsed_seconds, fallback_reason and missing_reason. An
    explicit null or empty proposal records no suggestion. Omit proposed when the
    proposal was not observed. An empty execution list means no action started.
    Omit executed or use null for missing dispatch evidence.

    Supply actual starts through the same budget or endpoint on both sides. A
    proposal can extend beyond its observed prefix. Use a common verifier and a
    common clock origin. direction is "higher" or "lower"; positive differences
    always favor the candidate. Missing outcomes stay unknown. Reported outcome
    differences are paired observations and can occur with identical dispatch.
    Integer and floating-point inputs are compared at their supplied precision.
    Nonintegral report values are rounded to floats and must fit their range.
    """
    _string(metric, "metric")
    if direction not in ("lower", "higher"):
        raise ValueError("direction must be lower or higher")
    ids = list(_rows(arrivals, "arrivals"))
    for job in ids:
        _string(job, "arrival id")
    roster = set(ids)
    if len(roster) != len(ids):
        raise ValueError("duplicate arrival id")
    controls = _index(baseline, roster, "baseline")
    trials = _index(candidate, roster, "candidate")
    cases = []
    for job in ids:
        control = controls.get(job, {})
        trial = trials.get(job, {})
        left, right = _prefix(control.get("executed")), _prefix(trial.get("executed"))
        dispatch = "unknown"
        if left is not None and right is not None:
            dispatch = "same" if left == right else "changed"
        delta = _difference(control, trial, "value", direction)
        outcome = "unknown"
        if delta is not None:
            outcome = "better" if delta > 0 else "worse" if delta < 0 else "equal"
        missing = {}
        for side, receipt in (("baseline", control), ("candidate", trial)):
            if receipt.get("value") is None:
                missing[side] = receipt.get("missing_reason", "outcome_unobserved") if receipt else "receipt_missing"
        cases.append({"id": job, "baseline": controls.get(job), "candidate": trials.get(job),
                      "baseline_check": _check(controls.get(job)), "candidate_check": _check(trials.get(job)),
                      "dispatch": dispatch, "outcome": outcome, "improvement": delta,
                      "seconds_saved": _difference(control, trial, "elapsed_seconds", "lower"),
                      "missing_outcomes": missing})

    deltas = [c["improvement"] for c in cases if c["improvement"] is not None]
    seconds = [c["seconds_saved"] for c in cases if c["seconds_saved"] is not None]
    by_dispatch = {status: dict.fromkeys(_OUTCOMES, 0) for status in _DISPATCH}
    for case in cases:
        by_dispatch[case["dispatch"]][case["outcome"]] += 1
    mean_delta = _report_number(mean(deltas), "mean improvement") if deltas else None
    paired_means = {}
    for side in ("baseline", "candidate"):
        values = (Fraction(c[side]["value"]) for c in cases if c["improvement"] is not None)
        paired_means[side] = _report_number(mean(values), f"{side} mean value") if deltas else None
    summary = {
        "arrivals": len(cases), "baseline": _arm_summary(cases, "baseline"),
        "candidate": _arm_summary(cases, "candidate"), "paired_outcomes": len(deltas),
        "paired_mean_values": paired_means,
        "mean_improvement_on_pairs": mean_delta,
        "mean_improvement_per_arrival": mean_delta if len(deltas) == len(cases) else None,
        "by_dispatch": by_dispatch,
        "outcomes": {status: sum(row[status] for row in by_dispatch.values()) for status in _OUTCOMES},
        "timing": {"paired": len(seconds), "missing_pairs": len(cases) - len(seconds),
                   "mean_seconds_saved_on_pairs": _report_number(mean(seconds), "mean seconds saved")
                   if seconds else None},
    }
    for case in cases:
        for field in ("seconds_saved", "improvement"):
            case[field] = _report_number(case[field], field)
    return {"metric": metric, "direction": direction, "summary": summary, "cases": cases}
