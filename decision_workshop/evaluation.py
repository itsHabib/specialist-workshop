"""Compare replayed suggestions and recorded incumbents on the same held-out decisions."""

from .core import digest, predict


def tally(pairs):
    total = len(pairs)
    suggested = sum(action is not None for action, _ in pairs)
    matched = sum(action == target for action, target in pairs)
    return {
        "rows": total, "suggested": suggested, "matched": matched,
        "wrong_suggestions": suggested - matched, "review": total - suggested,
        "coverage": suggested / total if total else None,
        "match_per_record": matched / total if total else None,
        "match_when_suggested": matched / suggested if suggested else None,
    }


def model_tally(cases, field):
    return tally([(case["decision"][field], case["target"]) for case in cases])


def baseline_comparison(cases, policy):
    paired = [case for case in cases if policy in case["baselines"]]
    known = [case for case in paired if case["candidates"] is not None]
    selectable = [case for case in known if case["target"] in case["candidates"]]

    def selection_tally(subset):
        selections = []
        for case in subset:
            answer = case["baselines"][policy]
            selections.append((answer if answer in case["candidates"] else None, case["target"]))
        return tally(selections)

    outside = sum(case["baselines"][policy] is not None
                  and case["baselines"][policy] not in case["candidates"] for case in known)
    return {
        "missing_records": len(cases) - len(paired),
        "outside_candidates": outside,
        "raw_incumbent": tally([(case["baselines"][policy], case["target"]) for case in paired]),
        "raw_model_preference": model_tally(paired, "preferred"),
        "incumbent": selection_tally(known),
        "model_on_same_records": model_tally(known, "action"),
        "when_reference_present": {
            "incumbent": selection_tally(selectable),
            "model_on_same_records": model_tally(selectable, "action"),
        },
    }


def evaluate(model, records, split, *, min_score=0.0):
    if split not in ("dev", "test"):
        raise ValueError("evaluate a dev or test split")
    rows = sorted((row for row in records if row["split"] == split), key=lambda row: row["id"])
    if not rows:
        raise ValueError("no records for the requested split")
    training = model["training"]
    groups = set(training["group_hashes"])
    ids = set(training["id_hashes"])
    if any(digest(row["group"]) in groups or digest(row["id"]) in ids for row in rows):
        raise ValueError("evaluation group or record ID was used in checkpoint training")
    policies = sorted({name for row in rows for name in row.get("baselines", {})})
    counts = training["label_counts"]
    majority = max(model["task"]["labels"], key=counts.__getitem__)
    cases = []
    for row in rows:
        candidates = row["candidates"]
        decision = predict(model, row["state"], candidates, min_score=min_score)
        cases.append({
            "id": row["id"], "group": row["group"], "target": row["target"],
            "candidates": candidates, "decision": decision,
            "majority_action": majority if candidates and majority in candidates else None,
            "baselines": row.get("baselines", {}),
        })
    known = [case for case in cases if case["candidates"] is not None]
    selectable = [case for case in known if case["target"] in case["candidates"]]
    raw = model_tally(cases, "preferred")
    return {
        "algorithm": model["algorithm"], "min_score": min_score,
        "brier_multiclass": sum(sum((p - (label == c["target"])) ** 2
                                    for label, p in c["decision"]["scores"].items())
                                for c in cases) / len(cases),
        "task": model["task"]["name"], "model_sha256": model["sha256"],
        "evaluation_records_sha256": digest(rows), "split": split, "groups": len({r["group"] for r in rows}),
        "rows": len(rows), "candidate_sets_recorded": len(known),
        "candidate_sets_unrecorded": len(rows) - len(known),
        "reference_in_candidates": len(selectable), "reference_missing": len(known) - len(selectable),
        "reference_in_candidates_rate": len(selectable) / len(known) if known else None,
        "preference_matches_before_candidate_check": raw["matched"],
        "raw_model_preference": raw, "model": model_tally(known, "action"),
        "model_when_reference_present": model_tally(selectable, "action"),
        "train_majority": tally([(case["majority_action"], case["target"]) for case in known]),
        "recorded_baselines": {policy: baseline_comparison(cases, policy) for policy in policies},
        "cases": cases,
    }
