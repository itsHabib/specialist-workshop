"""Train local routers and recovery specialists on the failures each router produces."""

from collections import Counter
from functools import lru_cache
import hashlib
import json
from pathlib import Path

from decision_workshop.checkpoint import train_bert
from decision_workshop.contracts import validate_records
from decision_workshop.core import digest, load_model, predict, train, write_new

from .host import attempt, jobs


ROUTE_TASK = {
    "name": "import-delimiter", "version": "1",
    "question": "Which delimiter should the importer try first?",
    "features": {"declared_hint": "category", "comma_count": "number", "semicolon_count": "number"},
    "labels": ["comma", "semicolon"],
}
RECOVERY_TASK = {
    "name": "import-recovery", "version": "1",
    "question": "Which single change should the importer try?",
    "features": {"status": "category"},
    "labels": ["switch_delimiter", "switch_encoding"],
}
BERT_RECIPE = {"epochs": 4, "batch_size": 16, "learning_rate": 0.0003}


def training_row(job_id, state, target, task):
    return {"id": job_id, "group": job_id, "split": "train", "state": state,
            "target": target, "candidates": task["labels"]}


def classify(model):
    """Cache deployment decisions by exactly the observable fields in this fixture."""
    task = model["task"]
    fields = list(task["features"])

    @lru_cache(maxsize=128)
    def decide(values):
        return predict(model, dict(zip(fields, values)), task["labels"])["action"]

    return lambda state: decide(tuple(state[field] for field in fields))


def fit_pair(task, rows, seed, base, directory, output):
    validate_records(task, rows)
    directory.mkdir(parents=True)
    write_new(directory / "task.json", task)
    data = directory / "records.jsonl"
    with data.open("x", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n")
    paths = {"tree": directory / "tree.json", "bert": directory / "bert"}
    write_new(paths["tree"], train(task, rows))
    train_bert(task, rows, base, paths["bert"], seed=seed, **BERT_RECIPE)
    classifiers, receipts = {}, {}
    records_digest = digest(sorted(rows, key=lambda row: row["id"]))
    for kind, path in paths.items():
        model = load_model(path)
        if model["training"]["records_sha256"] != records_digest:
            raise ValueError(f"{kind} checkpoint consumed different training rows")
        classifiers[kind] = classify(model)
        receipt = {
            "checkpoint": path.relative_to(output).as_posix(),
            "model_sha256": model["sha256"], "records_sha256": records_digest,
            "records": data.relative_to(output).as_posix(), "rows": len(rows),
            "label_counts": model["training"]["label_counts"],
        }
        if kind == "bert":
            stats = model["optimization"]
            if stats["optimizer_steps"] < 1 or stats["backbone_sha256_before"] == stats["backbone_sha256_after"]:
                raise ValueError("fine-tuning did not update the encoder")
            receipt.update(base=model["base"]["model"], base_revision=model["base"]["revision"],
                           optimizer_steps=stats["optimizer_steps"],
                           backbone_sha256_before=stats["backbone_sha256_before"],
                           backbone_sha256_after=stats["backbone_sha256_after"],
                           decision_head_sha256=model["artifacts"]["decision_head.safetensors"])
        receipts[kind] = receipt
    return classifiers, receipts


def collect_failures(collection, propose):
    """Only a flagged import with exactly one content-correct repair supplies a label."""
    statuses, outcomes = Counter(), Counter()
    fallback = 0
    examples = []
    for job in collection:
        route = propose(job)
        if route is None:
            fallback += 1
            route = job.state["declared_hint"]
        status, rows = attempt(job, route)
        statuses[status] += 1
        if status == "ok":
            outcomes["exact_first" if rows == job.expected else "wrong_first_output"] += 1
            continue
        passing = []
        for action in RECOVERY_TASK["labels"]:
            recovered, result = attempt(job, route, action)
            if recovered == "ok" and result == job.expected:
                passing.append(action)
        if len(passing) == 1:
            examples.append((job.id, status, passing[0]))
            outcomes["unique_fix"] += 1
        else:
            outcomes["no_fix" if not passing else "ambiguous_fix"] += 1
    return examples, {"arrivals": len(collection), "route_fallbacks": fallback,
                      "first_status_counts": dict(statuses), "outcome_counts": dict(outcomes),
                      "eligible_label_counts": dict(Counter(action for _, _, action in examples))}


def recovery_state(view, status):
    return {"status": status if view == "diagnostic" else "failed"}


def train_policies(output, base, *, seeds):
    """Return reloaded callables per seed and persist the evidence used to fit them."""
    output = Path(output)
    directory = output / "fits"
    directory.mkdir(parents=True, exist_ok=False)
    route_jobs = jobs("fit")
    # One public hash permutation is shared by all sources and seeds. Taking its
    # prefix after filtering gives equal budgets without stratifying on labels.
    collection = sorted(jobs("collect"), key=lambda j: hashlib.sha256(j.id.encode()).digest())
    if {j.id for j in route_jobs} & {j.id for j in collection}:
        raise ValueError("routing and recovery groups must be disjoint")
    route_rows = [training_row(j.id, j.state, j.route_target, ROUTE_TASK) for j in route_jobs]
    receipt = {"recipe": BERT_RECIPE, "seeds": {}}
    policies = {}
    for seed in seeds:
        run = directory / str(seed)
        route_models, routing_receipt = fit_pair(ROUTE_TASK, route_rows, seed, base, run / "route", output)
        proposals = {"hint": lambda j: j.state["declared_hint"]}
        for kind, classifier in route_models.items():
            proposals[kind] = lambda j, score=classifier: score(j.state)
        routes = {kind: (lambda j, propose=propose: propose(j) or j.state["declared_hint"])
                  for kind, propose in proposals.items()}

        observed, collection_receipts = {}, {}
        for source, propose in proposals.items():
            observed[source], collection_receipts[source] = collect_failures(collection, propose)
        budget = min(map(len, observed.values()))
        if not budget:
            raise ValueError("at least one router yielded no uniquely repairable training failure")
        repairs, recovery_receipts, fitted = {}, {}, {}
        for source, examples in observed.items():
            selected = examples[:budget]
            collection_receipts[source].update(
                selected_ids=[job_id for job_id, _, _ in selected],
                selected_status_counts=dict(Counter(status for _, status, _ in selected)),
                selected_label_counts=dict(Counter(action for _, _, action in selected)),
            )
            recovery_receipts[source] = {}
            for view in ("coarse", "diagnostic"):
                rows = [training_row(job_id, recovery_state(view, status), action, RECOVERY_TASK)
                        for job_id, status, action in selected]
                # Task, seed, base and recipe are constant across these recovery fits.
                # If routers produced identical records, use the very same checkpoint.
                key = digest(rows)
                if key not in fitted:
                    fitted[key] = fit_pair(RECOVERY_TASK, rows, seed, base, run / source / view, output)
                models, recovery_receipts[source][view] = fitted[key]
                for kind, classifier in models.items():
                    repairs[source, view, kind] = lambda status, score=classifier, v=view: score(recovery_state(v, status))
        policies[seed] = {"routes": routes, "repairs": repairs}
        receipt["seeds"][str(seed)] = {"routing": routing_receipt, "matched_rows": budget,
                                       "collection": collection_receipts, "recovery": recovery_receipts}
    write_new(directory / "receipt.json", receipt)
    return policies
