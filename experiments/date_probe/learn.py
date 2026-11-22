"""Learn the chance that one more date-format witness corrects a feed."""

from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path

from decision_workshop.checkpoint import train_bert
from decision_workshop.contracts import validate_records
from decision_workshop.core import digest, load_model, predict, train, write_new

from .host import batches, free_history, observe, probe


TASK = {
    "name": "date-format-witness", "version": "1",
    "question": "Would probing this feed correct an imported date?",
    "features": {"notice": "boolean", "peer_outcome": "category"},
    "labels": ["prefer", "defer"],
}
PEER_OUTCOMES = ("unseen", "right", "wrong", "mixed")
TREE_RECIPE = {"max_depth": 4, "min_leaf": 4}
BERT_RECIPE = {"epochs": 4, "batch_size": 16, "seed": 17, "learning_rate": 0.0003}


def state(feed, history):
    """Expose only the notice and already observed outcomes on this channel."""
    outcomes = {wrong for channel, wrong in history if channel == feed.channel}
    if not outcomes:
        peer_outcome = "unseen"
    elif len(outcomes) == 2:
        peer_outcome = "mixed"
    else:
        peer_outcome = "wrong" if True in outcomes else "right"
    return {"notice": feed.notice, "peer_outcome": peer_outcome}


def training_rows():
    rows = []
    for batch in batches("fit"):
        initial_history = list(free_history(batch))
        eligible = []
        for feed in batch.feeds:
            native, ambiguous_count, resolved = observe(feed)
            if resolved or ambiguous_count == 0:
                continue
            repaired, hint_wrong = probe(feed)
            if repaired != feed.expected:
                raise ValueError("training witness failed to recover the source dates")
            prefer = native != repaired
            if prefer != hint_wrong:
                raise ValueError("training witness and exact outcomes disagree")
            eligible.append((feed, prefer))
        for feed, prefer in eligible:
            # Fixed peer order precedes any label inspection. Never use the target
            # feed's own witness to construct its observed history.
            peers = [(peer.channel, peer_wrong) for peer, peer_wrong in eligible
                     if peer.id != feed.id and peer.channel == feed.channel][:2]
            for count in range(len(peers) + 1):
                rows.append({"id": f"{batch.id}/{feed.id}/{count}", "group": batch.id,
                             "split": "train", "state": state(feed, initial_history + peers[:count]),
                             "candidates": TASK["labels"], "target": "prefer" if prefer else "defer"})
    validate_records(TASK, rows)
    return rows


def fit(out: Path, base: Path):
    """Fit both architectures on the same rows and reload before scoring."""
    directory = Path(out) / "fits"
    directory.mkdir(parents=True, exist_ok=False)
    rows = training_rows()
    write_new(directory / "task.json", TASK)
    records = directory / "records.jsonl"
    with records.open("x", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n")

    tree_file = directory / "tree.json"
    bert_dir = directory / "bert"
    write_new(tree_file, train(TASK, rows, **TREE_RECIPE))
    train_bert(TASK, rows, base, bert_dir, **BERT_RECIPE)
    models = {"tree": load_model(tree_file), "bert": load_model(bert_dir)}
    records_digest = digest(sorted(rows, key=lambda r: r["id"]))
    for model in models.values():
        if model["training"]["records_sha256"] != records_digest:
            raise ValueError("model training records differ from the recorded rows")
    bert = models["bert"]
    optimization = bert["optimization"]
    if (optimization["optimizer_steps"] < 1
            or optimization["backbone_sha256_before"] == optimization["backbone_sha256_after"]):
        raise ValueError("BERT encoder weights did not change")

    counts, by_notice, overall = defaultdict(Counter), defaultdict(Counter), Counter()
    for row in rows:
        features, label = row["state"], row["target"]
        notice = features["notice"]
        counts[notice, features["peer_outcome"]][label] += 1
        by_notice[notice][label] += 1
        overall[label] += 1

    states = []
    for notice in (False, True):
        for peer_outcome in PEER_OUTCOMES:
            features = {"notice": notice, "peer_outcome": peer_outcome}
            group = counts[notice, peer_outcome]
            pool, source = (group, "state") if group else (by_notice[notice], "notice")
            if not pool:
                pool, source = overall, "global"
            scores = {"empirical": (pool["prefer"] + 1) / (pool.total() + 2)}
            # The numeric-only workshop roster helper cannot represent a category.
            # The general predictor exposes the same raw preference distribution.
            for name, model in models.items():
                scores[name] = predict(model, features, TASK["labels"])["scores"]["prefer"]
            states.append({**features, "train_rows": group.total(), "train_prefer": group["prefer"],
                           "empirical_source": source, "scores": scores})

    fits = {"tree": {"checkpoint": "fits/tree.json", "model_sha256": models["tree"]["sha256"],
                     "recipe": TREE_RECIPE},
            "bert": {"checkpoint": "fits/bert", "model_sha256": bert["sha256"], "recipe": BERT_RECIPE,
                     "base": bert["base"]["model"], "base_revision": bert["base"]["revision"],
                     "backbone_sha256_before": optimization["backbone_sha256_before"],
                     "backbone_sha256_after": optimization["backbone_sha256_after"],
                     "optimizer_steps": optimization["optimizer_steps"],
                     "decision_head_sha256": bert["artifacts"]["decision_head.safetensors"]}}
    receipt = {"groups": len({row["group"] for row in rows}), "training_rows": len(rows),
               "records": "fits/records.jsonl", "records_sha256": records_digest,
               "records_file_sha256": hashlib.sha256(records.read_bytes()).hexdigest(),
               "fits": fits, "states": states}
    write_new(directory / "receipt.json", receipt)
    # Two fields exhaust the observation vocabulary, so scoring each state once
    # is equivalent to invoking the loaded checkpoint for every file arrival.
    lookup = {(s["notice"], s["peer_outcome"]): s["scores"] for s in states}
    scorers = {name: (lambda features, key=name: lookup[features["notice"], features["peer_outcome"]][key])
               for name in ("empirical", "tree", "bert")}
    return scorers, receipt
