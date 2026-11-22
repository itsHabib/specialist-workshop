"""Train and grade a decision tree on the pinned, historical Paint Control dataset."""

import argparse
from collections import defaultdict, deque
import csv
from dataclasses import dataclass, field
import hashlib
import os
from pathlib import Path
from statistics import mean
from tempfile import TemporaryDirectory

from decision_workshop import contracts, core, steering, tree


CSV_SHA256 = "7b2aa489f944177b8d0ac3c461dfa10846fad5465d66e1ffa0ab3c9fd88157b0"
CSV_URL = ("https://api.bitbucket.org/2.0/repositories/HelgeS/retecs/src/"
           "ae0b732228e1c1d09005b02aae351142ff43ec8b/DATA/paintcontrol.csv")
TRAIN_END, DEV_END, TEST_END = 211, 281, 352
RULES = ("numeric_id", "recent_failure", "prior_rate", "recent_rate", "last_fraction")
FEATURES = ("observed_cycles", "cycles_since_seen", "cycles_since_failure",
            "prior_rate", "recent_rate", "last_fraction")
TASK = {"name": "paint-control-priority", "version": "1",
        "features": dict.fromkeys(FEATURES, "number"), "labels": list(steering.LABELS)}


def file_sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_cycles(path, end):
    if file_sha(path) != CSV_SHA256:
        raise ValueError("input differs from the pinned Paint Control CSV")
    cycles = defaultdict(lambda: defaultdict(list))
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle, delimiter=";"):
            cycle = int(row["Cycle"])
            if cycle <= end:
                cycles[cycle][int(row["Name"])].append(int(row["Verdict"]))
    if set(cycles) != set(range(1, end + 1)):
        raise ValueError("missing or unexpected cycles")
    return cycles


@dataclass
class History:
    observed_cycles: int = 0
    failure_sum: float = 0
    last_seen: int = 0
    last_failure: int = 0
    recent: deque = field(default_factory=lambda: deque(maxlen=5))

    def features(self, cycle):
        return {"observed_cycles": self.observed_cycles,
                "cycles_since_seen": cycle - self.last_seen,
                "cycles_since_failure": cycle - self.last_failure,
                "prior_rate": (self.failure_sum + 1) / (self.observed_cycles + 2),
                "recent_rate": mean(self.recent) if self.recent else 0.5,
                "last_fraction": self.recent[-1] if self.recent else 0.5}

    def observe(self, cycle, verdicts):
        # Weight each earlier observed cycle once, including days with retries.
        fraction = mean(verdicts)
        self.observed_cycles += 1
        self.failure_sum += fraction
        self.recent.append(fraction)
        self.last_seen = cycle
        if fraction:
            self.last_failure = cycle


def snapshots(cycles):
    history = defaultdict(History)
    for cycle, observations in sorted(cycles.items()):
        states = {name: history[name].features(cycle) for name in sorted(observations)}
        yield cycle, states, observations
        for name, verdicts in observations.items():
            history[name].observe(cycle, verdicts)


def cycle_status(observations):
    if any(len(outcomes) != 1 for outcomes in observations.values()):
        return "repeated_names"
    failures = sum(outcomes[0] for outcomes in observations.values())
    if failures == 0:
        return "all_pass"
    if failures == len(observations):
        return "all_fail"
    return "mixed"


def order(names_and_features, policy, model=None):
    if policy == "tree":
        scored = steering.rank_candidates(model, ((str(n), s) for n, s in names_and_features.items()))
        scores = {int(name): value for name, value in scored}
    elif policy == "numeric_id":
        return sorted(names_and_features)
    elif policy == "recent_failure":
        scores = {name: -s["cycles_since_failure"] for name, s in names_and_features.items()}
    else:
        scores = {name: s[policy] for name, s in names_and_features.items()}
    return sorted(scores, key=lambda name: (-scores[name], name))


def grade(cycles, start, policies, model=None):
    receipts = []
    for cycle, states, observations in snapshots(cycles):
        if cycle < start:
            continue
        status = cycle_status(observations)
        receipt = {"cycle": cycle, "status": status, "names": len(observations),
                   "executions": sum(map(len, observations.values())),
                   "disagreeing_names": sum(len(set(v)) > 1 for v in observations.values())}
        if status != "repeated_names":
            failures = {name for name, outcomes in observations.items() if outcomes[0]}
            receipt["failed_names"] = sorted(failures)
            receipt["orders"] = {policy: order(states, policy, model) for policy in policies}
            receipt["ranks"] = {
                policy: next((i for i, name in enumerate(ranked, 1) if name in failures), None)
                for policy, ranked in receipt["orders"].items()}
        receipts.append(receipt)
    return receipts


def mixed_rank_means(receipts, policies):
    mixed = [r for r in receipts if r["status"] == "mixed"]
    return {policy: mean(r["ranks"][policy] for r in mixed) for policy in policies}


def protocol():
    modules = [Path(__file__), *(Path(m.__file__) for m in (core, contracts, tree, steering))]
    return {"experiment": "paint-control-v1", "data_sha256": CSV_SHA256, "source": CSV_URL,
            "implementation": {p.name: file_sha(p) for p in modules},
            "train": [1, TRAIN_END], "dev": [TRAIN_END + 1, DEV_END],
            "held_out_cycles": list(range(DEV_END + 1, TEST_END + 1)),
            "features": list(FEATURES), "training_eligibility": "one observation per name in the cycle",
            "tree_settings": {"max_depth": 5, "min_leaf": 20, "balanced_ties": True},
            "controls_in_selection_tie_order": list(RULES), "score_tie": "ascending numeric Name",
            "selection_metric": "mean first-failure rank on unique-name cycles containing pass and fail"}


def freeze(csv_path, run):
    if os.path.lexists(run):
        raise FileExistsError(run)
    spec = protocol()
    cycles = load_cycles(csv_path, DEV_END)
    training = []
    for cycle, states, observations in snapshots(cycles):
        if cycle > TRAIN_END:
            break
        if cycle_status(observations) == "repeated_names":
            continue
        for name, state in states.items():
            target = "prefer" if observations[name][0] else "defer"
            training.append({"id": f"{cycle}/{name}", "group": str(cycle), "split": "train",
                             "state": state, "target": target, "candidates": steering.LABELS})
    contracts.validate_records(TASK, training)
    model = core.train(TASK, training, **spec["tree_settings"])
    controls = mixed_rank_means(grade(cycles, TRAIN_END + 1, RULES), RULES)
    winner = min(RULES, key=controls.get)
    manifest = {"protocol": spec, "checkpoint_sha256": model["sha256"], "control": winner,
                "development_mean_ranks": controls, "training_rows": len(training),
                "training_cycles": len({r["group"] for r in training})}
    run.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=run.parent, prefix=".paint-freeze-") as tmp:
        staged = Path(tmp) / "run"
        core.write_new(staged / "tree.json", model)
        core.write_new(staged / "freeze.json", {**manifest, "sha256": core.digest(manifest)})
        if os.path.lexists(run):
            raise FileExistsError(run)
        staged.rename(run)


def _holdout_result(csv_path, run):
    manifest = contracts.read_json(run / "freeze.json")
    signature = manifest.pop("sha256")
    if signature != core.digest(manifest) or manifest["protocol"] != protocol():
        raise ValueError("frozen protocol changed")
    model = core.load_model(run / "tree.json")
    if model["sha256"] != manifest["checkpoint_sha256"]:
        raise ValueError("model differs from frozen checkpoint")
    control = manifest["control"]
    policies = tuple(dict.fromkeys(("numeric_id", control, "tree")))
    receipts = grade(load_cycles(csv_path, TEST_END), DEV_END + 1, policies, model)
    means = mixed_rank_means(receipts, policies)
    deltas = [r["ranks"][control] - r["ranks"]["tree"] for r in receipts if r["status"] == "mixed"]
    counts = {s: sum(r["status"] == s for r in receipts)
              for s in ("mixed", "all_fail", "all_pass", "repeated_names")}
    return {"frozen_sha256": signature, "control": control, "all_cycles": len(receipts),
            "cycle_counts": counts, "mixed_mean_ranks": means,
            "mixed_tree_vs_control": {"earlier": sum(d > 0 for d in deltas),
                                      "equal": deltas.count(0), "later": sum(d < 0 for d in deltas)},
            "receipts": receipts}


def evaluate(csv_path, run):
    receipt = run / "holdout.json"
    if os.path.lexists(receipt):
        raise FileExistsError(receipt)
    # Keep the lock file in place so all contenders open the same inode.
    # Closing its handle releases the kernel lock, including on process exit.
    with (run / ".holdout.lock").open("a+b") as lock:
        if os.name == "nt":
            import msvcrt
            lock.seek(0)
            msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if os.path.lexists(receipt):
            raise FileExistsError(receipt)
        core.write_new(receipt, _holdout_result(csv_path, run))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["freeze", "evaluate"])
    parser.add_argument("--csv", type=Path, required=True, help="local copy of the pinned public CSV")
    parser.add_argument("--run", type=Path, required=True, help="new run directory for freeze")
    args = parser.parse_args()
    if args.stage == "freeze":
        freeze(args.csv, args.run)
        print(args.run / "freeze.json")
    else:
        evaluate(args.csv, args.run)
        print(args.run / "holdout.json")


if __name__ == "__main__":
    main()
