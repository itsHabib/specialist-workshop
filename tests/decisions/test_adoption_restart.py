"""Exercise the two-job adoption regression from the host's working directory."""

import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

from decision_workshop.core import predict, train, write_new
from decision_workshop.evaluation import evaluate
from decision_workshop.outcomes import compare_host_outcomes


TASK = {"name": "format", "version": "1", "features": {"compact_requested": "boolean"},
        "labels": ["compact", "indented"]}


def record(identifier, split, compact, target, candidates):
    return {"id": identifier, "group": identifier, "split": split,
            "state": {"compact_requested": compact}, "target": target, "candidates": candidates}


def execute(identifier, action):
    # Both host routines round-trip the payload. The emitting routine records its identity.
    def compact(value):
        return "compact", json.dumps(value, separators=(",", ":"))

    def indented(value):
        return "indented", json.dumps(value, indent=2)

    payload = {"job": identifier}
    executed, result = {"compact": compact, "indented": indented}[action](payload)
    return {"id": identifier, "executed": executed, "value": int(json.loads(result) == payload)}


class AdoptionRestartTests(unittest.TestCase):
    def test_restarted_policy_coverage_dispatch_and_verified_fallback(self):
        labels = TASK["labels"]
        practice = [record(str(i), "train", compact, target, labels)
                    for i, (compact, target) in enumerate([
                        (True, "compact"), (True, "compact"), (True, "compact"),
                        (False, "indented"), (False, "indented"), (False, "compact"),
                    ])]
        held_out = [record("explicit", "test", True, "compact", labels),
                    record("unrecorded", "test", True, "compact", None)]
        uncertain = record("low-score", "dev", False, "indented", labels)
        model = train(TASK, practice, min_score=0.8)
        before = evaluate(model, held_out, "test")

        with tempfile.TemporaryDirectory() as temporary:
            host = Path(temporary)
            package = Path(__file__).resolve().parents[2] / "decision_workshop"
            shutil.copytree(package, host / package.name, ignore=shutil.ignore_patterns("__pycache__"))
            write_new(host / "tree.json", model)
            (host / "held-out.jsonl").write_text(
                "".join(json.dumps(row) + "\n" for row in held_out), encoding="utf-8")

            def cli(*args):
                run = subprocess.run([sys.executable, "-E", "-S", "-m", "decision_workshop", *args],
                                     cwd=host, check=True, text=True, capture_output=True, timeout=15)
                return json.loads(run.stdout)

            summary = cli("evaluate", "--model", "tree.json", "--data", "held-out.jsonl",
                          "--split", "test", "--report", "test.json")
            self.assertEqual(before, json.loads((host / "test.json").read_text(encoding="utf-8")))
            headline = summary["model"]
            self.assertEqual([headline[k] for k in ("rows", "suggested", "matched", "wrong_suggestions",
                                                     "review", "coverage")], [2, 1, 1, 0, 1, 0.5])
            conditional = summary["model_with_recorded_candidates"]
            self.assertEqual((conditional["suggested"], conditional["rows"], conditional["coverage"]), (1, 1, 1))

            decisions = {}
            for row in [*held_out, uncertain]:
                request = {key: row[key] for key in ("state", "candidates")}
                write_new(host / (row["id"] + ".json"), request)
                decisions[row["id"]] = cli("predict", "--model", "tree.json", "--input", row["id"] + ".json")
                self.assertEqual(decisions[row["id"]], predict(model, **request))
            self.assertEqual(decisions["low-score"]["reasons"], ["below_min_score"])
            self.assertEqual(decisions["unrecorded"]["reasons"], ["candidates_unrecorded"])

        arrivals = [row["id"] for row in held_out]
        fallback = "indented"
        baseline = [execute(job, fallback) for job in arrivals]
        candidate = []
        for job in arrivals:
            decision = decisions[job]
            receipt = execute(job, decision["action"] or fallback)
            receipt["proposed"] = decision["action"]
            if decision["status"] == "review":
                receipt.update(source="fallback", fallback_reason=decision["reasons"][0])
            candidate.append(receipt)
        metric = {"metric": "round_trip_ok", "direction": "higher"}
        report = compare_host_outcomes(arrivals, baseline, candidate, **metric)
        self.assertEqual([case["candidate_check"] for case in report["cases"]], ["matched_prefix", "fallback"])
        self.assertEqual([case["dispatch"] for case in report["cases"]], ["changed", "same"])
        self.assertEqual(report["summary"]["paired_outcomes"], 2)
        self.assertEqual(report["summary"]["paired_mean_values"], {"baseline": 1, "candidate": 1})

        # A bypass still serializes valid JSON, so only the dispatch check catches it.
        bypass = {**candidate[0], **execute(arrivals[0], fallback)}
        control = compare_host_outcomes(arrivals, baseline, [bypass, candidate[1]], **metric)
        self.assertEqual(control["cases"][0]["candidate_check"], "mismatch")
        self.assertEqual(control["cases"][0]["outcome"], "equal")
