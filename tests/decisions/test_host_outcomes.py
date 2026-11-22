import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from decision_workshop.outcomes import compare_host_outcomes


def compare(ids, baseline, candidate):
    return compare_host_outcomes(ids, baseline, candidate, metric="wait_seconds", direction="lower")


class HostOutcomeTests(unittest.TestCase):
    def test_mixed_numeric_receipts_preserve_closeness_and_huge_integral_differences(self):
        for original, replacement, difference in [(2**53 + 1, float(2**53), 1), (10**400, 0.0, 10**400)]:
            with self.subTest(original=original):
                r = compare(["a"], [{"id": "a", "value": original, "elapsed_seconds": original}],
                            [{"id": "a", "value": replacement, "elapsed_seconds": replacement}])
                self.assertEqual(r["cases"][0]["outcome"], "better")
                self.assertIs(type(r["cases"][0]["improvement"]), int)
                self.assertEqual(r["cases"][0]["improvement"], difference)
                self.assertEqual(r["summary"]["mean_improvement_on_pairs"], difference)
                self.assertEqual(r["summary"]["timing"]["mean_seconds_saved_on_pairs"], difference)
                json.dumps(r, allow_nan=False)

    def test_averaging_preserves_cancellation_and_rejects_out_of_range_means(self):
        # Rounding each difference to float before averaging erases the 0.25.
        r = compare(["a", "b"], [{"id": "a", "value": 2**60}, {"id": "b", "value": 0.5}],
                    [{"id": "a", "value": 0.0}, {"id": "b", "value": 2**60}])
        self.assertEqual(r["summary"]["mean_improvement_per_arrival"], 0.25)
        for left, right in [(10**400, 1), (5e-324, 0)]:
            values = [{"id": "a", "value": left}, {"id": "b", "value": right}]
            with self.subTest(left=left), self.assertRaisesRegex(ValueError, "baseline mean value"):
                compare(["a", "b"], values, values)

    def test_cli_reports_missing_values_and_preserves_existing_report(self):
        with tempfile.TemporaryDirectory() as tmp:
            bundle, output = Path(tmp) / "receipts.json", Path(tmp) / "report.json"
            data = {"arrivals": ["one"], "baseline": [{"id": "one", "value": 1}],
                    "candidate": [{"id": "one", "missing_reason": "verifier stopped"}],
                    "direction": "higher", "metric": "verified_success"}
            bundle.write_text(json.dumps(data))
            command = [sys.executable, "-m", "decision_workshop", "compare-host",
                       "--input", str(bundle), "--report", str(output)]
            proc = subprocess.run(command, capture_output=True, text=True, check=True)
            saved = output.read_bytes()
            report = json.loads(saved)
            self.assertEqual(json.loads(proc.stdout)["summary"], report["summary"])
            self.assertEqual(report["cases"][0]["missing_outcomes"], {"candidate": "verifier stopped"})
            self.assertIsNone(report["summary"]["mean_improvement_per_arrival"])
            repeated = subprocess.run(command, capture_output=True, text=True)
            self.assertNotEqual(repeated.returncode, 0)
            self.assertIn("error:", repeated.stderr)
            self.assertEqual(output.read_bytes(), saved)
            data["candidate"] = None
            bundle.write_text(json.dumps(data))
            invalid = subprocess.run(command, capture_output=True, text=True)
            self.assertNotEqual(invalid.returncode, 0)
            self.assertIn("must be arrays", invalid.stderr)
            self.assertEqual(output.read_bytes(), saved)

    def test_roster_includes_missing_receipts_and_unknown_counterfactuals(self):
        out = compare(["both", "one", "unknown", "absent"],
                      [{"id": "both", "executed": "native", "value": 0},
                       {"id": "one", "executed": [], "value": 4},
                       {"id": "unknown", "value": 3, "elapsed_seconds": 8}],
                      [{"id": "both", "proposed": None, "source": "fallback", "executed": "native",
                        "value": 0, "fallback_reason": "missing candidates"},
                       {"id": "unknown", "missing_reason": "endpoint was not recorded",
                        "elapsed_seconds": 10}])
        s = out["summary"]
        self.assertEqual([c["id"] for c in out["cases"]], ["both", "one", "unknown", "absent"])
        self.assertEqual([c["dispatch"] for c in out["cases"]], ["same", "unknown", "unknown", "unknown"])
        self.assertEqual(s["outcomes"], {"better": 0, "equal": 1, "worse": 0, "unknown": 3})
        self.assertEqual(s["paired_outcomes"], 1)
        self.assertEqual(s["paired_mean_values"], {"baseline": 0, "candidate": 0})
        self.assertIsNone(s["mean_improvement_per_arrival"])
        self.assertEqual(s["candidate"]["missing_receipts"], 2)
        self.assertEqual(s["candidate"]["explicit_no_suggestion"], 1)
        self.assertEqual(s["candidate"]["unknown_proposals"], 3)
        self.assertEqual(s["timing"], {"paired": 1, "missing_pairs": 3, "mean_seconds_saved_on_pairs": -2})
        self.assertEqual(out["cases"][2]["missing_outcomes"], {"candidate": "endpoint was not recorded"})
        self.assertEqual(out["cases"][3]["missing_outcomes"],
                         {"baseline": "receipt_missing", "candidate": "receipt_missing"})

    def test_changes_to_unexecuted_tail_and_explicit_fallback_have_no_dispatch_effect(self):
        base = [{"id": "tail", "proposed": ["a", "b", "c"], "executed": ["a"], "value": 2},
                {"id": "bypass", "executed": "x", "value": 4},
                {"id": "fallback", "executed": "x", "value": 6}]
        new = [{"id": "tail", "proposed": ["a", "c", "b"], "executed": ["a"], "value": 2},
               {"id": "bypass", "source": "policy", "proposed": "y", "executed": "x", "value": 4},
               {"id": "fallback", "source": "fallback", "proposed": "y", "executed": "x", "value": 6,
                "fallback_reason": "host budget expired"}]
        r = compare(["tail", "bypass", "fallback"], base, reversed(new))
        self.assertEqual([c["candidate_check"] for c in r["cases"]], ["matched_prefix", "mismatch", "fallback"])
        self.assertEqual(r["summary"]["by_dispatch"]["same"]["equal"], 3)
        self.assertEqual(r["summary"]["mean_improvement_per_arrival"], 0)

    def test_separate_outcomes_when_dispatch_changed_or_stayed_the_same(self):
        base = [{"id": str(i), "executed": "a", "value": 5} for i in range(4)]
        new = [{"id": str(i), "executed": action, "value": value}
               for i, (action, value) in enumerate([("b", 4), ("b", 7), ("a", 3), ("a", 5)])]
        r = compare([v["id"] for v in base], base, new)["summary"]
        self.assertEqual(r["by_dispatch"]["changed"], {"better": 1, "worse": 1, "equal": 0, "unknown": 0})
        self.assertEqual(r["by_dispatch"]["same"], {"better": 1, "worse": 0, "equal": 1, "unknown": 0})
        self.assertEqual(r["mean_improvement_per_arrival"], 0.25)
        other = compare_host_outcomes([v["id"] for v in base], base, new, metric="value", direction="higher")
        self.assertEqual(other["summary"]["mean_improvement_per_arrival"], -0.25)

    def test_empty_executions_and_unobserved_dispatch_stay_distinct(self):
        ids = ["empty", "unknown", "repeat"]
        old = [{"id": "empty", "executed": []}, {"id": "unknown", "executed": []},
               {"id": "repeat", "executed": ["retry", "retry"]}]
        new = [{"id": "empty", "proposed": ["x"], "executed": []},
               {"id": "unknown", "proposed": ["x"]},
               {"id": "repeat", "proposed": ["retry", "repair"], "executed": ["retry", "retry"]}]
        r = compare(ids, old, new)
        self.assertEqual([c["candidate_check"] for c in r["cases"]], ["no_starts", "unknown", "mismatch"])
        self.assertEqual([c["dispatch"] for c in r["cases"]], ["same", "unknown", "same"])
        self.assertIsNone(compare([], [], [])["summary"]["mean_improvement_on_pairs"])

    def test_reject_invalid_joins_receipts_and_metrics(self):
        cases = [(["a", "a"], []), ([""], []), ("abc", []), (None, []), ({"a": 1}, []),
                 (["a"], None), (["a"], ""), (["a"], {}), (["a"], [{"id": "other"}]),
                 (["a"], [{"id": "a"}, {"id": "a"}])]
        for field, value in [("executed", ["x", ""]), ("proposed", 7), ("value", True),
                             ("value", float("nan")), ("elapsed_seconds", -1),
                             ("elapsed_seconds", float("inf")), ("source", "other"),
                             ("missing_reason", ""), ("outcomee", 1)]:
            cases.append((["a"], [{"id": "a", field: value}]))
        for ids, receipts in cases:
            with self.subTest(ids=ids, receipts=receipts), self.assertRaises(ValueError):
                compare(ids, [], receipts)
        for metric, direction in [("", "lower"), ("wait", "down")]:
            with self.subTest(metric=metric), self.assertRaises(ValueError):
                compare_host_outcomes([], [], [], metric=metric, direction=direction)
