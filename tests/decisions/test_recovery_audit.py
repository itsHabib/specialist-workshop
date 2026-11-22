import unittest
from copy import deepcopy

from experiments.decision_recovery.audit import AuditError, check_outcome, paired_outcomes


class RecoveryEvidenceTests(unittest.TestCase):
    def receipt(self):
        return {
            "request_id": "batch-7", "batches": [[-3, 5], [-2]], "expected_result": 0,
            "caller": {"request_id": "batch-7", "result": 0}, "elapsed": 3, "deadline": 3,
            "status": "verified", "event_count": 1, "duplicate_effects": 0,
            "ledger": {
                "requests": [{"request_id": "batch-7", "input_json": "[[-3,5],[-2]]",
                              "checkpoint_batches": 2, "subtotal": 0, "result": 0}],
                "completions": [{"request_id": "batch-7", "result": 0}],
            },
        }

    def test_exact_deadline_and_zero_result_are_success(self):
        self.assertTrue(check_outcome(self.receipt()))

    def test_independent_checker_rejects_wrong_results_keys_duplicates_and_unbacked_success(self):
        def wrong_total(r):
            r["caller"]["result"] = 1
            r["ledger"]["requests"][0]["result"] = 1
            r["ledger"]["completions"][0]["result"] = 1
            r["expected_result"] = 1

        def duplicate(r):
            r["ledger"]["completions"].append(deepcopy(r["ledger"]["completions"][0]))

        def phantom(r):
            r["ledger"]["requests"].clear()
            r["ledger"]["completions"].clear()
            r["event_count"] = 0

        def wrong_id(r):
            r["caller"]["request_id"] = "another-operation"

        def late(r):
            r["elapsed"] = r["deadline"] + 1

        for change in (wrong_total, duplicate, phantom, wrong_id, late):
            with self.subTest(change=change.__name__):
                receipt = self.receipt()
                change(receipt)
                with self.assertRaises(AuditError):
                    check_outcome(receipt)

    def test_pairing_rejects_missing_or_substituted_worlds_and_resamples_groups(self):
        rows = []
        for ident, group, challenger_wins in ((0, "g1", False), (1, "g1", False), (2, "g2", True)):
            for policy in ("control", "tree"):
                outcome = self.receipt()
                if challenger_wins != (policy == "tree"):
                    outcome.update(caller=None, status="failed")
                rows.append({"cohort": "ordinary", "case_id": ident, "group": group,
                             "scenario": {"key": ident}, "policy": policy, "outcome": outcome})
        result = paired_outcomes(rows, "tree", "control")
        self.assertEqual((result["win"], result["tie"], result["loss"]), (1, 0, 2))
        self.assertEqual(result["group_mean_difference"], 0.0)
        self.assertEqual(result["bootstrap_95pct"], [-1.0, 1.0])
        for damaged in (rows[:-1], rows + [rows[-1]], rows[:-1] + [{**rows[-1], "scenario": {"key": 3}}]):
            with self.assertRaises(AuditError):
                paired_outcomes(damaged, "tree", "control")

    def test_lost_reply_db_is_committed_before_response_and_identical_after_recovery(self):
        from pathlib import Path
        from tempfile import TemporaryDirectory
        from experiments.decision_recovery.audit import live_lost_reply

        with TemporaryDirectory() as tmp:
            recovered = live_lost_reply(Path(tmp) / "batch.sqlite")
        self.assertEqual(recovered["caller_after"], {"request_id": "audit-operation", "result": 0})
        self.assertEqual(len(recovered["durable_before"]["completions"]), 1)

    def test_physical_corruption_after_commit_is_detected_on_reopen(self):
        from pathlib import Path
        import sqlite3
        from tempfile import TemporaryDirectory
        from experiments.decision_recovery.audit import inspect_ledger, live_lost_reply, read_ledger

        with TemporaryDirectory() as temp:
            path = Path(temp) / "batch.sqlite"
            recovered = live_lost_reply(path)
            key = recovered["caller_after"]["request_id"]
            with sqlite3.connect(path) as db:
                db.execute("UPDATE completions SET result = 19 WHERE request_id = ?", (key,))
            with self.assertRaises(AuditError):
                inspect_ledger(key, [[7, -2], [-8], [3]], **read_ledger(path))

    def test_hidden_worlds_have_identical_observations_but_disjoint_winning_moves(self):
        from experiments.decision_recovery.audit import indistinguishable_pair

        pair = indistinguishable_pair()["first_actions"]
        winners = lambda name: {key for key, r in pair[name].items() if r["verified"]}
        self.assertEqual(winners("pending"), {"retry"})
        self.assertEqual(winners("committed"), {"lookup", "wait"})
