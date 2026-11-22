import copy
import unittest

from experiments.import_recovery import audit, host


class ImportRecoveryTests(unittest.TestCase):
    def setUp(self):
        self.rows = (("T1", 'Café, "one"\nroom', "4"),)
        self.wire = 'ticket,description,units\r\nT1,"Café, ""one""\nroom",4\r\n'
        self.job = host.ImportJob("example", "quoted", self.wire.encode(), self.rows, "comma", "comma")

    def assert_parses(self, job, route, action, status, exact):
        result, rows = host.attempt(job, route, action)
        self.assertEqual((result, result == "ok" and rows == job.expected), (status, exact))
        self.assertEqual(audit.decode(job, route, action), (status, exact))

    def test_exact_import_and_mojibake_share_parser_success(self):
        self.assert_parses(self.job, "comma", None, "ok", True)
        self.assert_parses(self.job, "comma", "switch_encoding", "ok", False)
        self.assert_parses(self.job, "semicolon", "switch_delimiter", "ok", True)
        result, data = host.attempt(self.job, "comma", "switch_encoding")
        self.assertEqual(result, "ok")
        self.assertIn("CafÃ©", data[0][1])

    def test_two_errors_exhaust_the_retry_budget_and_bad_values_require_review(self):
        legacy = host.ImportJob("legacy", "quoted", self.wire.encode("cp1252"),
                                self.rows, "comma", "semicolon")
        self.assert_parses(legacy, "semicolon", None, "decode_error", False)
        self.assert_parses(legacy, "semicolon", "switch_encoding", "shape_error", False)
        self.assert_parses(legacy, "semicolon", "switch_delimiter", "decode_error", False)
        self.assert_parses(legacy, "comma", "switch_encoding", "ok", True)
        malformed = host.ImportJob("corrupt", "quoted", self.wire.replace(",4", ",unknown").encode(),
                                   self.rows, "comma", "comma")
        self.assert_parses(malformed, "comma", None, "value_error", False)

    def sample_receipts(self):
        receipts, tables = [], []
        counts = {"jobs": 1, "first_exact": 1, "final_exact": 1, "attainable": 1, "flagged": 0,
                  "repair_calls": 0, "regressed": 0, "recovered": 0, "parser_accepted_wrong": 0}
        for arm in audit.ARMS:
            setting = {"split": "dev", "seed": 7, **dict(zip(audit.POLICY_KEYS, arm))}
            receipts.append({**setting, "family": "quoted", "job_id": "example", "route": "comma",
                             "action": None, "first_status": "ok", "final_status": "ok",
                             "first_exact": True, "final_exact": True, "attainable": True})
            for family in ("quoted", "all"):
                tables.append({**setting, "family": family, **counts})
        return receipts, {"seeds": [7], "tables": tables}

    def test_information_ceiling_counts_observable_error_classes(self):
        def options(status, exact, delimiter, encoding):
            return {None: (status, exact), "switch_delimiter": ("ok", delimiter),
                    "switch_encoding": ("ok", encoding)}
        cases = [options("ok", True, False, False), options("ok", False, False, True),
                 options("shape_error", False, True, False), options("decode_error", False, False, True),
                 options("decode_error", False, True, False), options("decode_error", False, False, True)]
        self.assertEqual(audit.observation_ceiling(cases), {"jobs": 6, "first_exact": 1,
                         "coarse": 3, "diagnostic": 4, "per_case": 5})

    def test_audit_rejects_a_missing_duplicate_or_corrupted_receipt_and_wrong_totals(self):
        rows, summary = self.sample_receipts()
        verify = lambda records, report: audit.check(records, report, {"dev": [self.job]})
        self.assertEqual(verify(rows, summary)["verified_receipts"], 48)
        for altered in (rows[1:], rows + [rows[0]],
                        [dict(rows[0], final_exact=False), *rows[1:]],
                        [dict(rows[0], action="switch_encoding"), *rows[1:]]):
            with self.subTest(size=len(altered)), self.assertRaises(ValueError):
                verify(altered, summary)
        wrong = copy.deepcopy(summary)
        wrong["tables"][0]["final_exact"] = 0
        with self.assertRaisesRegex(ValueError, "final_exact"):
            verify(rows, wrong)
