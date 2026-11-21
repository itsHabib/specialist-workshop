import unittest

from experiments.steering import rare_tests, test_selection as study


class TestSelectionTests(unittest.TestCase):
    def test_executed_failures_require_baseline_coverage_of_the_changed_line(self):
        for patch in study.patches("test"):
            known = [case for case in patch["cases"] if not study.passes(case["test"],patch["library"])]
            self.assertEqual(bool(known), not patch["invariant"])
            for case in known:
                self.assertTrue(case["failed"])
                self.assertGreater(case["features"]["changed_line_calls"],0)
            if not patch["invariant"]:
                remaining = dict(study.variants(patch))["failing_options_removed"]
                self.assertTrue(remaining)
                self.assertTrue(all(study.passes(r["test"],patch["library"]) for r in remaining))

    def test_renaming_opaque_ids_preserves_rule_order_and_prefix_detection(self):
        patch = next(study.patches("test"))
        views = dict(study.variants(patch))
        original = study.ordering("line_and_assertion",study.candidate_inputs(views["full"]),{})
        renamed = study.ordering("line_and_assertion",study.candidate_inputs(views["renamed"]),{})
        self.assertEqual([study.opaque("renamed:"+k) for k in original],renamed)
        by_id = {r["id"]:r for r in views["full"]}
        failing = next(r["id"] for r in views["full"] if r["failed"])
        passing = next(r["id"] for r in views["full"] if not r["failed"])
        self.assertIsNone(study.execute([passing,failing],by_id,patch["library"],1)["first_failure"])
        result = study.execute([passing,failing],by_id,patch["library"],5)
        self.assertEqual((result["first_failure"],result["tests_executed"]),(failing,2))
        self.assertGreaterEqual(result["host_cpu_ns"]*result["timing_batch_executions"],500000)

    def test_counter_shifts_and_clean_neighbor_feature(self):
        for offset in rare_tests.FRESH_OFFSETS:
            name,_,rows = rare_tests.suite(offset)
            self.assertEqual(len(rows),28)
            scores = rare_tests.clean_probes(study.LIBRARY[name],rows)
            exposed = set()
            for delta in (-1,1):
                library = {name:study.make_function(name,f"return (x+{offset+delta})//y")}
                failures = [r["id"] for r in rows if not study.passes(r["test"],library)]
                self.assertEqual(len(failures),1)
                exposed.update(failures)
            self.assertEqual(len(exposed),2)
            self.assertEqual(exposed,{test_id for test_id,score in scores.items() if score})
