import unittest

from experiments.decision_limits.release import (
    Episode, SIZES, episodes, expected_costs, incumbent, optimal, oracle, posterior, records, rollout, summarize,
)


class ReleaseTests(unittest.TestCase):
    def test_test_value_changes_at_fixed_prior(self):
        sharp = expected_costs(.25, .90, 1, 1)
        noisy = expected_costs(.25, .60, 1, 1)
        self.assertAlmostEqual(sharp["diagnose"], 2.5)
        self.assertAlmostEqual(noisy["diagnose"], 4)
        self.assertEqual(optimal(sharp), ["diagnose"])
        self.assertEqual(optimal(noisy), ["review"])

    def test_conditional_repeat_value_and_tie(self):
        sample = Episode("case", 25, 90, 1, True, (True, False))
        p = posterior(sample.state(1))
        self.assertAlmostEqual(p, .75)
        costs = expected_costs(p, .9, 1, 1)
        self.assertEqual(optimal(costs), ["diagnose"])
        self.assertAlmostEqual(costs["diagnose"], 2.15)
        negative = Episode("other", 25, 90, 1, False, (False, True))
        p = posterior(negative.state(1))
        self.assertAlmostEqual(p, 1 / 28)
        costs = expected_costs(p, .9, 1, 1)
        self.assertEqual(optimal(costs), ["publish"])
        self.assertAlmostEqual(costs["publish"], 15 / 14)
        self.assertAlmostEqual(costs["diagnose"], 1.492857142857143)
        self.assertEqual(optimal(expected_costs(.1, .9, 1, 0)), ["publish", "review"])

    def test_repeated_observations_and_realized_loss(self):
        source = Episode("case", 25, 90, 1, True, (True, False))
        trace = rollout(source, oracle)
        self.assertEqual([s["executed"] for s in trace["steps"]], ["diagnose", "diagnose", "review"])
        self.assertEqual(trace["total_cost"], 5)
        summary = summarize([trace])
        self.assertEqual(summary["oracle_justified_reviews"], 1)
        self.assertEqual(summary["policy_deferrals"], 0)
        self.assertEqual(summary["unnecessary_diagnostics"], 0)
        self.assertEqual(summary["mean_expected_regret"], 0.0)

    def test_known_missing_optimum_is_separate_from_optimal_review(self):
        episode = Episode("case", 25, 90, 1, True, (True, False))
        miss = rollout(episode, incumbent, omit=True)
        self.assertEqual(miss["steps"][0]["omitted_action"], "diagnose")
        stats = summarize([miss])
        self.assertEqual(stats["policy_deferrals"], 1)
        self.assertEqual(stats["oracle_justified_reviews"], 1)
        self.assertEqual(stats["total_cost"], 3)
        self.assertIsNone(stats["accuracy_when_suggested"])

    def test_oracle_expected_cost_matches_exact_complete_enumeration(self):
        from itertools import product
        total = 0
        for fault, a, b in product((False, True), repeat=3):
            prob = .25 if fault else .75
            prob *= .9 if a == fault else .1
            prob *= .9 if b == fault else .1
            trace = rollout(Episode("case", 25, 90, 1, fault, (a, b)), oracle)
            total += prob * trace["total_cost"]
        self.assertAlmostEqual(total, 2.395)
        self.assertAlmostEqual(total, min(expected_costs(.25, .9, 1, 2).values()))

    def test_grouped_prefixed_records_and_shifted_reliability(self):
        all_rows = list(records())
        self.assertEqual(len(all_rows), 3 * sum(SIZES.values()))
        self.assertEqual(len({r["id"] for r in all_rows}), len(all_rows))
        group_splits = {}
        for row in all_rows:
            self.assertEqual(group_splits.setdefault(row["group"], row["split"]), row["split"])
            self.assertNotIn("faulty", row["state"])
        self.assertEqual(list(episodes("iid")), list(episodes("iid")))
        self.assertTrue({ep.accuracy for ep in episodes("iid")}.isdisjoint(
                         ep.accuracy for ep in episodes("shift")))
