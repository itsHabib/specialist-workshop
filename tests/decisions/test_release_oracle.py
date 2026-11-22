"""Check the release grader against exhaustive conditional-loss tables."""

from functools import lru_cache
import unittest

from experiments.decision_limits.release import quotes, records


@lru_cache(maxsize=None)
def policies(remaining, reliability, price):
    # (first move, conditional cost if faulty, conditional cost if healthy).
    values = [("publish", 30, 0), ("rollback", 0, 10), ("review", 3, 3)]
    if remaining:
        children = policies(remaining - 1, reliability, price)
        for _, bad_pos, good_pos in children:
            for _, bad_neg, good_neg in children:
                values.append(("diagnose", price + reliability * bad_pos + (1 - reliability) * bad_neg,
                               price + (1 - reliability) * good_pos + reliability * good_neg))
    return values


class ReleaseOracleTest(unittest.TestCase):
    def test_every_generated_state_against_all_contingent_policies(self):
        for row in records():
            s = row["state"]
            q = s["diagnostic_accuracy_pct"] / 100
            prior = s["prior_fault_pct"] / 100
            odds = prior / (1 - prior) * (q / (1 - q)) ** (s["positive_checks"] - s["negative_checks"])
            p = odds / (1 + odds)
            horizon = 2 - s["positive_checks"] - s["negative_checks"]
            brute = {}
            for move, bad, good in policies(horizon, q, s["diagnostic_cost"]):
                value = p * bad + (1 - p) * good
                brute[move] = min(brute.get(move, float("inf")), value)
            grader = quotes(s)
            self.assertEqual(set(brute), set(grader))
            for move, cost in brute.items():
                self.assertAlmostEqual(cost, grader[move], places=9, msg=f"{row['id']} {move}")
