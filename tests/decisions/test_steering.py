import unittest

from decision_workshop.core import train
from decision_workshop.steering import LABELS, rank_candidates


class SteeringTests(unittest.TestCase):
    def setUp(self):
        task = {"name": "priority", "version": "1", "labels": LABELS, "features": {"cost": "number"}}
        rows = [{"id": str(i), "group": str(i), "split": "train", "state": {"cost": i},
                 "target": "prefer" if i < 3 else "defer", "candidates": LABELS} for i in range(6)]
        self.model = train(task, rows)

    def test_unseen_handles_and_input_order_have_no_effect_on_distinct_candidates(self):
        original = [("old", {"cost": 1}), ("new", {"cost": 4}), ("third", {"cost": 2})]
        result = rank_candidates(self.model, original)
        self.assertEqual(result, [("old", 1), ("third", 1), ("new", 0)])
        self.assertEqual(result, rank_candidates(self.model, reversed(original)))
        renamed = [("x", {"cost": 4}), ("z", {"cost": 2}), ("y", {"cost": 1})]
        self.assertEqual(rank_candidates(self.model, renamed), [("y", 1), ("z", 1), ("x", 0)])

    def test_empty_duplicate_invalid_or_identical_candidates(self):
        self.assertEqual(rank_candidates(self.model, []), [])
        self.assertEqual(rank_candidates(self.model, [("a", {"cost": 0}), ("b", {"cost": 0})]),
                         [("a", 1), ("b", 1)])
        with self.assertRaises(ValueError):
            rank_candidates(self.model, [("a", {"cost": 0}), ("a", {"cost": 1})])
        with self.assertRaises(ValueError):
            rank_candidates(self.model, [("a", {"cost": float("nan")})])
