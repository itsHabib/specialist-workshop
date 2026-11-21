import json
import math
import sys
import unittest

from decision_workshop.tree import fit, predict_counts


def records(feature, pairs):
    return [{"state": {feature: value}, "target": label} for value, label in pairs]


class TreeTests(unittest.TestCase):
    def test_numeric_three_class_tree_round_trips(self):
        rows = records("x", [(1, "a"), (2, "a"), (4, "b"), (5, "b"), (8, "c"), (9, "c")])
        tree = fit({"x": "number"}, rows, ["a", "b", "c"])
        tree = json.loads(json.dumps(tree, allow_nan=False))
        self.assertEqual(tree["value"], 3)
        self.assertEqual(present_counts(tree, {"x": 1.5}), {"a": 2})
        self.assertEqual(present_counts(tree, {"x": 4.5}), {"b": 2})
        self.assertEqual(present_counts(tree, {"x": 10}), {"c": 2})

    def test_categories_and_unseen_value(self):
        rows = records("color", [("blue", "n"), ("green", "n"), ("red", "y")] * 2)
        tree = fit({"color": "category"}, rows, ["y", "n"])
        self.assertEqual((tree["op"], tree["value"]), ("==", "red"))
        self.assertEqual(present_counts(tree, {"color": "red"}), {"y": 2})
        self.assertEqual(present_counts(tree, {"color": "orange"}), {"n": 4})

    def test_limits_preserve_empirical_counts(self):
        rows = records("x", [(1, "a"), (1, "a"), (1, "b"), (2, "b")])
        for limits in ({"max_depth": 0}, {"min_leaf": 2}):
            tree = fit({"x": "number"}, rows, ["a", "b"], **limits)
            self.assertEqual(tree, {"counts": {"a": 2, "b": 2}})
        split = fit({"x": "number"}, rows, ["a", "b"], min_leaf=1)
        self.assertEqual(predict_counts(split, {"x": 1}), {"a": 2, "b": 1})

    def test_boolean_interaction_and_feature_ties(self):
        rows = [
            {"state": {"z": z, "a": a}, "target": str(z != a)}
            for z in (True, False) for a in (True, False)
        ]
        features = {"z": "boolean", "a": "boolean"}
        labels = ["True", "False"]
        tree = fit(features, rows, labels, min_leaf=1)
        self.assertEqual((tree["feature"], tree["value"]), ("a", False))
        self.assertEqual(tree, fit(dict(reversed(features.items())), rows[::-1], labels, min_leaf=1))
        for row in rows:
            self.assertEqual(present_counts(tree, row["state"]), {row["target"]: 1})

    def test_float_midpoint_preserves_partition(self):
        largest = sys.float_info.max
        for low, high in [
            (-largest, largest), (largest / 2, largest),
            (-largest, -largest / 2), (math.nextafter(largest, 0.0), largest),
            (0.0, math.nextafter(0.0, 1.0)),
        ]:
            with self.subTest(low=low, high=high):
                rows = records("x", [(low, "below"), (high, "above")])
                tree = fit({"x": "number"}, rows, ["below", "above"], min_leaf=1)
                restored = json.loads(json.dumps(tree, allow_nan=False))
                self.assertEqual(present_counts(restored, {"x": low}), {"below": 1})
                self.assertEqual(present_counts(restored, {"x": high}), {"above": 1})

    def test_pure_and_constant_data_stop(self):
        tree = fit({"x": "category"}, records("x", [("hot", "a")] * 4), ["a", "b"])
        self.assertEqual(tree, {"counts": {"a": 4, "b": 0}})
        rows = records("x", [(0, "a"), (0, "b")] * 3)
        self.assertEqual(fit({"x": "number"}, rows, ["a", "b"]), {"counts": {"a": 3, "b": 3}})


def present_counts(tree, state):
    return {key: value for key, value in predict_counts(tree, state).items() if value}


if __name__ == "__main__":
    unittest.main()
