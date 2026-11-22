import json
import math
import sys
import unittest

from decision_workshop.tree import fit, predict_counts
from decision_workshop.core import digest, predict, train


def records(feature, pairs):
    return [{"state": {feature: value}, "target": label} for value, label in pairs]


class TreeTests(unittest.TestCase):
    def test_text_only_training_reports_a_single_leaf(self):
        task = {"name": "routing", "version": "1", "labels": ["A", "B"], "features": {"context": "text"}}
        rows = [
            dict(id=str(i), group=str(i), split="train", state={"context": context},
                 target=label, candidates=task["labels"])
            for i, (context, label) in enumerate([("route to A", "A"), ("send to A", "A"), ("route to B", "B")])
        ]
        with self.assertWarnsRegex(UserWarning, "no structured fields") as emitted:
            model = train(task, rows, min_leaf=1)
        diagnostics = model["tree_diagnostics"]
        self.assertEqual([str(w.message) for w in emitted.warnings], diagnostics["warnings"])
        self.assertIn("3 of 3 training rows", diagnostics["warnings"][1])
        self.assertEqual(diagnostics["split_features"], [])
        unseen = predict(model, {"context": "send to B"}, task["labels"], min_score=0.0)
        self.assertEqual(unseen["leaf_counts"], {"A": 2, "B": 1})

    def test_reports_only_features_that_split_and_counts_unique_text_over_training_rows(self):
        task = {"name": "xor", "version": "1", "labels": ["same", "different"],
                "features": {"a": "boolean", "b": "boolean", "constant": "number",
                             "notes": "text", "half_unique": "text"}}
        rows = []
        for i, (a, b) in enumerate([(True, True), (False, True), (False, False), (True, False)]):
            rows.append(dict(id=str(i), group=str(i), split="train", candidates=task["labels"],
                             target="same" if a == b else "different",
                             state={"a": a, "b": b, "constant": 1, "notes": str(i),
                                    "half_unique": str(i) if i < 2 else "repeated"}))
        # Duplicate each training text on dev. A diagnostic using both splits would miss the warning.
        dev = [{**row, "id": "dev-" + row["id"], "group": "dev", "split": "dev"} for row in rows]
        with self.assertWarnsRegex(UserWarning, "'notes'.*4 of 4 training rows"):
            model = train(task, rows + dev, min_leaf=1)
        self.assertEqual(model["tree_diagnostics"]["split_features"], ["a", "b"])
        self.assertEqual(len(model["tree_diagnostics"]["warnings"]), 1)

    def test_balanced_ties_are_opt_in_and_preserve_the_original_fit(self):
        labels = list("abcd")
        task = {"name": "ordered", "version": "1", "labels": labels, "features": {"x": "number"}}
        rows = [dict(id=str(i), group=str(i), split="train", state={"x": i}, target=labels[i // 2],
                     candidates=labels) for i in range(8)]
        original = train(task, rows, max_depth=1)
        legacy_fields = ("algorithm", "format", "task", "settings", "training", "tree")
        self.assertEqual(digest({k: original[k] for k in legacy_fields}),
                         "d64a6938f7d323091b9b29d927a97ca5602c5f03416eb9f8f834ec70ca50bf08")
        self.assertEqual(train(task, rows, max_depth=1, balanced_ties=False), original)
        self.assertEqual(original["tree"]["value"], 1.5)
        balanced = train(task, rows, max_depth=1, balanced_ties=True)
        self.assertEqual(balanced["tree"]["value"], 3.5)
        self.assertTrue(balanced["settings"]["balanced_ties"])

    def test_balancing_preserves_strictly_better_gain_and_equal_balance_order(self):
        # The pure 2:6 split beats a more evenly sized, mixed split.
        rows = records("x", enumerate("aabbbbbb"))
        best = fit({"x": "number"}, rows, ["a", "b"], max_depth=1, balanced_ties=True)
        self.assertEqual(best["value"], 1.5)
        # Splitting three equal bands yields two equivalent 2:4 or 4:2 choices.
        rows = records("x", enumerate("aabbcc"))
        tied = fit({"x": "number"}, rows, list("abc"), max_depth=1, balanced_ties=True)
        self.assertEqual(tied["value"], 1.5)

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
