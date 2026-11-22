import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from decision_workshop.core import train, write_new
from decision_workshop.policy import save_policy
from experiments.steering.pytest_history import ADAPTER, FALLBACK, candidate_features, order_tests


class PytestOrderTest(unittest.TestCase):
    def test_new_test_keeps_its_slot_and_missing_history_uses_native_order(self):
        native = ["tests/slow.py::test_a", "tests/new.py::test_b", "tests/changed.py::test_c"]
        changed = {"tests/changed.py"}
        history = {native[0]: 0, native[2]: 0}
        task = {"name": "pytest-order", "version": "1", "labels": ["prefer", "defer"],
                "features": dict.fromkeys(candidate_features(native[0], changed, history), "number")}
        rows = [{"id": str(i), "group": "training", "split": "train", "candidates": task["labels"],
                 "state": candidate_features(node, changed, history), "target": label}
                for i, (node, label) in enumerate([(native[0], "defer"), (native[2], "prefer")])]
        with tempfile.TemporaryDirectory() as tmp:
            bundle = Path(tmp)
            write_new(bundle / "tree.json", train(task, rows, min_leaf=1))
            pin = save_policy(bundle / "policy.json", bundle / "tree.json", kind="ranking",
                              adapter_revision=ADAPTER, fallback_revision=FALLBACK)
            for counts, sha, source, order in [
                    (history, pin, "policy", [native[2], native[1], native[0]]),
                    ({}, pin, "fallback", native), ({native[2]: -1}, pin, "fallback", native),
                    (history, None, "fallback", native)]:
                with self.subTest(counts=counts, sha=sha):
                    items = [SimpleNamespace(nodeid=node) for node in native]
                    decision = order_tests(items, bundle, policy_sha256=sha, changed_paths=changed,
                                           prior_failures=counts)
                    self.assertEqual(decision["source"], source)
                    self.assertEqual([item.nodeid for item in items], order)
                    self.assertEqual(decision["requested_order"], order)
            for node_ids, paths, counts, directory in [
                    (native, changed, {**history, None: 0}, bundle),
                    ([None, *native], changed, history, bundle),
                    (native, {"", *changed}, history, bundle),
                    (native, changed, history, None)]:
                with self.subTest(node_ids=node_ids, paths=paths, directory=directory):
                    items = [SimpleNamespace(nodeid=node) for node in node_ids]
                    result = order_tests(items, directory, policy_sha256=pin,
                                         changed_paths=paths, prior_failures=counts)
                    self.assertEqual(result["source"], "fallback")
                    self.assertEqual([item.nodeid for item in items], node_ids)
