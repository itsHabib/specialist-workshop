"""Example pytest host adapter for a tree trained on prior test results."""

from pathlib import Path

from decision_workshop.policy import load_policy


ADAPTER = "pytest-changed-file-parameter-prior-failures/v1"
FALLBACK = "pytest-native-order-unseen-slots/v1"


def candidate_features(node, changed_paths, prior_failures):
    """Use a tested diff and history that had completed before this job arrived."""
    test_path = node.partition("::")[0]
    return {"changed_test_file": int(test_path in changed_paths),
            "parameterized": int(node.endswith("]")),
            "prior_failures": prior_failures.get(node, 0)}


def order_tests(items, bundle, *, policy_sha256, changed_paths, prior_failures):
    """Reorder collected items in place. Keep tests without history in their slots.

    The caller supplies a count (zero included) for every previously observed
    test, and records actual execution independently of this collection hook.
    """
    native = [getattr(item, "nodeid", None) for item in items]
    try:
        if not all(isinstance(node, str) and node.strip() for node in native):
            raise ValueError("collected tests must have nonempty string IDs")
        by_node = dict(zip(native, items, strict=True))
        if len(by_node) != len(native):
            raise ValueError("collected test IDs must be unique")
        if not isinstance(policy_sha256, str) or len(policy_sha256) != 64:
            raise ValueError("host must pin a policy digest")
        if not isinstance(prior_failures, dict) or any(
                not isinstance(node, str) or not node.strip() or type(count) is not int or count < 0
                for node, count in prior_failures.items()):
            raise ValueError("prior failures must map test IDs to nonnegative integer counts")
        if not isinstance(changed_paths, (set, frozenset, list, tuple)) or not all(
                isinstance(path, str) and path.strip() for path in changed_paths):
            raise ValueError("changed paths must be a collection of filenames")
        try:
            bundle = Path(bundle)
        except TypeError as exc:
            raise ValueError("bundle must be a directory path") from exc
        policy = load_policy(bundle / "policy.json", bundle / "tree.json",
                             adapter_revision=ADAPTER, fallback_revision=FALLBACK,
                             expected_sha256=policy_sha256)
        changed_paths = set(changed_paths)
        known = [node for node in native if node in prior_failures]
        if native and not known:
            return {"source": "fallback", "reason": "no previously observed tests",
                    "requested_order": native}
        candidates = [(node, candidate_features(node, changed_paths, prior_failures))
                      for node in known]
        ranked = iter(node for node, _ in policy.rank(candidates))
        order = [next(ranked) if node in prior_failures else node for node in native]
    except (OSError, ValueError) as exc:
        return {"source": "fallback", "reason": str(exc), "requested_order": native}
    items[:] = [by_node[node] for node in order]
    return {"source": "policy", "policy_sha256": policy.sha256,
            "known_count": len(known), "requested_order": order}
