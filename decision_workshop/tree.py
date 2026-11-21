"""A deterministic classification tree for validated, fully observed records.

Numeric splits use thresholds. Boolean and categorical splits test equality to
one observed value. Candidate splits minimize weighted Gini impurity.
"""

from collections import Counter, defaultdict


def _midpoint(low, high):
    # Halving first keeps the midpoint finite across the full IEEE float range.
    # Adjacent floats can round to the upper endpoint. Keep that endpoint on
    # the right, as it was during training.
    try:
        middle = low / 2 + high / 2
    except OverflowError:
        # Integers in Python may be larger than the largest finite float.
        return low
    return middle if low <= middle < high else low


def _numeric_splits(rows, feature, totals, total_squared, min_leaf):
    ordered = sorted((row["state"][feature], row["target"]) for row in rows)
    left = Counter()
    left_squared = 0
    right_squared = total_squared
    for index in range(len(ordered) - min_leaf):
        value, label = ordered[index]
        # Moving one observation updates the two squared class counts in O(1).
        left_squared += 2 * left[label] + 1
        right_squared -= 2 * (totals[label] - left[label]) - 1
        left[label] += 1
        size = index + 1
        next_value = ordered[size][0]
        if size >= min_leaf and value != next_value:
            yield _midpoint(value, next_value), size, left_squared, right_squared


def _category_splits(rows, feature, totals, total_squared, min_leaf):
    groups = defaultdict(Counter)
    for row in rows:
        groups[row["state"][feature]][row["target"]] += 1
    for value, counts in sorted(groups.items()):
        size = sum(counts.values())
        if min(size, len(rows) - size) < min_leaf:
            continue
        left_squared = sum(count * count for count in counts.values())
        # Each right count is total - left. Terms for classes absent on the
        # left are already accounted for by total_squared.
        overlap = sum(count * totals[label] for label, count in counts.items())
        right_squared = total_squared + left_squared - 2 * overlap
        yield value, size, left_squared, right_squared


def _goes_left(value, operator, threshold):
    return value <= threshold if operator == "<=" else value == threshold


def fit(features: dict, rows: list, labels: list, *, max_depth=5, min_leaf=2) -> dict:
    """Fit a JSON-safe tree and store empirical class counts in its leaves.

    Equal-quality splits prefer the first feature name, then the first split
    value in sorted order. Splits with zero immediate gain are allowed so a
    child can discover interactions such as XOR.
    """
    ordered_features = sorted(features.items())

    def grow(subset, depth):
        totals = Counter(row["target"] for row in subset)
        leaf = {"counts": {label: totals[label] for label in labels}}
        if len(totals) <= 1 or depth >= max_depth or len(subset) < 2 * min_leaf:
            return leaf

        total_squared = sum(count * count for count in totals.values())
        best = None
        best_num, best_den = 0, 1
        for feature, kind in ordered_features:
            if kind == "number":
                candidates = _numeric_splits(subset, feature, totals, total_squared, min_leaf)
                operator = "<="
            else:
                candidates = _category_splits(subset, feature, totals, total_squared, min_leaf)
                operator = "=="
            for value, left_size, left_squared, right_squared in candidates:
                right_size = len(subset) - left_size
                # Minimizing weighted Gini is equivalent to maximizing
                # left_squared / left_size + right_squared / right_size.
                # Compare fractions exactly so tie-breaking is reproducible.
                num = left_squared * right_size + right_squared * left_size
                den = left_size * right_size
                if best is None or num * best_den > best_num * den:
                    best = feature, operator, value
                    best_num, best_den = num, den
        if best is None:
            return leaf

        feature, operator, value = best
        left, right = [], []
        for row in subset:
            branch = left if _goes_left(row["state"][feature], operator, value) else right
            branch.append(row)
        return {
            "feature": feature,
            "op": operator,
            "value": value,
            "left": grow(left, depth + 1),
            "right": grow(right, depth + 1),
        }

    return grow(rows, 0)


def predict_counts(tree: dict, state: dict) -> dict:
    """Return the unsmoothed label counts at the leaf reached by state."""
    node = tree
    while "counts" not in node:
        value = state[node["feature"]]
        branch = "left" if _goes_left(value, node["op"], node["value"]) else "right"
        node = node[branch]
    return dict(node["counts"])
