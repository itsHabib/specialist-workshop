"""Choose a paid witness using only information the importer has observed."""

from dataclasses import dataclass
from functools import lru_cache


@dataclass(frozen=True)
class Candidate:
    id: str
    channel: str
    notice: bool
    differences: int


def plan(remaining, history, chance, budget):
    """Maximize expected corrected rows through the remaining number of calls."""
    if not remaining or budget < 1:
        raise ValueError("a positive budget and at least one candidate are required")

    @lru_cache(maxsize=None)
    def best(roster, seen, depth):
        if not depth or not roster:
            return 0.0, None
        best_value, selected = -1.0, None
        for i, candidate in enumerate(roster):
            p = chance(candidate, seen)
            others = roster[:i] + roster[i + 1:]
            when_wrong, _ = best(others, seen + ((candidate.channel, True),), depth - 1)
            when_right, _ = best(others, seen + ((candidate.channel, False),), depth - 1)
            value = p * (candidate.differences + when_wrong) + (1 - p) * when_right
            if value > best_value + 1e-10:
                best_value, selected = value, candidate
        return best_value, selected

    return best(tuple(remaining), tuple(history), min(len(remaining), budget))[1]
