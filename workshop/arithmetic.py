"""Deliberately narrow arithmetic baseline; unsupported prose stays unknown."""
import re

LABELS = {
    "consistent": "Explicit arithmetic agrees with the stated distance and fits the time budget. This is not coaching clearance.",
    "time-overrun": "The explicitly specified work plus rest exceeds its stated time budget.",
    "distance-conflict": "The explicitly specified distances disagree with the claimed total distance.",
    "needs-context": "A required duration, repetition count, or distance is unspecified; do not invent it.",
}


def arithmetic_rule(text):
    """Parse a finite supported language; no hidden dataset metadata."""
    patterns = [
        (r"(\d+) repetitions of (\d+) seconds; (\d+) seconds rest between repetitions; budget (\d+) seconds", "repeat"),
        (r"Benchmark completion (\d+) seconds; block budget (\d+) seconds", "benchmark"),
        (r"(\d+) down-and-back trips on a (\d+) metre lane; claimed distance (\d+) metres", "shuttle"),
        (r"(\d+) lengths of (\d+) metres; claimed distance (\d+) metres", "lengths"),
        (r"(\d+) metres at (\d+) seconds per kilometre; budget (\d+) seconds", "pace"),
        (r"Warmup (\d+) seconds, (\d+) repetitions of (\d+) seconds, (\d+) seconds rest between repetitions, cooldown (\d+) seconds; budget (\d+) seconds", "compound"),
        (r"Technique (\d+) lengths of (\d+) metres plus test (\d+) metres; claimed total (\d+) metres", "distance"),
    ]
    for pattern, kind in patterns:
        match = re.search(pattern, text)
        if match is None:
            continue
        n = list(map(int, match.groups()))
        if kind == "repeat":
            actual, limit = n[0] * n[1] + (n[0] - 1) * n[2], n[3]
        if kind == "benchmark":
            actual, limit = n
        if kind == "pace":
            actual, limit = n[0] * n[1] / 1000, n[2]
        if kind == "compound":
            actual, limit = n[0] + n[1] * n[2] + (n[1] - 1) * n[3] + n[4], n[5]
        if kind == "shuttle":
            return "consistent" if 2 * n[0] * n[1] == n[2] else "distance-conflict"
        if kind == "lengths":
            return "consistent" if n[0] * n[1] == n[2] else "distance-conflict"
        if kind == "distance":
            return "consistent" if n[0] * n[1] + n[2] == n[3] else "distance-conflict"
        return "consistent" if actual <= limit else "time-overrun"
    return "needs-context"
