"""Exact rectangle-union repair tasks with host-only final evaluation."""

from __future__ import annotations

from fractions import Fraction
import json
import random
import textwrap


_STARTER = '''\
"""Compute the union area of axis-aligned rectangles."""
import argparse
import json
import sys


def read_rectangles(lines):
    rectangles = []
    for number, raw in enumerate(lines, 1):
        line = raw.strip()
        if not line:
            continue
        values = [float(value) for value in line.split()]
        if len(values) != 4:
            raise ValueError(f"line {number}: expected four coordinates")
        rectangles.append(tuple(values))
    return rectangles


def union_area(rectangles):
    # Add each rectangle and subtract overlap with rectangles already seen.
    total = 0.0
    seen = []
    for x1, y1, x2, y2 in rectangles:
        total += (x2 - x1) * (y2 - y1)
        for a1, b1, a2, b2 in seen:
            width = max(0.0, min(x2, a2) - max(x1, a1))
            height = max(0.0, min(y2, b2) - max(y1, b1))
            total -= width * height
        seen.append((x1, y1, x2, y2))
    return total


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--input")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)
    stream = open(args.input) if args.input else sys.stdin
    try:
        area = union_area(read_rectangles(stream))
    finally:
        if args.input:
            stream.close()
    text = str(int(area)) if area.is_integer() else str(area)
    print(json.dumps({"area": text}) if args.json else text)


if __name__ == "__main__":
    main()
'''

_SOLUTION = '''\
"""Compute exact union area for axis-aligned rectangles."""
import argparse
from fractions import Fraction
import json
import sys


def read_rectangles(lines):
    rectangles = []
    for number, raw in enumerate(lines, 1):
        line = raw.split("#", 1)[0].strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) != 4:
            raise ValueError(f"line {number}: expected four coordinates")
        x1, y1, x2, y2 = map(Fraction, parts)
        x1, x2 = sorted((x1, x2))
        y1, y2 = sorted((y1, y2))
        if x1 != x2 and y1 != y2:
            rectangles.append((x1, y1, x2, y2))
    return rectangles


def union_area(rectangles):
    events = []
    for x1, y1, x2, y2 in rectangles:
        events.extend(((x1, 1, y1, y2), (x2, -1, y1, y2)))
    events.sort()
    active = []
    total = Fraction()
    previous = events[0][0] if events else Fraction()
    index = 0
    while index < len(events):
        x = events[index][0]
        intervals = sorted(active)
        covered = Fraction()
        if intervals:
            start, end = intervals[0]
            for lo, hi in intervals[1:]:
                if lo > end:
                    covered += end - start
                    start, end = lo, hi
                elif hi > end:
                    end = hi
            covered += end - start
        total += (x - previous) * covered
        while index < len(events) and events[index][0] == x:
            _, direction, lo, hi = events[index]
            if direction == 1:
                active.append((lo, hi))
            else:
                active.remove((lo, hi))
            index += 1
        previous = x
    return total


def _text(value):
    return str(value.numerator) if value.denominator == 1 else f"{value.numerator}/{value.denominator}"


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--input")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)
    stream = open(args.input, encoding="utf-8") if args.input else sys.stdin
    try:
        text = _text(union_area(read_rectangles(stream)))
    finally:
        if args.input:
            stream.close()
    print(json.dumps({"area": text}, separators=(",", ":")) if args.json else text)


if __name__ == "__main__":
    main()
'''

_PUBLIC_TESTS = '''\
import subprocess
import sys
import unittest

import rectarea


class RectangleAreaTests(unittest.TestCase):
    def test_disjoint_and_overlapping_rectangles(self):
        source = ["0 0 3 2\\n", "1 1 4 3\\n", "10 10 11 12\\n"]
        self.assertEqual(str(rectarea.union_area(rectarea.read_rectangles(source))), "12")

    def test_three_way_overlap_is_not_subtracted_twice(self):
        source = ["0 0 4 4\\n", "1 1 5 5\\n", "2 2 3 3\\n"]
        self.assertEqual(str(rectarea.union_area(rectarea.read_rectangles(source))), "23")

    def test_reversed_and_degenerate_rectangles(self):
        source = ["3 2 0 0\\n", "1 1 1 9\\n"]
        self.assertEqual(str(rectarea.union_area(rectarea.read_rectangles(source))), "6")

    def test_cli_reads_stdin(self):
        run = subprocess.run(
            [sys.executable, "rectarea.py"], input="0 0 3/2 2\\n",
            text=True, capture_output=True, check=False,
        )
        self.assertEqual((run.returncode, run.stdout), (0, "3\\n"), run.stderr)
'''

_V1_TEST = '''\
    def test_comments_are_ignored(self):
        source = ["# measured tiles\\n", "0 0 3/2 2 # west\\n"]
        self.assertEqual(str(rectarea.union_area(rectarea.read_rectangles(source))), "3")
'''


def build_task(variant: int) -> dict:
    """Return a worker-visible repair task. Final inputs stay in this module."""
    if variant not in (0, 1):
        raise ValueError("variant must be 0 or 1")
    extra = ""
    goal = (
        "Repair the rectangle union library and stdin CLI. Coordinates are exact "
        "integer or rational strings, may be reversed, and may be arbitrarily large. "
        "Print the nonnegative union area as a reduced integer or n/d."
    )
    if variant == 1:
        goal += (
            " Also ignore # comments, support --input PATH, and make --json emit "
            "a compact object whose area value is the same canonical string."
        )
        extra = _V1_TEST
    tests = _PUBLIC_TESTS + extra
    readme = f"""# Rectangle union repair\n\n{goal}\n\nBug report: overlapping imports sometimes undercount, reversed boxes produce negative\narea, and surveyed fractional coordinates drift on large maps. Keep the public API\n`read_rectangles(lines)` and `union_area(rectangles)` plus the CLI in `rectarea.py`.\n"""
    return {
        "name": f"rectarea-union-v{variant}",
        "goal": goal,
        "files": {
            "README.md": readme,
            "rectarea.py": _STARTER,
            "tests/test_rectarea.py": tests,
        },
        "test_command": "python -m unittest discover -s tests -v",
        "entrypoint": "rectarea.py",
    }


def _canonical(value: Fraction) -> str:
    return str(value.numerator) if value.denominator == 1 else f"{value.numerator}/{value.denominator}"


def _oracle(rectangles) -> str:
    """Independent coordinate-cell oracle used only on withheld inputs."""
    normalized = []
    for values in rectangles:
        x1, y1, x2, y2 = map(Fraction, values)
        x1, x2 = sorted((x1, x2)); y1, y2 = sorted((y1, y2))
        if x1 != x2 and y1 != y2:
            normalized.append((x1, y1, x2, y2))
    xs = sorted({x for rectangle in normalized for x in (rectangle[0], rectangle[2])})
    ys = sorted({y for rectangle in normalized for y in (rectangle[1], rectangle[3])})
    area = Fraction()
    for left, right in zip(xs, xs[1:]):
        for bottom, top in zip(ys, ys[1:]):
            if any(x1 <= left and right <= x2 and y1 <= bottom and top <= y2
                   for x1, y1, x2, y2 in normalized):
                area += (right - left) * (top - bottom)
    return _canonical(area)


def final_requests(variant: int, seed: int) -> list[dict]:
    """Create deterministic withheld CLI requests for an execution adapter."""
    if variant not in (0, 1):
        raise ValueError("variant must be 0 or 1")
    rng = random.Random(seed)
    batches = [
        [["9007199254740992", "0", "9007199254740993", "1"]],
        [["0", "0", "4", "4"], ["1", "1", "5", "5"], ["2", "2", "3", "3"]],
        [["7/3", "5/2", "-2/3", "-1/2"], ["0", "0", "1/3", "9"]],
        [],
    ]
    for _ in range(10):
        rows = []
        for _ in range(rng.randint(1, 7)):
            scale = rng.choice((1, 2, 3, 5))
            rows.append([str(Fraction(rng.randint(-12, 12), scale)) for _ in range(4)])
        batches.append(rows)
    requests = []
    for index, rows in enumerate(batches):
        lines = [" ".join(row) for row in rows]
        if variant == 1:
            lines = ["# withheld survey"] + [line + (" # box" if i % 2 else "") for i, line in enumerate(lines)]
        argv = ["--json"] if variant == 1 and index % 2 else []
        request = {"argv": argv, "stdin": "\n".join(lines) + "\n",
                   "expected": _oracle(rows)}
        if variant == 1 and index == 2:
            request.update(argv=["--input", "withheld.txt"], input_file=True)
        requests.append(request)
    return requests


def _evaluate(files: dict, execute: callable, requests: list[dict]) -> dict:
    """Run requests through the experiment's isolated command adapter."""
    failures = []
    checks = 0
    for request in requests:
        command = "python rectarea.py " + " ".join(request["argv"])
        run_files = files
        stdin = request["stdin"]
        if request.get("input_file"):
            run_files = {**files, "withheld.txt": stdin}
            stdin = ""
        result = execute(run_files, command.strip(), stdin)
        checks += 1
        if result.get("error") is not None or not result.get("cleanup_confirmed", False):
            failures.append({"kind": "infrastructure", "error": result.get("error")})
            continue
        if result.get("returncode") != 0:
            failures.append({"kind": "exit", "stderr": result.get("stderr", "")[-300:]})
            continue
        output = result.get("stdout", "").strip()
        if request["argv"] == ["--json"]:
            try:
                parsed = json.loads(output)["area"]
            except (json.JSONDecodeError, KeyError, TypeError):
                failures.append({"kind": "json", "output": output[:200]})
                continue
            expected_json = json.dumps({"area": request["expected"]}, separators=(",", ":"))
            if output != expected_json:
                failures.append({"kind": "json_format", "expected": expected_json,
                                 "actual": output[:200]})
                continue
            output = parsed
        if output != request["expected"]:
            failures.append({"kind": "area", "expected": request["expected"], "actual": output[:200]})
    return {"passed": not failures, "checks": checks, "failures": failures[:8]}


def verify(files: dict, execute: callable, seed: int, variant: int = 0) -> dict:
    """Evaluate withheld cases with execute(files, command, stdin='')."""
    if variant not in (0, 1):
        raise ValueError("variant must be 0 or 1")
    return _evaluate(files, execute, final_requests(variant, seed))


def development_probe(files: dict, execute: callable, seed: int, variant: int = 0) -> dict:
    """Expose reproducible repair feedback from a distribution separate from final."""
    if variant not in (0, 1):
        raise ValueError("variant must be 0 or 1")
    rng = random.Random(seed ^ 0x5EED)
    batches = [[
        [str(x), str(y), str(x + rng.randint(1, 5)), str(y + rng.randint(1, 5))]
        for x, y in ((0, 0), (1, 1), (2, 2))
    ]]
    for _ in range(4):
        batches.append([[str(rng.randint(-6, 3)), str(rng.randint(-6, 3)),
                         str(rng.randint(0, 8)), str(rng.randint(0, 8))]
                        for _ in range(rng.randint(3, 6))])
    requests = [{"argv": [], "stdin": "\n".join(" ".join(row) for row in rows) + "\n",
                 "expected": _oracle(rows)} for rows in batches]
    if variant == 1:
        requests[0]["argv"] = ["--json"]
    return _evaluate(files, execute, requests)


def reference_files(variant: int) -> dict:
    """Return a known-good control without exposing it in the task manifest."""
    files = dict(build_task(variant)["files"])
    files["rectarea.py"] = _SOLUTION
    return files
