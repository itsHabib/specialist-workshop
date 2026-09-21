from fractions import Fraction
import sys


def evaluate(request):
    edges = []
    cuts = set()

    for ring in request.get('rings', []):
        n = len(ring)
        if not n:
            continue
        for i in range(n):
            x1, y1 = ring[i]
            x2, y2 = ring[(i + 1) % n]
            cuts.add(Fraction(x1))
            dx = x2 - x1
            if dx == 0:
                continue
            slope = Fraction(y2 - y1, dx)
            intercept = Fraction(y1) - slope * x1
            edges.append((min(x1, x2), max(x1, x2),
                          slope, intercept, 1 if dx > 0 else -1))

    if not edges:
        return {'area': '0'}

    for i, edge in enumerate(edges):
        lo1, hi1, m1, b1, _ = edge
        for j in range(i):
            lo2, hi2, m2, b2, _ = edges[j]
            lo = max(lo1, lo2)
            hi = min(hi1, hi2)
            if lo >= hi or m1 == m2:
                continue
            crossing = (b2 - b1) / (m1 - m2)
            if lo < crossing < hi:
                cuts.add(crossing)

    cuts = sorted(cuts)
    evenodd = request['rule'] == 'evenodd'
    area = Fraction(0)

    for left, right in zip(cuts, cuts[1:]):
        midpoint = (left + right) / 2
        crossings = []
        for lo, hi, slope, intercept, delta in edges:
            if lo < midpoint < hi:
                crossings.append((slope * midpoint + intercept, delta))
        crossings.sort()

        winding = 0
        previous_y = None
        filled_length = Fraction(0)
        for y, delta in crossings:
            filled = (winding % 2 != 0) if evenodd else (winding != 0)
            if previous_y is not None and filled:
                filled_length += y - previous_y
            winding += delta
            previous_y = y

        area += filled_length * (right - left)

    # Exact results can exceed Python's default decimal conversion limit.
    if hasattr(sys, 'set_int_max_str_digits'):
        sys.set_int_max_str_digits(0)
    return {'area': str(area)}
