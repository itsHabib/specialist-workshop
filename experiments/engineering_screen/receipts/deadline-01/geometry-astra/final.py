from fractions import Fraction


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
            if x1 == x2:
                continue
            slope = Fraction(y2 - y1, x2 - x1)
            intercept = Fraction(y1) - slope * x1
            edges.append((min(x1, x2), max(x1, x2), slope,
                          intercept, 1 if x2 > x1 else -1))

    if not edges:
        return {'area': '0'}

    for i, first in enumerate(edges):
        lo1, hi1, m1, b1, _ = first
        for j in range(i + 1, len(edges)):
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
        events = sorted(
            (slope * midpoint + intercept, delta)
            for lo, hi, slope, intercept, delta in edges
            if lo < midpoint < hi
        )
        winding = 0
        previous = None
        filled_length = Fraction(0)
        i = 0
        while i < len(events):
            height = events[i][0]
            filled = (winding % 2 != 0) if evenodd else (winding != 0)
            if previous is not None and filled:
                filled_length += height - previous
            change = 0
            while i < len(events) and events[i][0] == height:
                change += events[i][1]
                i += 1
            winding += change
            previous = height
        area += filled_length * (right - left)

    return {'area': str(area)}
