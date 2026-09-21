from fractions import Fraction


def evaluate(request):
    rings = request["rings"]
    rule = request["rule"]

    edges = []
    for ring in rings:
        n = len(ring)
        if n < 2:
            continue
        for i in range(n):
            x1, y1 = ring[i]
            x2, y2 = ring[(i + 1) % n]
            if x1 == x2:
                continue
            edges.append((Fraction(x1), Fraction(y1), Fraction(x2), Fraction(y2)))

    if not edges:
        return {"area": "0"}

    bps = set()
    for (x1, y1, x2, y2) in edges:
        bps.add(x1)
        bps.add(x2)

    for i in range(len(edges)):
        x1, y1, x2, y2 = edges[i]
        for j in range(i + 1, len(edges)):
            x3, y3, x4, y4 = edges[j]
            denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
            if denom == 0:
                continue
            t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / denom
            u = ((x1 - x3) * (y1 - y2) - (y1 - y3) * (x1 - x2)) / denom
            if 0 < t < 1 and 0 < u < 1:
                px = x1 + t * (x2 - x1)
                bps.add(px)

    bps = sorted(bps)

    total_area = Fraction(0)

    for i in range(len(bps) - 1):
        a, b = bps[i], bps[i + 1]
        if a == b:
            continue
        xm = (a + b) / 2

        crossings = []
        for (x1, y1, x2, y2) in edges:
            lo, hi = (x1, x2) if x1 < x2 else (x2, x1)
            if lo < xm < hi:
                y = y1 + (y2 - y1) * (xm - x1) / (x2 - x1)
                sign = 1 if x2 > x1 else -1
                crossings.append((y, sign))

        if not crossings:
            continue

        crossings.sort(key=lambda t: t[0])

        grouped = []
        j = 0
        while j < len(crossings):
            yv = crossings[j][0]
            s = 0
            while j < len(crossings) and crossings[j][0] == yv:
                s += crossings[j][1]
                j += 1
            grouped.append((yv, s))

        width = b - a
        running = 0
        for k in range(len(grouped) - 1):
            running += grouped[k][1]
            if rule == "evenodd":
                filled = (running % 2) != 0
            else:
                filled = running != 0
            if filled:
                dy = grouped[k + 1][0] - grouped[k][0]
                total_area += width * dy

    if total_area < 0:
        total_area = -total_area

    if total_area.denominator == 1:
        return {"area": str(total_area.numerator)}
    return {"area": f"{total_area.numerator}/{total_area.denominator}"}
