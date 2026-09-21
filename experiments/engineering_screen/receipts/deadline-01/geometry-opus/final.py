from fractions import Fraction


def _tree_sum(terms):
    """Balanced summation keeps intermediate rational denominators small."""
    if not terms:
        return Fraction(0)
    terms = list(terms)
    while len(terms) > 1:
        nxt = []
        for i in range(0, len(terms) - 1, 2):
            nxt.append(terms[i] + terms[i + 1])
        if len(terms) % 2:
            nxt.append(terms[-1])
        terms = nxt
    return terms[0]


def _collect_edges(rings):
    edges = []
    for ring in rings or []:
        if not ring:
            continue
        pts = []
        for p in ring:
            pts.append((int(p[0]), int(p[1])))
        n = len(pts)
        if n < 2:
            continue
        for i in range(n):
            a = pts[i]
            b = pts[(i + 1) % n]
            if a != b:
                edges.append((a[0], a[1], b[0], b[1]))
    return edges


def _critical_xs(edges):
    xs = set()
    for (x1, _y1, x2, _y2) in edges:
        xs.add(Fraction(x1))
        xs.add(Fraction(x2))
    m = len(edges)
    for i in range(m):
        ax1, ay1, ax2, ay2 = edges[i]
        d1x = ax2 - ax1
        d1y = ay2 - ay1
        for j in range(i + 1, m):
            bx1, by1, bx2, by2 = edges[j]
            d2x = bx2 - bx1
            d2y = by2 - by1
            den = d1x * d2y - d1y * d2x
            if den == 0:
                continue  # parallel: overlap breakpoints are vertex x's
            ex = bx1 - ax1
            ey = by1 - ay1
            tn = ex * d2y - ey * d2x
            un = ex * d1y - ey * d1x
            if den < 0:
                den, tn, un = -den, -tn, -un
            if 0 <= tn <= den and 0 <= un <= den:
                xs.add(Fraction(ax1 * den + tn * d1x, den))
    return sorted(xs)


def evaluate(request):
    request = request or {}
    rings = request.get("rings") or []
    rule = request.get("rule", "evenodd")
    even_odd = (rule == "evenodd")

    edges = _collect_edges(rings)
    if not edges:
        return {"area": "0"}

    # Non-vertical edges, normalized left-to-right, with traversal sign.
    segs = []
    for (x1, y1, x2, y2) in edges:
        if x1 == x2:
            continue  # vertical: measure zero, never interior to a slab
        if x1 < x2:
            segs.append((x1, x2, y1, y2 - y1, x2 - x1, 1))
        else:
            segs.append((x2, x1, y2, y1 - y2, x1 - x2, -1))
    if not segs:
        return {"area": "0"}

    xlist = _critical_xs(edges)

    slab_terms = []
    for k in range(len(xlist) - 1):
        xa = xlist[k]
        xb = xlist[k + 1]
        if xa == xb:
            continue
        xm = (xa + xb) / 2
        active = []
        for (xlo, xhi, ylo, dy, dx, s) in segs:
            if xlo <= xa and xhi >= xb:
                y = Fraction(ylo * dx + dy * (xm - xlo), dx)
                active.append((y, s))
        if len(active) < 2:
            continue
        active.sort(key=lambda t: t[0])

        w = 0
        gap = Fraction(0)
        for idx in range(len(active) - 1):
            w += active[idx][1]
            filled = (w % 2 != 0) if even_odd else (w != 0)
            if filled:
                d = active[idx + 1][0] - active[idx][0]
                if d:
                    gap += d
        if gap:
            slab_terms.append((xb - xa) * gap)

    total = _tree_sum(slab_terms)
    if total < 0:
        total = -total
    if total.denominator == 1:
        return {"area": str(total.numerator)}
    return {"area": "%d/%d" % (total.numerator, total.denominator)}
