from fractions import Fraction

def evaluate(request):
    def line_eq(p1, p2):
        x1, y1 = p1
        x2, y2 = p2
        if x1 == x2:
            return None, x1
        a = (y2 - y1) / (x2 - x1)
        b = y1 - a * x1
        return Fraction(a).limit_denominator(), Fraction(b).limit_denominator()

def find_intersections(rings):
    intersections = set()
    for i, ring in enumerate(rings):
        for j in range(len(ring) - 1):
            p1, p2 = ring[j], ring[(j + 1) % len(ring)]
            line_a, line_b = line_eq(p1, p2)
            if not line_a:
                continue
            for k, other_ring in enumerate(rings):
                if i == k:
                    continue
                for l in range(len(other_ring) - 1):
                    q1, q2 = other_ring[l], other_ring[(l + 1) % len(other_ring)]
                    if line_eq(q1, q2)[0] == line_a:
                        x_intersect = (q1[1] - line_b) / line_a
                        y_intersect = line_a * x_intersect + line_b
                        intersections.add((x_intersect, y_intersect))
    return sorted(intersections)

def process_slab(y, left, right, rule):
    active_edges = []
    for edge in left:
        if edge[1] < y <= edge[2]:
            active_edges.append(edge)
    for edge in right:
        if edge[0] < y <= edge[1]:
            active_edges.append(edge)
    active_edges.sort()
    x_values = [edge[0] for edge in active_edges]
    x_values += [edge[2] for edge in active_edges]
    x_values = sorted(set(x_values))
    total_winding = 0
    i = 0
    while i < len(active_edges) - 1:
        j = i + 1
        while j < len(active_edges) and active_edges[j][0] == active_edges[i][2]:
            j += 1
        for x in range(int(active_edges[i][2]) + 1, int(active_edges[j - 1][0])):
            total_winding += sum(1 if active_edges[k][0] < x <= active_edges[k][2] else -1 for k in range(i, j))
        i = j
    return total_winding % 2 != 0 or (rule == 'nonzero' and total_winding != 0)

def integrate_slab(y, height):
    if not process_slab(y, left_edges, right_edges, rule):
        return Fraction(0)
    x_left = min(edge[0] for edge in left_edges if y <= edge[2])
    x_right = max(edge[2] for edge in right_edges if y >= edge[0])
    area = (x_right - x_left) * height / 2
    return Fraction(area).limit_denominator()

def evaluate(request):
    rings = request['rings']
    rule = request['rule']
    intersections = find_intersections(rings)
    y_values = sorted(set([p[1] for p in intersections])) + [max(p[1] for ring in rings for p in ring)]
    total_area = Fraction(0)
    left_edges, right_edges = [], []
    for i in range(len(y_values) - 1):
        y1, y2 = y_values[i], y_values[i + 1]
        height = y2 - y1
        slab_intersections = [p for p in intersections if y1 < p[1] <= y2]
        left_edges = [(x, y1, y) for x, y in rings[0]]
        right_edges = [(x, y, y2) for x, y in rings[0]]
        for ring in rings[1:]:
            new_left_edges = []
            new_right_edges = []
            for p in slab_intersections:
                if rule == 'evenodd':
                    total_area += integrate_slab(p[1], height)
                else:
                    left_edges.append((p[0], y1, p[1]))
                    right_edges.append((p[0], p[1], y2))
    return {'area': str(total_area)}