Python code to evaluate the signed winding numbers of all walks in a list of closed polygonal walks with a given rule.

def evaluate(request):
    rings = request['rings']
    rule = request['rule']
    area = 0
    for walk in rings:
        # Calculate signed winding number using Shoelace formula
        n = len(walk) - 1
        winding = 0
        for i in range(n):
            j = (i + 1) % n
            x1, y1 = walk[i]
            x2, y2 = walk[j]
            winding += ((x2 - x1) * (y1 - y2))
        # Apply the given rule
        if rule == 'evenodd':
            winding *= 1
        elif rule == 'nonzero':
            winding *= -1
        # Add to total area
        area += winding
    return {'area': str(area)}