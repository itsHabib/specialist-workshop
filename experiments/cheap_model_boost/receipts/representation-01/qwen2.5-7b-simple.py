from math import gcd

def evaluate(request):
    rings = request['rings']
    rule = request['rule']
    area = 0
    for ring in rings:
        if not ring: continue
        x, y = zip(*ring)
        n = len(x)
        winding_numbers = [0] * n
        for i in range(n):
            j = (i + 1) % n
            dx1, dy1 = x[i+1] - x[i], y[i+1] - y[i]
            dx2, dy2 = x[j] - x[i], y[j] - y[i]
            cross_product = dx1 * dy2 - dy1 * dx2
            if rule == 'evenodd':
                winding_numbers[i] += int(cross_product > 0)
                winding_numbers[j] -= int(cross_product > 0)
            else:
                winding_numbers[i] += (cross_product > 0) - (cross_product < 0)
        for i in range(n):
            area += x[i] * y[(i + 1) % n] - y[i] * x[(i + 1) % n]
    if rule == 'evenodd':
        area = sum(winding_numbers[i] * (x[i] * y[(i + 1) % n] - y[i] * x[(i + 1) % n]) for i in range(n)) // 2
    else:
        area = sum(winding_numbers[i] * (x[i] * y[(i + 1) % n] - y[i] * x[(i + 1) % n]) for i in range(n))
    common_divisor = gcd(area, abs(area)) if area != 0 else 1
    return {'area': f'{abs(area) // common_divisor}/{common_divisor}' if area != 0 else '0'}