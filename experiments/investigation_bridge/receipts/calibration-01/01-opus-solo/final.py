"""Affine frame evaluator. Exact rational composition over root-ward chains."""

from fractions import Fraction


def _number(value):
    return Fraction(str(value))


def _text(value):
    if value.denominator == 1:
        return str(value.numerator)
    return f"{value.numerator}/{value.denominator}"


def _valid_scene(frames):
    if "world" in frames:
        return False
    limit = len(frames) + 1
    for start in frames:
        current = start
        steps = 0
        while current != "world":
            if current not in frames:
                return False
            if steps > limit:
                return False
            current = frames[current]["parent"]
            steps += 1
    return True


def _chain(name, frames):
    """Frames from `name` up to but not including world, child first."""
    chain = []
    current = name
    while current != "world":
        chain.append(current)
        current = frames[current]["parent"]
    return chain


def _affine(frame):
    """Local -> parent as (matrix, offset): p |-> L*p + offset."""
    a, b = map(_number, frame["linear"][0])
    c, d = map(_number, frame["linear"][1])
    tx, ty = map(_number, frame["translation"])
    px, py = map(_number, frame["pivot"])
    # translation + pivot + L * (p - pivot)
    return (a, b, c, d,
            tx + px - (a * px + b * py),
            ty + py - (c * px + d * py))


def _compose(outer, inner):
    """Apply `inner` first, then `outer`."""
    a1, b1, c1, d1, e1, f1 = outer
    a2, b2, c2, d2, e2, f2 = inner
    return (a1 * a2 + b1 * c2, a1 * b2 + b1 * d2,
            c1 * a2 + d1 * c2, c1 * b2 + d1 * d2,
            a1 * e2 + b1 * f2 + e1,
            c1 * e2 + d1 * f2 + f1)


def _to_world(name, frames):
    """Composite frame -> world transform."""
    result = (Fraction(1), Fraction(0), Fraction(0), Fraction(1),
              Fraction(0), Fraction(0))
    # Child first: each ancestor wraps around what has been built so far.
    for frame_name in _chain(name, frames):
        result = _compose(_affine(frames[frame_name]), result)
    return result


def _apply(transform, point):
    a, b, c, d, e, f = transform
    x, y = point
    return (a * x + b * y + e, c * x + d * y + f)


def _invert(transform):
    a, b, c, d, e, f = transform
    determinant = a * d - b * c
    ia = d / determinant
    ib = -b / determinant
    ic = -c / determinant
    id_ = a / determinant
    return (ia, ib, ic, id_,
            -(ia * e + ib * f),
            -(ic * e + id_ * f))


def evaluate(request):
    frames = request["frames"]
    if not _valid_scene(frames):
        return {"error": "invalid_scene"}

    source = request["source"]
    target = request["target"]
    if source != "world" and source not in frames:
        return {"error": "unknown_frame"}
    if target != "world" and target not in frames:
        return {"error": "unknown_frame"}

    target_transform = _to_world(target, frames)
    if target_transform[0] * target_transform[3] - target_transform[1] * target_transform[2] == 0:
        return {"error": "singular"}

    point = tuple(map(_number, request["point"]))
    point = _apply(_to_world(source, frames), point)
    point = _apply(_invert(target_transform), point)

    return {"point": [_text(point[0]), _text(point[1])]}
