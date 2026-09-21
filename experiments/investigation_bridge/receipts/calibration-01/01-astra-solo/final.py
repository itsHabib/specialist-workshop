"""Evaluate affine frame coordinates using exact rational arithmetic."""

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
    for start in frames:
        current = start
        seen = set()
        while current != "world":
            if current not in frames or current in seen:
                return False
            seen.add(current)
            current = frames[current]["parent"]
    return True


def _chain(name, frames):
    chain = []
    while name != "world":
        chain.append(name)
        name = frames[name]["parent"]
    return chain


def _determinant(frame):
    linear = frame["linear"]
    return (_number(linear[0][0]) * _number(linear[1][1])
            - _number(linear[0][1]) * _number(linear[1][0]))


def _to_parent(frame, point):
    x, y = point
    tx, ty = map(_number, frame["translation"])
    px, py = map(_number, frame["pivot"])
    a, b = map(_number, frame["linear"][0])
    c, d = map(_number, frame["linear"][1])
    dx, dy = x - px, y - py
    return (tx + px + a * dx + b * dy,
            ty + py + c * dx + d * dy)


def _from_parent(frame, point):
    x, y = point
    tx, ty = map(_number, frame["translation"])
    px, py = map(_number, frame["pivot"])
    a, b = map(_number, frame["linear"][0])
    c, d = map(_number, frame["linear"][1])
    determinant = a * d - b * c
    dx, dy = x - tx - px, y - ty - py
    return (px + (d * dx - b * dy) / determinant,
            py + (-c * dx + a * dy) / determinant)


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

    source_chain = _chain(source, frames)
    target_chain = _chain(target, frames)
    if any(_determinant(frames[name]) == 0 for name in target_chain):
        return {"error": "singular"}

    point = tuple(map(_number, request["point"]))
    for name in source_chain:
        point = _to_parent(frames[name], point)
    for name in reversed(target_chain):
        point = _from_parent(frames[name], point)

    return {"point": [_text(point[0]), _text(point[1])]}
