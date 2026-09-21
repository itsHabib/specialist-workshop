"""Affine frame evaluator. Exact rational arithmetic over a rooted frame tree.

For a frame, local -> parent is

    translation + pivot + linear * (point - pivot)

so parent -> local is

    pivot + linear^-1 * (point - translation - pivot)
"""

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
    """Frame names from `name` up to world, nearest first, world excluded."""
    chain = []
    current = name
    while current != "world":
        chain.append(current)
        current = frames[current]["parent"]
    return chain


def _parts(frame):
    tx, ty = map(_number, frame["translation"])
    px, py = map(_number, frame["pivot"])
    linear = frame["linear"]
    a, b = map(_number, linear[0])
    c, d = map(_number, linear[1])
    return tx, ty, px, py, a, b, c, d


def _determinant(frame):
    _, _, _, _, a, b, c, d = _parts(frame)
    return a * d - b * c


def _to_parent(frame, point):
    x, y = point
    tx, ty, px, py, a, b, c, d = _parts(frame)
    ux = x - px
    uy = y - py
    return (tx + px + a * ux + b * uy,
            ty + py + c * ux + d * uy)


def _from_parent(frame, point):
    x, y = point
    tx, ty, px, py, a, b, c, d = _parts(frame)
    determinant = a * d - b * c
    ux = x - tx - px
    uy = y - ty - py
    return (px + (d * ux - b * uy) / determinant,
            py + (-c * ux + a * uy) / determinant)


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

    source_chain = [] if source == "world" else _chain(source, frames)
    target_chain = [] if target == "world" else _chain(target, frames)

    # Only the target leg is inverted; a singular source maps toward world fine.
    if any(_determinant(frames[name]) == 0 for name in target_chain):
        return {"error": "singular"}

    point = tuple(map(_number, request["point"]))
    for frame_name in source_chain:
        point = _to_parent(frames[frame_name], point)
    for frame_name in reversed(target_chain):
        point = _from_parent(frames[frame_name], point)

    return {"point": [_text(point[0]), _text(point[1])]}
