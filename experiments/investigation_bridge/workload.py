"""Deterministic affine-frame debugging workload.

The public task gives a candidate a standalone ``solution.py`` containing an
``evaluate(request)`` function.  The evaluator keeps ``reference`` and final
requests out of the model context.
"""

from __future__ import annotations

from fractions import Fraction
import hashlib
import json
import random
import textwrap
from typing import Any


VARIANTS = (
    "composition",
    "inverse_translation",
    "pivot",
    "singular_conversion",
    "ancestor_omission",
    "sibling_paths",
)

CONTROL_MUTANTS = (*VARIANTS, "target_local_singularity")

CONTRACT = """\
Write solution.py with evaluate(request) -> JSON-compatible dict.

request has:
  frames: {name: {parent, translation, linear, pivot}}
  source: a frame name or "world"
  target: a frame name or "world"
  point: [x, y] in source coordinates

All numeric values are rational strings. For a frame, local -> parent is:
  translation + pivot + linear * (point - pivot)

Frames form a rooted graph whose root is the reserved name "world". Validate
the entire graph, including frames unrelated to the query. A missing parent or
cycle returns {"error":"invalid_scene"}. After scene validation, an unknown
source or target returns {"error":"unknown_frame"}. After those checks, a
singular target-to-world transform returns {"error":"singular"}. A singular
source is allowed because mapping from source toward world needs no inverse.

On success return {"point":[x,y]}. Coordinates must be canonical rational
strings: integers have no denominator, and other values use reduced n/d form
with a positive denominator. Inputs have the documented shapes; malformed
numeric arrays and other unspecified invalid inputs are outside this task.
"""


def _fraction(value: Any) -> Fraction:
    return Fraction(str(value))


def _canonical(value: Fraction) -> str:
    if value.denominator == 1:
        return str(value.numerator)
    return f"{value.numerator}/{value.denominator}"


def _matmul(left: tuple[tuple[Fraction, ...], ...], right: tuple[tuple[Fraction, ...], ...]):
    return tuple(
        tuple(sum(left[row][k] * right[k][column] for k in range(3)) for column in range(3))
        for row in range(3)
    )


def _local_matrix(frame: dict[str, Any]):
    a = _fraction(frame["linear"][0][0])
    b = _fraction(frame["linear"][0][1])
    c = _fraction(frame["linear"][1][0])
    d = _fraction(frame["linear"][1][1])
    tx, ty = map(_fraction, frame["translation"])
    px, py = map(_fraction, frame["pivot"])
    # A*x + (translation + pivot - A*pivot)
    bx = tx + px - a * px - b * py
    by = ty + py - c * px - d * py
    return (
        (a, b, bx),
        (c, d, by),
        (Fraction(0), Fraction(0), Fraction(1)),
    )


_IDENTITY = (
    (Fraction(1), Fraction(0), Fraction(0)),
    (Fraction(0), Fraction(1), Fraction(0)),
    (Fraction(0), Fraction(0), Fraction(1)),
)


def _scene_is_valid(frames: dict[str, dict[str, Any]]) -> bool:
    if "world" in frames:
        return False
    for origin in frames:
        seen: set[str] = set()
        current = origin
        while current != "world":
            if current not in frames or current in seen:
                return False
            seen.add(current)
            current = frames[current]["parent"]
    return True


def _world_matrix(name: str, frames: dict[str, dict[str, Any]]):
    result = _IDENTITY
    current = name
    while current != "world":
        result = _matmul(_local_matrix(frames[current]), result)
        current = frames[current]["parent"]
    return result


def _apply(matrix, point: tuple[Fraction, Fraction]):
    x, y = point
    return (
        matrix[0][0] * x + matrix[0][1] * y + matrix[0][2],
        matrix[1][0] * x + matrix[1][1] * y + matrix[1][2],
    )


def _inverse_affine(matrix):
    a, b, tx = matrix[0]
    c, d, ty = matrix[1]
    determinant = a * d - b * c
    if determinant == 0:
        return None
    return (
        (d / determinant, -b / determinant, (b * ty - d * tx) / determinant),
        (-c / determinant, a / determinant, (c * tx - a * ty) / determinant),
        (Fraction(0), Fraction(0), Fraction(1)),
    )


def reference(request: dict[str, Any]) -> dict[str, Any]:
    """Return the exact evaluator result using homogeneous affine matrices."""

    frames = request["frames"]
    if not _scene_is_valid(frames):
        return {"error": "invalid_scene"}

    source = request["source"]
    target = request["target"]
    if source != "world" and source not in frames:
        return {"error": "unknown_frame"}
    if target != "world" and target not in frames:
        return {"error": "unknown_frame"}

    source_matrix = _IDENTITY if source == "world" else _world_matrix(source, frames)
    target_matrix = _IDENTITY if target == "world" else _world_matrix(target, frames)
    target_inverse = _inverse_affine(target_matrix)
    if target_inverse is None:
        return {"error": "singular"}

    point = tuple(map(_fraction, request["point"]))
    converted = _apply(target_inverse, _apply(source_matrix, point))
    return {"point": [_canonical(converted[0]), _canonical(converted[1])]}


_CORRECT_SOURCE = textwrap.dedent(
    '''\
    """Affine frame evaluator. Uses scalar path traversal and exact fractions."""

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
        result = []
        current = name
        while current != "world":
            result.append(current)
            current = frames[current]["parent"]
        result.append("world")
        return result


    def _determinant(frame):
        linear = frame["linear"]
        return (_number(linear[0][0]) * _number(linear[1][1])
                - _number(linear[0][1]) * _number(linear[1][0]))


    def _to_parent(frame, point):
        x, y = point
        tx, ty = map(_number, frame["translation"])
        px, py = map(_number, frame["pivot"])
        linear = frame["linear"]
        a, b = map(_number, linear[0])
        c, d = map(_number, linear[1])
        return (tx + px + a * (x - px) + b * (y - py),
                ty + py + c * (x - px) + d * (y - py))


    def _from_parent(frame, point):
        x, y = point
        tx, ty = map(_number, frame["translation"])
        px, py = map(_number, frame["pivot"])
        linear = frame["linear"]
        a, b = map(_number, linear[0])
        c, d = map(_number, linear[1])
        determinant = a * d - b * c
        shifted_x = x - tx - px
        shifted_y = y - ty - py
        return (px + (d * shifted_x - b * shifted_y) / determinant,
                py + (-c * shifted_x + a * shifted_y) / determinant)


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
        if target != "world" and any(
                _determinant(frames[name]) == 0 for name in target_chain[:-1]):
            return {"error": "singular"}

        target_names = set(target_chain)
        common = next(name for name in source_chain if name in target_names)
        source_stop = source_chain.index(common)
        target_stop = target_chain.index(common)

        point = tuple(map(_number, request["point"]))
        for frame_name in source_chain[:source_stop]:
            point = _to_parent(frames[frame_name], point)

        target_leg = target_chain[:target_stop]
        for frame_name in reversed(target_leg):
            point = _from_parent(frames[frame_name], point)

        return {"point": [_text(point[0]), _text(point[1])]}
    '''
)


def _replace_once(source: str, old: str, new: str) -> str:
    if source.count(old) != 1:
        raise AssertionError(f"mutant edit expected one match, found {source.count(old)}")
    return source.replace(old, new)


def _apply_mutation(source: str, name: str) -> str:
    if name == "composition":
        return _replace_once(
            source,
            "for frame_name in source_chain[:source_stop]:\n",
            "for frame_name in reversed(source_chain[:source_stop]):\n",
        )
    if name == "inverse_translation":
        pivot_version = """    shifted_x = x - tx
    shifted_y = y - ty
    return ((d * shifted_x - b * shifted_y) / determinant,
            (-c * shifted_x + a * shifted_y) / determinant)
"""
        if pivot_version in source:
            return _replace_once(
                source,
                pivot_version,
                """    shifted_x = x
    shifted_y = y
    return ((d * shifted_x - b * shifted_y) / determinant - tx,
            (-c * shifted_x + a * shifted_y) / determinant - ty)
""",
            )
        return _replace_once(
            source,
            """    shifted_x = x - tx - px
    shifted_y = y - ty - py
    return (px + (d * shifted_x - b * shifted_y) / determinant,
            py + (-c * shifted_x + a * shifted_y) / determinant)
""",
            """    shifted_x = x - px
    shifted_y = y - py
    return (px + (d * shifted_x - b * shifted_y) / determinant - tx,
            py + (-c * shifted_x + a * shifted_y) / determinant - ty)
""",
        )
    if name == "pivot":
        source = _replace_once(
            source,
            """    return (tx + px + a * (x - px) + b * (y - py),
            ty + py + c * (x - px) + d * (y - py))
""",
            """    return (tx + a * x + b * y,
            ty + c * x + d * y)
""",
        )
        return _replace_once(
            source,
            """    shifted_x = x - tx - px
    shifted_y = y - ty - py
    return (px + (d * shifted_x - b * shifted_y) / determinant,
            py + (-c * shifted_x + a * shifted_y) / determinant)
""",
            """    shifted_x = x - tx
    shifted_y = y - ty
    return ((d * shifted_x - b * shifted_y) / determinant,
            (-c * shifted_x + a * shifted_y) / determinant)
""",
        )
    if name == "singular_conversion":
        return _replace_once(
            source,
            """    if target != "world" and any(
            _determinant(frames[name]) == 0 for name in target_chain[:-1]):
        return {"error": "singular"}
""",
            """    if source != "world" and any(
            _determinant(frames[name]) == 0 for name in source_chain[:-1]):
        return {"error": "singular"}
    if target != "world" and any(
            _determinant(frames[name]) == 0 for name in target_chain[:-1]):
        return {"error": "singular"}
""",
        )
    if name == "ancestor_omission":
        return _replace_once(
            source,
            """    result = []
    current = name
    while current != "world":
        result.append(current)
        current = frames[current]["parent"]
    result.append("world")
    return result
""",
            """    if name == "world":
        return ["world"]
    return [name, "world"]
""",
        )
    if name == "sibling_paths":
        return _replace_once(
            source,
            """    target_leg = target_chain[:target_stop]
    for frame_name in reversed(target_leg):
""",
            """    target_leg = target_chain[:target_stop]
    if source != "world" and target != "world":
        target_leg = target_leg[:1]
    for frame_name in reversed(target_leg):
            """,
        )
    if name == "target_local_singularity":
        return _replace_once(
            source,
            """    if target != "world" and any(
            _determinant(frames[name]) == 0 for name in target_chain[:-1]):
        return {"error": "singular"}
""",
            """    if target != "world" and _determinant(frames[target]) == 0:
        return {"error": "singular"}
""",
        )
    raise ValueError(f"unknown mutation: {name}")


def correct_source() -> str:
    """Return a standalone known-good ``solution.py`` for sandbox controls."""

    return _CORRECT_SOURCE


def mutants() -> dict[str, str]:
    """Return the six starters plus a held-out regression control."""

    return {name: _apply_mutation(_CORRECT_SOURCE, name) for name in CONTROL_MUTANTS}


def _rational(value: Fraction | int) -> str:
    return _canonical(Fraction(value))


def _pair(x: Fraction | int, y: Fraction | int) -> list[str]:
    return [_rational(x), _rational(y)]


def _frame(parent: str, translation, linear, pivot):
    return {
        "parent": parent,
        "translation": _pair(*translation),
        "linear": [_pair(*linear[0]), _pair(*linear[1])],
        "pivot": _pair(*pivot),
    }


def _stable_rng(seed: int | str, label: str) -> random.Random:
    material = json.dumps([seed, label], separators=(",", ":"), ensure_ascii=True)
    digest = hashlib.sha256(material.encode("utf-8")).digest()
    return random.Random(int.from_bytes(digest, "big"))


def _salt(rng: random.Random) -> Fraction:
    numerator = rng.randint(11, 997)
    if rng.randrange(2):
        numerator = -numerator
    return Fraction(numerator, rng.choice((2, 3, 5, 7, 11)))


def _token(seed: int | str, suite: int) -> str:
    material = json.dumps([seed, "affine-suite", suite], separators=(",", ":"))
    return hashlib.sha256(material.encode("utf-8")).hexdigest()[:10]


_REGULAR_LINEAR = (
    ((1, 1), (0, 1)),
    ((0, -1), (1, 0)),
    ((2, 0), (1, 1)),
    ((1, 0), (-1, 1)),
    ((2, 1), (1, 1)),
)


def _node(token: str, case: int, index: int) -> str:
    return f"f{token}{case:02x}{index:02x}"


def _regular_chain(
    frames: dict[str, dict[str, Any]],
    token: str,
    case: int,
    parent: str,
    depth: int,
    salt: Fraction,
    flavor: int,
) -> str:
    current = parent
    for index in range(depth):
        name = _node(token, case, index)
        linear = _REGULAR_LINEAR[(case + index + flavor) % len(_REGULAR_LINEAR)]
        translation = (salt + index - case, Fraction((flavor + 1) * (index + 1), 3))
        pivot = (Fraction((case + index) % 5 - 2, 2), Fraction((flavor + index) % 7 - 3, 2))
        frames[name] = _frame(current, translation, linear, pivot)
        current = name
    return current


def _case_suite(token: str, salt: Fraction, depth: int, flavor: int) -> list[dict[str, Any]]:
    cases: list[dict[str, Any]] = []

    # World identity with an unrelated, valid branch exercises whole-scene traversal.
    frames: dict[str, dict[str, Any]] = {}
    _regular_chain(frames, token, 0, "world", max(1, depth), salt, flavor)
    cases.append({"frames": frames, "source": "world", "target": "world", "point": _pair(salt, -salt)})

    # Pivoted forward and inverse conversions sit below different ancestor depths.
    frames = {}
    parent = _regular_chain(frames, token, 1, "world", 1 + depth % 3, salt, flavor)
    pivoted = _node(token, 1, 12)
    frames[pivoted] = _frame(parent, (salt, Fraction(-3, 2)), ((2, 1), (0, 3)), (Fraction(3, 2), -2))
    cases.append({"frames": frames, "source": pivoted, "target": "world", "point": _pair(2, 1)})

    frames = {}
    parent = _regular_chain(frames, token, 2, "world", 1 + (depth + 1) % 4, salt, flavor)
    inverse = _node(token, 2, 12)
    frames[inverse] = _frame(parent, (salt, -2), ((2, 1), (1, 2)), (1, -1))
    cases.append({"frames": frames, "source": "world", "target": inverse, "point": _pair(salt + 7, 5)})

    # A non-commuting source chain catches reversed composition and missing ancestors.
    frames = {}
    parent = _regular_chain(frames, token, 3, "world", depth, salt, flavor)
    upper = _node(token, 3, 12)
    lower = _node(token, 3, 13)
    frames[upper] = _frame(parent, (salt, 1), ((0, -1), (1, 0)), (0, 0))
    frames[lower] = _frame(upper, (-1, Fraction(2, 3)), ((2, 1), (0, 1)), (1, -1))
    cases.append({"frames": frames, "source": lower, "target": "world", "point": _pair(2, 3)})

    # Cousin frames have asymmetric branch depths and a varying common-ancestor depth.
    frames = {}
    common = _regular_chain(frames, token, 4, "world", 1 + depth % 3, salt, flavor)
    left = _regular_chain(frames, token, 5, common, 1 + depth % 2, salt + 1, flavor)
    right = _regular_chain(frames, token, 6, common, 2 + (depth + flavor) % 3, salt - 1, flavor)
    cases.append({"frames": frames, "source": left, "target": right, "point": _pair(Fraction(5, 2), -3)})

    # A singular source is legal because no inverse is required.
    frames = {}
    parent = _regular_chain(frames, token, 7, "world", depth % 3, salt, flavor)
    singular_source = _node(token, 7, 12)
    frames[singular_source] = _frame(parent, (salt, 2), ((1, 2), (2, 4)), (1, -1))
    cases.append({"frames": frames, "source": singular_source, "target": "world", "point": _pair(2, -1)})

    # Direct target singularity remains distinct from singularity inherited from an ancestor.
    frames = {}
    parent = _regular_chain(frames, token, 8, "world", depth % 2, salt, flavor)
    singular_target = _node(token, 8, 12)
    frames[singular_target] = _frame(parent, (salt, 2), ((1, 2), (2, 4)), (1, -1))
    cases.append({"frames": frames, "source": "world", "target": singular_target, "point": _pair(1, 4)})

    frames = {}
    parent = _regular_chain(frames, token, 9, "world", 1 + depth % 2, salt, flavor)
    collapsed = _node(token, 9, 12)
    frames[collapsed] = _frame(parent, (1, -2), ((1, 2), (2, 4)), (0, 1))
    inherited_target = _regular_chain(
        frames, token, 10, collapsed, 1 + depth % 3, salt + 2, flavor
    )
    cases.append({"frames": frames, "source": "world", "target": inherited_target, "point": _pair(-2, 3)})

    # Unknown-frame precedence is tested against an otherwise singular target path.
    ghost_source = _node(token, 11, 31)
    cases.append({"frames": frames, "source": ghost_source, "target": inherited_target, "point": _pair(0, 0)})

    frames = {}
    known_source = _regular_chain(frames, token, 12, "world", 1 + depth, salt, flavor)
    ghost_target = _node(token, 12, 31)
    cases.append({"frames": frames, "source": known_source, "target": ghost_target, "point": _pair(1, 1)})

    # Invalid nodes and cycles are disconnected from the query to enforce whole-graph precedence.
    frames = {}
    _regular_chain(frames, token, 13, "world", 1 + depth % 3, salt, flavor)
    missing = _node(token, 14, 0)
    absent = _node(token, 14, 31)
    frames[missing] = _frame(absent, (0, 0), ((1, 0), (0, 1)), (0, 0))
    cases.append({"frames": frames, "source": ghost_source, "target": ghost_target, "point": _pair(0, 0)})

    cycle_depth = 2 + depth % 4
    cycle_names = [_node(token, 15, index) for index in range(cycle_depth)]
    frames = {
        name: _frame(cycle_names[(index + 1) % cycle_depth], (index, -index), ((1, 0), (0, 1)), (0, 0))
        for index, name in enumerate(cycle_names)
    }
    cases.append({"frames": frames, "source": "world", "target": "world", "point": _pair(1, 2)})

    # The deepest valid target also carries an unused sibling branch.
    frames = {}
    deep_target = _regular_chain(frames, token, 16, "world", 3 + depth, salt, flavor)
    _regular_chain(frames, token, 17, "world", 1 + (depth + 1) % 4, -salt, flavor)
    cases.append({"frames": frames, "source": "world", "target": deep_target, "point": _pair(salt - 4, salt + 3)})
    return cases


def _requests(seed: int | str):
    topology_rng = _stable_rng(seed, "topology-order")
    depths = list(range(1, 7))
    topology_rng.shuffle(depths)
    suites = []
    for suite, depth in enumerate(depths[:5]):
        rng = _stable_rng(seed, f"suite-{suite}")
        suites.append(_case_suite(_token(seed, suite), _salt(rng), depth, suite))
    development = suites[0]
    final = [request for suite in suites[1:] for request in suite]
    _stable_rng(seed, "case-order").shuffle(final)
    return development, final


def task(seed: int | str, variant: str) -> dict[str, Any]:
    """Build one deterministic repair task.

    ``variant`` may be one of ``VARIANTS`` or ``"hard"``. The hard starter
    combines the six single faults; it is not included in ``mutants()``.
    """

    if variant not in VARIANTS and variant != "hard":
        allowed = ", ".join((*VARIANTS, "hard"))
        raise ValueError(f"unknown variant {variant!r}; expected one of {allowed}")
    development, final = _requests(seed)
    starter = _CORRECT_SOURCE
    chosen = (
        ("composition", "pivot", "inverse_translation", "singular_conversion", "ancestor_omission", "sibling_paths")
        if variant == "hard"
        else (variant,)
    )
    for mutation in chosen:
        starter = _apply_mutation(starter, mutation)
    task_digest = hashlib.sha256(
        json.dumps([seed, variant], separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    ).hexdigest()[:12]
    return {
        "id": f"investigation-bridge-{variant}-{task_digest}",
        "contract": CONTRACT,
        "starter": starter,
        "development": development,
        "final": final,
    }
