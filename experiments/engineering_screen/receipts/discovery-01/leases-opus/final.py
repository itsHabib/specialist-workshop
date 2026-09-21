"""Linearizability checker for a one-job leased queue.

State: (generation, lease, done) where lease is None or (worker, token, deadline).
Search: backtracking over all real-time-respecting linearizations, with
memoization of failed (remaining, state) pairs.
"""

_MISSING = object()


def _is_bool(x):
    return isinstance(x, bool)


def _num_eq(a, b):
    """Strict numeric equality: booleans never equal numbers."""
    if _is_bool(a) or _is_bool(b):
        return False
    if not isinstance(a, (int, float)) or not isinstance(b, (int, float)):
        return False
    return a == b


def _results_match(expected, actual):
    """Compare a recorded result against the spec's result, type-strictly."""
    if actual is None:
        return expected is None
    if _is_bool(actual):
        return _is_bool(expected) and expected == actual
    # actual is a numeric token/generation
    if expected is None or _is_bool(expected):
        return False
    return _num_eq(expected, actual)


def _lt(a, b):
    """now < deadline, guarding against non-numeric junk."""
    if _is_bool(a) or _is_bool(b):
        return False
    if not isinstance(a, (int, float)) or not isinstance(b, (int, float)):
        return False
    return a < b


def _apply(opname, args, state):
    """Return (new_state, result) or None if the operation is not applicable."""
    gen, lease, done = state
    if opname == "claim":
        worker = args.get("worker", _MISSING)
        now = args.get("now", _MISSING)
        ttl = args.get("ttl", _MISSING)
        if worker is _MISSING or now is _MISSING or ttl is _MISSING:
            return None
        if not isinstance(now, (int, float)) or _is_bool(now):
            return None
        if not isinstance(ttl, (int, float)) or _is_bool(ttl):
            return None
        if done or (lease is not None and _lt(now, lease[2])):
            return state, None
        gen += 1
        return (gen, (worker, gen, now + ttl), done), gen
    if opname == "finish":
        worker = args.get("worker", _MISSING)
        token = args.get("token", _MISSING)
        now = args.get("now", _MISSING)
        if worker is _MISSING or token is _MISSING or now is _MISSING:
            return None
        if (not done) and lease is not None and lease[0] == worker \
                and _num_eq(token, lease[1]) and _lt(now, lease[2]):
            return (gen, lease, True), True
        return state, False
    if opname == "reset":
        return (gen + 1, None, False), gen + 1
    return None


def _extract_history(request):
    hist = None
    if isinstance(request, dict):
        hist = request.get("history")
    else:
        hist = getattr(request, "history", None)
        if hist is None and hasattr(request, "get"):
            try:
                hist = request.get("history")
            except Exception:
                hist = None
    if hist is None:
        return []
    try:
        return list(hist)
    except TypeError:
        return []


def _field(entry, name, default=None):
    if isinstance(entry, dict):
        return entry.get(name, default)
    return getattr(entry, name, default)


def evaluate(request):
    entries = _extract_history(request)

    ops = []
    for e in entries:
        opname = _field(e, "op")
        args = _field(e, "args") or {}
        if not isinstance(args, dict):
            args = {}
        start = _field(e, "start")
        end = _field(e, "end")
        result = _field(e, "result")
        ops.append({
            "op": opname,
            "args": args,
            "start": start,
            "end": end,
            "result": result,
            "pending": end is None,
        })

    n = len(ops)
    if n == 0:
        return {"linearizable": True}

    # preds[i] = indices that must be linearized before i (A.end < B.start).
    preds = []
    for i in range(n):
        si = ops[i]["start"]
        s = set()
        if isinstance(si, (int, float)) and not _is_bool(si):
            for j in range(n):
                if j == i:
                    continue
                ej = ops[j]["end"]
                if ej is None:
                    continue
                if isinstance(ej, (int, float)) and not _is_bool(ej) and ej < si:
                    s.add(j)
        preds.append(s)

    completed = frozenset(i for i in range(n) if not ops[i]["pending"])

    failed = set()
    init_state = (0, None, False)

    def rec(remaining, state):
        # Any pending ops still left may simply be dropped.
        if not (remaining & completed):
            return True
        key = (remaining, state)
        if key in failed:
            return False
        for i in remaining:
            if preds[i] & remaining:
                continue  # something that must precede i hasn't run yet
            applied = _apply(ops[i]["op"], ops[i]["args"], state)
            if applied is None:
                continue
            new_state, res = applied
            if not ops[i]["pending"]:
                if not _results_match(ops[i]["result"], res):
                    continue
            if rec(remaining - {i}, new_state):
                return True
        failed.add(key)
        return False

    ok = rec(frozenset(range(n)), init_state)
    return {"linearizable": bool(ok)}
