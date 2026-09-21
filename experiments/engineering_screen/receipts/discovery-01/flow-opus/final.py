import heapq


class _MCMF(object):
    """Successive shortest paths min-cost max-flow (Dijkstra + potentials).

    All arc costs added here are non-negative, so potentials start at zero.
    """

    def __init__(self, n):
        self.n = n
        self.g = [[] for _ in range(n)]

    def add_edge(self, u, v, cap, cost):
        if cap <= 0:
            return
        self.g[u].append([v, cap, cost, len(self.g[v])])
        self.g[v].append([u, 0, -cost, len(self.g[u]) - 1])

    def run(self, s, t, maxf):
        n = self.n
        INF = float('inf')
        h = [0] * n
        flow = 0
        total = 0
        while flow < maxf:
            dist = [INF] * n
            dist[s] = 0
            prevv = [-1] * n
            preve = [-1] * n
            pq = [(0, s)]
            while pq:
                d, u = heapq.heappop(pq)
                if d > dist[u]:
                    continue
                gu = self.g[u]
                hu = h[u]
                for i in range(len(gu)):
                    e = gu[i]
                    if e[1] <= 0:
                        continue
                    v = e[0]
                    nd = d + e[2] + hu - h[v]
                    if nd < dist[v]:
                        dist[v] = nd
                        prevv[v] = u
                        preve[v] = i
                        heapq.heappush(pq, (nd, v))
            if dist[t] == INF:
                break
            for v in range(n):
                if dist[v] < INF:
                    h[v] += dist[v]
            # bottleneck
            d = maxf - flow
            v = t
            while v != s:
                cap = self.g[prevv[v]][preve[v]][1]
                if cap < d:
                    d = cap
                v = prevv[v]
            v = t
            while v != s:
                e = self.g[prevv[v]][preve[v]]
                e[1] -= d
                self.g[v][e[3]][1] += d
                v = prevv[v]
            flow += d
            total += d * h[t]
        return flow, total


def _as_int(x):
    return int(x)


def evaluate(request):
    if isinstance(request, (str, bytes)):
        import json
        request = json.loads(request)

    n = _as_int(request.get("n", 0))
    source = _as_int(request.get("source", 0))
    sink = _as_int(request.get("sink", 0))
    amount = _as_int(request.get("amount", 0))
    raw_edges = request.get("edges") or []

    # Normalize edges and determine the node universe defensively.
    edges = []
    max_idx = n - 1
    for e in raw_edges:
        if isinstance(e, dict):
            u = _as_int(e.get("u", 0))
            v = _as_int(e.get("v", 0))
            lo = _as_int(e.get("lower", 0))
            hi = _as_int(e.get("upper", 0))
            c = _as_int(e.get("cost", 0))
        else:
            u, v, lo, hi, c = (_as_int(x) for x in e)
        if lo > hi:
            return {"cost": None}
        if u < 0 or v < 0:
            return {"cost": None}
        if u > max_idx:
            max_idx = u
        if v > max_idx:
            max_idx = v
        edges.append((u, v, lo, hi, c))

    if source > max_idx:
        max_idx = source
    if sink > max_idx:
        max_idx = sink
    if amount < 0:
        return {"cost": None}
    if source == sink and amount != 0:
        return {"cost": None}

    N = max_idx + 1
    if N <= 0:
        return {"cost": 0 if amount == 0 else None}

    # Required net flow: a virtual arc sink -> source with lower = upper = amount
    # turns the problem into a pure min-cost circulation.
    if source != sink:
        edges.append((sink, source, amount, amount, 0))

    base = 0
    div = [0] * N          # divergence (outflow - inflow) of the base flow
    residual = []          # (u, v, capacity, non-negative cost)

    for (u, v, lo, hi, c) in edges:
        if u == v:
            # Self-loops never affect conservation: optimize independently.
            base += (hi * c) if c < 0 else (lo * c)
            continue
        if c >= 0:
            f0 = lo
            if hi > lo:
                residual.append((u, v, hi - lo, c))
        else:
            f0 = hi                      # saturate negative-cost arcs
            if hi > lo:
                residual.append((v, u, hi - lo, -c))
        if f0:
            base += f0 * c
            div[u] += f0
            div[v] -= f0

    # Fix the imbalance of the base flow with a min-cost b-flow on the
    # residual network (all residual costs are non-negative by construction).
    S = N
    T = N + 1
    mc = _MCMF(N + 2)
    need = 0
    for v in range(N):
        d = div[v]
        if d > 0:
            mc.add_edge(v, T, d, 0)
            need += d
        elif d < 0:
            mc.add_edge(S, v, -d, 0)
    for (u, v, cap, cost) in residual:
        mc.add_edge(u, v, cap, cost)

    if need == 0:
        return {"cost": int(base)}

    got, extra = mc.run(S, T, need)
    if got < need:
        return {"cost": None}
    return {"cost": int(base + extra)}
