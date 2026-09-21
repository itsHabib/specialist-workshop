from collections import deque
from heapq import heappush, heappop


def evaluate(request):
    n = request['n']
    source = request['source']
    sink = request['sink']
    amount = request['amount']
    ss, tt = n, n + 1
    size = n + 2
    graph = [[] for _ in range(size)]
    balance = [0] * n
    balance[source] = amount
    balance[sink] = -amount
    cost = 0

    def add_pair(u, v, forward, backward, weight):
        # Self-loops are optimized separately below.
        a = [v, len(graph[v]), forward, weight]
        b = [u, len(graph[u]), backward, -weight]
        graph[u].append(a)
        graph[v].append(b)

    for edge in request['edges']:
        u, v = edge['u'], edge['v']
        lower, upper = edge['lower'], edge['upper']
        weight = edge['cost']
        flow = upper if weight < 0 else lower
        cost += flow * weight
        if u == v:
            continue
        balance[u] -= flow
        balance[v] += flow
        if upper > lower:
            add_pair(u, v, upper - flow, flow - lower, weight)

    required = 0
    for v, b in enumerate(balance):
        if b > 0:
            add_pair(ss, v, b, 0, 0)
            required += b
        elif b < 0:
            add_pair(v, tt, -b, 0, 0)

    if required == 0:
        return {'cost': cost}

    potential = [0] * size
    sent = 0
    infinity = float('inf')

    def send_zero(v, limit):
        if v == tt:
            return limit
        while cursor[v] < len(graph[v]):
            edge = graph[v][cursor[v]]
            w, reverse, capacity, weight = edge
            if (capacity > 0 and level[w] == level[v] + 1
                    and weight + potential[v] - potential[w] == 0):
                pushed = send_zero(w, min(limit, capacity))
                if pushed:
                    edge[2] -= pushed
                    graph[w][reverse][2] += pushed
                    return pushed
            cursor[v] += 1
        return 0

    while sent < required:
        distance = [infinity] * size
        distance[ss] = 0
        heap = [(0, ss)]
        while heap:
            current, v = heappop(heap)
            if current != distance[v]:
                continue
            for w, reverse, capacity, weight in graph[v]:
                if capacity <= 0:
                    continue
                candidate = current + weight + potential[v] - potential[w]
                if candidate < distance[w]:
                    distance[w] = candidate
                    heappush(heap, (candidate, w))

        if distance[tt] == infinity:
            return {'cost': None}

        # Capping all potential increments preserves nonnegative reduced
        # costs even on edges entering the reachable region.
        increment = distance[tt]
        for v in range(size):
            potential[v] += min(distance[v], increment)

        unit_cost = potential[tt] - potential[ss]

        # Exhaust all paths at this marginal cost before running Dijkstra
        # again. Each augmentation sends a whole capacity bottleneck.
        while sent < required:
            level = [-1] * size
            level[ss] = 0
            queue = deque([ss])
            while queue:
                v = queue.popleft()
                for w, reverse, capacity, weight in graph[v]:
                    if (capacity > 0 and level[w] < 0
                            and weight + potential[v] - potential[w] == 0):
                        level[w] = level[v] + 1
                        queue.append(w)
            if level[tt] < 0:
                break

            cursor = [0] * size
            while sent < required:
                pushed = send_zero(ss, required - sent)
                if pushed == 0:
                    break
                sent += pushed
                cost += pushed * unit_cost

    return {'cost': cost}
