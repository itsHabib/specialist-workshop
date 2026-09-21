from functools import lru_cache


def evaluate(request):
    history = request['history']
    n = len(history)
    required = 0
    predecessors = [0] * n
    for i, operation in enumerate(history):
        if operation['end'] is not None:
            required |= 1 << i
        for j, earlier in enumerate(history):
            if earlier['end'] is not None and earlier['end'] < operation['start']:
                predecessors[i] |= 1 << j

    order = sorted(range(n), key=lambda i: history[i]['end'] is None)

    def equal(a, b):
        if isinstance(a, bool) or isinstance(b, bool):
            return isinstance(a, bool) and isinstance(b, bool) and a == b
        return a == b

    @lru_cache(maxsize=None)
    def search(mask, generation, lease, done):
        if (mask & required) == required:
            return True
        for i in order:
            bit = 1 << i
            if mask & bit or predecessors[i] & ~mask:
                continue
            operation = history[i]
            args = operation['args']
            kind = operation['op']
            next_generation, next_lease, next_done = generation, lease, done

            if kind == 'reset':
                next_generation = generation + 1
                next_lease = -1
                next_done = False
                result = next_generation
            elif kind == 'claim':
                active = False
                if lease >= 0:
                    previous = history[lease]['args']
                    active = args['now'] < previous['now'] + previous['ttl']
                if done or active:
                    result = None
                else:
                    next_generation = generation + 1
                    next_lease = i
                    result = next_generation
            elif kind == 'finish':
                result = False
                if not done and lease >= 0:
                    previous = history[lease]['args']
                    # A reset clears the lease; a successful claim installs
                    # a lease with the newly incremented generation.
                    result = bool(
                        equal(args['worker'], previous['worker'])
                        and equal(args['token'], generation)
                        and args['now'] < previous['now'] + previous['ttl']
                    )
                if result:
                    next_done = True
            else:
                continue

            if operation['end'] is not None and not equal(result, operation['result']):
                continue
            if search(mask | bit, next_generation, next_lease, next_done):
                return True
        return False

    return {'linearizable': search(0, 0, -1, False)}
