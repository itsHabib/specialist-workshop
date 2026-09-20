"""Bounded inference policies; deployment feedback never consults held-out answers.

This runner orchestrates a transport supplied by the caller. It neither launches
workers nor claims that additional calls improve capability. Every requested call
has a checkpoint event before transport, including calls that fail or lose usage.
"""
from dataclasses import asdict, dataclass
import hashlib
import json
import math
import re
import time

from workshop import packages
from workshop.domain import digest


def _positive(value, name, minimum=1):
    if type(value) is not int or value < minimum:
        raise ValueError(f'{name} must be an integer >= {minimum}')


@dataclass(frozen=True)
class Policy:
    name: str
    solver: str
    critic: str | None = None
    rounds: int = 1
    examples: int = 0

    def __post_init__(self):
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError('name must be nonempty')
        if not isinstance(self.solver, str) or not self.solver.strip():
            raise ValueError('solver must be nonempty')
        if self.critic is not None and (not isinstance(self.critic, str) or not self.critic.strip()):
            raise ValueError('critic must be nonempty when supplied')
        _positive(self.rounds, 'rounds')
        _positive(self.examples, 'examples', 0)


@dataclass(frozen=True)
class Limits:
    max_calls: int = 6
    max_output_tokens: int = 1024
    max_total_tokens: int = 24000

    def __post_init__(self):
        for name, value in asdict(self).items():
            _positive(value, name)


def _text(value):
    return value if isinstance(value, str) else json.dumps(value, sort_keys=True, ensure_ascii=False)


def select_examples(package, input_value, count):
    """Fixed, lexical input-only retrieval from TRAIN; stable ties by case ID."""
    words = set(re.findall(r'\w+', _text(input_value).casefold()))
    def rank(case):
        overlap = len(words & set(re.findall(r'\w+', _text(case.input).casefold())))
        return -overlap, case.id
    return sorted(package.train, key=rank)[:count]


def input_token_bound(messages):
    """Conservative UTF-8 byte bound plus chat framing for plain text transport.

    Providers must send these messages without adding hidden tools/documents and
    account for all billed output (including reasoning) in output_tokens. This
    cannot reserve unknown provider-side prompt additions.
    """
    return 1024 + sum(len(row['content'].encode('utf-8')) + 256 for row in messages)


def _candidate(raw):
    return dict(raw=raw, hash=hashlib.sha256(raw.encode('utf-8')).hexdigest())


def _usage(value):
    if not isinstance(value, dict):
        return None
    keys = ('input_tokens', 'output_tokens')
    if any(type(value.get(key)) is not int or value[key] < 0 for key in keys):
        return None
    return {key: value[key] for key in keys}



_TOKEN_METADATA_KEYS = frozenset({
    'input_tokens', 'output_tokens', 'total_tokens', 'prompt_tokens', 'completion_tokens',
    'cache_creation_input_tokens', 'cache_read_input_tokens', 'cache_write_tokens', 'cached_tokens',
    'reasoning_tokens', 'audio_tokens', 'accepted_prediction_tokens', 'rejected_prediction_tokens',
    'input_tokens_details', 'output_tokens_details', 'prompt_tokens_details', 'completion_tokens_details',
    'cache_creation', 'ephemeral_5m_input_tokens', 'ephemeral_1h_input_tokens',
})


def _provider_usage(value, depth=0):
    """Retain auditable token counters, excluding free-form provider metadata."""
    if not isinstance(value, dict) or depth > 4:
        return None
    clean = {}
    for key, item in value.items():
        if key not in _TOKEN_METADATA_KEYS:
            continue
        if isinstance(item, dict):
            nested = _provider_usage(item, depth + 1)
            if nested:
                clean[key] = nested
            continue
        if (type(item) is int and item >= 0) or (type(item) is float and math.isfinite(item) and item >= 0):
            clean[key] = item
    return clean


def _response_id(value):
    if isinstance(value, str) and re.fullmatch(r'[A-Za-z0-9_-]{1,256}', value):
        return value
    return None


def _provider_error(value):
    if isinstance(value, str) and re.fullmatch(r'HTTP_[0-9]{3}|incomplete_response', value):
        return value
    return 'ProviderError'


def _cost(value):
    if type(value) not in (float, int) or not math.isfinite(value) or value < 0:
        return None
    return value


def run_case(package, case, policy, limits, complete, record=None):
    """Run one policy without scoring or selecting against case.expected.

    ``rounds`` counts solver candidates, not calls. Critic calls consume the same
    budget. A valid candidate does not end the configured refinement policy;
    schema validity alone is not semantic success. The last solver candidate wins.
    """
    started = time.monotonic()
    selected = select_examples(package, case.input, policy.examples)
    contract = dict(instructions=package.instructions, output_schema=package.output_schema)
    demonstrations = [dict(id=row.id, input=row.input, output=row.expected) for row in selected]
    base = [dict(role='system', content='Solve the task under this contract. Return only the requested output.\n' + _text(contract)),
            dict(role='user', content=_text(dict(training_examples=demonstrations, input=case.input)))]
    history = list(base)
    events, calls = [], []
    known_tokens, known_cost = 0, 0.0
    tokens_known, costs_known = True, True
    candidate, validation = None, None
    stop_reason, error = 'rounds_completed', None

    def emit(event):
        # Deep copies prevent a transport/checkpoint callback from mutating history.
        event = json.loads(json.dumps(event))
        events.append(event)
        if record is not None:
            record(json.loads(json.dumps(event)))

    def invoke(role, model, messages):
        nonlocal known_tokens, known_cost, tokens_known, costs_known, stop_reason, error
        if len(calls) >= limits.max_calls:
            stop_reason = 'call_budget'
            return None
        bound = input_token_bound(messages)
        if known_tokens + bound + limits.max_output_tokens > limits.max_total_tokens:
            stop_reason = 'token_budget'
            return None
        call = dict(call_id=len(calls) + 1, role=role, requested_model=model,
                    messages=json.loads(json.dumps(messages)), input_token_bound=bound,
                    max_output_tokens=limits.max_output_tokens)
        call['request_digest'] = digest(dict(model=model, messages=messages, max_output_tokens=limits.max_output_tokens))
        try:
            emit(dict(type='before_call', **call))
        except Exception as exc:
            stop_reason, error = 'record_error', type(exc).__name__
            return None
        calls.append(call)
        begin = time.monotonic()
        try:
            result = complete(model, json.loads(json.dumps(messages)), limits.max_output_tokens)
            if not isinstance(result, dict):
                raise TypeError('Transport response must be a dict')
            usage, cost = _usage(result.get('usage')), _cost(result.get('cost_usd'))
            call.update(raw=result.get('raw') if isinstance(result.get('raw'), str) else None,
                        model=result.get('model') if isinstance(result.get('model'), str) else None,
                        usage=usage, cost_usd=cost, provider_usage=_provider_usage(result.get('provider_usage')),
                        response_id=_response_id(result.get('response_id')))
            # Provider error details can include credentials/request bodies. Keep
            # the failure class only; transports should preserve sensitive logs separately.
            if result.get('error'):
                call['error'] = _provider_error(result['error'])
        except Exception as exc:
            call.update(raw=None, model=None, usage=None, cost_usd=None, provider_usage=None,
                        response_id=None, error=type(exc).__name__)
        call['elapsed_seconds'] = time.monotonic() - begin
        if call['usage'] is None:
            tokens_known = False
        else:
            known_tokens += sum(call['usage'].values())
        if call['cost_usd'] is None:
            costs_known = False
        else:
            known_cost += call['cost_usd']
        if call.get('error'):
            stop_reason, error = 'transport_error', call['error']
        elif call['raw'] is None or not call['model']:
            stop_reason, error = 'transport_error', 'InvalidResponse'
        elif not tokens_known:
            stop_reason = 'usage_unknown'
        elif (known_tokens > limits.max_total_tokens or call['usage']['input_tokens'] > bound
              or call['usage']['output_tokens'] > limits.max_output_tokens):
            stop_reason = 'budget_exceeded'
        try:
            emit(dict(type='after_call', **call))
        except Exception as exc:
            stop_reason, error = 'record_error', type(exc).__name__
        return call

    for round_number in range(policy.rounds):
        if round_number:
            feedback = dict(candidate=candidate, validation=validation)
            if policy.critic:
                critic_messages = [dict(role='system', content='Independently critique the candidate against the original contract. '
                                        'Find concrete mistakes or counterexamples. You are feedback, not an oracle.\n' + _text(contract)),
                                   dict(role='user', content=_text(dict(training_examples=demonstrations,
                                                                      input=case.input, **feedback)))]
                critique = invoke('critic', policy.critic, critic_messages)
                if critique is None or stop_reason != 'rounds_completed':
                    break
                feedback['critique'] = critique['raw']
            history.append(dict(role='user', content='Critique and improve your answer using this deployable feedback. '
                                'Return the complete revised output; feedback may be mistaken.\n' + _text(feedback)))
        result = invoke('solver', policy.solver, history)
        if result is None:
            break
        if result['raw'] is not None and not result.get('error'):
            candidate = _candidate(result['raw'])
            validation = packages.score(package, case.input, result['raw'], expected=None)
            history.append(dict(role='assistant', content=result['raw']))
        if stop_reason != 'rounds_completed':
            break

    return dict(policy=asdict(policy), limits=asdict(limits),
                config_id=digest(dict(version=1, policy=asdict(policy), limits=asdict(limits))),
                case_id=case.id, input_hash=packages.input_hash(case.input),
                package_hash=digest(package.model_dump()),
                selected_example_ids=[row.id for row in selected],
                events=events, calls=calls, candidate=candidate, validation=validation,
                stop_reason=stop_reason, error=error, elapsed_seconds=time.monotonic() - started,
                total_tokens=known_tokens if tokens_known else None, known_total_tokens=known_tokens,
                cost_usd=known_cost if costs_known else None)
