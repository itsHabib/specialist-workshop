"""Fake transports test orchestration mechanics, never model capability."""
import json
from pathlib import Path

import pytest

from workshop import packages
from workshop.policies import Limits, Policy, input_token_bound, run_case, select_examples


@pytest.fixture
def package():
    payload = json.loads((Path(__file__).resolve().parents[1] / 'examples/support-intake.package.json').read_text())
    return packages.Package.model_validate(payload)


class Transport:
    def __init__(self, outputs):
        self.outputs = iter(outputs)
        self.requests = []

    def __call__(self, model, messages, max_output_tokens):
        self.requests.append(dict(model=model, messages=messages, max_output_tokens=max_output_tokens))
        value = next(self.outputs)
        if isinstance(value, Exception):
            raise value
        if isinstance(value, dict):
            return value
        return dict(raw=value, model=model + '-resolved', usage=dict(input_tokens=12, output_tokens=8), cost_usd=.01)


def run(package, transport, **kwargs):
    return run_case(package, package.development[0], kwargs.pop('policy', Policy('test', 'cheap')),
                    kwargs.pop('limits', Limits()), transport, **kwargs)


def test_no_heldout_answer_or_other_eval_input_reaches_model(package):
    altered = package.model_copy(deep=True)
    altered.development[0].expected = {'secret': 'GOLD_NEVER_PROMPT_THIS'}
    altered.final[0].input = 'FINAL_INPUT_NEVER_PROMPT_THIS'
    altered.final[0].expected = {'secret': 'FINAL_GOLD_NEVER_PROMPT_THIS'}
    transport = Transport(['bad', 'find issues', '{}'])
    row = run(altered, transport, policy=Policy('pair', 'cheap', 'critic', rounds=2, examples=1))
    prompts = json.dumps(transport.requests)
    assert 'GOLD_NEVER_PROMPT_THIS' not in prompts
    assert 'FINAL_INPUT_NEVER_PROMPT_THIS' not in prompts
    assert row['validation']['correct'] is None
    assert all(c['role'] in ('solver', 'critic') for c in row['calls'])


def test_critic_sees_current_candidate_not_solver_conversation(package):
    transport = Transport(['old candidate', 'first critique', 'new candidate', 'second critique', 'last candidate'])
    row = run(package, transport, policy=Policy('pair', 'cheap', 'critic', rounds=3))
    assert [c['role'] for c in row['calls']] == ['solver', 'critic', 'solver', 'critic', 'solver']
    second_critic = json.dumps(transport.requests[3]['messages'])
    assert 'new candidate' in second_critic
    assert 'old candidate' not in second_critic and 'first critique' not in second_critic
    assert all(m['role'] != 'assistant' for m in transport.requests[3]['messages'])
    assert row['candidate']['raw'] == 'last candidate'


def test_last_solver_candidate_wins_even_when_worse(package):
    first = json.dumps(package.development[0].expected)
    transport = Transport([first, 'this critique is not an answer', 'invalid json'])
    row = run(package, transport, policy=Policy('pair', 'cheap', 'critic', rounds=2))
    assert row['candidate']['raw'] == 'invalid json'
    assert row['validation']['valid'] is False
    assert row['validation']['correct'] is None
    assert row['stop_reason'] == 'rounds_completed'
    assert row['total_tokens'] == 60 and row['cost_usd'] == pytest.approx(.03)


def test_critics_share_call_budget_and_never_become_candidate(package):
    transport = Transport(['candidate', 'critique'])
    row = run(package, transport, policy=Policy('pair', 'cheap', 'critic', rounds=5), limits=Limits(max_calls=2))
    assert len(transport.requests) == 2
    assert row['stop_reason'] == 'call_budget'
    assert row['candidate']['raw'] == 'candidate'
    assert row['total_tokens'] == 40


def test_total_budget_reserves_output_before_initial_call(package):
    transport = Transport([])
    row = run(package, transport, limits=Limits(max_total_tokens=1024))
    assert row['stop_reason'] == 'token_budget'
    assert transport.requests == [] and row['calls'] == []
    assert row['candidate'] is None


def test_prior_usage_and_critic_prompt_share_total_budget(package):
    probe = Transport(['answer'])
    initial = run(package, probe)
    bound = initial['calls'][0]['input_token_bound']
    transport = Transport([dict(raw='answer', model='resolved', usage=dict(input_tokens=bound, output_tokens=100))])
    row = run(package, transport, policy=Policy('pair', 'cheap', 'critic', rounds=3),
              limits=Limits(max_total_tokens=bound+1024))
    assert len(transport.requests) == 1
    assert row['stop_reason'] == 'token_budget'
    assert row['total_tokens'] == bound+100


def test_byte_bound_counts_unicode_without_truncation():
    text = '🦊' * 10
    assert input_token_bound([dict(role='user', content=text)]) == 1024 + 256 + len(text.encode())


@pytest.mark.parametrize('usage', [None, {}, {'input_tokens': True, 'output_tokens': 2},
                                  {'input_tokens': -1, 'output_tokens': 2},
                                  {'input_tokens': 1.5, 'output_tokens': 2}])
def test_missing_or_invalid_usage_stops_and_is_not_free(package, usage):
    transport = Transport([dict(raw='answer', model='resolved', usage=usage)])
    row = run(package, transport, policy=Policy('solo', 'cheap', rounds=3))
    assert row['stop_reason'] == 'usage_unknown'
    assert row['total_tokens'] is None and row['cost_usd'] is None
    assert row['candidate']['raw'] == 'answer'
    assert len(row['calls']) == 1


def test_failed_call_preserves_previous_candidate_and_uncertain_spend(package):
    transport = Transport(['candidate', RuntimeError('SECRET_KEY=do-not-log')])
    events = []
    row = run(package, transport, policy=Policy('solo', 'cheap', rounds=3), record=events.append)
    assert row['stop_reason'] == 'transport_error' and row['error'] == 'RuntimeError'
    assert row['candidate']['raw'] == 'candidate'
    assert row['total_tokens'] is None and row['known_total_tokens'] == 20
    assert row['cost_usd'] is None
    assert [e['type'] for e in events] == ['before_call', 'after_call', 'before_call', 'after_call']
    assert 'SECRET_KEY' not in json.dumps(row)


def test_provider_failure_usage_is_accounted_but_message_redacted(package):
    transport = Transport([dict(raw='', model='resolved', usage=dict(input_tokens=9, output_tokens=4),
                                cost_usd=.02, error='token SECRET leaked here')])
    row = run(package, transport)
    assert row['total_tokens'] == 13 and row['cost_usd'] == .02
    assert row['candidate'] is None and row['error'] == 'ProviderError'
    assert 'SECRET' not in json.dumps(row)


def test_usage_beyond_reservation_halts_next_call(package):
    transport = Transport([dict(raw='answer', model='resolved', usage=dict(input_tokens=9, output_tokens=1025))])
    row = run(package, transport, policy=Policy('solo', 'cheap', rounds=2))
    assert row['stop_reason'] == 'budget_exceeded'
    assert row['total_tokens'] == 1034 and len(row['calls']) == 1


def test_train_retrieval_fixed_and_ties_stable(package):
    # Compare to the declared input-only overlap rank, not evaluation quality.
    selected = select_examples(package, package.development[0].input, 2)
    transport = Transport(['candidate', 'critique', 'revised'])
    row = run(package, transport, policy=Policy('pair', 'cheap', 'critic', rounds=2, examples=2))
    ids = [c.id for c in selected]
    assert row['selected_example_ids'] == ids
    for request in transport.requests:
        assert all(case_id in json.dumps(request['messages']) for case_id in ids)
    assert set(ids) <= {c.id for c in package.train}
    backwards = package.model_copy(deep=True)
    backwards.train.reverse()
    assert [r.id for r in select_examples(backwards, 'unlikely_no_overlap', 2)] == [r.id for r in select_examples(package, 'unlikely_no_overlap', 2)]


def test_config_identity_stable_and_sensitive_to_limits(package):
    first = run(package, Transport(['a']))
    second = run(package, Transport(['different']))
    third = run(package, Transport(['a']), limits=Limits(max_calls=5))
    assert first['config_id'] == second['config_id']
    assert first['config_id'] != third['config_id']
    assert first['calls'][0]['request_digest'] == second['calls'][0]['request_digest']
    assert first['candidate']['hash'] != second['candidate']['hash']


def test_checkpoint_failure_before_call_prevents_spend(package):
    def record(event):
        raise OSError('secret filesystem path')
    transport = Transport([])
    row = run(package, transport, record=record)
    assert not transport.requests
    assert row['stop_reason'] == 'record_error' and row['error'] == 'OSError'
    assert len(row['events']) == 1 and row['events'][0]['type'] == 'before_call'


def test_checkpoint_failure_after_call_preserves_paid_receipt(package):
    def record(event):
        if event['type'] == 'after_call':
            raise OSError('secret filesystem path')
    row = run(package, Transport(['answer']), record=record, policy=Policy('solo', 'cheap', rounds=2))
    assert row['stop_reason'] == 'record_error'
    assert row['candidate']['raw'] == 'answer'
    assert row['total_tokens'] == 20 and row['cost_usd'] == .01
    assert len(row['calls']) == 1


@pytest.mark.parametrize('kwargs', [dict(rounds=0), dict(rounds=True), dict(examples=-1), dict(solver='')])
def test_invalid_policy_rejected(kwargs):
    with pytest.raises(ValueError):
        Policy(**(dict(name='test', solver='cheap') | kwargs))


@pytest.mark.parametrize('kwargs', [dict(max_calls=0), dict(max_output_tokens=0), dict(max_total_tokens=True)])
def test_invalid_limits_rejected(kwargs):
    with pytest.raises(ValueError):
        Limits(**kwargs)


def test_provider_token_breakdown_and_response_id_survive_receipts(package):
    breakdown = dict(input_tokens=100, output_tokens=40,
                     input_tokens_details=dict(cached_tokens=50, cache_write_tokens=10),
                     output_tokens_details=dict(reasoning_tokens=20),
                     cache_creation=dict(ephemeral_5m_input_tokens=10, ephemeral_1h_input_tokens=0))
    transport = Transport([dict(raw='answer', model='actual', usage=dict(input_tokens=110, output_tokens=40),
                                provider_usage=breakdown, response_id='resp_test-123', cost_usd=.05)])
    recorded = []
    row = run(package, transport, record=recorded.append)
    call = row['calls'][0]
    assert call['provider_usage'] == breakdown
    assert call['response_id'] == 'resp_test-123'
    after = [e for e in recorded if e['type'] == 'after_call'][0]
    assert after['provider_usage'] == breakdown and after['response_id'] == 'resp_test-123'
    assert row['total_tokens'] == 150 and row['cost_usd'] == .05


def test_provider_usage_filters_strings_nonfinite_and_unrecognized_metadata(package):
    transport = Transport([dict(raw='answer', model='actual', usage=dict(input_tokens=1, output_tokens=1),
                                provider_usage=dict(input_tokens=1, output_tokens='SECRET',
                                    cached_tokens=float('nan'), reasoning_tokens=True,
                                    arbitrary_secret='private-token', unknown_numeric_field=42,
                                    input_tokens_details=dict(cached_tokens=7, output_tokens='SECRET')),
                                response_id='not an id\nSECRET')])
    row = run(package, transport)
    assert row['calls'][0]['provider_usage'] == dict(input_tokens=1, input_tokens_details=dict(cached_tokens=7))
    assert row['calls'][0]['response_id'] is None
    assert 'SECRET' not in json.dumps(row, allow_nan=False)


@pytest.mark.parametrize('error', ['HTTP_401', 'HTTP_429', 'HTTP_503', 'incomplete_response'])
def test_safe_provider_error_codes_remain_diagnosable(package, error):
    row = run(package, Transport([dict(raw='', model='actual', usage=None, response_id='msg_test', error=error)]))
    assert row['stop_reason'] == 'transport_error' and row['error'] == error
    assert row['calls'][0]['error'] == error
    assert row['events'][-1]['error'] == error
    assert row['calls'][0]['response_id'] == 'msg_test'
    assert row['total_tokens'] is None
