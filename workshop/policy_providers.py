"""Explicit model aliases over OpenAI Responses or Anthropic Messages.

No tools or filesystem access is exposed to models. Credentials remain in the
process environment; no provider exception bodies or credentials enter receipts.
"""
import math
import os
import httpx


def validate_models(models):
    if not isinstance(models, dict) or not models:
        raise ValueError('models must be a nonempty alias map')
    for alias, spec in models.items():
        if not isinstance(spec, dict) or spec.get('provider') not in ('openai', 'anthropic'):
            raise ValueError('each model needs provider openai or anthropic')
        if not isinstance(spec.get('model'), str) or not spec['model'].strip():
            raise ValueError('each model needs an explicit model ID')
        if spec['provider'] == 'anthropic' and spec.get('effort'):
            raise ValueError('Anthropic effort is not supported in this adapter')
        if set(spec) - {'provider', 'model', 'effort', 'rates'}:
            raise ValueError('unknown model configuration field')
        rates = spec.get('rates')
        if rates is not None:
            if set(rates) != {'input', 'output', 'cached', 'cache_write'}:
                raise ValueError('rates require input/output/cached/cache_write per million tokens')
            if any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) or v < 0 for v in rates.values()):
                raise ValueError('invalid rates')


def credentials_available(models):
    return all(os.environ.get('OPENAI_API_KEY' if spec['provider'] == 'openai' else 'ANTHROPIC_API_KEY') for spec in models.values())


class Providers:
    def __init__(self, models, client=None):
        validate_models(models)
        self.models = models
        self.client = client or httpx.Client(timeout=60, follow_redirects=False)

    def close(self):
        self.client.close()

    def __call__(self, model, messages, max_output_tokens):
        spec = self.models[model]
        if spec['provider'] == 'openai':
            payload = dict(model=spec['model'], input=messages, max_output_tokens=max_output_tokens, store=False)
            if spec.get('effort'):
                payload['reasoning'] = {'effort': spec['effort']}
            response = self.client.post('https://api.openai.com/v1/responses', json=payload,
                                       headers={'Authorization': 'Bearer ' + os.environ['OPENAI_API_KEY']})
        else:
            payload = dict(model=spec['model'], max_tokens=max_output_tokens,
                           system='\n'.join(m['content'] for m in messages if m['role'] == 'system'),
                           messages=[m for m in messages if m['role'] != 'system'])
            if spec.get('effort'):
                raise ValueError('Anthropic effort configuration not implemented; omit effort')
            response = self.client.post('https://api.anthropic.com/v1/messages', json=payload,
                                       headers={'x-api-key': os.environ['ANTHROPIC_API_KEY'], 'anthropic-version': '2023-06-01'})
        if response.status_code != 200:
            return dict(raw='', model=spec['model'], usage=None, cost_usd=None,
                        error='HTTP_' + str(response.status_code))
        body = response.json()
        return decode(spec, body)


def decode(spec, body):
    usage = body.get('usage')
    raw_usage = usage
    if spec['provider'] == 'openai':
        raw = ''.join(c['text'] for item in body.get('output', []) for c in item.get('content', []) if c.get('type') == 'output_text')
        error = None if body.get('status') == 'completed' else 'incomplete_response'
        cached = (usage or {}).get('input_tokens_details', {}).get('cached_tokens', 0)
        writes = (usage or {}).get('input_tokens_details', {}).get('cache_write_tokens', 0)
        billable = (usage or {}).get('input_tokens', 0) - cached - writes
    else:
        raw = ''.join(c['text'] for c in body.get('content', []) if c.get('type') == 'text')
        error = None if body.get('stop_reason') == 'end_turn' else 'incomplete_response'
        cached = (usage or {}).get('cache_read_input_tokens', 0)
        writes = (usage or {}).get('cache_creation_input_tokens', 0)
        billable = (usage or {}).get('input_tokens', 0)
        if usage is not None:
            usage = dict(usage, input_tokens=billable + cached + writes)
    cost = None
    rates = spec.get('rates')
    if usage is not None and rates is not None:
        cost = (billable * rates['input'] + cached * rates['cached'] + writes * rates['cache_write'] + usage['output_tokens'] * rates['output']) / 1e6
    return dict(raw=raw, model=body.get('model', spec['model']), usage=usage,
                provider_usage=raw_usage, response_id=body.get('id'), cost_usd=cost, error=error)
