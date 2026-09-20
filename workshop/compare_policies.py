"""Compare complete model policies on existing task packages; no gold feedback.

python -m workshop.compare_policies --package examples/support-intake.package.json
  --config examples/capability-policies.json --output .state/policy-run --dry-run
"""
import argparse
import html
import json
import math
from pathlib import Path
import random
import time

from workshop import packages, store
from workshop.domain import digest
from workshop.policies import Policy, Limits, run_case
from workshop.policy_providers import Providers, validate_models, credentials_available


def configuration(path):
    data = json.loads(path.read_text())
    if set(data) - {'models', 'policies', 'limits', 'max_run_calls', 'max_run_cost_usd', 'seed'}:
        raise ValueError('unknown comparison configuration field')
    validate_models(data['models'])
    policies = [Policy(**p) for p in data['policies']]
    limits = Limits(**data.get('limits', {}))
    if not policies or len({p.name for p in policies}) != len(policies):
        raise ValueError('policies require unique names')
    for p in policies:
        if p.solver not in data['models'] or (p.critic is not None and p.critic not in data['models']):
            raise ValueError('policy references unknown model alias')
    maximum = data.get('max_run_calls', 24)
    if isinstance(maximum, bool) or not isinstance(maximum, int) or not 1 <= maximum <= 1000:
        raise ValueError('max_run_calls must be 1..1000')
    budget = data.get('max_run_cost_usd')
    if budget is not None:
        if isinstance(budget, bool) or not isinstance(budget, (int, float)) or not math.isfinite(budget) or budget <= 0:
            raise ValueError('max_run_cost_usd must be finite and positive')
        if any('rates' not in spec for spec in data['models'].values()):
            raise ValueError('dollar reservation requires rates for every model')
    return data, policies, limits


def summarize(rows):
    groups = {}
    for row in rows:
        name = row['policy']['name']
        groups.setdefault(name, []).append(row)
    result = []
    for name, group in groups.items():
        success = sum(row['final_score']['accepted'] for row in group)
        costs = [row['cost_usd'] for row in group]
        cost = sum(costs) if all(c is not None for c in costs) else None
        result.append(dict(policy=name, tasks=len(group), successful=success,
                           calls=sum(len(row['calls']) for row in group),
                           total_cost_usd=cost, cost_per_success_usd=cost/success if cost is not None and success else None,
                           elapsed_seconds=sum(row['elapsed_seconds'] for row in group)))
    return result


def report(record):
    columns = ['policy', 'tasks', 'successful', 'calls', 'total_cost_usd', 'cost_per_success_usd', 'elapsed_seconds']
    rows = summarize(record['rows'])
    cells = lambda row: ''.join('<td>' + html.escape(str(row.get(k))) + '</td>' for k in columns)
    return ('<!doctype html><meta charset="utf-8"><meta name="viewport" content="width=device-width">'
            '<title>Model policy comparison</title><style>body{font:16px system-ui;max-width:1000px;margin:3rem auto;padding:1rem}td,th{text-align:left;padding:.6rem;border-bottom:1px solid #ddd}table{display:block;overflow:auto}</style>'
            '<h1>Model policy comparison</h1><p>Whole-run call accounting: ' + html.escape(json.dumps(record.get('call_accounting', {}))) + '</p><p>Status: ' + html.escape(record['status']) +
            ' · Split: ' + html.escape(record['split']) + '</p><p>Costs include every recorded role and failed attempt. '
            'Only attempted tasks appear below; compare plan and status before interpreting incomplete runs. None means unavailable, not free. Synthetic starter tasks do not establish general model parity.</p><table><tr>' +
            ''.join('<th>' + k + '</th>' for k in columns) + '</tr>' + ''.join('<tr>' + cells(row) + '</tr>' for row in rows) +
            '</table><p>Read run.json for candidate identities, critiques, usage and failures. '
            'This report contains only final scores; they were not fed back into model calls.</p>')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--package', type=Path, required=True)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--split', choices=['development', 'final'], default='development')
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args(argv)
    package = packages.Package.model_validate_json(args.package.read_text())
    config, policies, limits = configuration(args.config)
    plan = [(case.id, p.name) for case in getattr(package, args.split) for p in policies]
    random.Random(config.get('seed', 0)).shuffle(plan)
    record = dict(status='planned', split=args.split, package=packages.summary(package),
                  config=config, config_hash=digest(config), plan=plan, rows=[],
                  implementation_hash=digest({p.name:p.read_text() for p in Path(__file__).parent.glob('*.py')}),
                  created_at=time.time(), started_calls=0, reserved_usd=0.0,
                  call_accounting=dict(finished=0, known_cost_usd=0.0, unknown_cost_calls=0, total_cost_usd=0.0))
    if args.dry_run:
        print(json.dumps(record, indent=2))
        return 0
    if not credentials_available(config['models']):
        raise ValueError('Missing provider API credentials; dry-run needs none')
    args.output.mkdir(parents=True, exist_ok=False)
    store.atomic_json(args.output/'package.json', package.model_dump())
    provider = Providers(config['models'])
    record['status'] = 'running'
    store.atomic_json(args.output/'run.json', record)
    def checkpoint(event):
        event = dict(event, case_id=case_id, policy=name)
        record['in_progress'] = event
        if event.get('type') == 'before_call':
            if record['started_calls'] >= config.get('max_run_calls', 24):
                record['status'] = 'budget_exhausted'
                raise RuntimeError('run_call_budget_exhausted')
            rates = config['models'][event['requested_model']].get('rates')
            budget = config.get('max_run_cost_usd')
            if budget is not None:
                if event['input_token_bound'] > 128000:
                    raise ValueError('priced adapter supports at most 128000 input tokens')
                reserve = (event['input_token_bound'] * max(rates['input'], rates['cached'], rates['cache_write']) + event['max_output_tokens'] * rates['output']) / 1e6
                if record['reserved_usd'] + reserve > budget:
                    record['status'] = 'budget_exhausted'
                    raise RuntimeError('run_cost_budget_exhausted')
                record['reserved_usd'] += reserve
            record['started_calls'] += 1
        accounting = record['call_accounting']
        if event.get('type') == 'after_call':
            accounting['finished'] += 1
            cost = event.get('cost_usd')
            if cost is None:
                accounting['unknown_cost_calls'] += 1
            else:
                accounting['known_cost_usd'] += cost
        accounting['total_cost_usd'] = accounting['known_cost_usd'] if accounting['finished'] == record['started_calls'] and not accounting['unknown_cost_calls'] else None
        with (args.output/'events.jsonl').open('a') as stream:
            stream.write(json.dumps(event) + '\n')
            stream.flush()
            import os
            os.fsync(stream.fileno())
        store.atomic_json(args.output/'run.json', record)
    try:
        cases = {case.id:case for case in getattr(package, args.split)}
        by_name = {p.name:p for p in policies}
        for case_id, name in plan:
            if record['started_calls'] >= config.get('max_run_calls', 24):
                record['status'] = 'budget_exhausted'
                break
            case = cases[case_id]
            row = run_case(package, case, by_name[name], limits, provider, record=checkpoint)
            # Gold comparison happens after policy termination, never in feedback.
            row['final_score'] = packages.score(package, case.input, (row['candidate'] or {}).get('raw', ''), case.expected)
            if row['stop_reason'] in {'transport_error', 'usage_unknown', 'budget_exceeded', 'record_error'}:
                row['final_score']['accepted'] = False
                if record['status'] != 'budget_exhausted':
                    record['status'] = 'interrupted_or_failed'
            record['rows'].append(row)
            record.pop('in_progress', None)
            store.atomic_json(args.output/'run.json', record)
            if record['status'] in {'interrupted_or_failed', 'budget_exhausted'}:
                break
        if record['status'] == 'running':
            record['status'] = 'completed'
    except BaseException as exc:
        record['status'] = 'interrupted_or_failed'
        record['error'] = type(exc).__name__
        raise
    finally:
        provider.close()
        record['finished_at'] = time.time()
        record['summary'] = summarize(record['rows'])
        store.atomic_json(args.output/'run.json', record)
        (args.output/'report.html').write_text(report(record))
    print(args.output/'report.html')
    return 0 if record['status'] == 'completed' else 1


if __name__ == '__main__':
    raise SystemExit(main())
