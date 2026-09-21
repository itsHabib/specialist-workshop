"""Freeze and run independent model investigations on synthetic repair tasks."""
import argparse
import fcntl
import os
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import random
import time

import models
import sandbox
import workload

ROOT = Path(__file__).resolve().parent
GUIDANCE = ("Diagnose the earliest incorrect intermediate result. Check input interpretation and "
            "assumptions as well as the central algorithm. State a hypothesis and prediction. "
            "Use discriminating probes, preserve passing cases, and prefer a small justified repair. "
            "Do not change the requested contract. You may implement a complete solution if simpler. ")


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def write(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def plan(root, seed, variant, repeats, backend="docker"):
    if root.exists():
        raise ValueError("plan already exists; use a new directory")
    if not 1 <= repeats <= 3:
        raise ValueError("pilot repeats must be 1..3")
    root.mkdir(parents=True, mode=0o700)
    trials = []
    for repeat in range(repeats):
        task = workload.task(seed + repeat, variant)
        order = ['opus-solo', 'opus-team', 'astra-solo']
        random.Random(seed + repeat).shuffle(order)
        for arm in order:
            name = f'{repeat + 1:02d}-{arm}'
            write(root / name / 'task.json', task)
            trials.append({'name': name, 'arm': arm, 'task_sha256': digest(task)})
    manifest = {'seed': seed, 'variant': variant, 'trials': trials, 'max_calls': 6,
                'call_timeout': 150, 'image': sandbox.IMAGE, 'backend': backend,
                'models': {'opus': 'claude-opus-5', 'astra': 'gpt-6-astra'},
                'effort': 'high', 'protocol_sha256': hashlib.sha256((ROOT/'PROTOCOL.md').read_bytes()).hexdigest(),
                'code': {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(ROOT.glob('*.py'))}}
    if backend == 'rooms':
        config_path = Path(os.environ['BRIDGE_ROOMS_CONFIG']).resolve()
        manifest['rooms_config'] = str(config_path)
        manifest['rooms_config_sha256'] = hashlib.sha256(config_path.read_bytes()).hexdigest()
    write(root / 'plan.json', manifest)
    return manifest


def grade(source, requests, image, stop):
    expected = [workload.reference(request) for request in requests]
    backend = sandbox
    if os.environ.get("BRIDGE_ACTIVE_BACKEND") == "rooms":
        import rooms_backend
        backend = rooms_backend
        image = os.environ["BRIDGE_ROOMS_CONFIG"]
    actual = backend.evaluate(source, requests, image=image, stop=stop)
    failures = []
    complete = isinstance(actual.get('outputs'), list) and len(actual['outputs']) == len(requests)
    if not complete:
        actual['error'] = actual.get('error') or 'missing_or_truncated_outputs'
    if complete:
        failures = [{'input': request, 'expected': answer, 'actual': got}
                    for request, answer, got in zip(requests, expected, actual['outputs']) if got != answer]
    return {'passed': actual['error'] is None and not failures, 'total': len(requests),
            'failed': len(failures) if complete else len(requests),
            'failures': failures, 'execution': actual}


def prompt(task, source, evidence, role, history, remaining=6):
    instruction = "You are the sole solver. "
    if role.startswith('investigator'):
        instruction = ('Independently investigate this candidate. Do not edit it. Return action hypothesis '
                       'with a concrete diagnosis, prediction and optionally up to four distinguishing '
                       'input probes as probes_json. Source must be empty. ')
        if role.endswith('2'):
            instruction += 'Focus on representation, conventions, invariants, and overlooked assumptions. '
    if role == 'integrator':
        instruction = ('You integrate independent investigations. Decide using the contract and executable '
                       'evidence, not majority agreement. You alone edit the candidate. ')
    return instruction + GUIDANCE + (
        'Return the action JSON only. edit supplies the full self-contained solution.py as source; '
        'probe requests up to four JSON inputs in probes_json for execution on the current candidate; '
        'finish seals the current candidate. An edit passing all development checks is automatically '
        'sealed for final evaluation; the last permitted call also seals. '
        'Empty fields are empty strings; unused probes_json is []. '
        'No tools, filesystem, network or outside context. The harness executes code and supplies feedback. '
        'Use only Python standard library.\n' + json.dumps({
            'contract': task['contract'], 'source': source, 'remaining_calls': remaining,
            'development_inputs': task['development'],
            'development_feedback': {k:v for k,v in evidence.items() if k != 'execution'},
            'history': history}))


def ask(directory, number, provider, text, manifest, stop):
    output = directory / 'calls' / str(number)
    # A reserved call is never silently retried after interruption.
    if output.exists():
        receipt = output / 'receipt.json'
        if not receipt.exists():
            raise RuntimeError(f'call {number} outcome unknown; retain this run and start a new trial')
        return json.loads(receipt.read_text())
    return models.complete(text, provider, output, timeout=manifest['call_timeout'], stop=stop)


def validate(response, investigator=False):
    if not isinstance(response, dict) or set(response) != set(models.SCHEMA['required']):
        raise ValueError('wrong action fields')
    if any(not isinstance(value, str) for value in response.values()):
        raise ValueError('action fields must be strings')
    if response['action'] not in ('hypothesis', 'probe', 'edit', 'finish'):
        raise ValueError('unknown action')
    if investigator and (response['action'] != 'hypothesis' or response['source']):
        raise ValueError('investigators provide hypotheses, not edits')
    if response['action'] != 'edit' and response['source']:
        raise ValueError('only edit supplies source')
    if len(response['source'].encode()) > 100_000:
        raise ValueError('source exceeds 100 KB')
    probes = json.loads(response['probes_json'])
    if not isinstance(probes, list) or len(probes) > 4 or len(json.dumps(probes)) > 30_000:
        raise ValueError('at most four bounded probes')
    return probes


def trial(root, entry, manifest):
    directory = root / entry['name']; stop = root / 'STOP'
    if (directory / 'result.json').exists():
        return json.loads((directory / 'result.json').read_text())
    task = json.loads((directory / 'task.json').read_text())
    if digest(task) != entry['task_sha256']:
        raise ValueError('task changed after planning')
    source = task['starter']; history = []; calls = []
    development = grade(source, task['development'], manifest['image'], stop)
    write(directory / 'baseline.json', development)
    provider = 'astra' if entry['arm'] == 'astra-solo' else 'opus'
    error = None; began = time.monotonic()
    if entry['arm'] == 'opus-team' and not stop.exists():
        def investigate(number):
            role = f'investigator-{number}'
            return role, ask(directory, number, provider, prompt(task, source, development, role, []), manifest, stop)
        with ThreadPoolExecutor(max_workers=2) as pool:
            investigations = list(pool.map(investigate, (1, 2)))
        for role, receipt in investigations:
            calls.append(dict(receipt, role=role))
            if receipt['error']:
                history.append({'role':role,'unavailable_advice':receipt['error']}); continue
            try:
                probes = validate(receipt['response'], investigator=True)
                observation = grade(source, probes, manifest['image'], stop) if probes else None
                history.append({'role': role, 'response': receipt['response'], 'probe_result': observation})
            except (ValueError, TypeError, KeyError, IndexError, ZeroDivisionError) as failure:
                history.append({'role':role,'response':receipt['response'],'invalid_probe':str(failure)})
    while len(calls) < manifest['max_calls'] and error is None and not stop.exists():
        role = 'integrator' if entry['arm'] == 'opus-team' else 'solver'
        receipt = ask(directory, len(calls)+1, provider, prompt(task, source, development, role, history,
                      manifest['max_calls']-len(calls)), manifest, stop)
        calls.append(dict(receipt, role=role))
        if receipt['error']:
            error = receipt['error']; break
        response = receipt['response']
        try:
            probes = validate(response)
            action = response['action']
            row = {'role': role, 'response': response}
            if action == 'edit':
                if not response['source']:
                    raise ValueError('empty edit')
                source = response['source']
                (directory / f'candidate-{len(calls)}.py').write_text(source)
                development = grade(source, task['development'], manifest['image'], stop)
                row['development'] = development
            if probes:
                try:
                    row['probe_result'] = grade(source, probes, manifest['image'], stop)
                except (ValueError, TypeError, KeyError, IndexError, ZeroDivisionError) as failure:
                    row['invalid_probe'] = str(failure)
            history.append(row)
            write(directory / 'history.json', history)
            if action == 'finish' or (action == 'edit' and development['passed']):
                break
        except (ValueError, TypeError, KeyError, IndexError, ZeroDivisionError) as failure:
            history.append({'harness_feedback': str(failure)})
    (directory / 'final.py').write_text(source)
    # The final set is used once, after the final source is sealed. No repair feedback.
    final = grade(source, task['final'], manifest['image'], stop) if error is None and not stop.exists() else None
    result = {'arm': entry['arm'], 'task_id': task['id'], 'task_sha256': entry['task_sha256'],
              'candidate_sha256': hashlib.sha256(source.encode()).hexdigest(),
              'status': 'finished' if final else 'interrupted', 'error': error,
              'accepted': bool(final and final['passed'] and development['passed']),
              'calls': calls, 'development': development, 'final': final,
              'seconds': time.monotonic()-began}
    write(directory / 'result.json', result)
    return result


def run_locked(root):
    manifest = json.loads((root / 'plan.json').read_text())
    for name, expected in manifest['code'].items():
        if hashlib.sha256((ROOT/name).read_bytes()).hexdigest() != expected:
            raise ValueError('runner changed after planning: ' + name)
    if hashlib.sha256((ROOT/'PROTOCOL.md').read_bytes()).hexdigest() != manifest['protocol_sha256']:
        raise ValueError('protocol changed after planning')
    os.environ['BRIDGE_ACTIVE_BACKEND'] = manifest['backend']
    if manifest['backend'] == 'rooms':
        config = Path(manifest['rooms_config'])
        if hashlib.sha256(config.read_bytes()).hexdigest() != manifest['rooms_config_sha256']:
            raise ValueError('Rooms configuration changed after planning')
        os.environ['BRIDGE_ROOMS_CONFIG'] = str(config)
    rows = [{'name': e['name'], 'arm': e['arm'], 'accepted': False, 'status': 'not_started',
             'calls': 0, 'error': None} for e in manifest['trials']]
    write(root/'summary.json', rows)
    for index, entry in enumerate(manifest['trials']):
        if (root/'STOP').exists():
            break
        try:
            result = trial(root, entry, manifest)
        except (RuntimeError, OSError) as failure:
            attempts = list((root/entry['name']/'calls').glob('*'))
            rows[index].update(status='interrupted', calls=len(attempts), error=str(failure))
            write(root/'summary.json', rows)
            raise
        rows[index] = {'name': entry['name'], 'arm': result['arm'], 'accepted': result['accepted'],
                       'calls': len(result['calls']), 'status': result['status'], 'error': result['error']}
        print(json.dumps(rows[index]), flush=True)
        write(root/'summary.json', rows)
        if result['status'] == 'interrupted':
            break
    return rows


def run(root):
    with (root/'driver.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        return run_locked(root)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['plan','run'])
    parser.add_argument('directory', type=Path)
    parser.add_argument('--seed', type=int, default=1729)
    parser.add_argument('--variant', default='hard')
    parser.add_argument('--repeats', type=int, default=1)
    parser.add_argument('--backend', choices=['docker','rooms'], default='docker')
    args = parser.parse_args(); root = args.directory.resolve()
    if args.command == 'plan':
        print(json.dumps(plan(root,args.seed,args.variant,args.repeats,args.backend), indent=2))
        return
    run(root)


if __name__ == '__main__':
    main()
