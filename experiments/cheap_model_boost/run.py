"""A frozen, paired pilot: solo retries versus a same-model critic and repair."""
import hashlib
import json
from pathlib import Path
import random
import sys
import urllib.request

EXP = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(EXP/'engineering_screen'), str(EXP/'investigation_bridge')]
from tasks import GEOMETRY, cases, geometry
from screen import grade, reference_source
from models import complete


def write(path, value):
    path.write_text(json.dumps(value, indent=2)+'\n')


def hashes():
    paths = [Path(__file__), EXP/'engineering_screen/tasks.py', EXP/'engineering_screen/screen.py',
             EXP/'investigation_bridge/models.py', EXP/'investigation_bridge/model_supervisor.py',
             EXP/'investigation_bridge/sandbox.py', Path(__file__).with_name('PROTOCOL.md')]
    return {str(p.relative_to(EXP)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}


def freeze(root):
    root.mkdir(parents=True, exist_ok=False, mode=0o700)
    development = cases(48123)['geometry']
    development = development[:2]+development[-2:]
    seen = {json.dumps(q, sort_keys=True) for q in development}
    final = [q for q in cases(902713)['geometry'] if json.dumps(q, sort_keys=True) not in seen]
    tags = json.load(urllib.request.urlopen('http://127.0.0.1:11434/api/tags'))
    model = next(m for m in tags['models'] if m['name']=='llama3.2:1b')
    order = ['sonnet', 'haiku', 'llama3.2:1b']; random.Random(724).shuffle(order)
    write(root/'plan.json', {'files': hashes(), 'models': order, 'local_model': model,
          'development': development, 'final': final, 'timeout_seconds': 240,
          'max_calls_per_model': 5, 'native_per_call_budget_usd': 0.8})


def evaluate(root, label, source, queries):
    result = grade('geometry', source, queries, root/'STOP')
    write(root/(label+'.json'), result)
    return result


def feedback(result):
    # Development cases and outputs only. Never final cases or answers.
    return json.dumps(result)


def run(root):
    import fcntl
    with (root/'driver.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX|fcntl.LOCK_NB)
        plan = json.loads((root/'plan.json').read_text())
        if plan['files'] != hashes(): raise RuntimeError('frozen source changed')
        if (root/'started').exists(): raise RuntimeError('reserved run: inspect receipts, do not retry')
        (root/'started').touch()
        for name, source, passes in [('reference', reference_source('geometry'), True), ('invalid', 'def evaluate(q): return {}', False)]:
            result = evaluate(root, 'control-'+name, source, plan['final'])
            if result['passed'] != passes: raise RuntimeError('control failed')
        base = GEOMETRY+'\nReturn the required JSON envelope. action=edit, source=complete Python code; hypothesis/prediction may be brief, probes_json="[]". No tools.\nDevelopment examples:\n'+json.dumps([{'input': q, 'expected': geometry(q)} for q in plan['development']])
        summary = []
        for model in plan['models']:
            folder = root/model.replace(':','-'); folder.mkdir()
            calls = []
            def call(label, prompt):
                if (root/'STOP').exists(): raise RuntimeError('stopped')
                receipt = complete(prompt, model, folder/label, timeout=240, stop=root/'STOP')
                calls.append({'label':label, 'receipt':receipt})
                print(model, label, receipt['error'], round(receipt['seconds'],1), flush=True)
                return (receipt.get('response') or {}).get('source','') if not receipt['error'] else ''
            def dev(label, source):
                (folder/(label+'.py')).write_text(source)
                return evaluate(folder,label+'-dev',source,plan['development'])
            initial = call('initial',base); initial_dev = dev('initial',initial)
            current = initial; result = initial_dev
            for n in (2,3):
                current = call('solo-'+str(n),base+'\nRevise this candidate using development execution feedback.\n'+current+'\nFeedback: '+feedback(result))
                result = dev('solo-'+str(n),current)
            critic_prompt = GEOMETRY+'\nIndependently audit the following candidate and development results. Identify concrete mathematical or implementation failures, or state uncertainty. Return action=hypothesis, put your concise review in hypothesis, source="", prediction="", probes_json="[]".\nCandidate:\n'+initial+'\nDevelopment execution:\n'+feedback(initial_dev)
            call('critic',critic_prompt)
            critic = calls[-1]['receipt'].get('response') or {'hypothesis':'Critic failed; no advice available.'}
            boosted = call('boost-repair',base+'\nRevise candidate after considering a same-model critic; verify its claims yourself.\nCandidate:\n'+initial+'\nDevelopment execution:\n'+feedback(initial_dev)+'\nCritic:\n'+json.dumps(critic))
            dev('boost-repair',boosted)
            # Both treatments sealed before final answers are evaluated.
            finals = {label:evaluate(folder,label+'-final',src,plan['final']) for label,src in [('initial',initial),('solo',current),('boost',boosted)]}
            item = {'model':model,'calls':calls,'finals':finals}
            write(folder/'result.json',item); summary.append(item)
            write(root/'summary.json',summary)


if __name__=='__main__':
    action, path = sys.argv[1:]; {'freeze':freeze,'run':run}[action](Path(path).resolve())
