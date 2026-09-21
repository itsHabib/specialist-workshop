"""Bounded discovery screen. Reuses the bridge's supervised models and sandbox."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import fcntl
import hashlib
import inspect
import json
from pathlib import Path
import random
import sys
import time

ROOT=Path(__file__).resolve().parent
BRIDGE=ROOT.parent/'investigation_bridge'
sys.path.insert(0,str(BRIDGE))
import models
import sandbox
sys.path.insert(0,str(ROOT))
import tasks


def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_suffix('.tmp');temporary.write_text(json.dumps(value,indent=2)+'\n');temporary.replace(path)


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def grade(name,source,inputs,stop):
    expected=[tasks.REFERENCES[name](q) for q in inputs]
    execution=sandbox.evaluate(source,inputs,stop=stop)
    outputs=execution['outputs'];complete=isinstance(outputs,list) and len(outputs)==len(inputs)
    failures=[]
    if complete:
        failures=[{'input':q,'expected':want,'actual':got} for q,want,got in zip(inputs,expected,outputs) if type(want)!=type(got) or json.dumps(want,sort_keys=True)!=json.dumps(got,sort_keys=True)]
    return {'passed':execution['error'] is None and complete and not failures,'total':len(inputs),'failed':len(failures) if complete else len(inputs),'failures':failures,'execution':execution}


def freeze(root, diagnostic=False):
    root.mkdir(parents=True,exist_ok=False,mode=0o700)
    discovery=tasks.cases(1729);heldout=tasks.cases(5772 if diagnostic else 2718)
    packages={}
    for name,qs in discovery.items():
        if diagnostic and name!='geometry':continue
        # All edge cases are in the contract. Development examples are fixed before calls.
        selected=[qs[0],qs[1],qs[-2],qs[-1]]
        seen={json.dumps(q,sort_keys=True) for q in selected}
        final=[q for q in heldout[name] if json.dumps(q,sort_keys=True) not in seen]
        packages[name]={'contract':tasks.CONTRACTS[name],'development':selected,'final':final}
    write(root/'tasks.json',packages)
    trials=[{'task':name,'provider':provider} for name in packages for provider in ('opus','astra')]
    random.Random(573).shuffle(trials)
    files=[ROOT/'tasks.py',ROOT/'screen.py',ROOT/('DEADLINE-PROTOCOL.md' if diagnostic else 'PROTOCOL.md'),BRIDGE/'models.py',BRIDGE/'model_supervisor.py',BRIDGE/'sandbox.py']
    plan={'schema':'engineering-screen.v1','trials':trials,'calls_per_arm':1 if diagnostic else 2,'timeout':300 if diagnostic else 150,'tasks_sha256':sha(root/'tasks.json'),'files':{str(p.relative_to(ROOT.parent)):sha(p) for p in files},'image':sandbox.IMAGE,'models':{'opus':'claude-opus-5','astra':'gpt-6-astra'},'effort':'high','backend':'docker','seeds':{'development':1729,'final':5772 if diagnostic else 2718},'label':'deadline-diagnostic' if diagnostic else 'discovery'}
    write(root/'plan.json',plan)
    return plan


def reference_source(name):
    prelude='from fractions import Fraction as F\nfrom functools import lru_cache\n'
    if name=='leases':prelude+=inspect.getsource(tasks.step)+'\n'
    return prelude+inspect.getsource(tasks.REFERENCES[name])+f'\nevaluate={name}\n'


def controls(root):
    rows=[]
    for name,qs in tasks.cases(2718).items():
        good=grade(name,reference_source(name),qs,root/'STOP')
        bad=grade(name,"def evaluate(q): return {}",qs,root/'STOP')
        rows.append({'task':name,'good':good,'empty_control':bad})
        if not good['passed'] or bad['passed']:raise RuntimeError('sandbox control failed: '+name)
    write(root/'controls.json',rows)


def prompt(package,history,max_calls=2):
    return ('Solve this synthetic engineering implementation task. Return the required JSON action object. '
            'Use action="edit", source=the complete self-contained Python implementation defining evaluate(request), '
            'hypothesis=a concise design explanation, prediction=what your implementation handles, probes_json="[]". '
            'Only standard library; no tools/network/filesystem access. The harness executes your program. '
            f'You have at most {max_calls} model calls; remaining calls receive development feedback. '
            'A passing development implementation is sealed and evaluated on final inputs once. '
            'Reason carefully about edge cases; choose an algorithm fitting the stated bounds.\n'+json.dumps({'contract':package['contract'],'examples':[{'input':q,'expected':tasks.REFERENCES[history['task']](q)} for q in package['development']], 'previous_attempts':history['attempts']}))


def trial(root,entry,package,plan):
    name=entry['task'];provider=entry['provider'];directory=root/f'{name}-{provider}';directory.mkdir(exist_ok=True)
    result_path=directory/'result.json'
    if result_path.exists():return json.loads(result_path.read_text())
    history={'task':name,'attempts':[]};calls=[];source=None;development=None;error=None;started=time.monotonic()
    for i in range(plan['calls_per_arm']):
        if (root/'STOP').exists():error='stopped';break
        out=directory/f'call-{i+1}'
        if out.exists():
            if not (out/'receipt.json').exists():raise RuntimeError('uncertain paid call: '+str(out))
            receipt=json.loads((out/'receipt.json').read_text())
        else:receipt=models.complete(prompt(package,history,plan['calls_per_arm']),provider,out,timeout=plan['timeout'],stop=root/'STOP')
        calls.append(receipt)
        if receipt['error']:error=receipt['error'];break
        response=receipt['response']
        if not isinstance(response,dict) or response.get('action')!='edit' or not isinstance(response.get('source'),str) or not response['source']:
            history['attempts'].append({'format_error':'Expected edit with full source','response':response});continue
        source=response['source'];(directory/f'candidate-{i+1}.py').write_text(source)
        development=grade(name,source,package['development'],root/'STOP')
        feedback={k:v for k,v in development.items() if k!='execution'}
        feedback['execution_error']=development['execution']['error']
        history['attempts'].append({'response':response,'development':feedback})
        write(directory/'history.json',history)
        if development['passed']:break
    final=None
    if source is not None and error is None and not (root/'STOP').exists():
        (directory/'final.py').write_text(source)
        final=grade(name,source,package['final'],root/'STOP')
    result={'task':name,'provider':provider,'accepted':bool(development and development['passed'] and final and final['passed']), 'status':'finished' if final else 'incomplete','error':error,'calls':calls,'development':development,'final':final,'seconds':time.monotonic()-started,'source_sha256':hashlib.sha256(source.encode()).hexdigest() if source else None}
    write(result_path,result);print(json.dumps({k:result[k] for k in ('task','provider','accepted','status','seconds')}),flush=True)
    return result


def execute(root):
    with (root/'driver.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        plan=json.loads((root/'plan.json').read_text());packages=json.loads((root/'tasks.json').read_text())
        assert sha(root/'tasks.json')==plan['tasks_sha256'],'task changed'
        for path,want in plan['files'].items():assert sha(ROOT.parent/path)==want,'frozen code changed: '+path
        # At most two native calls in flight, both independently solving frozen tasks.
        with ThreadPoolExecutor(max_workers=2) as pool:
            results=list(pool.map(lambda e:trial(root,e,packages[e['task']],plan),plan['trials']))
        write(root/'summary.json',[{k:r[k] for k in ('task','provider','accepted','status','error','seconds')} for r in results])


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('command',choices=['freeze','freeze-deadline','controls','run']);parser.add_argument('directory',type=Path);args=parser.parse_args();root=args.directory.resolve()
    if args.command in ('freeze','freeze-deadline'):freeze(root,diagnostic=args.command=='freeze-deadline')
    elif args.command=='controls':controls(root)
    else:execute(root)
