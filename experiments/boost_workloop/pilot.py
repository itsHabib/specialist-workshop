"""Freeze/run a small paired pilot, keeping all grading outside worker feedback."""
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT))
import agent
from runtime import execute
from taskpack.taskpack import build_task,verify,reference_files


def freeze(root,model,variant):
    root.mkdir(parents=True,exist_ok=False,mode=0o700)
    hashes={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in ROOT.rglob('*.py') if 'receipts' not in p.parts}
    plan={'model':model,'variant':variant,'seed':90137 if variant==0 else 90149,
          'max_calls':12,'seconds':7200,'max_dollars':4,'task':build_task(variant),'source_hashes':hashes}
    agent.write(root/'plan.json',plan)
    for label,files,want in [('reference',reference_files(variant),True),('initial',plan['task']['files'],False)]:
        result=verify(files,execute,plan['seed'],variant=variant)
        agent.write(root/(label+'-control.json'),result)
        if result['passed']!=want:raise RuntimeError('grader control failed')


def grade(root,policy):
    plan=json.loads((root/'plan.json').read_text());state=json.loads((root/policy/'state.json').read_text())
    # Restore original visible tests during acceptance; do not trust edited tests.
    seen={};scores=[]
    for snapshot in state['snapshots']:
        key=snapshot['digest']
        if key not in seen:
            source={**snapshot['files'],**{k:v for k,v in plan['task']['files'].items() if k.startswith('tests/')}}
            regression=execute(source,plan['task']['test_command'])
            hidden=verify(snapshot['files'],execute,plan['seed'],variant=plan['variant'])
            seen[key]={'passed':regression['error'] is None and regression['returncode']==0 and hidden['passed'],
                       'regression':regression,'hidden':hidden}
        scores.append({'label':snapshot['label'],'elapsed':snapshot['elapsed'],'calls':snapshot['calls'],
                       'digest':key,**seen[key]})
    result={'policy':policy,'status':state['status'],'finished':state['finished'],'elapsed':state['elapsed'],
            'first_valid_seconds':next((r['elapsed'] for r in scores if r['passed']),None),
            'final_passed':scores[-1]['passed'],'calls':len(state['calls']),'snapshots':scores}
    agent.write(root/policy/'grading.json',result)
    print(policy,result['final_passed'],result['first_valid_seconds'],state['status'],flush=True)
    return result


def run(root,policy):
    plan=json.loads((root/'plan.json').read_text())
    result=agent.run(plan['task'],plan['model'],root/policy,policy,plan['max_calls'],plan['seconds'],max_dollars=plan['max_dollars'])
    return grade(root,policy)


if __name__=='__main__':
    action=sys.argv[1];root=Path(sys.argv[2]).resolve()
    if action=='freeze':freeze(root,sys.argv[3],int(sys.argv[4]))
    if action=='run':run(root,sys.argv[3])
    if action=='grade':grade(root,sys.argv[3])
