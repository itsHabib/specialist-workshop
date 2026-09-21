"""Inspect/edit/test loop with targeted assistance and resumable call receipts."""
import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parent
sys.path[:0]=[str(ROOT),str(ROOT.parent/'investigation_bridge')]
import models
from runtime import execute,validate

SCHEMA={'type':'object','additionalProperties':False,'properties':{
 'action':{'type':'string','enum':['read','write','edit','run','help','finish']},
 'path':{'type':'string'},'text':{'type':'string'},'old_text':{'type':'string'}},'required':['action','path','text']}
HELP_SCHEMA={'type':'object','properties':{'advice':{'type':'string'},'test_source':{'type':'string'}},'required':['advice','test_source'],'additionalProperties':False}


def write(path,value):
    temporary=path.with_suffix('.tmp')
    with temporary.open('w') as stream:
        json.dump(value,stream,indent=2);stream.write('\n');stream.flush()
        import os
        os.fsync(stream.fileno())
    temporary.replace(path)


def digest(files):return hashlib.sha256(json.dumps(files,sort_keys=True).encode()).hexdigest()


def script_checker(source,stop=None):
    """A caller-owned development check. Its diagnostics are shown to the worker."""
    def check(files):
        result=execute({**files,'_boost_external_check.py':source},'python _boost_external_check.py',stop=stop)
        return {'passed':result['error'] is None and result['returncode']==0 and result.get('cleanup_confirmed',False),
                'execution':result}
    return check,'script-sha256:'+hashlib.sha256(source.encode()).hexdigest()


def evidence(history):
    observations=deepcopy(history[-6:])
    for event in observations:
        action=event.get('action')
        if isinstance(action,dict) and action.get('action')=='write':action['text']='[content is in current files]'
    text=json.dumps(observations)
    return text if len(text)<=10000 else '[Earlier observations omitted]\n'+text[-10000:]


def prompt_for(task,files,history,policy,lessons):
    return ('Repair this Python repository. Inspect code, edit files, run tests, diagnose failures and continue. '
       'Passing the current tests may miss stated requirements: add tests when useful. Return one JSON action.\n'
       'read: path=relative file,text="". write: path=relative file,text=complete executable file contents. '
       'edit: path=relative file, old_text=exact unique old substring, text=replacement. Prefer edit for small changes. '
       'run: path="",text=shell command in a networkless Python container with current files. '
       'finish: path="",text=brief conclusion after testing final edits. '
       +('help: path="",text=concrete question for a fresh investigator; use when useful.\n' if policy=='assist' else 'No external helper; investigate independently.\n')
       +'Commands start in the snapshot root /tmp/repo, not the model client working directory. '
       'Shell changes do not persist; use write/edit to retain edits/tests. No installs.\n'
       +'Goal:\n'+task['goal']+'\nTest command: '+task['test_command']+'\nCurrent files:\n'+json.dumps(files)
       +'\nPrior actions and observations:\n'+evidence(history)
       +'\nPrior lessons (check applicability):\n'+json.dumps(lessons))


def run(task,model,out,policy='solo',max_calls=12,seconds=7200,lessons=None,resume=False,max_dollars=4,auto_test=False,helper_model=None,checker=None,checker_name=None):
    import fcntl
    validate(task['files']);out=Path(out).resolve()
    if resume:
        plan=json.loads((out/'plan.json').read_text())
        if plan['task']!=task or plan['model']!=model or plan['policy']!=policy:
            raise ValueError('resume must use the original task/model/policy')
    else:
        out.mkdir(parents=True,exist_ok=False,mode=0o700)
        plan={'task':task,'model':model,'policy':policy,'max_calls':max_calls,'seconds':seconds,'max_dollars':max_dollars,'auto_test':auto_test,'helper_model':helper_model or model,'checker_name':checker_name,'lessons':lessons or []}
        write(out/'plan.json',plan)
    if bool(plan.get('checker_name')) != bool(checker) or (checker and checker_name!=plan.get('checker_name')):
        raise ValueError('resume/run requires the named development checker; its output is agent-visible')
    with (out/'driver.lock').open('w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        return _drive(plan,out,resume,checker)


def _drive(plan,out,resume,checker=None):
    task=plan['task'];model=plan['model'];policy=plan['policy'];stop=out/'STOP'
    began=time.monotonic()
    state=json.loads((out/'state.json').read_text()) if resume else {
       'files':deepcopy(task['files']),'history':[],'calls':[],'snapshots':[], 'processed':0,
       'pending':None,'finished':False,'elapsed':0,'helped':False,'failed_tests':0,
       'need_helper':None,'status':'running','transport_failures':0}
    if resume and state['finished']:
        return state
    if resume:
        state['history'].append({'action':'resume','result':'Continue original plan; prior receipts and spend retained.'})
    previous=state['elapsed']
    def elapsed():return previous+time.monotonic()-began
    def save():
        state['elapsed']=elapsed();write(out/'state.json',state);write(out/'snapshots.json',state['snapshots'])
    def snapshot(label):
        state['snapshots'].append({'label':label,'elapsed':elapsed(),'files':deepcopy(state['files']),
                                  'digest':digest(state['files']),'calls':len(state['calls'])})
    def checked_execute(files,command):
        result=execute(files,command,stop=stop)
        if checker and command==task['test_command'] and result['error'] is None and result['returncode']==0:
            result['extra_check']=checker(files)
            result['accepted']=result['extra_check'].get('passed') is True
        return result
    if not resume:
        snapshot('initial')
        initial=checked_execute(state['files'],task['test_command'])
        state['history'].append({'action':'initial_test','result':initial});save()
    if state['pending']:
        path=out/state['pending']['directory']/'receipt.json'
        if not path.exists():raise RuntimeError('reserved call has no receipt; inspect supervisor before resuming, never silently repeat')
        state['calls'].append({'role':state['pending']['role'],'receipt':json.loads(path.read_text())})
        state['pending']=None;save()
    state['status']='running'
    try:
        while not state['finished'] and not stop.exists():
            if elapsed()>=plan['seconds']:state['status']='time_budget';break
            known=sum(c['receipt'].get('estimated_cost_usd') or 0 for c in state['calls'])
            if known>=plan['max_dollars']:state['status']='cost_budget';break
            if state['processed']>=len(state['calls']):
                if len(state['calls'])>=plan['max_calls']:state['status']='call_budget';break
                role='helper' if state['need_helper'] else 'worker'
                prompt=prompt_for(task,state['files'],state['history'],policy,plan['lessons'])
                schema=SCHEMA
                if role=='helper':
                    prompt=('Independently investigate the following goal, code and observed test results. Return brief actionable advice and test_source: a standalone Python assert script importing candidate modules. Produce a concrete counterexample or repair direction; empty test_source is fine if no defensible probe. Do not invent requirements or assume the worker diagnosis is correct.\n'
                       +json.dumps({'goal':task['goal'],'files':state['files'],'test_command':task['test_command']})
                       +'\nObserved evidence:\n'+evidence(state['history'])+'\nQuestion:\n'+state['need_helper'])
                    schema=HELP_SCHEMA
                call_model=plan.get('helper_model',model) if role=='helper' else model
                if call_model in ('llama3.2:1b','qwen2.5:7b') and len(prompt)>24000:
                    state['status']='prompt_budget';break
                directory=f'call-{len(state["calls"])+1:02d}-{role}'
                state['pending']={'role':role,'directory':directory};save()
                receipt=models.complete(prompt,call_model,out/directory,timeout=max(1,min(1800,plan['seconds']-elapsed())),
                        stop=stop,schema=schema,max_budget_usd=min(2,plan['max_dollars']-known))
                state['calls'].append({'role':role,'receipt':receipt});state['pending']=None;save()
                print(out.name,role,round(receipt['seconds'],1),receipt['error'],flush=True)
            call=state['calls'][state['processed']];receipt=call['receipt']
            if receipt['error']:
                state['history'].append({'action':'transport','result':receipt['error']});state['processed']+=1
                state['transport_failures']+=1
                # Unknown-usage interruptions need an explicit resume decision. Other
                # known failures get at most one retry, retaining their cost and time.
                state['status']='transport_incomplete';save()
                usage_known=receipt.get('usage') is not None
                if not usage_known or state['transport_failures']>=2:break
                continue
            state['transport_failures']=0
            response=receipt.get('response') or {};observation=None
            if call['role']=='helper':
                observation=deepcopy(response)
                if response.get('test_source'):
                    observation['test_execution']=execute({**state['files'],'_boost_probe.py':response['test_source']},'python _boost_probe.py',stop=stop)
                state['history'].append({'action':'helper_advice','result':observation})
                state['need_helper']=None;state['helped']=True
            else:
                kind=response.get('action');path=response.get('path','');text=response.get('text','')
                if kind=='read':observation=state['files'].get(path,'No such file')
                elif kind in ('write','edit'):
                    content=text
                    if kind=='edit':
                        old=response.get('old_text','')
                        if not old or path not in state['files'] or state['files'][path].count(old)!=1:
                            state['history'].append({'action':response,'result':'Edit rejected: old_text must match exactly once.'})
                            state['processed']+=1;save();continue
                        content=state['files'][path].replace(old,text,1)
                    candidate={**state['files'],path:content}
                    try:validate(candidate)
                    except ValueError as error:observation=str(error)
                    else:
                        state['files']=candidate;snapshot('edit');observation='File written.'
                        if plan.get('auto_test'):
                            observation=checked_execute(state['files'],task['test_command'])
                            passed=observation['error'] is None and observation['returncode']==0 and observation.get('accepted',True)
                            state['failed_tests']=0 if passed else state['failed_tests']+1
                            if policy=='assist' and state['failed_tests']>=2 and not state['helped']:
                                state['need_helper']='Two successive edits still fail tests. Find a concrete reproducer and repair direction.'
                elif kind=='help' and policy=='assist':
                    state['need_helper']=text;observation='Investigator requested.'
                elif kind in ('run','finish'):
                    command=task['test_command'] if kind=='finish' else text
                    observation=checked_execute(state['files'],command)
                    if command==task['test_command']:
                        passed=observation['error'] is None and observation['returncode']==0 and observation.get('accepted',True)
                        state['failed_tests']=0 if passed else state['failed_tests']+1
                        if kind=='finish' and passed:state['finished']=True;state['status']='self_reported_complete'
                        if kind=='finish' and not passed:observation={'finish_rejected':'tests fail; continue repair','test':observation}
                        if policy=='assist' and state['failed_tests']>=2 and not state['helped']:
                            state['need_helper']='Two consecutive official test attempts failed. Find the earliest concrete mismatch and a useful reproducer.'
                else:observation='Invalid or unavailable action.'
                state['history'].append({'action':response,'result':observation})
            state['processed']+=1;save()
        if stop.exists():state['status']='stopped'
        snapshot('final');save()
    except BaseException:
        save();raise
    return state


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('task',type=Path);parser.add_argument('--model',default='haiku');parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--policy',choices=['solo','assist'],default='assist');parser.add_argument('--calls',type=int,default=12)
    parser.add_argument('--seconds',type=int,default=7200);parser.add_argument('--max-dollars',type=float,default=4)
    parser.add_argument('--helper-model');parser.add_argument('--lessons',type=Path);parser.add_argument('--resume',action='store_true');parser.add_argument('--auto-test',action='store_true')
    parser.add_argument('--check',type=Path,help='caller-owned Python development check; diagnostics are returned to worker, never use hidden final cases')
    args=parser.parse_args()
    checker,name=script_checker(args.check.read_text(),args.out/'STOP') if args.check else (None,None)
    run(json.loads(args.task.read_text()),args.model,args.out,args.policy,args.calls,args.seconds,
        json.loads(args.lessons.read_text()) if args.lessons else None,args.resume,args.max_dollars,args.auto_test,args.helper_model,checker,name)
