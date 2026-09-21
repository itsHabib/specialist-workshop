"""Give a fresh worker a stalled run's current code and observed failure."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parent))
import agent


def packet(source):
    state=json.loads((source/'state.json').read_text())
    plan=json.loads((source/'plan.json').read_text())
    # Preserve the original regression suite even if the earlier worker edited it.
    tests={k:v for k,v in plan['task']['files'].items() if k.startswith('tests/') or Path(k).name.startswith('test_')}
    task={**plan['task'],'files':{**state['files'],**tests}}
    task['goal']+='\nContinue this stalled repair. Keep correct work intact, reproduce the remaining problems and deliver a tested patch. Recent observed evidence:\n'+agent.evidence(state['history'])
    return task,{'source_digest':agent.digest(state['files']),'source_status':state['status'],
                 'source_finished':state['finished'],'result':'continuation only; requires independent validation',
                 'previous_calls':len(state['calls']),'previous_seconds':state['elapsed'],
                 'previous_known_cost_usd':sum(c['receipt'].get('estimated_cost_usd') or 0 for c in state['calls']),
                 'previous_cost_complete':all(c['receipt'].get('estimated_cost_usd') is not None for c in state['calls'])}


def rescue(source,out,model,calls=8,check=None):
    task,provenance=packet(source)
    out.mkdir(parents=True,exist_ok=False,mode=0o700)
    agent.write(out/'task.json',task);agent.write(out/'parent.json',provenance)
    checker,name=agent.script_checker(check.read_text(),out/'run'/'STOP') if check else (None,None)
    return agent.run(task,model,out/'run',policy='solo',auto_test=True,max_calls=calls,checker=checker,checker_name=name)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('source',type=Path);p.add_argument('--out',type=Path,required=True);p.add_argument('--model',default='haiku');p.add_argument('--calls',type=int,default=8)
    p.add_argument('--check',type=Path,help='agent-visible Python development check, not a hidden final grader')
    a=p.parse_args();rescue(a.source,a.out,a.model,a.calls,a.check)
