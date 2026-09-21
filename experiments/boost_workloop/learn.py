"""Distill a candidate lesson from a verified practice repair, without final cases."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parent))
import agent

SCHEMA={'type':'object','properties':{'trigger':{'type':'string'},'guidance':{'type':'string'},'check':{'type':'string'}},'required':['trigger','guidance','check'],'additionalProperties':False}


def practice_packet(run,grading):
    state=json.loads((run/'state.json').read_text());plan=json.loads((run/'plan.json').read_text())
    grade=json.loads(grading.read_text())
    if grade.get('schema')!='boost-grading.v1' or grade.get('run_plan_sha256')!=agent.digest(plan) or grade.get('task_sha256')!=agent.digest(plan['task']) or not grade.get('evaluator'):
        raise ValueError('grading must bind this exact run plan, task and evaluator')
    if not grade.get('final_passed') or grade['snapshots'][-1]['digest']!=agent.digest(state['files']):
        raise ValueError('independent acceptance must name the current source snapshot')
    return {'goal':plan['task']['goal'],'files':state['files']},grade['snapshots'][-1]['digest']


def learn(run,grading,out,model):
    packet,digest=practice_packet(run,grading)
    prompt=('Extract one reusable lesson from this verified practice repair. Return a concise trigger, concrete guidance and a check for NEW tasks. Under180 words total. Do not include full implementation, test answers, fixed input constants, or claim untested generalization. Identify the representation or debugging move that matters, not generic advice. No hidden evaluator cases are provided.\n'+json.dumps(packet))
    receipt=agent.models.complete(prompt,model,out,timeout=1800,schema=SCHEMA,max_budget_usd=.5)
    if receipt['error']:raise RuntimeError(receipt['error'])
    agent.write(out/'lessons.json',[receipt['response']])
    agent.write(out/'provenance.json',{'status':'candidate lesson; transfer not established','source_digest':digest,'teacher_requested':model,'estimated_cost_usd':receipt['estimated_cost_usd'],'evidence':'independently accepted practice snapshot; final inputs excluded'})
    return out/'lessons.json'


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('run',type=Path);p.add_argument('--grading',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--model',default='haiku')
    a=p.parse_args();print(learn(a.run,a.grading,a.out,a.model))
