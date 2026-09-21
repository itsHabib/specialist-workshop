"""Core execution/feedback checks; no live inference."""
import importlib.util
import json
import os
from pathlib import Path
import sys

HERE=Path(__file__).parent
sys.path.insert(0,str(HERE))
import agent
from runtime import execute,validate
import pytest


def test_paths_cannot_leave_source_snapshot():
    for name in ('../outside','/tmp/host','a/../x','a//x',''):
        with pytest.raises(ValueError):validate({name:'x'})


def test_failed_finish_returns_test_feedback_and_repairs(monkeypatch,tmp_path):
    actions=iter([
        {'action':'finish','path':'','text':'done'},
        {'action':'write','path':'app.py','text':'print(1)'},
        {'action':'finish','path':'','text':'fixed'},
    ])
    monkeypatch.setattr(agent.models,'complete',lambda *args,**kwargs:{'response':next(actions),'error':None,'seconds':0,'usage':{},'estimated_cost_usd':0})
    monkeypatch.setattr(agent,'execute',lambda files,*a,**kw:{'error':None,'returncode':0 if files['app.py']=='print(1)' else 1,'stdout':'','stderr':''})
    task={'files':{'app.py':'bad'},'goal':'print 1','test_command':'python app.py'}
    result=agent.run(task,'haiku',tmp_path/'run',policy='solo',max_calls=3)
    assert result['finished']
    assert 'finish_rejected' in result['history'][1]['result']
    assert len(result['calls'])==3


def test_helper_spend_is_in_shared_budget_and_cannot_finish_task(monkeypatch,tmp_path):
    responses=iter([
      {'action':'help','path':'','text':'check boundaries'},
      {'advice':'check negative inputs','test_source':'assert 1==1'},
      {'action':'finish','path':'','text':'checked'},
    ])
    monkeypatch.setattr(agent.models,'complete',lambda *args,**kwargs:{'response':next(responses),'error':None,'seconds':0,'usage':{},'estimated_cost_usd':0})
    monkeypatch.setattr(agent,'execute',lambda *a,**kw:{'error':None,'returncode':0,'stdout':'','stderr':''})
    result=agent.run({'files':{'app.py':'pass'},'goal':'test','test_command':'python app.py'},'haiku',tmp_path/'run',policy='assist',max_calls=4)
    assert [r['role'] for r in result['calls']]==['worker','helper','worker']
    assert result['finished'] and len(result['snapshots'])==2


@pytest.mark.skipif(os.environ.get("BRIDGE_DOCKER_TESTS") != "1", reason="opt-in Docker integration")
def test_runtime_uses_actual_files_and_cleans_container():
    result=execute({'thing.py':'print(7)'},'python thing.py')
    assert result['error'] is None and result['returncode']==0
    assert result['stdout']=='7\n' and result['cleanup_confirmed']


def test_resume_reconciles_completed_call_without_repeating_inference(monkeypatch,tmp_path):
    task={'files':{'app.py':'bad'},'goal':'print 1','test_command':'python app.py'}
    receipt={'response':{'action':'write','path':'app.py','text':'print(1)'},'error':None,'seconds':1,'usage':{},'estimated_cost_usd':.01}
    def interrupted(prompt,model,out,**kwargs):
        out.mkdir();(out/'receipt.json').write_text(json.dumps(receipt))
        raise KeyboardInterrupt()
    monkeypatch.setattr(agent.models,'complete',interrupted)
    monkeypatch.setattr(agent,'execute',lambda files,*a,**kw:{'error':None,'returncode':0 if files['app.py']=='print(1)' else 1,'stdout':'','stderr':''})
    with pytest.raises(KeyboardInterrupt):agent.run(task,'haiku',tmp_path/'run',policy='solo')
    invoked=[]
    def finish(*args,**kwargs):
        invoked.append(1)
        return {**receipt,'response':{'action':'finish','path':'','text':'done'}}
    monkeypatch.setattr(agent.models,'complete',finish)
    result=agent.run(task,'haiku',tmp_path/'run',policy='solo',resume=True)
    assert result['finished'] and result['files']['app.py']=='print(1)'
    assert len(invoked)==1 and len(result['calls'])==2


def test_auto_feedback_exposes_failure_without_worker_run_action(monkeypatch,tmp_path):
    actions=iter([{'action':'write','path':'app.py','text':'broken'}, {'action':'write','path':'app.py','text':'print(1)'}, {'action':'finish','path':'','text':'done'}])
    monkeypatch.setattr(agent.models,'complete',lambda *a,**kw:{'response':next(actions),'error':None,'seconds':0,'usage':{},'estimated_cost_usd':0})
    monkeypatch.setattr(agent,'execute',lambda files,*a,**kw:{'error':None,'returncode':0 if files['app.py']=='print(1)' else 1,'stdout':'','stderr':'failure'})
    task={'files':{'app.py':'bad'},'goal':'print 1','test_command':'python app.py'}
    result=agent.run(task,'haiku',tmp_path/'run',policy='solo',auto_test=True)
    assert result['finished'] and result['history'][1]['result']['returncode']==1
    resumed=agent.run(task,'haiku',tmp_path/'run',policy='solo',resume=True)
    assert resumed['status']==result['status'] and len(resumed['snapshots'])==len(result['snapshots'])


def test_local_known_usage_failure_gets_one_retry(monkeypatch,tmp_path):
    responses=iter([
      {'response':None,'error':'local_incomplete_response','seconds':0,'usage':{'eval_count':6000},'estimated_cost_usd':None},
      {'response':{'action':'finish','path':'','text':'done'},'error':None,'seconds':0,'usage':{'eval_count':30},'estimated_cost_usd':None},
    ])
    monkeypatch.setattr(agent.models,'complete',lambda *a,**kw:next(responses))
    monkeypatch.setattr(agent,'execute',lambda *a,**kw:{'error':None,'returncode':0,'stdout':'','stderr':''})
    result=agent.run({'files':{'app.py':'pass'},'goal':'test','test_command':'python app.py'},'qwen2.5:7b',tmp_path/'run',max_calls=2)
    assert result['finished'] and len(result['calls'])==2


def test_learning_requires_acceptance_of_exact_current_snapshot(tmp_path):
    import learn
    agent.write(tmp_path/'state.json',{'files':{'app.py':'pass'}})
    plan={'task':{'goal':'practice'}}
    agent.write(tmp_path/'plan.json',plan)
    grading=tmp_path/'grading.json'
    agent.write(grading,{'final_passed':True,'snapshots':[{'digest':'stale'}]})
    with pytest.raises(ValueError):learn.practice_packet(tmp_path,grading)
    valid={'schema':'boost-grading.v1','run_plan_sha256':agent.digest(plan),'task_sha256':agent.digest(plan['task']),
           'evaluator':{'task_sha256':agent.digest(plan['task']),'seed':1,'variant':0,'source_sha256':{name:'test-source' for name in ('pilot.py','taskpack/taskpack.py','runtime.py')}},'final_passed':True,'snapshots':[{'digest':agent.digest({'app.py':'pass'})}]}
    agent.write(grading,valid)
    packet,_=learn.practice_packet(tmp_path,grading)
    assert set(packet)=={'goal','files'}
    valid['evaluator']['task_sha256']='different contract'
    agent.write(grading,valid)
    with pytest.raises(ValueError):learn.practice_packet(tmp_path,grading)
    agent.write(tmp_path/'plan.json',{'task':{'goal':'different acceptance contract'}})
    with pytest.raises(ValueError):learn.practice_packet(tmp_path,grading)


def test_final_grader_rejects_a_different_task_before_execution(tmp_path):
    import pilot
    agent.write(tmp_path/'plan.json',{'task':{'goal':'original'}})
    agent.write(tmp_path/'state.json',{})
    with pytest.raises(ValueError):pilot.grade_run(tmp_path,{'goal':'different'},0,1)


def test_component_reuse_checks_integrity_and_never_overwrites_task():
    import hashlib
    import reuse
    component={'source':'pass','source_sha256':hashlib.sha256(b'pass').hexdigest(),'accepted_goal':'practice'}
    task={'files':{'app.py':'bad'},'goal':'new work'}
    result=reuse.apply(task,component,'learned.py')
    assert result['files']['app.py']=='bad' and result['files']['learned.py']=='pass'
    with pytest.raises(ValueError):reuse.apply(task,component,'app.py')
    component['source']='changed'
    with pytest.raises(ValueError):reuse.apply(task,component,'learned.py')


def test_partial_edit_is_exact_and_rejects_ambiguous_matches(monkeypatch,tmp_path):
    actions=iter([
      {'action':'edit','path':'app.py','old_text':'bad','text':'fixed'},
      {'action':'edit','path':'app.py','old_text':'bad bad','text':'print(1)'},
      {'action':'finish','path':'','text':'done'},
    ])
    monkeypatch.setattr(agent.models,'complete',lambda *a,**kw:{'response':next(actions),'error':None,'seconds':0,'usage':{},'estimated_cost_usd':0})
    monkeypatch.setattr(agent,'execute',lambda files,*a,**kw:{'error':None,'returncode':0 if files['app.py']=='print(1)' else 1,'stdout':'','stderr':''})
    result=agent.run({'files':{'app.py':'bad bad'},'goal':'test','test_command':'python app.py'},'haiku',tmp_path/'run',max_calls=3)
    assert result['finished'] and result['files']['app.py']=='print(1)'
    assert 'rejected' in result['history'][1]['result']


def test_external_checker_catches_green_test_suite_and_drives_repair(monkeypatch,tmp_path):
    actions=iter([{'action':'finish','path':'','text':'done'}, {'action':'edit','path':'app.py','old_text':'bad','text':'good'}, {'action':'finish','path':'','text':'fixed'}])
    monkeypatch.setattr(agent.models,'complete',lambda *a,**kw:{'response':next(actions),'error':None,'seconds':0,'usage':{},'estimated_cost_usd':0})
    monkeypatch.setattr(agent,'execute',lambda *a,**kw:{'error':None,'returncode':0,'stdout':'','stderr':''})
    check=lambda files:{'passed':files['app.py']=='good','why':'requirement'}
    task={'files':{'app.py':'bad'},'goal':'test','test_command':'python app.py'}
    result=agent.run(task,'haiku',tmp_path/'run',checker=check,checker_name='independent')
    assert result['finished'] and 'finish_rejected' in result['history'][1]['result']
    with pytest.raises(ValueError):agent.run(task,'haiku',tmp_path/'run',resume=True)


def test_external_script_replaces_worker_copy_and_is_bound_on_resume(monkeypatch):
    observed=[]
    def execute(files,command,**kwargs):
        observed.append((files,command))
        return {'error':None,'returncode':1,'cleanup_confirmed':True,'stderr':'contract failed'}
    monkeypatch.setattr(agent,'execute',execute)
    check,name=agent.script_checker('assert False')
    assert not check({'app.py':'pass','_boost_external_check.py':'pass'})['passed']
    assert observed[0][0]['_boost_external_check.py']=='assert False'
    assert name!=agent.script_checker('assert True')[1]


def test_rescue_preserves_original_tests_and_upstream_accounting(tmp_path):
    import rescue
    agent.write(tmp_path/'plan.json',{'task':{'goal':'fix','files':{'app.py':'old','tests/test_app.py':'original'},'test_command':'test'}})
    agent.write(tmp_path/'state.json',{'files':{'app.py':'new','tests/test_app.py':'weakened'},'history':[],
       'status':'call_budget','finished':False,'elapsed':20,'calls':[{'receipt':{'estimated_cost_usd':.2}}]})
    task,parent=rescue.packet(tmp_path)
    assert task['files']=={'app.py':'new','tests/test_app.py':'original'}
    assert parent['previous_known_cost_usd']==.2 and parent['previous_seconds']==20
