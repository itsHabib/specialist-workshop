"""Core execution/feedback checks; no live inference."""
import importlib.util
import json
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


def test_runtime_uses_actual_files_and_cleans_container():
    result=execute({'thing.py':'print(7)'},'python thing.py')
    assert result['error'] is None and result['returncode']==0
    assert result['stdout']=='7\n' and result['cleanup_confirmed']
