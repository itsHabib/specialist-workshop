from pathlib import Path
import json
import pytest
from workshop.compare_policies import main, summarize, report

ROOT=Path(__file__).resolve().parents[1]


def test_dry_run_no_state_or_credentials(tmp_path, monkeypatch, capsys):
    monkeypatch.delenv('OPENAI_API_KEY',raising=False)
    out=tmp_path/'run'
    assert main(['--package',str(ROOT/'examples/support-intake.package.json'),
                 '--config',str(ROOT/'examples/capability-policies.json'),'--output',str(out),'--dry-run'])==0
    assert not out.exists()
    result=json.loads(capsys.readouterr().out)
    assert result['split']=='development'
    assert result['status']=='planned'
    assert len(result['plan'])>0
    assert 'expected' not in result


def test_summary_includes_failed_attempt_cost():
    rows=[{'policy':{'name':'pair'},'final_score':{'accepted':good},'cost_usd':cost,
           'calls':[{},{}],'elapsed_seconds':2} for good,cost in [(True,1),(False,3)]]
    stats=summarize(rows)[0]
    assert stats['cost_per_success_usd']==4
    rows[1]['cost_usd']=None
    assert summarize(rows)[0]['total_cost_usd'] is None


def test_report_escapes_names():
    result=report({'rows':[{'policy':{'name':'<script>alert(1)</script>'},'final_score':{'accepted':False},
                             'cost_usd':None,'calls':[],'elapsed_seconds':0}], 'status':'completed','split':'development'})
    assert '<script>' not in result


def test_interruption_keeps_paid_and_unknown_accounting(tmp_path, monkeypatch):
    import workshop.compare_policies as cli
    class FakeProvider:
        def __init__(self, models): self.calls=0
        def close(self): pass
        def __call__(self, *args):
            self.calls+=1
            if self.calls==2: raise KeyboardInterrupt()
            return {'raw':'{}','model':'fake','usage':{'input_tokens':10,'output_tokens':10},'cost_usd':1}
    monkeypatch.setattr(cli,'Providers',FakeProvider)
    monkeypatch.setattr(cli,'credentials_available',lambda models:True)
    out=tmp_path/'run'
    args=['--package',str(ROOT/'examples/support-intake.package.json'),
          '--config',str(ROOT/'examples/capability-policies.json'),'--output',str(out)]
    try: cli.main(args)
    except KeyboardInterrupt: pass
    result=json.loads((out/'run.json').read_text())
    assert result['status']=='interrupted_or_failed'
    assert result['call_accounting']['known_cost_usd']==1
    assert result['call_accounting']['total_cost_usd'] is None
    assert result['started_calls']==2


def test_global_budget_prevents_extra_transport(tmp_path, monkeypatch):
    import workshop.compare_policies as cli
    counter=[]
    class FakeProvider:
        def __init__(self, models): pass
        def close(self): pass
        def __call__(self, *args):
            counter.append(1)
            return {'raw':'{}','model':'fake','usage':{'input_tokens':10,'output_tokens':10},'cost_usd':.1}
    monkeypatch.setattr(cli,'Providers',FakeProvider)
    monkeypatch.setattr(cli,'credentials_available',lambda models:True)
    config=json.loads((ROOT/'examples/capability-policies.json').read_text())
    config['max_run_calls']=1
    path=tmp_path/'config.json';path.write_text(json.dumps(config))
    out=tmp_path/'run'
    assert cli.main(['--package',str(ROOT/'examples/support-intake.package.json'),'--config',str(path),'--output',str(out)])==1
    result=json.loads((out/'run.json').read_text())
    assert result['status']=='budget_exhausted'
    assert len(counter)==1
    assert result['started_calls']==1
