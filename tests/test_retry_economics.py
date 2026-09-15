"""Guard the experiment against oracle retries and optimistic cost accounting."""
import importlib.util
from pathlib import Path
from types import SimpleNamespace

spec=importlib.util.spec_from_file_location('retry_economics',Path(__file__).resolve().parents[1]/'scripts/retry_economics.py')
experiment=importlib.util.module_from_spec(spec)
spec.loader.exec_module(experiment)


def test_retry_gets_validation_error_but_never_expected_answer(monkeypatch):
    calls=[]
    responses=iter(['bad','{"answer": 7}'])
    monkeypatch.setattr(experiment,'validated',lambda p,v,r: {'valid':r.startswith('{'),'error':'JSON required'})
    def predict(messages):
        calls.append([dict(m) for m in messages])
        return {'raw':next(responses)}
    row=experiment.policy(SimpleNamespace(instructions='Return JSON'), 'Question',predict)
    assert len(calls)==2
    assert calls[1][-2]['content']=='bad'
    assert 'JSON required' in calls[1][-1]['content']
    assert row['stop_reason']=='valid'
    assert all('expected' not in str(m) for m in calls)


def test_valid_output_stops_without_consulting_gold(monkeypatch):
    monkeypatch.setattr(experiment,'validated',lambda *args: {'valid':True,'correct':None})
    calls=[]
    def predict(messages):
        calls.append(messages)
        return {'raw':'{"plausible_but_wrong":true}'}
    assert len(experiment.policy(SimpleNamespace(instructions='Task'), 'Question', predict)['attempts'])==1
    assert len(calls)==1


def test_cost_per_success_charges_failed_tasks():
    def row(ok,cost):
        return {'elapsed_seconds':1,'attempts':[{'elapsed_seconds':1,'cost_usd':cost,'score':{'accepted':ok},'validation':{'valid':True}}]}
    summary=experiment.summarize([row(True,.01),row(False,.02)],3)
    assert summary['cost_per_success_usd']==.03
    assert summary['valid_but_wrong']==1
    assert experiment.summarize([row(True,None)],3)['total_cost_usd'] is None
    assert experiment.summarize([row(False,.01)],3)['cost_per_success_usd'] is None


def test_usage_prices_cached_input_and_all_output():
    assert experiment.cost('gpt-6-astra',{'input_tokens':1000,'input_tokens_details':{'cached_tokens':500},'output_tokens':100})==.0105
    assert experiment.cost('gpt-5.6-luna',{'input_tokens':1000,'input_tokens_details':{'cache_write_tokens':1000},'output_tokens':0})==.00025
