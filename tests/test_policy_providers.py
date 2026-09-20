import json
import httpx
import pytest
from workshop.policy_providers import Providers, decode, validate_models


def test_anthropic_messages_and_cached_accounting(monkeypatch):
    monkeypatch.setenv('ANTHROPIC_API_KEY', 'not-a-real-key')
    spec = {'provider':'anthropic','model':'chosen-opus',
            'rates':{'input':10,'output':20,'cached':1,'cache_write':12.5}}
    def handle(req):
        body=json.loads(req.content)
        assert body['system']=='contract'
        assert body['messages']==[{'role':'user','content':'task'}]
        assert body['max_tokens']==100
        return httpx.Response(200,json={'id':'msg-test','model':'actual-opus','stop_reason':'end_turn',
            'content':[{'type':'text','text':'{"ok":true}'}],
            'usage':{'input_tokens':10,'cache_read_input_tokens':20,'cache_creation_input_tokens':30,'output_tokens':40}})
    p=Providers({'m':spec},httpx.Client(transport=httpx.MockTransport(handle)))
    result=p('m',[{'role':'system','content':'contract'},{'role':'user','content':'task'}],100)
    p.close()
    assert result['model']=='actual-opus'
    assert result['usage']['input_tokens']==60
    assert result['cost_usd']==pytest.approx((100+20+375+800)/1e6)


def test_openai_partial_output_not_success():
    result=decode({'provider':'openai','model':'requested'},
                  {'model':'actual','status':'incomplete','output':[{'content':[{'type':'output_text','text':'{'}]}],
                   'usage':{'input_tokens':12,'output_tokens':10}})
    assert result['raw']=='{'
    assert result['error']=='incomplete_response'
    assert result['cost_usd'] is None


def test_http_errors_do_not_record_provider_body(monkeypatch):
    monkeypatch.setenv('OPENAI_API_KEY','fake-secret')
    p=Providers({'m':{'provider':'openai','model':'test'}},
                httpx.Client(transport=httpx.MockTransport(lambda req:httpx.Response(401,text='private diagnostic'))))
    result=p('m',[{'role':'user','content':'x'}],100)
    p.close()
    assert result['error']=='HTTP_401'
    assert result['usage'] is None
    assert 'private' not in json.dumps(result)


def test_bad_price_refused():
    with pytest.raises(ValueError):
        validate_models({'m':{'provider':'openai','model':'test','rates':{'input':-1}}})
