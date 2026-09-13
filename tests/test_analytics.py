import json
import pytest
from workshop.analytics import Plan, canonical, execute, grade, messages, validate_plan
from workshop.store import ROOT

CASES=json.loads((ROOT/'experiments/analytics/test-adjudicated.json').read_text())


def test_independent_plans_satisfy_contract_without_making_validity_an_intent_oracle():
    for case in CASES:
        assert grade(json.dumps(case['expected']),case)['intent_correct']
    case=CASES[0]
    wrong=dict(case['expected'],metrics=['total_time'])
    result=grade(json.dumps(wrong),case)
    assert result['valid'] and not result['intent_correct']


def test_model_input_excludes_labels_family_and_other_examples():
    for case in CASES:
        content=messages(case)
        assert len(content)==2
        payload=json.loads(content[1]['content'])
        assert set(payload)=={'request','context'}
        assert 'expected' not in payload and 'family' not in payload


def test_delta_direction_is_semantic_but_metric_order_is_not():
    case=CASES[17]
    reverse=dict(case['expected'],result_ids=list(reversed(case['expected']['result_ids'])))
    assert not grade(json.dumps(reverse),case)['intent_correct']
    case=CASES[5]
    reorder=dict(case['expected'],metrics=list(reversed(case['expected']['metrics'])))
    assert grade(json.dumps(reorder),case)['intent_correct']


@pytest.mark.parametrize('change',[
    dict(result_ids=[999999]),dict(result_ids=[3011],metrics=['sled_pull']),
    dict(action='benchmark',metrics=['run_total'],scope='global'),
    dict(metrics=['penalties']),dict(result_ids=[True]),dict(sql='DROP TABLE results'),
])
def test_invalid_id_null_metric_schema_distractor_and_extra_code_are_rejected(change):
    case=CASES[1]
    raw=dict(case['expected']);raw.update(change)
    assert not grade(json.dumps(raw),case)['valid']


def test_executor_checks_data_and_does_arithmetic_without_model_numbers():
    case=CASES[0]
    plan=Plan.model_validate(case['expected'])
    snapshot={'results':{'3011':{'run_total':2500,'roxzone':500},'3012':{'run_total':2300,'roxzone':540}}}
    result=execute(plan,case['context'],snapshot)
    assert result['delta_seconds']=={'run_total':-200,'roxzone':40}
    assert result['fixture_only']


def test_unsupported_request_does_not_execute():
    case=CASES[7]
    result=execute(Plan.model_validate(case['expected']),case['context'],{})
    assert result['status']=='unsupported' and result['rows']==[]


def test_blind_adjudication_hashes_and_split_identities():
    import hashlib
    root=ROOT/'experiments/analytics'
    record=json.loads((root/'adjudication.json').read_text())
    assert not record['training_or_predictions_seen']
    assert hashlib.sha256((root/'test-adjudicated.json').read_bytes()).hexdigest()==record['test_sha256']
    assert hashlib.sha256((root/'test-inputs.json').read_bytes()).hexdigest()==record['input_sha256']
    train=json.loads((root/'train.json').read_text())
    dev=json.loads((root/'development.json').read_text())
    ids=lambda rows:{result['id'] for row in rows for result in row['context']['results']}
    assert ids(train).isdisjoint(ids(CASES)|ids(dev))
    assert ids(dev).isdisjoint(ids(CASES))


def test_pooled_reference_does_not_shift_when_an_unrelated_split_is_missing():
    import importlib.util
    spec=importlib.util.spec_from_file_location('analytics_runner',ROOT/'scripts/run_analytics.py')
    runner=importlib.util.module_from_spec(spec);spec.loader.exec_module(runner)
    snapshot=runner.fixture(CASES[0])
    assert snapshot['pooled']['3011']['division']['wall_balls']==snapshot['pooled']['3012']['division']['wall_balls']
