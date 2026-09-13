import copy
import json
import pytest
from workshop import packages,store,experiment
from workshop.completion import decode


@pytest.fixture
def package():return packages.Package.model_validate(store.read_json(store.ROOT/'examples/support-intake.package.json'))


@pytest.fixture
def isolated(tmp_path,monkeypatch):
    monkeypatch.setattr(store,'STATE',tmp_path)
    store.initialize()
    return tmp_path


def test_import_and_correction_keep_old_version_and_evaluation_immutable(package,isolated):
    original=packages.import_package(package)
    correction=packages.Case(id='new',family='profile-link',input={'message':'My account recovery link expired.'},expected={'owner':'accounts','urgency':'normal','reason':'access'},provenance='Reviewed test fixture')
    updated=packages.correct(original['reference'],correction)
    assert updated['reference']!=original['reference']
    assert updated['final_hash']==original['final_hash']
    assert updated['contract_hash']==original['contract_hash']
    assert len(packages.load(original['reference']).train)==6
    assert len(packages.load(updated['reference']).train)==7


def test_reserved_inputs_and_families_cannot_become_corrections(package,isolated):
    ref=packages.import_package(package)['reference']
    row=package.final[0].model_copy(update={'id':'new','family':'new-family'})
    with pytest.raises(ValueError,match='reserved'):packages.correct(ref,row)
    row=package.train[0].model_copy(update={'id':'new','family':package.final[0].family})
    with pytest.raises(ValueError,match='reserved'):packages.correct(ref,row)


def test_nested_whitespace_variants_do_not_cross_splits(package):
    payload=package.model_dump();payload['final'][0]['input']={'message':'  MY  PASSWORD RESET LINK DOES NOT WORK. '}
    with pytest.raises(ValueError,match='overlaps'):packages.Package.model_validate(payload)


def test_family_split_and_schema_contract_are_enforced(package):
    payload=package.model_dump();payload['final'][0]['family']=payload['train'][0]['family']
    with pytest.raises(ValueError,match='Family overlaps'):packages.Package.model_validate(payload)
    payload=package.model_dump();payload['train'][0]['expected']['owner']='invented-team'
    with pytest.raises(ValueError):packages.Package.model_validate(payload)
    payload=package.model_dump();payload['output_schema']={'$ref':'https://example.com/private'}
    with pytest.raises(ValueError,match='self-contained'):packages.Package.model_validate(payload)


def test_validity_and_semantic_correctness_remain_separate(package):
    row=package.final[0];wrong={'owner':'payments','urgency':'normal','reason':'charge'}
    scored=packages.score(package,row.input,json.dumps(wrong),row.expected)
    assert scored['valid'] and not scored['correct'] and not scored['accepted']
    live=packages.score(package,row.input,json.dumps(wrong))
    assert live['correct'] is None


def test_empty_model_channel_codec_is_narrow_and_keeps_raw():
    raw='<think>\n\n</think>\n\n{"owner":"accounts"}'
    assert decode(raw)['content']=='{"owner":"accounts"}'
    assert decode(raw)['empty_channel_removed']
    assert decode('<think>do something</think>{"x":1}')['content'].startswith('<think>')
    assert decode('Here is JSON: {"x":1}')['content'].startswith('Here')


def test_training_serialization_never_contains_final_cases(package,isolated):
    _,hashes=experiment.training_data(package,isolated/'training')
    text=(isolated/'training/data/train.jsonl').read_text()
    assert package.final[0].input['message'] not in text
    assert set(hashes)=={'train','valid'}
    assert len(text.splitlines())==6


def test_imported_package_hash_is_checked(package,isolated):
    ref=packages.import_package(package)['reference']
    path=next((isolated/'packages').glob('*/*.json'));body=store.read_json(path);body['name']='changed';store.atomic_json(path,body)
    with pytest.raises(ValueError,match='hash changed'):packages.load(ref)


def test_comparison_rejects_different_final_or_grader_versions(package,isolated):
    ref=packages.import_package(package)['reference'];a=experiment.new_run(ref,'evaluate');b=experiment.new_run(ref,'evaluate')
    for r in (a,b):
        r.update(status='completed',started_at=0,finished_at=1)
        store.atomic_json(experiment.runs_root()/r['id']/'run.json',r)
    assert len(experiment.compare([a['id'],b['id']]))==2
    b['grader_hash']='different';store.atomic_json(experiment.runs_root()/b['id']/'run.json',b)
    with pytest.raises(ValueError,match='Different'):experiment.compare([a['id'],b['id']])


def test_environment_interface_is_data_driven_and_rejects_unknown_adapters(isolated):
    from workshop.environments import create
    scenario={'id':'demo','family':'demo','commit':'abcdef','log':'timeout','history':'commit=abcdef conclusion=success','expected':'advise-retry'}
    with pytest.raises(ValueError):create('upload-python',scenario,isolated/'bad')
    env=create('ci-evidence-v1',scenario,isolated/'env')
    while not env.done:env.step(env.baseline())
    assert env.outcome()['success']
