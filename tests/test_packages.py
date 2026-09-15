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


def fake_checkpoint(package,isolated):
    ref=packages.import_package(package)['reference'];run=experiment.new_run(ref,'train')
    directory=experiment.runs_root()/run['id'];adapter=directory/'adapter';adapter.mkdir()
    (adapter/'adapters.safetensors').write_bytes(b'test fixture, not model weights')
    (adapter/'adapter_config.json').write_text('{"lora_parameters":{"scale":1}}')
    import hashlib
    run.update(status='completed',adapter_sha256=hashlib.sha256((adapter/'adapters.safetensors').read_bytes()).hexdigest(),adapter_manifest=experiment.adapter_manifest(adapter))
    store.atomic_json(directory/'run.json',run)
    return run


def test_checkpoint_cannot_evaluate_training_rows_rotated_into_final(package,isolated):
    run=fake_checkpoint(package,isolated)
    swapped=package.model_dump();swapped['train'],swapped['final']=swapped['final'],swapped['train']
    target=packages.Package.model_validate(swapped)
    with pytest.raises(ValueError,match='reserved evaluation'):experiment.checkpoint(target,run['id'])
    corrected=package.model_copy(deep=True)
    corrected.train.append(package.train[0].model_copy(update={'id':'extra','input':{'message':'A new practice example'}}))
    assert experiment.checkpoint(corrected,run['id']).is_dir()


def test_checkpoint_binds_config_not_just_tensor_bytes(package,isolated):
    run=fake_checkpoint(package,isolated)
    directory=experiment.checkpoint(package,run['id'])
    (directory/'adapter_config.json').write_text('{"lora_parameters":{"scale":99}}')
    with pytest.raises(ValueError,match='adapter files changed'):experiment.checkpoint(package,run['id'])


def test_evaluation_records_actual_raw_validity(package,isolated):
    ref=packages.import_package(package)['reference'];run=experiment.new_run(ref,'evaluate',policy='majority');run=experiment.worker(run['id'])
    assert run['status']=='completed'
    assert run['metrics']['raw_valid']==run['metrics']['valid']==2
    assert all(row['raw_valid'] for row in run['rows'])
    assert experiment.metrics([{'valid':True}])['raw_valid'] is None


def test_compute_child_keeps_lock_after_worker_death(tmp_path):
    import fcntl,os,signal,subprocess,sys,time
    if os.name!='posix':pytest.skip('POSIX runtime contract')
    lock_path=tmp_path/'accelerator.lock';marker=tmp_path/'child.pid'
    child_code=f"import os,time;from pathlib import Path;Path({str(marker)!r}).write_text(str(os.getpid()));time.sleep(20)"
    parent_code=f"import fcntl,sys;from workshop.experiment import run_compute;f=open({str(lock_path)!r},'a');fcntl.flock(f,fcntl.LOCK_EX);log=open({str(tmp_path/'child.log')!r},'w');run_compute([sys.executable,'-c',{child_code!r}],log,f.fileno())"
    parent=subprocess.Popen([sys.executable,'-c',parent_code],cwd=store.ROOT)
    child=None
    try:
        deadline=time.monotonic()+5
        while not marker.exists() and time.monotonic()<deadline:time.sleep(.02)
        assert marker.exists();child=int(marker.read_text())
        parent.kill();parent.wait(timeout=3)
        with lock_path.open('a') as contender:
            with pytest.raises(BlockingIOError):fcntl.flock(contender,fcntl.LOCK_EX|fcntl.LOCK_NB)
        os.kill(child,signal.SIGTERM);child=None
        deadline=time.monotonic()+3
        with lock_path.open('a') as contender:
            while True:
                try:fcntl.flock(contender,fcntl.LOCK_EX|fcntl.LOCK_NB);break
                except BlockingIOError:
                    if time.monotonic()>deadline:raise
                    time.sleep(.02)
    finally:
        if parent.poll() is None:parent.kill();parent.wait(timeout=3)
        if child:
            try:os.kill(child,signal.SIGTERM)
            except ProcessLookupError:pass


def test_evaluation_defaults_to_development_and_final_is_explicit(package,isolated):
    ref=packages.import_package(package)['reference']
    dev=experiment.worker(experiment.new_run(ref,'evaluate',policy='majority')['id'])
    final=experiment.worker(experiment.new_run(ref,'qualify',policy='majority')['id'])
    assert dev['evaluation_split']=='development'
    assert {r['id'] for r in dev['rows']}=={r.id for r in package.development}
    assert final['evaluation_split']=='final'
    assert {r['id'] for r in final['rows']}=={r.id for r in package.final}
    with pytest.raises(ValueError,match='Different'):experiment.compare([dev['id'],final['id']])


def test_development_hash_prevents_comparing_changed_dev_cases(package,isolated):
    ref=packages.import_package(package)['reference']
    first=experiment.worker(experiment.new_run(ref,'evaluate',policy='majority')['id'])
    changed=package.model_copy(deep=True)
    changed.development[0].input={'message':'Different development input'}
    ref2=packages.import_package(changed)['reference']
    second=experiment.worker(experiment.new_run(ref2,'evaluate',policy='majority')['id'])
    assert first['final_hash']==second['final_hash']
    with pytest.raises(ValueError,match='Different'):experiment.compare([first['id'],second['id']])


def test_legacy_queued_evaluation_keeps_final_semantics(package,isolated):
    ref=packages.import_package(package)['reference']
    run=experiment.new_run(ref,'evaluate',policy='majority')
    del run['evaluation_split'];del run['evaluation_hash']
    store.atomic_json(experiment.runs_root()/run['id']/'run.json',run)
    finished=experiment.worker(run['id'])
    assert len(finished['rows'])==len(package.final)
    assert experiment.compare([run['id']])[0]['evaluation_split']=='final'


def test_checkpoint_rejects_training_rows_moved_to_development(package,isolated):
    run=fake_checkpoint(package,isolated)
    changed=package.model_dump()
    changed['development'].append(changed['train'].pop())
    target=packages.Package.model_validate(changed)
    with pytest.raises(ValueError,match='reserved evaluation'):
        experiment.checkpoint(target,run['id'])
