import json
import threading

from fastapi.testclient import TestClient
import pytest

from workshop import app as module
from workshop import store


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setattr(store, "STATE", tmp_path)
    monkeypatch.setattr(module, "operation_lock", threading.Lock())
    monkeypatch.delenv("WORKSHOP_REFERENCE_URL", raising=False)
    monkeypatch.delenv("WORKSHOP_REFERENCE_MODEL", raising=False)
    with TestClient(module.app) as instance:
        yield instance


def test_import_generic_skill_and_conflict(client):
    skill = json.loads((store.ROOT / "static/skill-template.json").read_text())
    assert client.post("/api/skills", json=skill).status_code == 201
    assert client.get("/api/skills/support-routing").json()["labels"] == skill["labels"]
    assert client.post("/api/skills", json=skill).status_code == 409


def test_reserved_test_cannot_enter_correction_loop(client):
    skill = client.get("/api/skills/ci-triage").json()
    row = skill["test"][0]
    response = client.post("/api/skills/ci-triage/corrections", json=dict(input=row["input"],expected=row["expected"],evidence=row["input"].splitlines()[0]))
    assert response.status_code == 409
    assert client.get("/api/skills/ci-triage").json() == skill


def test_correction_persists_and_preserves_test(client):
    before = client.get("/api/skills/ci-triage").json()
    row = dict(input="New API smoke: error TS9999 invalid deployment fixture",expected="real-break",evidence="error TS9999")
    response = client.post("/api/skills/ci-triage/corrections", json=row)
    assert response.status_code == 201
    after = client.get("/api/skills/ci-triage").json()
    assert len(after["train"]) == len(before["train"]) + 1
    assert after["test"] == before["test"]
    # Revising the same example replaces it; it does not silently multiply its weight.
    response = client.post("/api/skills/ci-triage/corrections", json=row)
    assert response.status_code == 201
    assert len(client.get("/api/skills/ci-triage").json()["train"]) == len(after["train"])


def test_invalid_correction_does_not_mutate_data(client):
    before = client.get("/api/skills/ci-triage").json()
    response = client.post("/api/skills/ci-triage/corrections", json=dict(input="some log",expected="infra",evidence="invented quote"))
    assert response.status_code == 422
    assert client.get("/api/skills/ci-triage").json() == before


def test_reference_must_be_configured_and_selected(client):
    response = client.post("/api/infer",json=dict(skill_id="ci-triage",input="test",model="reference"))
    assert response.status_code == 409


def test_untrained_specialist_is_not_a_fake_model(client):
    assert client.post("/api/infer",json=dict(skill_id="ci-triage",input="test",model="specialist")).status_code == 409


def test_exclusive_runtime_rejects_overlap(client):
    module.operation_lock.acquire()
    try:
        response = client.post("/api/runs",json=dict(skill_id="ci-triage",kind="train"))
        assert response.status_code == 409
        assert client.get("/api/status").json()["busy"]
    finally:
        module.operation_lock.release()


def test_foreign_origin_cannot_start_training(client):
    response = client.post("/api/runs",json=dict(skill_id="ci-triage",kind="train"),headers={"Origin":"https://unrelated.example"})
    assert response.status_code == 403


def test_same_origin_requests_and_host_check(client):
    assert client.post("/api/infer",json=dict(skill_id="ci-triage",input="test",model="specialist"),headers={"Origin":"http://testserver"}).status_code == 409
    assert client.get("/api/status",headers={"Host":"attacker.example"}).status_code == 400


def test_stale_run_is_marked_interrupted(client):
    # Startup recovery is exercised in an isolated app lifespan, not a simulated success.
    path = store.STATE / "jobs/job-aaaaaaaaaaaa/job.json"
    store.atomic_json(path,dict(id="job-aaaaaaaaaaaa",status="running",created_at=0))
    with TestClient(module.app) as restarted:
        assert restarted.get("/api/runs/job-aaaaaaaaaaaa").json()["status"] == "interrupted"


def test_practice_replays_only_known_fixtures_and_rejects_overlong_trace(client):
    catalog = client.get("/api/practice").json()
    identity = next(s["id"] for s in catalog["scenarios"] if s["family"] == "timeout-pass")
    response = client.post("/api/practice/step", json={"scenario_id":identity,"actions":["read-log","read-history","advise-retry"]})
    assert response.status_code == 200
    assert response.json()["outcome"]["success"]
    assert client.post("/api/practice/step",json={"scenario_id":"../../x","actions":[]}).status_code == 404
    assert client.post("/api/practice/step",json={"scenario_id":identity,"actions":["read-log"]*6}).status_code == 422
    reset = client.post("/api/practice/step",json={"scenario_id":identity,"actions":[]}).json()
    assert not reset["outcome"]["done"]
    assert "expected" not in reset["outcome"]
    assert client.post("/api/practice/step",json={"scenario_id":identity,"actions":["read-log","read-history","advise-retry","read-log"]}).status_code == 422


def test_analytics_contract_is_local_and_narrow(client, monkeypatch):
    catalog=client.get('/api/analytics').json()
    assert catalog['context']['selected_athlete_id']==301
    assert len(catalog['examples'])==24
    assert client.post('/api/analytics/infer',json={'question':'hi','policy':'shell'}).status_code==422
    assert client.post('/api/analytics/infer',json={'question':'hi','context':{}}).status_code==422
    monkeypatch.setattr(store,'runtime_busy',lambda:True)
    assert client.post('/api/analytics/infer',json={'question':'hi'}).status_code==409
    assert not module.operation_lock.locked()


def test_analytics_failure_releases_operation_lock(client, monkeypatch):
    monkeypatch.setattr(store,'runtime_busy',lambda:False)
    original=store.read_json
    def broken(path):
        if str(path).endswith('test-inputs.json'):
            raise FileNotFoundError('fixture unavailable')
        return original(path)
    monkeypatch.setattr(store,'read_json',broken)
    with pytest.raises(FileNotFoundError):
        client.post('/api/analytics/infer',json={'question':'hi'})
    assert not module.operation_lock.locked()


def test_package_import_schema_errors_and_correction_versioning(client):
    payload=store.read_json(store.ROOT/'examples/support-intake.package.json')
    imported=client.post('/api/packages',json=payload)
    assert imported.status_code==200
    ref=imported.json()['reference']
    assert client.get('/api/packages/'+ref).json()['id']=='support-intake'
    bad=dict(payload,output_schema={'type':'not-a-type'})
    assert client.post('/api/packages',json=bad).status_code==422
    row=dict(payload['train'][0],id='correction-test',input={'message':'Need a new password link.'})
    updated=client.post('/api/packages/'+ref+'/corrections',json={'case':row})
    assert updated.status_code==200
    assert updated.json()['final_hash']==imported.json()['final_hash']
    assert updated.json()['reference']!=ref
    assert client.get('/api/packages/'+ref).json()['train']==payload['train']


def test_package_run_rejects_bad_policy_and_releases_lock(client):
    assert client.post('/api/package-runs',json={'reference':'bad','operation':'train'}).status_code==422
    assert not module.operation_lock.locked()
    assert client.post('/api/package-runs',json={'reference':'bad','operation':'merge'}).status_code==422


def test_orphaned_package_run_is_interrupted_on_restart(client):
    path=store.STATE/'package-runs/pkg-aaaaaaaaaaaa/run.json'
    store.atomic_json(path,{'id':'pkg-aaaaaaaaaaaa','status':'running'})
    with TestClient(module.app):
        assert store.read_json(path)['status']=='interrupted'
