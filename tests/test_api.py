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
