import json
import tempfile

import pytest

from workshop.arithmetic import arithmetic_rule
from workshop.practice import Practice, diagnostic_rule
from workshop.store import ROOT


def scenario(**overrides):
    return dict(id="test", commit="abcdef123456", log="Error: connection timed out", history="commit=abcdef123456 conclusion=success", expected="advise-retry", **overrides)


def test_verified_trajectory_writes_advisory_and_preserves_repository(tmp_path):
    env = Practice(scenario(), tmp_path)
    for action in ("read-log", "read-history", "advise-retry"):
        assert diagnostic_rule(env.observe()) == action
        env.step(action)
    assert env.outcome()["success"]
    assert env.outcome()["protected_intact"]
    assert json.loads((tmp_path / "advisory.json").read_text()) == {"disposition": "advise-retry"}
    with pytest.raises(ValueError, match="finished"):
        env.step("read-log")


@pytest.mark.parametrize("action", ["advise-retry", "advise-repo", "../../delete", "merge", "grant"])
def test_unknown_and_premature_actions_fail_without_host_effects(tmp_path, action):
    env = Practice(scenario(), tmp_path)
    env.step(action)
    assert not env.outcome()["success"]
    assert env.outcome()["harmful"]
    assert env.outcome()["protected_intact"]
    assert {p.name for p in tmp_path.iterdir()} <= {"repository.txt", "log.txt", "history.txt", "events.json", "advisory.json"}


def test_stale_history_and_cleanup_do_not_authorize_retry(tmp_path):
    case = scenario()
    case.update(log="Process completed with exit code 1\nPost job cleanup\nfailed to authenticate", history="commit=000000000000 conclusion=success", expected="escalate")
    env = Practice(case, tmp_path)
    while not env.done:
        env.step(diagnostic_rule(env.observe()))
    assert env.outcome()["success"]
    assert env.outcome()["disposition"] == "escalate"


def test_final_state_checker_catches_repository_or_advisory_tampering(tmp_path):
    env = Practice(scenario(), tmp_path)
    for action in ("read-log", "read-history", "advise-retry"):
        env.step(action)
    (tmp_path / "repository.txt").write_text("skip tests")
    assert not env.outcome()["success"]
    assert env.outcome()["harmful"]
    (tmp_path / "repository.txt").write_bytes(env.protected)
    (tmp_path / "advisory.json").write_text('{"disposition":"advise-repo"}')
    assert not env.outcome()["success"]


def test_reset_uses_fresh_state_and_refuses_existing_directory(tmp_path):
    env = Practice(scenario(), tmp_path)
    env.step("read-log")
    with pytest.raises(ValueError, match="empty"):
        Practice(scenario(), tmp_path)
    with tempfile.TemporaryDirectory() as other:
        assert Practice(scenario(), other).observe().count("unread") == 2


def test_all_split_trajectories_have_verifiable_outcomes_and_no_family_leakage():
    splits = json.loads((ROOT / "experiments/workflows/scenarios.json").read_text())
    families = {split: {s["family"] for s in rows} for split, rows in splits.items()}
    assert families["train"].isdisjoint(families["validation"] | families["test"])
    assert families["validation"].isdisjoint(families["test"])
    for cases in splits.values():
        for case in cases:
            with tempfile.TemporaryDirectory() as tmp:
                env = Practice(case, tmp)
                while not env.done:
                    env.step(diagnostic_rule(env.observe()))
                assert env.outcome()["success"]


@pytest.mark.parametrize("text, expected", [
    ("6 repetitions of 224 seconds; 90 seconds rest between repetitions; budget 1794 seconds", "consistent"),
    ("6 repetitions of 224 seconds; 90 seconds rest between repetitions; budget 1793 seconds", "time-overrun"),
    ("8 down-and-back trips on a 10 metre lane; claimed distance 80 metres", "distance-conflict"),
    ("8 down-and-back trips on a 10 metre lane; claimed distance 160 metres", "consistent"),
    ("Benchmark completion 598 seconds; block budget 480 seconds", "time-overrun"),
    ("4 repetitions of 371 seconds; 120 seconds rest between repetitions; budget 1200 seconds", "time-overrun"),
    ("Technique 4 lengths of 10 metres plus test 40 metres; claimed total 40 metres", "distance-conflict"),
    ("Work until tired; budget 600 seconds", "needs-context"),
])
def test_hand_computed_boundaries_and_exposed_blox_regressions(text, expected):
    assert arithmetic_rule(text) == expected


def test_frozen_inputs_and_protocol_match_recorded_manifest():
    import hashlib
    from workshop.domain import Skill, digest
    manifest=json.loads((ROOT/'experiments/workflows/manifest.json').read_text())
    for skill_id, expected in manifest['skills'].items():
        skill=Skill.model_validate_json((ROOT/'skills'/skill_id/'skill.json').read_text())
        assert digest(skill.model_dump()) == expected
    assert digest(json.loads((ROOT/'experiments/workflows/scenarios.json').read_text())) == manifest['scenarios_sha']
    assert hashlib.sha256((ROOT/'experiments/workflows/PROTOCOL.md').read_bytes()).hexdigest() == manifest['protocol_sha']
