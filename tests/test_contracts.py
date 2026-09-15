import copy
import json

import pytest

from workshop.domain import Skill, grade, input_key, messages_for, summarize
from workshop.store import ROOT


@pytest.fixture
def payload():
    return json.loads((ROOT / "static" / "skill-template.json").read_text())


def test_seed_has_no_cross_split_leakage_and_source_is_pinned():
    skill = Skill.model_validate_json((ROOT / "skills/ci-triage/skill.json").read_text())
    assert (len(skill.train), len(skill.validation), len(skill.test)) == (240, 9, 51)
    import hashlib
    data = (ROOT / "skills/ci-triage/workbench-evaluation.jsonl").read_bytes()
    assert hashlib.sha256(data).hexdigest() == skill.provenance["sha256"]
    assert {input_key(r.input) for r in skill.train}.isdisjoint(input_key(r.input) for r in skill.test)


def test_training_rejects_reserved_input_even_with_whitespace_changes(payload):
    payload["train"][0]["input"] = "  " + payload["test"][0]["input"] + "\n"
    payload["train"][0]["evidence"] = "account is locked"
    with pytest.raises(ValueError, match="overlaps"):
        Skill.model_validate(payload)


def test_evidence_cannot_be_invented(payload):
    payload["train"][0]["evidence"] = "NOT PRESENT IN THE INPUT"
    with pytest.raises(ValueError, match="nonempty substring"):
        Skill.model_validate(payload)


def test_empty_evidence_is_not_a_verifier_pass(payload):
    skill = Skill.model_validate(payload)
    result = grade(skill, "account locked", '{"bucket":"access","evidence":""}', "access")
    assert result["correct"] is True
    assert result["evidence_valid"] is False
    assert summarize([{**result, "elapsed_ms": 5}])["accepted_rate"] == 0


@pytest.mark.parametrize("output", ["[]", '"access"', '{"bucket":[],"evidence":"hi"}',
                                   '{"bucket":{},"evidence":"hi"}', '{"bucket":"access","evidence":3}',
                                   '{"bucket":"access","evidence":"hi","confidence":1}'])
def test_malformed_outputs_fail_without_crashing(payload, output):
    result = grade(Skill.model_validate(payload), "hi", output, "access")
    assert not result["valid"]
    assert summarize([{**result, "elapsed_ms": 5}])["accepted_rate"] == 0


def test_markdown_wrapping_is_not_confused_with_label_accuracy(payload):
    result = grade(Skill.model_validate(payload), "invoice", '```json\n{"bucket":"billing","evidence":"invoice"}\n```', "billing")
    assert result["correct"] and result["valid"]
    assert not result["raw_json_valid"]


def test_line_selection_returns_original_source_line(payload):
    payload["evidence_mode"] = "line"
    skill = Skill.model_validate(payload)
    result = grade(skill, "INFO: started\nInvoice overdue\nwrapper error", '{"bucket":"billing","line":2}', "billing")
    assert result["correct"] and result["evidence_valid"]
    assert result["output"]["evidence"] == "Invoice overdue"


@pytest.mark.parametrize("number", [-1, 0, 999, True, "1", None])
def test_invalid_line_selections_fail(payload, number):
    payload["evidence_mode"] = "line"
    result = grade(Skill.model_validate(payload), "one line", json.dumps(dict(bucket="access",line=number)), "access")
    assert not result["valid"] and not result["evidence_valid"]


def test_label_and_quote_are_independent_signals(payload):
    result = grade(Skill.model_validate(payload), "invoice unpaid", '{"bucket":"access","evidence":"invoice"}', "billing")
    assert result["evidence_valid"]
    assert not result["correct"]


def test_model_messages_never_include_test_labels_or_training_examples(payload):
    skill = Skill.model_validate(payload)
    messages = messages_for(skill, "new request")
    assert len(messages) == 2
    text = json.dumps(messages)
    assert all(r.input not in text for r in skill.test + skill.train)


def test_identifiers_cannot_traverse_paths(payload):
    payload["id"] = "../escape"
    with pytest.raises(ValueError):
        Skill.model_validate(payload)


def test_unmeasured_metrics_remain_null():
    assert summarize([])["accuracy"] is None


def test_token_limit_counts_ids_not_batchencoding_keys():
    from workshop.domain import check_training_length
    class Tokenizer:
        def apply_chat_template(self, messages, **kwargs):
            if kwargs.get("return_dict") is False:
                return list(range(1025))
            return {"input_ids": list(range(1025)), "attention_mask": [1] * 1025}
    with pytest.raises(ValueError, match="1024-token"):
        check_training_length(Tokenizer(), [], "oversized")
