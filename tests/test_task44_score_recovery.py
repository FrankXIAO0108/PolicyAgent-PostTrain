"""Offline evidence integrity tests; optional private frozen replay integration."""

import copy
import json
from pathlib import Path

import pytest

from scripts.recover_task44_prescreen_scores import (
    content_value,
    reconstruct,
    verify_pair,
)
from src.rl.retail_agentic_env import (
    TRANSPORT_INVALID_COMPLETION_FIELDS,
    _canonical_sha256,
    _ensure_tau2_importable,
)


def bind(raw, evidence):
    evidence["state_hashes"] = {
        f"{name}_sha256": _canonical_sha256(evidence[f"{name}_state"])
        for name in ("initial", "final")
    }
    evidence["evidence_sha256"] = _canonical_sha256(
        {k: v for k, v in evidence.items() if k != "evidence_sha256"}
    )
    raw["evidence_sha256"] = evidence["evidence_sha256"]


def pair():
    completion = {key: False for key in TRANSPORT_INVALID_COMPLETION_FIELDS}
    completion["stop_reason"] = "MODEL_EOS_BEFORE_USER_STOP"
    raw = {"task_id": "44", "user_seed": 7, "completion": completion}
    evidence = {
        **copy.deepcopy(raw),
        "initial_state": {"x": 0},
        "final_state": {"x": 1},
    }
    bind(raw, evidence)
    return raw, evidence


def test_early_eos_is_not_filtered_as_transport_error():
    verify_pair(*pair())


@pytest.mark.parametrize(
    "field", ["task_id", "user_seed", "completion", "evidence_sha256"]
)
def test_cross_row_binding_mismatch_rejected(field):
    raw, evidence = pair()
    raw[field] = "changed"
    with pytest.raises(ValueError, match="Frozen evidence mismatch"):
        verify_pair(raw, evidence)


@pytest.mark.parametrize("name", ["initial", "final"])
def test_tampered_state_rejected_even_with_rebound_outer_hash(name):
    raw, evidence = pair()
    evidence[f"{name}_state"] = {"x": 99}
    evidence["evidence_sha256"] = _canonical_sha256(
        {k: v for k, v in evidence.items() if k != "evidence_sha256"}
    )
    raw["evidence_sha256"] = evidence["evidence_sha256"]
    with pytest.raises(ValueError, match="state hash"):
        verify_pair(raw, evidence)


def test_truncation_cannot_be_recovered_as_eligible():
    raw, evidence = pair()
    raw["completion"]["model_completion_truncated"] = True
    evidence["completion"] = copy.deepcopy(raw["completion"])
    bind(raw, evidence)
    with pytest.raises(ValueError, match="Transport-invalid"):
        verify_pair(raw, evidence)


def test_json_object_order_ignored_but_values_not_relaxed():
    assert content_value('{"a":1,"b":2}') == content_value('{"b":2,"a":1}')
    assert content_value('{"amount":17.99}') != content_value('{"amount":18.99}')


@pytest.fixture
def frozen():
    root = (
        Path(__file__).resolve().parents[1]
        / "_local_private_runs/20260903-task44-hybrid-prescreen-v1/run"
    )
    if not root.exists():
        pytest.skip("Private frozen run not distributed with repository")
    _ensure_tau2_importable()
    from tau2.domains.retail.environment import get_tasks, get_environment
    from loguru import logger

    logger.disable("tau2")
    return (
        [
            json.loads(s)
            for s in (root / "raw_rollouts.jsonl")
            .read_text(encoding="utf-8")
            .splitlines()
        ],
        [
            json.loads(s)
            for s in (root / "rollout_evidence.jsonl")
            .read_text(encoding="utf-8")
            .splitlines()
        ],
        json.loads((root / "config.json").read_text(encoding="utf-8")),
        next(t for t in get_tasks("train") if t.id == "44"),
        get_environment,
    )


@pytest.mark.parametrize("index,expected", [(0, 1.0), (1, 0.2), (2, 0.98), (3, 0.25)])
def test_frozen_replay_reproduces_base_without_extractor(
    frozen, monkeypatch, index, expected
):
    from src.evaluation import task44_hybrid_reward as hybrid

    def forbidden(*a, **kw):
        pytest.fail("Offline base recovery called semantic API")

    monkeypatch.setattr(hybrid, "call_extractor", forbidden)
    raw, evidence, config, task, constructor = frozen
    result = reconstruct(raw[index], evidence[index], config, task, constructor)
    assert result["base"]["staged_reward"] == expected
    assert result["initial_state_matches"] and result["final_state_matches"]


@pytest.mark.parametrize("change", ["initial_db", "final_db", "tool_return"])
def test_rebound_fake_evidence_still_fails_real_replay(frozen, change):
    from pydantic import TypeAdapter
    from tau2.data_model.message import Message
    from src.rl.retail_agentic_env import _tool_trace

    rows, evs, config, task, constructor = frozen
    raw, evidence = copy.deepcopy(rows[0]), copy.deepcopy(evs[0])
    if change in {"initial_db", "final_db"}:
        state = "initial_state" if change == "initial_db" else "final_state"
        evidence[state]["agent"]["orders"]["#W9300146"]["status"] = "fake"
    else:
        message = next(m for m in raw["messages"] if m["role"] == "tool")
        message["content"] = '"fake"'
        evidence["tool_trace"] = _tool_trace(
            [TypeAdapter(Message).validate_python(m) for m in raw["messages"]]
        )
    bind(raw, evidence)
    with pytest.raises(ValueError, match="Frozen evidence mismatch"):
        reconstruct(raw, evidence, config, task, constructor)
