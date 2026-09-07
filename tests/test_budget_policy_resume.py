"""Explicit budget-policy continuation; unrelated changes remain forbidden."""
from copy import deepcopy
import json
from pathlib import Path

import pytest

from src.training import run_retail_agentic_grpo as runner

ROOT = Path(__file__).resolve().parents[1]
PREFIX = "configs/retail_agentic_qwen3_4b_task44_staged_v6_n4_50step_c8192"


@pytest.fixture
def resume_case(tmp_path):
    parent = runner.load_json(ROOT / (PREFIX + "_v2.json"))
    config = runner.load_json(ROOT / (PREFIX + "_budget0_v3.json"))
    run = tmp_path / "old"
    ckpt = run / "trainer" / "checkpoint-10"
    ckpt.mkdir(parents=True)
    for name in ("adapter_model.safetensors", "adapter_config.json", "optimizer.pt",
                 "scheduler.pt", "rng_state.pth"):
        (ckpt / name).write_bytes(b"fixture")
    (ckpt / "trainer_state.json").write_text('{"global_step":10}', encoding="utf-8")
    (run / "config.json").write_text(json.dumps(parent), encoding="utf-8")
    for name in ("raw_rollouts.jsonl", "rollout_evidence.jsonl"):
        (run / name).write_text("{}\n" * 60, encoding="utf-8")
    return ckpt, config, parent


def test_budget_policy_resume_records_change_and_checkpoint_rows(resume_case):
    ckpt, config, _ = resume_case
    result = runner.validate_resume_checkpoint(ckpt, config)
    assert result["completed_steps"] == 10
    assert result["prior_rollouts"] == 40
    assert result["completion_budget_policy_changed"] is True
    assert result["exact_same_policy_resume"] is False
    assert result["parent_completion_budget_as_terminal_failure"] is False
    assert result["new_completion_budget_as_terminal_failure"] is True


@pytest.mark.parametrize("section,key,value", [
    ("grpo", "learning_rate", 1e-4),
    ("grpo", "beta", 0.5),
    ("engineering_acceptance", "require_transport_valid_before_reward", False),
    ("engineering_acceptance", "completion_budget_as_terminal_failure", "true"),
    ("grpo", "mask_truncated_completions", True),
])
def test_resume_rejects_other_changes(resume_case, section, key, value):
    ckpt, config, _ = resume_case
    config = deepcopy(config)
    config[section][key] = value
    with pytest.raises(ValueError):
        runner.validate_resume_checkpoint(ckpt, config)


def test_resume_rejects_parent_true_mask(resume_case):
    ckpt, config, parent = resume_case
    parent["grpo"]["mask_truncated_completions"] = True
    (ckpt.parent.parent / "config.json").write_text(json.dumps(parent), encoding="utf-8")
    with pytest.raises(ValueError, match="parent truncation loss mask"):
        runner.validate_resume_checkpoint(ckpt, config)
