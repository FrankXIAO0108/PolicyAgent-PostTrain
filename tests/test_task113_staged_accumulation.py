"""Offline contracts for the new SFT30 / Task113 run; no model or API calls."""
from copy import deepcopy
from pathlib import Path
import json

import pytest

from src.evaluation.staged_reward_shadow import _claim_evidence_diagnostic, score_rollout
from src.training import run_retail_agentic_grpo as runner

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/retail_agentic_qwen3_4b_task113_sft30_staged_v7_1_n4_acc8_s50_v1.json"


def config():
    return runner.load_json(CONFIG)


def claim(text):
    spec = config()["reward"]["staged_reward_spec"]["tasks"]["113"]
    trace = [{"name": "get_user_details", "arguments": {"user_id": "yara_muller_8652"},
              "result": {"error": False, "content": '{"payment_methods":{"credit_card":{}}}'}}]
    return _claim_evidence_diagnostic([{"role": "assistant", "content": text}], trace, spec)


@pytest.mark.parametrize("text", [
    "The refund will be processed immediately if the original payment was a gift card, otherwise it will take 5-7 business days for a credit card.",
    "If the original payment was a gift card, the refund will be processed immediately.",
    "The refund will not be processed immediately.",
    "A refund of $243.62 will be processed to your credit card within 5-7 business days.",
])
def test_correct_policy_or_timing_is_not_a_false_immediate_refund(text):
    assert claim(text)["verdict"] == "PASS"


@pytest.mark.parametrize("text", [
    "A refund of **$243.62** will be processed to your original payment method (**visa ending in 6918**) immediately.",
    "The refund will be processed immediately if the original payment was a gift card. Your credit card refund will be processed immediately.",
    "Your credit card refund will be processed immediately. The refund will be processed immediately if the original payment was a gift card.",
    "Your credit card refund will be processed immediately. If the original payment was a gift card, the refund will be processed immediately.",
    "Your credit card refund will be processed immediately, while the refund will be processed immediately if the original payment was a gift card.",
    "Your credit card refund will be processed immediately, and the refund will be processed immediately if the original payment was a gift card.",
    "Your credit card refund will be processed immediately and the refund will be processed immediately if the original payment was a gift card.",
    "Your credit card refund will be processed immediately while the refund will be processed immediately if the original payment was a gift card.",
])
def test_conditional_cannot_hide_a_separate_wrong_promise(text):
    assert claim(text)["verdict"] == "FAIL"


def test_two_generation_groups_per_update():
    c = config()
    v = runner.validate_optimization_contract(c, selected_task_count=1)
    assert (v["expected_groups"], v["expected_rollouts"]) == (100, 400)
    assert v["generation_batch_size"] == 4
    assert c["grpo"]["gradient_accumulation_steps"] == 8
    assert runner.explicit_grpo_training_kwargs(c) == {
        "max_grad_norm": 1.0, "warmup_steps": 2, "lr_scheduler_type": "linear",
        "optim": "adamw_torch", "scale_rewards": "group", "top_p": 1.0, "top_k": 0,
    }
    assert runner.completion_budget_training_kwargs(c) == {"mask_truncated_completions": False}


@pytest.mark.parametrize("field,value", [
    ("expected_groups", 50), ("expected_rollouts", 200),
    ("one_prompt_group_per_optimizer_step", True),
    ("steps_per_generation_equals_gradient_accumulation_required", True),
])
def test_reject_wrong_batch_accounting(field, value):
    c = config()
    c["engineering_acceptance"][field] = value
    with pytest.raises(ValueError):
        runner.validate_optimization_contract(c)


@pytest.mark.parametrize("field,value", [("num_iterations", 2), ("gradient_accumulation_steps", 6)])
def test_reject_unreviewed_reuse_or_partial_generation_batches(field, value):
    c = config()
    c["grpo"][field] = value
    with pytest.raises(ValueError):
        runner.validate_optimization_contract(c)


def test_reward_change_is_only_declared_fixes_not_weights():
    old = runner.load_json(ROOT / "configs/retail_agentic_qwen3_4b_task113_staged_claim_v7_grpo_10step_n4_v2.json")
    new = deepcopy(config()["reward"])
    spec = new["staged_reward_spec"]
    spec["spec_id"] = old["reward"]["staged_reward_spec"]["spec_id"]
    del spec["tasks"]["113"]["claim_evidence_rules"][0]["trigger_exclusion_patterns"]
    del spec["reward"]["authorization_scope"]
    spec["reward"]["authorization_fail_hard_cap"] = 0.15
    assert new == old["reward"]


def test_new_sft_binding_and_inputs():
    result = runner.validate_config_and_split(CONFIG)
    assert result["sft_manifest_binding"]["binding_type"] == "DERIVED"
    assert result["config"]["model"]["expected_sha256"].startswith("2215F09D")
    assert result["config"]["data"]["task_ids"] == ["113"]


def test_legacy_defaults_unchanged():
    assert runner.explicit_grpo_training_kwargs({"grpo": {}}) == {}


def test_multi_group_run_cannot_resume_old_wrong_batch_checkpoint(tmp_path):
    with pytest.raises(ValueError, match="Multi-group resume is not audited"):
        runner.validate_resume_checkpoint(tmp_path / "checkpoint-10", config())


def test_separate_groups_are_not_jointly_standardized():
    rows = [{"task_id": "113", "user_seed": 1, "reward": {"reward": r}}
            for r in [1, 1, 1, 1, 0, 0, 0, 0]]
    separate = runner.summarize_sequential_reward_groups(rows, 4)
    joint = runner.summarize_sequential_reward_groups(rows, 8)
    assert separate["group_count"] == 2
    assert separate["nonzero_std_group_count"] == 0
    assert joint["nonzero_std_group_count"] == 1
    assert separate["groups"][0]["advantages"] == [0, 0, 0, 0]


def test_frozen_eval_configs_change_only_model_and_arm():
    from scripts.prepare_task113_staged_eval import evaluation_configs
    c = config()
    before = evaluation_configs(c, c["model"], "SFT")
    after = evaluation_configs(c, {**c["model"], "expected_sha256": "FUTURE_TEST_ONLY"}, "GRPO")
    assert len(before) == 9
    assert sum(x["diagnostic"]["expected_rollouts"] for x in before.values()) == 33
    assert c["seed"] not in {x["seed"] for x in before.values()}
    for name in before:
        a, b = deepcopy(before[name]), deepcopy(after[name])
        del a["model"], b["model"]
        del a["claims"]["paired_eval_arm"], b["claims"]["paired_eval_arm"]
        assert a == b
        assert a["execution_mode"] == "ROLLOUT_DIAGNOSTIC"
        assert a["grpo"]["learning_rate"] == a["grpo"]["beta"] == 0


@pytest.mark.parametrize("confirmed,expected", [(False, 0.0), (True, 0.2)])
def test_partial_cancel_cannot_bypass_authorization(confirmed, expected):
    # Synthetic unit fixture, not an experiment trajectory or outcome claim.
    spec = config()["reward"]["staged_reward_spec"]
    task = spec["tasks"]["113"]
    identity = task["identity_link"]
    trace = [
        {"name": "find_user_id_by_email", "arguments": {},
         "result": {"error": False, "content": identity["required_user_id"]}},
        {"name": "get_user_details", "arguments": {"user_id": identity["required_user_id"]},
         "result": {"error": False, "content": json.dumps({"orders": identity["required_order_ids"]})}},
    ] + [{**call, "result": {"error": False, "content": "{}"}}
         for call in task["target_evidence_calls"]]
    raw = {"task_id": "113", "evidence_sha256": "UNIT_FIXTURE", "messages": [
        {"role": "assistant", "tool_calls": [{"id": "cancel1", "name": "cancel_pending_order",
          "arguments": {"order_id": "#W5056519", "reason": "ordered by mistake"}}]},
        {"role": "tool", "id": "cancel1", "name": "cancel_pending_order", "error": False,
         "content": '{"status":"canceled"}'},
    ]}
    evidence = {"task_id": "113", "evidence_sha256": "UNIT_FIXTURE", "tool_trace": trace,
                "terminal_evaluator": {"reward": 0, "user_stopped": False, "action_progress": {
                    "unexpected_write_count": 0, "matches": [
                        {"action_id": "114_0", "name": "cancel_pending_order", "matched": True,
                         "matched_call_index": 0},
                        {"action_id": "114_1", "name": "cancel_pending_order", "matched": False},
                    ]}}}
    confirmation = {"write_count": 1, "confirmed_write_count": int(confirmed), "checks": [
        {"confirmed": confirmed, "parameter_binding": {"verdict": "PASS" if confirmed else "FAIL"}},
    ]}
    score = score_rollout(raw, evidence, spec, confirmation)
    assert not score["write_complete"]
    assert score["staged_reward"] == expected
    assert score["authorization_fail_cap_applied"] is (not confirmed)
