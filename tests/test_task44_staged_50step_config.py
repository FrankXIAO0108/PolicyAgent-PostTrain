"""Freeze the existing Task44 v6 reward while matching final Terminal settings."""
import json
from pathlib import Path

from src.training.run_retail_agentic_grpo import (
    validate_config_and_split,
    validate_optimization_contract,
)

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / 'configs/retail_agentic_qwen3_4b_task44_staged_v6_n4_50step_c8192_v1.json'


def read(name):
    return json.loads((ROOT / 'configs' / name).read_text(encoding='utf-8'))


def test_only_reward_differs_from_matching_terminal_protocol():
    staged = json.loads(CONFIG.read_text(encoding='utf-8'))
    terminal = read('retail_agentic_qwen3_4b_task44_reward_ab_terminal_n4_50step_c8192_resume20_toollimit0_v1.json')
    old_staged = read('retail_agentic_qwen3_4b_task44_reward_ab_staged_v6_n4_10step_v1.json')
    assert staged.pop('reward') == old_staged['reward']
    terminal.pop('reward')
    assert staged == terminal


def test_fifty_steps_two_hundred_rollouts_and_same_sft_start():
    config = validate_config_and_split(CONFIG)['config']
    contract = validate_optimization_contract(config, selected_task_count=1)
    assert contract['expected_optimizer_steps'] == 50
    assert contract['expected_rollouts'] == 200
    assert contract['expected_groups'] == 50
    assert contract['generation_batch_size'] == 4
    assert config['model']['source_stage'] == 'SFT_PROTOCOL_BRIDGE'
    assert config['grpo']['max_completion_length'] == 8192
    assert config['grpo']['learning_rate'] == 5e-6
    assert config['grpo']['beta'] == 0.02
    assert config['engineering_acceptance']['tool_iteration_limit_as_terminal_failure'] is True
    assert config['engineering_acceptance']['transport_complete_groups_required'] is True


def test_v6_component_weights_caps_and_claim_boundaries():
    config = json.loads(CONFIG.read_text(encoding='utf-8'))
    reward = config['reward']['staged_reward_spec']['reward']
    assert reward['composition_mode'] == 'hierarchical_state_authorization_review_v6'
    assert reward['additive_component_weights'] == {
        'environment_state': 0.35, 'interaction_complete': 0.1,
        'identity_link': 0.08, 'target_evidence': 0.12,
        'write_authorization': 0.2, 'post_write_communication': 0.15,
    }
    assert reward['no_verified_write_cap'] == 0.25
    assert reward['authorization_fail_hard_cap'] == 0.15
    assert reward['authorization_review_cap'] == 0.75
    assert reward['unexpected_write_hard_cap'] == 0
    assert config['claims']['business_improvement_claim_allowed'] is False
    assert config['claims']['generalization_claim_allowed'] is False


def test_plan_never_authorizes_cloud_or_resumes_terminal_checkpoint():
    plan = read('execution/task44_staged_v6_50step_plan_v1.json')
    assert plan['current_task'] == 'LOCAL_DESIGN_AND_VALIDATION'
    assert plan['cloud_execution_authorized'] is False
    assert plan['external_api_authorized'] is False
    assert plan['resume_from_checkpoint'] is None
    assert plan['historical_terminal_is_strict_control'] is False
