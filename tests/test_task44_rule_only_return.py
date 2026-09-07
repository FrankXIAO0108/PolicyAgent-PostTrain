"""The selected staged reward must not invoke semantic scoring or its gate."""

import json
from pathlib import Path
from types import SimpleNamespace

from src.evaluation import task44_hybrid_reward as hybrid
from src.rl.retail_agentic_env import tiered_terminal_process_reward
from src.training.run_retail_agentic_grpo import (
    validate_config_and_split,
    validate_optimization_contract,
)
from test_task44_reward_evidence import (
    test_v2_score_entrypoint_checks_amount_instead_of_trusting_legacy_pass as check_bad_refund,
)

ROOT = Path(__file__).resolve().parents[1]


def selected_config():
    plan = json.loads((ROOT / 'configs/execution/task44_rule_only_n4_s50_return_v1.json').read_text(encoding='utf-8'))
    return ROOT / plan['runtime_config']


def forbid(*args, **kwargs):
    raise AssertionError('Rule-only mode must not enter LLM scoring')


def test_selected_config_keeps_training_protocol_and_only_removes_semantics():
    config = validate_config_and_split(selected_config())['config']
    original = json.loads((ROOT / 'configs/retail_agentic_qwen3_4b_task44_hybrid_n4_50step_v1.json').read_text(encoding='utf-8'))
    assert 'semantic_assistance' not in config['reward']['staged_reward_spec']
    original['reward']['staged_reward_spec'].pop('semantic_assistance')
    # Pure-rule config predates an explicit spelling of the same default.
    config['grpo'].setdefault('num_iterations', 1)
    # Historical A/B intention flags are not evidence of a completed comparison.
    for flag in ('reward_ab_comparison', 'reward_only_variable_between_arms'):
        assert config['claims'].pop(flag) is True
        assert original['claims'].pop(flag) is False
    assert config == original
    contract = validate_optimization_contract(config, selected_task_count=1)
    assert contract['expected_optimizer_steps'] == 50
    assert contract['expected_rollouts'] == 200
    assert config['grpo']['beta'] == 0.02


def test_group_preparation_is_noop_without_semantic_mode(monkeypatch, tmp_path):
    monkeypatch.setattr(hybrid, 'call_extractor', forbid)
    monkeypatch.setattr(hybrid, '_save_new', forbid)
    monkeypatch.delenv('POLICYAGENT_ROLLOUT_LOG', raising=False)
    environments = [SimpleNamespace(_uses_semantic_reward=lambda: False) for _ in range(4)]
    assert hybrid.prepare_semantic_group(environments, None, optimizer_step=0, group_index=0) is None
    assert not list(tmp_path.iterdir())


def test_online_rule_scoring_never_calls_hybrid_and_keeps_bad_refund_penalty(monkeypatch):
    monkeypatch.setattr(hybrid, 'hybrid_score', forbid)
    monkeypatch.setattr(hybrid, 'call_extractor', forbid)
    monkeypatch.delenv('SHADOW_JUDGE_API_KEY', raising=False)
    check_bad_refund()
    config = json.loads(selected_config().read_text(encoding='utf-8'))
    score = tiered_terminal_process_reward(
        task_id='44', messages=[],
        action_progress={'matches': [{'action_id': '44_4', 'matched': False}]},
        environment_payload={'reward': 0}, communication_payload={},
        environment_state_reward=0, user_stopped=False, completion={},
        staged_reward_spec=config['reward']['staged_reward_spec'],
    )
    assert 'semantic_assistance' not in score
    assert 0 <= score['staged_reward'] <= 0.25
