"""Budget-only negatives continue; mixed/system failures remain fail-closed."""
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import test_retail_agentic_rl as fixtures

from src.rl.retail_agentic_env import COMPLETION_BUDGET_AS_TERMINAL_FAILURE_ENV
from src.training.rollout_diagnostics import make_guarded_grpo_trainer
from src.training.run_retail_agentic_grpo import (
    completion_budget_training_kwargs,
    validate_config_and_split,
    validate_optimization_contract,
)

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / 'configs/retail_agentic_qwen3_4b_task44_staged_v6_n4_50step_c8192_budget0_v3.json'


def budget(**extra):
    values = dict(stop_reason='COMPLETION_BUDGET_EXHAUSTED',
                  stop_flags=['COMPLETION_BUDGET_EXHAUSTED'],
                  model_eos_observed=False, completion_token_budget_exhausted=True,
                  model_completion_truncated=True)
    values.update(extra)
    return fixtures._completion_telemetry(**values)


def environment(monkeypatch, *, enabled=True):
    monkeypatch.setenv('POLICYAGENT_REQUIRE_TRANSPORT_COMPLETE_GROUPS', '1')
    monkeypatch.setenv(COMPLETION_BUDGET_AS_TERMINAL_FAILURE_ENV, '1' if enabled else '0')
    monkeypatch.delenv('POLICYAGENT_ROLLOUT_LOG', raising=False)
    env = fixtures.RetailAgenticEnvironmentTests().make_env(reward=1.0)
    env.reset(task_id='1', initial_user_message='Need help')
    return env


def test_budget_negative_skips_evaluator_but_keeps_full_audit(monkeypatch, tmp_path):
    env = environment(monkeypatch)
    monkeypatch.setenv('POLICYAGENT_ROLLOUT_LOG', str(tmp_path/'raw.jsonl'))
    monkeypatch.setenv('POLICYAGENT_ROLLOUT_EVIDENCE_LOG', str(tmp_path/'evidence.jsonl'))
    def forbidden(*args):
        raise AssertionError('Censored trajectory must not be judged as complete')
    env._evaluator = forbidden
    env._set_trainer_completion_telemetry(budget())
    assert env._blocking_transport_reasons() == []
    assert env.get_reward() == 0
    assert env.get_reward() == 0
    rows = (tmp_path/'raw.jsonl').read_text().splitlines()
    assert len(rows) == 1
    row = json.loads(rows[0])
    assert row['reward']['reward'] == 0
    assert row['reward']['evaluator_called'] is False
    assert row['completion']['model_completion_truncated'] is True
    assert not (tmp_path/'rejected_rollouts.jsonl').exists()


def test_default_still_rejects_truncation(monkeypatch):
    env = environment(monkeypatch, enabled=False)
    env._set_trainer_completion_telemetry(budget())
    with pytest.raises(RuntimeError, match='Transport-invalid'):
        env.get_reward()


@pytest.mark.parametrize('field,flag', [
    ('context_limit_reached','CONTEXT_LIMIT'),
    ('unresolved_tool_call','UNRESOLVED_TOOL_CALL'),
    ('tool_iteration_limit_reached','TOOL_ITERATION_LIMIT'),
    ('framework_loop_abnormal_end',None),
])
def test_mixed_failure_not_converted_to_zero(monkeypatch, field, flag):
    env = environment(monkeypatch)
    flags = ['COMPLETION_BUDGET_EXHAUSTED'] + ([flag] if flag else [])
    env._set_trainer_completion_telemetry(budget(**{field: True}, stop_flags=flags))
    assert field in env._blocking_transport_reasons()
    with pytest.raises(RuntimeError, match='Transport-invalid'):
        env.get_reward()


def test_tool_observation_exhaustion_is_not_model_budget_failure(monkeypatch):
    env = environment(monkeypatch)
    env._set_trainer_completion_telemetry(budget(
        stop_reason='TOOL_RESULT_BUDGET_EXCEEDED',
        stop_flags=['TOOL_RESULT_BUDGET_EXCEEDED'], model_completion_truncated=False))
    with pytest.raises(RuntimeError, match='Transport-invalid'):
        env.get_reward()


def test_missing_telemetry_and_evaluator_error_still_raise(monkeypatch):
    env = environment(monkeypatch)
    with pytest.raises(RuntimeError, match='missing_completion_telemetry'):
        env.get_reward()
    env._set_trainer_completion_telemetry(fixtures._completion_telemetry())
    def broken(*args):
        raise ConnectionError('simulated system error')
    env._evaluator = broken
    with pytest.raises(ConnectionError):
        env.get_reward()


def test_invalid_env_flag_rejected(monkeypatch):
    monkeypatch.setenv(COMPLETION_BUDGET_AS_TERMINAL_FAILURE_ENV, 'true')
    with pytest.raises(RuntimeError, match="exactly '0' or '1'"):
        fixtures.RetailAgenticEnvironmentTests().make_env()


def test_new_config_explicitly_keeps_truncated_samples_in_loss():
    config = validate_config_and_split(CONFIG)['config']
    contract = validate_optimization_contract(config, selected_task_count=1)
    assert contract['expected_rollouts'] == 200
    assert completion_budget_training_kwargs(config) == {'mask_truncated_completions': False}
    assert config['grpo']['beta'] == 0.02
    original = json.loads((ROOT/'configs/retail_agentic_qwen3_4b_task44_staged_v6_n4_50step_c8192_v2.json').read_text())
    assert completion_budget_training_kwargs(original) == {}
    config['engineering_acceptance'].pop('completion_budget_as_terminal_failure')
    config['grpo'].pop('mask_truncated_completions')
    for key in ('reward_ab_comparison','reward_only_variable_between_arms'):
        assert config['claims'][key] is False
        config['claims'][key] = original['claims'][key]
    assert config == original


@pytest.mark.parametrize('bad', [True, None, 'false', 0])
def test_mask_cannot_silently_discard_budget_negative(bad):
    config = json.loads(CONFIG.read_text())
    config['grpo']['mask_truncated_completions'] = bad
    with pytest.raises(ValueError, match='mask_truncated'):
        validate_optimization_contract(config, selected_task_count=1)


def test_group_keeps_truncated_member_and_other_rewards(monkeypatch):
    envs = [environment(monkeypatch) for _ in range(4)]
    events = []
    class Native:
        def __init__(self):
            self.model = SimpleNamespace(config=SimpleNamespace(max_position_embeddings=100))
            self._is_vlm = False
            self.state = SimpleNamespace(global_step=15)
            self.max_completion_length = 3
            self.max_tool_calling_iterations = 32
            self._tokenizer = SimpleNamespace(eos_token_id=99, decode=lambda ids, **kw: str(ids))
            self.environments = envs

        def _generate(self, prompts):
            trace = self._policyagent_stop_trace
            ids = [[5,6,7,8], [5,99], [6,99], [7,99]]
            trace.generated(trace.start(prompts), prompts, ids, 99)
            kept = [ids[0][:3], *ids[1:]]
            return prompts, kept, [[1]*len(s) for s in kept], [[],[],[],[]]

    trainer = make_guarded_grpo_trainer(Native, events.append)()
    output = trainer._generate([[1]]*4)
    assert output[2][0] == [1,1,1]  # not all-masked/dropped
    assert [e._blocking_transport_reasons() for e in envs] == [[],[],[],[]]
    assert [e.get_reward() for e in envs] == [0,1,1,1]
