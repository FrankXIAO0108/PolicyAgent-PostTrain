import base64
from copy import deepcopy
import sys
from types import ModuleType, SimpleNamespace
from unittest.mock import patch

import pytest

from src.training.rollout_diagnostics import (
    _remaining_batch_generation_budget,
    BudgetTrace,
    GuardedTrajectoryTrace,
    bind_sampling_runtime,
    make_guarded_grpo_trainer,
    make_sampling_trainer,
    validate_sampling_request,
    verify_trl_source,
)


def test_remaining_batch_budget_uses_max_and_restores_even_on_exception():
    trace = GuardedTrajectoryTrace(8, 100, lambda e: None)
    trace.start([[1], [2]])
    config = SimpleNamespace(max_new_tokens=8)
    kwargs = {"max_new_tokens": 8, "temperature": 0.8}
    trainer = SimpleNamespace(generation_config=config, generation_kwargs=kwargs)
    events = []
    with pytest.raises(RuntimeError, match="test interruption"):
        with _remaining_batch_generation_budget(
            trainer, trace, [0, 1], [[1, 3, 4, 5], [2, 6]], events.append
        ):
            assert trainer.generation_config.max_new_tokens == 7
            assert trainer.generation_kwargs["max_new_tokens"] == 7
            assert config.max_new_tokens == 8
            raise RuntimeError("test interruption")
    assert trainer.generation_config is config
    assert trainer.generation_kwargs is kwargs
    assert events[0]["remaining_tokens"] == [5, 7]


def test_remaining_budget_refuses_zero_and_keeps_configured_smaller_limit():
    trace = GuardedTrajectoryTrace(2, 100, lambda e: None)
    trace.start([[1]])
    trainer = SimpleNamespace(
        generation_config=SimpleNamespace(max_new_tokens=1), generation_kwargs={}
    )
    with _remaining_batch_generation_budget(trainer, trace, [0], [[1]], lambda e: None):
        assert trainer.generation_config.max_new_tokens == 1
    with pytest.raises(RuntimeError, match="no remaining"):
        with _remaining_batch_generation_budget(
            trainer, trace, [0], [[1, 3, 4]], lambda e: None
        ):
            pytest.fail("Zero-budget generation must not start")
    trainer.generation_config.max_new_tokens = 8
    trainer.generation_kwargs["max_new_tokens"] = 1
    with _remaining_batch_generation_budget(trainer, trace, [0], [[1]], lambda e: None):
        assert trainer.generation_config.max_new_tokens == 1


def test_guarded_rejected_group_preserves_all_candidates_before_reward():
    events, saved = [], []

    class Environment:
        _runtime_stop_signals = []
        _user_stopped = True

        def _set_trainer_completion_telemetry(self, payload):
            self.telemetry = payload

        def _blocking_transport_reasons(self):
            return ["model_completion_truncated"] if self.telemetry[
                "model_completion_truncated"
            ] else []

        def _persist_rejected_rollout(self, *, trainer_evidence):
            saved.append(trainer_evidence)

        def get_reward(self):
            pytest.fail("Group preservation must precede all reward calls")

    class Native:
        def __init__(self):
            self.model = SimpleNamespace(config=SimpleNamespace(max_position_embeddings=100))
            self._is_vlm = False
            self.state = SimpleNamespace(global_step=49)
            self.max_completion_length = 3
            self.max_tool_calling_iterations = 32
            self._tokenizer = SimpleNamespace(
                eos_token_id=99, decode=lambda ids, **kw: str(ids)
            )
            self.environments = [Environment() for _ in range(4)]

        def _generate(self, prompts):
            trace = self._policyagent_stop_trace
            ids = [[5, 99], [5, 6, 7, 8], [6, 99], [7, 99]]
            trace.generated(trace.start(prompts), prompts, ids, 99)
            kept = [ids[0], ids[1][:3], ids[2], ids[3]]
            return (prompts, kept, [[1] * len(s) for s in kept],
                    [[{"role": "assistant", "content": str(s)}] for s in kept])

    trainer = make_guarded_grpo_trainer(Native, events.append)()
    trainer._generate([[1]] * 4)
    assert [s["row"] for s in saved] == [0, 1, 2, 3]
    assert saved[1]["completion_token_ids"] == [5, 6, 7]
    assert saved[1]["optimizer_step"] == 49
    assert saved[0]["group_blocking_reasons"] == [[], ["model_completion_truncated"], [], []]
    censored = next(e for e in events if e["event"] == "censored_generation")
    assert censored["generated_token_ids"] == [5, 6, 7, 8]
    assert not hasattr(trainer, "_policyagent_stop_trace")


def _guarded_environment(*, user_stopped=False, signals=None):
    return SimpleNamespace(
        _user_stopped=user_stopped,
        _runtime_stop_signals=list(signals or []),
    )


def _guarded_summary(
    trace,
    completion_ids,
    completions,
    environments,
    *,
    masks=None,
    iterations=4,
):
    masks = masks or [[1] * len(ids) for ids in completion_ids]
    output = (
        [row["prompt_ids"] for row in trace.rows],
        completion_ids,
        masks,
        completions,
    )
    return trace.summarize(
        output,
        environments,
        SimpleNamespace(eos_token_id=99),
        iterations,
    )


def test_guarded_trace_distinguishes_eos_and_completion_budget():
    events = []
    eos_trace = GuardedTrajectoryTrace(8, 100, events.append)
    indices = eos_trace.bind_generation([[1, 2]])
    eos_trace.generated(indices, [[1, 2]], [[10, 99]], 99)
    eos = _guarded_summary(
        eos_trace,
        [[10, 99]],
        [[{"role": "assistant", "content": "done"}]],
        [_guarded_environment(user_stopped=True)],
    )[0]
    assert eos["stop_reason"] == "USER_STOP_AND_MODEL_EOS"
    assert eos["model_eos_observed"] is True
    assert eos["model_completion_truncated"] is False

    budget_trace = GuardedTrajectoryTrace(2, 100, events.append)
    indices = budget_trace.bind_generation([[1, 2]])
    budget_trace.generated(indices, [[1, 2]], [[10, 11]], 99)
    budget = _guarded_summary(
        budget_trace,
        [[10, 11]],
        [[{"role": "assistant", "content": "unfinished"}]],
        [_guarded_environment()],
    )[0]
    assert budget["stop_reason"] == "COMPLETION_BUDGET_EXHAUSTED"
    assert budget["completion_token_budget_exhausted"] is True
    assert budget["model_completion_truncated"] is True


def test_guarded_trace_distinguishes_tool_result_rollback_from_unresolved_call():
    trace = GuardedTrajectoryTrace(3, 100, lambda event: None)
    indices = trace.bind_generation([[1, 2]])
    trace.generated(indices, [[1, 2]], [[10, 99]], 99)
    trace.tool_called(0)
    trace.suffix([20, 21])
    report = _guarded_summary(
        trace,
        [[10, 99]],
        [[{"role": "assistant", "tool_calls": [{"type": "function"}]}]],
        [_guarded_environment()],
    )[0]
    assert report["stop_reason"] == "TOOL_RESULT_BUDGET_EXCEEDED"
    assert report["completion_token_budget_exhausted"] is True
    assert report["unresolved_tool_call"] is True
    assert report["model_completion_truncated"] is False


def test_guarded_trace_uses_only_causal_environment_limit_signals():
    trace = GuardedTrajectoryTrace(8, 100, lambda event: None)
    indices = trace.bind_generation([[1, 2]])
    trace.generated(indices, [[1, 2]], [[10, 99]], 99)
    report = _guarded_summary(
        trace,
        [[10, 99]],
        [[{"role": "assistant", "content": "stopped after error"}]],
        [_guarded_environment(signals=["CUSTOMER_TURN_LIMIT"])],
    )[0]
    assert report["stop_reason"] == "MODEL_EOS_BEFORE_USER_STOP"
    assert report["stop_flags"] == ["CUSTOMER_TURN_LIMIT"]

    recovered = GuardedTrajectoryTrace(8, 100, lambda event: None)
    indices = recovered.bind_generation([[1, 2]])
    recovered.generated(indices, [[1, 2]], [[10, 99]], 99)
    report = _guarded_summary(
        recovered,
        [[10, 99]],
        [[{"role": "assistant", "content": "recovered after tool error"}]],
        [
            _guarded_environment(
                user_stopped=True,
                signals=["TOOL_CALL_LIMIT"],
            )
        ],
    )[0]
    assert report["stop_reason"] == "USER_STOP_AND_MODEL_EOS"
    assert report["stop_flags"] == ["TOOL_CALL_LIMIT"]

    control = GuardedTrajectoryTrace(8, 100, lambda event: None)
    indices = control.bind_generation([[1, 2]])
    control.generated(indices, [[1, 2]], [[10, 99]], 99)
    report = _guarded_summary(
        control,
        [[10, 99]],
        [[{"role": "assistant", "content": "ended"}]],
        [_guarded_environment()],
    )[0]
    assert report["stop_reason"] == "MODEL_EOS_BEFORE_USER_STOP"


def test_guarded_trace_reports_context_iteration_and_non_eos_stops():
    context = GuardedTrajectoryTrace(20, 5, lambda event: None)
    indices = context.bind_generation([[1, 2]])
    context.generated(indices, [[1, 2]], [[10, 99]], 99)
    context.tool_called(0)
    context.suffix([20])
    report = _guarded_summary(
        context,
        [[10, 99]],
        [[{"role": "assistant", "tool_calls": [{"type": "function"}]}]],
        [_guarded_environment()],
    )[0]
    assert report["stop_reason"] == "CONTEXT_LIMIT"
    assert report["context_limit_reached"] is True

    iteration = GuardedTrajectoryTrace(20, 100, lambda event: None)
    indices = iteration.bind_generation([[1, 2]])
    iteration.generated(indices, [[1, 2]], [[10, 99]], 99)
    iteration.tool_called(0)
    iteration.suffix([20])
    continuation = [[1, 2, 10, 99, 20]]
    indices = iteration.bind_generation(continuation)
    iteration.generated(indices, continuation, [[11, 99]], 99)
    report = _guarded_summary(
        iteration,
        [[10, 99, 20, 11, 99]],
        [[{"role": "assistant", "tool_calls": [{"type": "function"}]}]],
        [_guarded_environment()],
        masks=[[1, 1, 0, 1, 1]],
        iterations=1,
    )[0]
    assert report["stop_reason"] == "TOOL_ITERATION_LIMIT"
    assert report["tool_iteration_limit_reached"] is True

    no_eos = GuardedTrajectoryTrace(20, 100, lambda event: None)
    indices = no_eos.bind_generation([[1, 2]])
    no_eos.generated(indices, [[1, 2]], [[10]], 99)
    report = _guarded_summary(
        no_eos,
        [[10]],
        [[{"role": "assistant", "content": "done"}]],
        [_guarded_environment(user_stopped=True)],
    )[0]
    assert report["stop_reason"] == "MODEL_END_WITHOUT_EOS"
    assert report["model_eos_observed"] is False


def test_guarded_trace_binding_failure_is_deferred_until_before_reward():
    events = []
    trace = GuardedTrajectoryTrace(20, 100, events.append)
    indices = trace.bind_generation([[1, 2]])
    trace.generated(indices, [[1, 2]], [[10, 99]], 99)

    # An unsupported non-function tool call never performs the native mapping
    # membership check, so its suffix cannot be assigned to a rollout safely.
    trace.suffix([20])
    assert trace.bind_generation([[1, 2, 10, 99, 20]]) is None
    assert events[-1] == {
        "event": "telemetry_binding_error",
        "reason": "Unbound guarded tool suffix",
    }
    with pytest.raises(RuntimeError, match="Unbound guarded tool suffix"):
        _guarded_summary(
            trace,
            [[10, 99, 20]],
            [[{"role": "assistant", "content": "native generation completed"}]],
            [_guarded_environment()],
            masks=[[1, 1, 0]],
        )


def test_guarded_generate_binds_telemetry_before_returning_native_tuple():
    events = []

    class Environment:
        _user_stopped = True
        _runtime_stop_signals = []

        def _set_trainer_completion_telemetry(self, payload):
            self.telemetry = payload

    class Base:
        def __init__(self):
            self.model = SimpleNamespace(
                config=SimpleNamespace(max_position_embeddings=100)
            )
            self._is_vlm = False
            self.max_completion_length = 8
            self.max_tool_calling_iterations = 4
            self._tokenizer = SimpleNamespace(eos_token_id=99)
            self.environments = [Environment(), Environment()]
            self.state = SimpleNamespace(global_step=7)

        def _generate(self, prompts):
            trace = self._policyagent_stop_trace
            prompt_ids = [[1, 2], [1, 2]]
            indices = trace.bind_generation(prompt_ids)
            outputs = [[10, 99], [11, 99]]
            trace.generated(indices, prompt_ids, outputs, 99)
            self.native_output = (
                prompt_ids,
                outputs,
                [[1, 1], [1, 1]],
                [
                    [{"role": "assistant", "content": "done"}],
                    [{"role": "assistant", "content": "done"}],
                ],
                object(),
            )
            return self.native_output

    trainer = make_guarded_grpo_trainer(Base, events.append)()
    output = trainer._generate(["same", "same"])
    assert output is trainer.native_output
    assert all(
        environment.telemetry["stop_reason"] == "USER_STOP_AND_MODEL_EOS"
        for environment in trainer.environments
    )
    stops = [event for event in events if event["event"] == "rollout_stop"]
    assert [event["row"] for event in stops] == [0, 1]
    assert all(event["optimizer_step"] == 7 for event in stops)
    assert not hasattr(trainer, "_policyagent_stop_trace")


def test_guarded_tool_observer_preserves_function_arguments_and_result():
    events = []
    calls = []

    def write(value):
        calls.append(value)
        return {"saved": value}

    class Environment:
        _user_stopped = True
        _runtime_stop_signals = []

        def _set_trainer_completion_telemetry(self, payload):
            self.telemetry = payload

    class Base:
        def __init__(self):
            self.model = SimpleNamespace(
                config=SimpleNamespace(max_position_embeddings=100)
            )
            self._is_vlm = False
            self.max_completion_length = 8
            self.max_tool_calling_iterations = 4
            self._tokenizer = SimpleNamespace(eos_token_id=99)
            self.environments = [Environment()]
            self.state = SimpleNamespace(global_step=0)
            self._sync_tool_dicts = [{"write": write}]

        def _get_tool_suffix_ids(self, tool_messages):
            assert tool_messages == [{"role": "tool", "content": "{'saved': 3}"}]
            return [20]

        def _tool_call_loop(self):
            mapping = self._sync_tool_dicts[0]
            assert "write" in mapping
            function = mapping["write"]
            assert function is write
            result = function(value=3)
            return self._get_tool_suffix_ids([{"role": "tool", "content": str(result)}])

        def _generate(self, prompts):
            trace = self._policyagent_stop_trace
            prompt_ids = [[1, 2]]
            indices = trace.bind_generation(prompt_ids)
            trace.generated(indices, prompt_ids, [[10, 99]], 99)
            suffix = self._tool_call_loop()
            continuation = [[1, 2, 10, 99, *suffix]]
            indices = trace.bind_generation(continuation)
            trace.generated(indices, continuation, [[11, 99]], 99)
            completion_ids = [[10, 99, *suffix, 11, 99]]
            return (
                prompt_ids,
                completion_ids,
                [[1, 1, 0, 1, 1]],
                [[{"role": "assistant", "content": "done"}]],
                object(),
            )

    trainer = make_guarded_grpo_trainer(Base, events.append)()
    output = trainer._generate(["prompt"])
    assert calls == [3]
    assert output[1] == [[10, 99, 20, 11, 99]]
    assert trainer.environments[0].telemetry["observation_tokens_retained"] == 1
    assert trainer.environments[0].telemetry["model_tokens_retained"] == 4


def test_guarded_generate_records_abnormal_end_without_replacing_exception():
    events = []

    class Base:
        def __init__(self):
            self.model = SimpleNamespace(
                config=SimpleNamespace(max_position_embeddings=100)
            )
            self._is_vlm = False
            self.max_completion_length = 8
            self.max_tool_calling_iterations = 4
            self._tokenizer = SimpleNamespace(eos_token_id=99)
            self.environments = []
            self.state = SimpleNamespace(global_step=3)

        def _generate(self, prompts):
            raise LookupError("native failure")

    trainer = make_guarded_grpo_trainer(Base, events.append)()
    with pytest.raises(LookupError, match="native failure"):
        trainer._generate(["prompt"])
    abnormal = events[-1]
    assert abnormal["event"] == "rollout_group_abnormal_end"
    assert abnormal["exception_type"] == "LookupError"
    assert abnormal["optimizer_step"] == 3
    assert not hasattr(trainer, "_policyagent_stop_trace")


def test_guarded_trainer_keeps_native_training_and_scoring_methods():
    class Base:
        def train(self):
            return "train"

        def create_optimizer(self):
            return "optimizer"

        def compute_loss(self):
            return "loss"

        def _generate_and_score_completions(self):
            return "score"

    guarded = make_guarded_grpo_trainer(Base, lambda event: None)
    for name in (
        "train",
        "create_optimizer",
        "compute_loss",
        "_generate_and_score_completions",
    ):
        assert getattr(guarded, name) is getattr(Base, name)


def new_trace(budget=8):
    events = []
    trace = BudgetTrace(budget, 100, events.append)
    trace.start([[1, 2], [1, 2]])
    trace.generated([0, 1], [[1, 2], [1, 2]], [[10, 99], [11, 99]], 99)
    return trace, events


def test_tool_overflow_keeps_raw_return_but_not_training_tokens():
    trace, events = new_trace()
    trace.tool_called(0)
    trace.suffix([20] * 7, [{"role": "tool", "content": "write was executed"}])
    assert trace.rows[0]["retained"] == [10, 99]
    assert trace.rows[0]["flags"] == ["TOOL_RESULT_BUDGET_EXCEEDED"]
    assert events[-1]["messages"][0]["content"] == "write was executed"
    assert events[-1]["would_rollback"] is True
    assert trace.next_generation == []


def test_identical_prompts_do_not_mix_up_candidate_binding():
    trace, _ = new_trace()
    trace.tool_called(0)
    trace.tool_called(1)
    trace.suffix([20] * 7, [{"role": "tool"}])  # first candidate cannot continue
    trace.suffix([30], [{"role": "tool"}])
    assert trace.next_generation == [(1, [1, 2, 11, 99, 30])]
    trace.generated([1], [trace.next_generation[0][1]], [[12, 99]], 99)
    assert trace.rows[0]["retained"] == [10, 99]
    assert trace.rows[1]["retained"] == [11, 99, 30, 12, 99]


def test_model_tail_clipping_recorded_even_if_raw_generation_ended_in_eos():
    trace, _ = new_trace(budget=4)
    trace.tool_called(0)
    trace.suffix([20], [{"role": "tool"}])
    trace.generated([0], [[1, 2, 10, 99, 20]], [[30, 99]], 99)
    assert trace.rows[0]["retained"] == [10, 99, 20, 30]
    assert "MODEL_GENERATION_LIMIT" in trace.rows[0]["flags"]


def test_unbound_suffix_and_small_context_fail_closed():
    trace, _ = new_trace()
    with pytest.raises(RuntimeError, match="Unbound"):
        trace.suffix([20], [])
    with pytest.raises(ValueError, match="headroom"):
        BudgetTrace(8, 10, lambda e: None).start([[1, 2]])


def test_budget_error_has_priority_over_unresolved_tool_call():
    trace, _ = new_trace()
    trace.tool_called(0)
    trace.suffix([20] * 7, [{"role": "tool"}])
    output = (
        [[1, 2]] * 2,
        [[10, 99], [11, 99]],
        [[1, 1]] * 2,
        [
            [{"role": "assistant", "tool_calls": [{}]}],
            [{"role": "assistant", "content": "done"}],
        ],
    )
    tokenizer = SimpleNamespace(decode=lambda ids, **kw: str(ids))
    result = trace.summarize(
        output,
        [SimpleNamespace(_user_stopped=False), SimpleNamespace(_user_stopped=True)],
        tokenizer,
        40,
    )
    assert result[0]["stop_reason"] == "TOOL_RESULT_BUDGET_EXCEEDED"
    assert result[1]["stop_reason"] == "USER_STOP_AND_MODEL_END"
    assert not result[0]["trajectory_transport_complete"]
    assert result[1]["trajectory_transport_complete"]
    assert not any(row["training_eligible"] for row in result)


def test_sampling_request_cannot_enable_training_or_use_malformed_task_scope():
    config = {
        "execution_mode": "ROLLOUT_DIAGNOSTIC",
        "grpo": {
            "learning_rate": 0.0,
            "beta": 0.0,
            "num_generations": 2,
            "use_vllm": False,
        },
        "data": {"task_ids": ["43", "72"]},
        "rollout": {"stage": "FULL_TASK"},
    }
    contract = validate_sampling_request(config, 8192, 1)
    assert contract["mode"] == "STOCHASTIC_GROUP_SAMPLING"
    assert contract["do_sample"] is True
    for key, value in [("learning_rate", 1e-6), ("beta", 0.02)]:
        changed = deepcopy(config)
        changed["grpo"][key] = value
        with pytest.raises(ValueError):
            validate_sampling_request(changed, 8192, 1)
    with pytest.raises(ValueError):
        validate_sampling_request(config, None, 1)
    for task_ids in ([], ["43", "43"]):
        changed = deepcopy(config)
        changed["data"]["task_ids"] = task_ids
        with pytest.raises(ValueError, match="frozen task scope"):
            validate_sampling_request(changed, 8192, 1)
    changed = deepcopy(config)
    changed["data"]["max_tasks"] = 1
    with pytest.raises(ValueError, match="data.max_tasks"):
        validate_sampling_request(changed, 8192, 1)


def test_explicit_stochastic_sampling_accepts_frozen_task95_scope():
    config = {
        "execution_mode": "ROLLOUT_DIAGNOSTIC",
        "grpo": {
            "learning_rate": 0.0,
            "beta": 0.0,
            "num_generations": 2,
            "temperature": 0.8,
            "max_completion_length": 13312,
            "use_vllm": False,
        },
        "diagnostic": {
            "expected_tasks": 1,
            "expected_rollouts_per_task": 4,
            "expected_rollouts": 4,
            "groups_per_task": 2,
            "group_size": 2,
            "strict_opening_manifest_binding": True,
        },
        "data": {
            "task_ids": ["95"],
            "max_tasks": 1,
            "train_subset": "development_audit",
        },
        "rollout": {"stage": "FULL_TASK"},
        "sampling": {
            "mode": "STOCHASTIC_GROUP_SAMPLING",
            "do_sample": True,
            "temperature": 0.8,
            "top_p": 1.0,
            "top_k": 0,
        },
    }
    contract = validate_sampling_request(config, 13312, 2)
    assert contract["mode"] == "STOCHASTIC_GROUP_SAMPLING"
    assert contract["groups_per_task"] == 2


def test_explicit_stochastic_sampling_accepts_strict_rl_train_scope_only():
    config = {
        "execution_mode": "ROLLOUT_DIAGNOSTIC",
        "claims": {"rl_train_task_only": True},
        "grpo": {
            "learning_rate": 0.0,
            "beta": 0.0,
            "num_generations": 4,
            "temperature": 0.8,
            "max_completion_length": 4096,
            "use_vllm": False,
        },
        "diagnostic": {
            "expected_tasks": 1,
            "expected_rollouts_per_task": 4,
            "expected_rollouts": 4,
            "groups_per_task": 1,
            "group_size": 4,
            "strict_opening_manifest_binding": True,
        },
        "data": {
            "task_ids": ["44"],
            "max_tasks": 1,
            "train_subset": "rl_train",
        },
        "rollout": {"stage": "FULL_TASK"},
        "sampling": {
            "mode": "STOCHASTIC_GROUP_SAMPLING",
            "contract_version": "fixed-n4-single-group-v1",
            "do_sample": True,
            "temperature": 0.8,
            "top_p": 1.0,
            "top_k": 0,
        },
    }

    contract = validate_sampling_request(config, 4096, 1)
    assert contract["actual_num_generations"] == 4
    assert contract["groups_per_task"] == 1

    wrong_group_size = deepcopy(config)
    wrong_group_size["grpo"]["num_generations"] = 6
    wrong_group_size["diagnostic"]["expected_rollouts_per_task"] = 6
    wrong_group_size["diagnostic"]["expected_rollouts"] = 6
    wrong_group_size["diagnostic"]["group_size"] = 6
    with pytest.raises(ValueError, match="exactly four"):
        validate_sampling_request(wrong_group_size, 4096, 1)

    missing_claim = deepcopy(config)
    missing_claim["claims"] = {}
    with pytest.raises(ValueError, match="rl_train_task_only"):
        validate_sampling_request(missing_claim, 4096, 1)

    validation_scope = deepcopy(config)
    validation_scope["data"]["train_subset"] = "rl_validation"
    with pytest.raises(ValueError, match="restricted"):
        validate_sampling_request(validation_scope, 4096, 1)


def test_true_greedy_requires_explicit_decode_contract_not_temperature():
    config = {
        "execution_mode": "ROLLOUT_DIAGNOSTIC",
        "grpo": {
            "learning_rate": 0.0,
            "beta": 0.0,
            "num_generations": 1,
            "temperature": 1e-8,
            "max_completion_length": 8192,
            "use_vllm": False,
        },
        "data": {"task_ids": ["43", "72"]},
        "diagnostic": {
            "expected_tasks": 2,
            "expected_rollouts_per_task": 1,
            "expected_rollouts": 2,
        },
        "rollout": {"stage": "FULL_TASK"},
    }
    with pytest.raises(ValueError, match="stochastic"):
        validate_sampling_request(config, 8192, 1)
    config["sampling"] = {"mode": "TRUE_GREEDY", "do_sample": True}
    with pytest.raises(ValueError, match="do_sample=false"):
        validate_sampling_request(config, 8192, 1)
    config["sampling"]["do_sample"] = False
    contract = validate_sampling_request(config, 8192, 1)
    assert contract["mode"] == "TRUE_GREEDY"
    assert contract["actual_num_generations"] == 1
    assert contract["trl_constructor_num_generations"] == 2
    assert contract["trl_constructor_steps_per_generation"] == 2
    with pytest.raises(ValueError, match="frozen config"):
        validate_sampling_request(config, 1, 1)


def test_explicit_stochastic_sampling_requires_exact_s7_contract():
    config = {
        "execution_mode": "ROLLOUT_DIAGNOSTIC",
        "grpo": {
            "learning_rate": 0.0,
            "beta": 0.0,
            "num_generations": 2,
            "temperature": 0.8,
            "max_completion_length": 13312,
            "use_vllm": False,
        },
        "data": {"task_ids": ["43", "72"]},
        "diagnostic": {
            "expected_tasks": 2,
            "expected_rollouts_per_task": 4,
            "expected_rollouts": 8,
            "groups_per_task": 2,
            "group_size": 2,
        },
        "rollout": {"stage": "FULL_TASK"},
        "sampling": {
            "mode": "STOCHASTIC_GROUP_SAMPLING",
            "do_sample": True,
            "temperature": 0.8,
            "top_p": 1.0,
            "top_k": 0,
        },
    }
    contract = validate_sampling_request(config, 13312, 2)
    assert contract == {
        **config["sampling"],
        "actual_num_generations": 2,
        "trl_constructor_num_generations": 2,
        "groups_per_task": 2,
        "trainer_max_steps_unused": True,
    }

    mutations = (
        ("sampling", "temperature", 0.7, "exact S7"),
        ("sampling", "top_p", 0.9, "exact S7"),
        ("sampling", "top_k", 10, "exact S7"),
        ("grpo", "temperature", 0.7, "temperature"),
        (
            "grpo",
            "num_generations",
            4,
            "expected_rollouts_per_task|two candidates",
        ),
    )
    for section, key, value, message in mutations:
        changed = deepcopy(config)
        changed[section][key] = value
        with pytest.raises(ValueError, match=message):
            validate_sampling_request(changed, 13312, 2)
    with pytest.raises(ValueError, match="frozen config"):
        validate_sampling_request(config, 8192, 2)
    with pytest.raises(
        ValueError,
        match="expected_rollouts_per_task|two groups",
    ):
        validate_sampling_request(config, 13312, 1)


def test_sampling_runtime_binds_true_greedy_and_rejects_decode_drift():
    contract = {
        "mode": "TRUE_GREEDY",
        "do_sample": False,
        "actual_num_generations": 1,
        "trl_constructor_num_generations": 2,
        "groups_per_task": 1,
        "trainer_max_steps_unused": True,
    }
    trainer = SimpleNamespace(
        args=SimpleNamespace(num_generations=2),
        generation_config=SimpleNamespace(do_sample=False),
        generation_kwargs={
            "do_sample": False,
            "num_beams": 1,
            "num_return_sequences": 1,
        },
        num_generations=2,
    )
    runtime = bind_sampling_runtime(trainer, contract)
    assert runtime["effective_do_sample"] is False
    assert runtime["effective_num_generations"] == trainer.num_generations == 1
    assert trainer.args.num_generations == 1
    drift = SimpleNamespace(
        generation_config=SimpleNamespace(do_sample=True),
        generation_kwargs={
            "do_sample": False,
            "num_beams": 1,
            "num_return_sequences": 1,
        },
        num_generations=2,
    )
    with pytest.raises(RuntimeError, match="GenerationConfig"):
        bind_sampling_runtime(drift, contract)


def test_sampling_runtime_binds_explicit_stochastic_decode_contract():
    contract = {
        "mode": "STOCHASTIC_GROUP_SAMPLING",
        "do_sample": True,
        "temperature": 0.8,
        "top_p": 1.0,
        "top_k": 0,
        "actual_num_generations": 2,
        "trl_constructor_num_generations": 2,
        "groups_per_task": 2,
        "trainer_max_steps_unused": True,
    }

    def trainer():
        return SimpleNamespace(
            args=SimpleNamespace(num_generations=2),
            generation_config=SimpleNamespace(
                do_sample=True, temperature=0.8, top_p=1.0, top_k=0
            ),
            generation_kwargs={
                "do_sample": True,
                "temperature": 0.8,
                "top_p": 1.0,
                "top_k": 0,
            },
            num_generations=2,
        )

    runtime = bind_sampling_runtime(trainer(), contract)
    assert runtime["effective_do_sample"] is True
    assert runtime["effective_num_generations"] == 2
    assert runtime["effective_temperature"] == 0.8
    assert runtime["effective_top_p"] == 1.0
    assert runtime["effective_top_k"] == 0

    for owner, name, value, message in (
        ("generation_config", "temperature", 0.7, "GenerationConfig temperature"),
        ("generation_config", "top_p", 0.9, "GenerationConfig top_p"),
        ("generation_config", "top_k", 10, "GenerationConfig top_k"),
        ("generation_kwargs", "temperature", 0.7, "generation kwargs temperature"),
        ("generation_kwargs", "top_p", 0.9, "generation kwargs top_p"),
        ("generation_kwargs", "top_k", 10, "generation kwargs top_k"),
    ):
        changed = trainer()
        if owner == "generation_config":
            setattr(changed.generation_config, name, value)
        else:
            changed.generation_kwargs[name] = value
        with pytest.raises(RuntimeError, match=message):
            bind_sampling_runtime(changed, contract)

    for owner in ("trainer", "args"):
        changed = trainer()
        if owner == "trainer":
            changed.num_generations = 4
        else:
            changed.args.num_generations = 4
        with pytest.raises(RuntimeError, match="generation count"):
            bind_sampling_runtime(changed, contract)


def test_runtime_source_hash_mismatch_fails_closed():
    with patch.dict("sys.modules", {"trl": SimpleNamespace(__version__="1.9.0")}):
        with pytest.raises(RuntimeError, match="source hash"):
            verify_trl_source(BudgetTrace)


@pytest.mark.parametrize("generation_source_matches", [True, False])
def test_generation_source_is_also_hash_bound(
    tmp_path, monkeypatch, generation_source_matches
):
    import hashlib
    from src.training import rollout_diagnostics as module

    trainer_path, generation_path = tmp_path / "trainer.py", tmp_path / "generation.py"
    trainer_path.write_text("# CPU fixture, not TRL\n", encoding="utf-8")
    generation_path.write_text("# CPU fixture, not Transformers\n", encoding="utf-8")
    trainer_type, generation_type = (
        type("Trainer", (), {}),
        type("GenerationMixin", (), {}),
    )
    monkeypatch.setattr(
        module.inspect,
        "getfile",
        lambda cls: str(trainer_path if cls is trainer_type else generation_path),
    )
    monkeypatch.setitem(
        sys.modules, "trl", SimpleNamespace(__version__=module.TRL_VERSION)
    )
    monkeypatch.setitem(
        sys.modules,
        "transformers",
        SimpleNamespace(__version__=module.TRANSFORMERS_VERSION),
    )
    monkeypatch.setitem(
        sys.modules,
        "transformers.generation.utils",
        SimpleNamespace(GenerationMixin=generation_type),
    )
    monkeypatch.setattr(
        module,
        "TRL_SOURCE_SHA256",
        hashlib.sha256(trainer_path.read_bytes()).hexdigest().upper(),
    )
    digest = hashlib.sha256(generation_path.read_bytes()).hexdigest().upper()
    monkeypatch.setattr(
        module,
        "TRANSFORMERS_SOURCE_SHA256",
        digest if generation_source_matches else "0" * 64,
    )
    if generation_source_matches:
        assert verify_trl_source(trainer_type)["transformers_source_sha256"] == digest
    else:
        with pytest.raises(RuntimeError, match="transformers source hash"):
            verify_trl_source(trainer_type)


class FakeTrainer:
    """Boundary test double, not evidence of GPU or real-TRL integration."""

    def __init__(self):
        import torch

        class Environment:
            def __init__(self):
                self._user_stopped = True
                self._runtime_stop_signals = []
                self.completion_telemetry = None
                self.reward_calls = 0

            def _set_trainer_completion_telemetry(self, payload):
                assert self.completion_telemetry is None
                self.completion_telemetry = deepcopy(payload)

            def get_reward(self):
                self.reward_calls += 1
                assert self.completion_telemetry is not None
                return 1.0

        class CoreModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = torch.nn.Parameter(torch.ones(1))
                self.config = SimpleNamespace(max_position_embeddings=100)
                self.nonfinite_value = None
                self.forward_calls = 0
                self.last_result = None

            def _get_logits_processor(self, generation_config, input_ids_seq_length):
                # This raw-forward boundary double never enters native sampling.
                # Actual processor installation is covered by the native-loop suite.
                return []

            def forward(self, input_ids, **kwargs):
                self.forward_calls += 1
                logits = torch.arange(3.0).reshape(1, 1, 3).repeat(len(input_ids), 1, 1)
                if self.nonfinite_value is not None:
                    logits[0, -1, 1] = self.nonfinite_value
                self.last_result = SimpleNamespace(logits=logits)
                return self.last_result

        class PeftStyleWrapper(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.core = CoreModel()
                self.config = self.core.config

            def get_base_model(self):
                return self.core

            def forward(self, *args, **kwargs):
                raise AssertionError(
                    "Generation must use core.forward, not the wrapper"
                )

        self.model = PeftStyleWrapper()
        self.accelerator = SimpleNamespace(num_processes=1)
        self.optimizer = self.lr_scheduler = None
        self.use_vllm = self._is_vlm = self.use_transformers_continuous_batching = False
        self.num_generations = 2
        self.generation_config = SimpleNamespace(do_sample=True)
        self.generation_kwargs = {"do_sample": True}
        self.max_completion_length = 8
        self.max_tool_calling_iterations = 40
        self._tokenizer = SimpleNamespace(
            eos_token_id=99, decode=lambda ids, **kw: str(ids)
        )
        self.environments = [Environment(), Environment()]
        self.scoring_reached = False
        self.native_generation_calls = 0
        self.sampling_results = []
        self.sampling_draws = []
        self.forward_kwargs = {}

    def _generate_and_score_completions(self, inputs):
        self._generate([row["prompt"] for row in inputs])
        self.scoring_reached = True
        raise AssertionError("Must not reach scoring")

    def _generate(self, prompts):
        ids, _ = self._generate_single_turn([[1, 2]] * 2, None, {})
        return (
            [[1, 2]] * 2,
            ids,
            [[1, 1]] * 2,
            [[{"role": "assistant", "content": "done"}]] * 2,
        )

    def _generate_single_turn(self, prompts, images, fields):
        import torch

        assert not torch.is_grad_enabled()
        assert not self.model.training
        self.native_generation_calls += 1
        core = self.model.get_base_model()
        processors = core._get_logits_processor(
            generation_config=SimpleNamespace(
                num_beams=1,
                num_return_sequences=1,
                remove_invalid_values=False,
                _eos_token_tensor=torch.tensor([99]),
                _pad_token_tensor=torch.tensor(0),
            ),
            input_ids_seq_length=max(map(len, prompts)),
        )
        result = core(torch.tensor(prompts), **self.forward_kwargs)
        # This fixture models only the raw-forward boundary; the separate frozen
        # native-loop tests execute the actual terminal processor and multinomial.
        processors[-1].calls += 1
        # This represents the sampler boundary: a failing forward hook must
        # prevent both sampling and its RNG consumption.
        self.sampling_results.append(result)
        self.sampling_draws.append(torch.rand(()).item())
        return [[10 + i, 99] for i in range(len(prompts))], None


def test_pure_sampling_exits_before_scoring_and_forbids_optimizer():
    import torch

    trainer = make_sampling_trainer(FakeTrainer)()
    before = [parameter.detach().clone() for parameter in trainer.model.parameters()]
    with (
        patch.object(
            torch.Tensor,
            "backward",
            side_effect=AssertionError("pure sampling called backward"),
        ),
        patch.object(
            torch.optim.Optimizer,
            "step",
            side_effect=AssertionError("pure sampling stepped an optimizer"),
        ),
    ):
        _, report = trainer.sample_group(
            [{"task_id": "43", "prompt": []}] * 2, lambda e: None
        )
    assert len(report) == 2
    assert not trainer.scoring_reached
    assert [environment.get_reward() for environment in trainer.environments] == [
        1.0,
        1.0,
    ]
    assert all(environment.reward_calls == 1 for environment in trainer.environments)
    assert all(
        environment.completion_telemetry["stop_reason_source"]
        == "guarded_grpo_trainer_v1"
        for environment in trainer.environments
    )
    assert all(
        environment.completion_telemetry["stop_reason"] == "USER_STOP_AND_MODEL_EOS"
        for environment in trainer.environments
    )
    assert all(
        p.grad is None and not p.requires_grad for p in trainer.model.parameters()
    )
    assert all(
        torch.equal(parameter, original)
        for parameter, original in zip(trainer.model.parameters(), before, strict=True)
    )
    for operation in (trainer.train, trainer.create_optimizer, trainer.compute_loss):
        with pytest.raises(RuntimeError, match="forbids"):
            operation()


def test_sampling_telemetry_binding_failure_prevents_observational_reward():
    trainer = make_sampling_trainer(FakeTrainer)()

    def reject_telemetry(payload):
        del payload
        raise RuntimeError("telemetry binding rejected")

    trainer.environments[1]._set_trainer_completion_telemetry = reject_telemetry

    def sample_then_reward():
        trainer.sample_group([{"task_id": "43", "prompt": []}] * 2, lambda event: None)
        return [environment.get_reward() for environment in trainer.environments]

    with pytest.raises(RuntimeError, match="telemetry binding rejected"):
        sample_then_reward()
    assert all(environment.reward_calls == 0 for environment in trainer.environments)
    assert not trainer.scoring_reached


def test_same_task_different_resets_cannot_share_group():
    trainer = make_sampling_trainer(FakeTrainer)()
    with pytest.raises(ValueError, match="exact reset"):
        trainer.sample_group(
            [{"task_id": "43", "user_seed": 1}, {"task_id": "43", "user_seed": 2}],
            lambda e: None,
        )


def test_raw_logits_guard_observes_core_without_changing_finite_results():
    import torch

    trainer = make_sampling_trainer(FakeTrainer)()
    core = trainer.model.get_base_model()
    events = []
    with (
        patch.object(
            trainer.model,
            "register_forward_hook",
            side_effect=AssertionError("The wrapper is not the generation core"),
        ),
        patch(
            "src.training.rollout_diagnostics._nonfinite_forward_context",
            side_effect=AssertionError("No failure dump on the finite path"),
        ),
    ):
        trainer.sample_group([{"task_id": "43", "prompt": []}] * 2, events.append)
    assert core.forward_calls == trainer.native_generation_calls == 1
    assert trainer.sampling_results[0] is core.last_result
    assert torch.equal(
        core.last_result.logits,
        torch.tensor([[[0.0, 1.0, 2.0]], [[0.0, 1.0, 2.0]]]),
    )
    assert not core._forward_hooks
    assert not core._forward_pre_hooks
    assert not any(e["event"] == "nonfinite_raw_logits" for e in events)
    assert not any("failure_context" in e for e in events)


@pytest.mark.parametrize("bad_value", [float("nan"), float("inf"), -float("inf")])
def test_nonfinite_raw_logits_stop_before_sampling_and_remove_hook(bad_value):
    import torch

    trainer = make_sampling_trainer(FakeTrainer)()
    core = trainer.model.get_base_model()
    core.nonfinite_value = bad_value
    rng_before = torch.get_rng_state().clone()
    events = []
    with pytest.raises(FloatingPointError, match="NONFINITE_RAW_MODEL_LOGITS"):
        trainer.sample_group([{"task_id": "43", "prompt": []}] * 2, events.append)
    assert core.forward_calls == trainer.native_generation_calls == 1
    assert trainer.sampling_results == trainer.sampling_draws == []
    assert not trainer.scoring_reached
    assert not core._forward_hooks
    assert not core._forward_pre_hooks
    assert torch.equal(torch.get_rng_state(), rng_before)
    assert [e["event"] for e in events] == [
        "generation_start",
        "nonfinite_raw_logits",
        "generation_error",
    ]
    assert events[1]["rows"] == [0, 1]
    assert events[1]["generation_index"] == events[1]["forward_index"] == 0
    assert events[1]["logits_shape"] == [2, 1, 3]
    assert events[1]["finite_counts"] == [2, 3]
    assert events[2]["observed_forward_count"] == 1
    assert base64.b64decode(events[0]["rng_before"]["cpu"]) == bytes(
        rng_before.tolist()
    )
    assert all(p.grad is None for p in trainer.model.parameters())


@pytest.mark.parametrize("legacy", [False, True])
def test_failure_captures_actual_forward_kwargs_and_cache_metadata_only(legacy):
    import torch

    class MetadataOnlyKV(torch.Tensor):
        def cpu(self, *args, **kwargs):
            raise AssertionError("KV values must never be copied")

        def tolist(self):
            raise AssertionError("KV values must never be dumped")

    keys = torch.zeros(2, 2, 7, 4).as_subclass(MetadataOnlyKV)
    values = torch.ones(2, 2, 7, 4).as_subclass(MetadataOnlyKV)
    cache = (
        ((keys, values),)
        if legacy
        else SimpleNamespace(
            layers=[SimpleNamespace(keys=keys, values=values)], get_seq_length=lambda: 7
        )
    )
    trainer = make_sampling_trainer(FakeTrainer)()
    core = trainer.model.get_base_model()
    core.nonfinite_value = float("nan")
    trainer.forward_kwargs = {
        "attention_mask": torch.tensor([[0, 1], [1, 1]]),
        "position_ids": torch.tensor([[0, 0], [0, 1]]),
        "cache_position": torch.tensor([7, 8]),
        "past_key_values": cache,
    }
    events = []
    with (
        patch.object(
            torch.cuda, "get_rng_state", side_effect=AssertionError("CUDA read")
        ),
        patch.object(
            torch.cuda, "get_rng_state_all", side_effect=AssertionError("CUDA read")
        ),
        pytest.raises(FloatingPointError, match="NONFINITE_RAW_MODEL_LOGITS"),
    ):
        trainer.sample_group([{"task_id": "43", "prompt": []}] * 2, events.append)
    failure = next(e for e in events if e["event"] == "nonfinite_raw_logits")
    context = failure["failure_context"]
    assert context["forward_inputs"]["input_ids"]["values"] == [[1, 2], [1, 2]]
    for name in ("attention_mask", "position_ids", "cache_position"):
        assert (
            context["forward_inputs"][name]["values"]
            == trainer.forward_kwargs[name].tolist()
        )
        assert context["forward_inputs"][name]["device"] == "cpu"
    cached = context["past_key_values"]
    assert not cached["tensor_values_saved"]
    assert cached["observation_time"] == "after_forward_before_sampling"
    assert cached["layers"][0]["key"] == {
        "status": "metadata_only",
        "shape": [2, 2, 7, 4],
        "dtype": "torch.float32",
        "device": "cpu",
    }
    assert cached["layers"][0]["value"]["shape"] == [2, 2, 7, 4]
    assert cached["layers"][0]["stored_sequence_dimension"] == 7
    if not legacy:
        assert cached["sequence_length"] == 7
    assert context["generation_state"]["status"] == "unavailable"
    assert not core._forward_hooks and not core._forward_pre_hooks
    assert trainer.forward_kwargs["past_key_values"] is cache


def test_dense_mask_is_metadata_only_and_bad_cache_getter_does_not_lose_inputs():
    import torch

    def fail_length():
        raise RuntimeError("PRIVATE error payload must not be copied")

    trainer = make_sampling_trainer(FakeTrainer)()
    trainer.model.get_base_model().nonfinite_value = float("inf")
    trainer.forward_kwargs = {
        "attention_mask": torch.zeros(2, 1, 4, 4),
        "past_key_values": SimpleNamespace(layers=[], get_seq_length=fail_length),
    }
    events = []
    with pytest.raises(FloatingPointError, match="NONFINITE_RAW_MODEL_LOGITS"):
        trainer.sample_group([{"task_id": "43", "prompt": []}] * 2, events.append)
    context = events[1]["failure_context"]
    mask = context["forward_inputs"]["attention_mask"]
    assert mask["status"] == "metadata_only" and "values" not in mask
    assert mask["shape"] == [2, 1, 4, 4]
    assert context["forward_inputs"]["input_ids"]["status"] == "available"
    assert context["forward_inputs"]["position_ids"]["status"] == "unavailable"
    assert context["past_key_values"]["sequence_length"] == {
        "status": "unavailable",
        "exception_type": "RuntimeError",
    }
    assert "PRIVATE" not in str(events)


@pytest.fixture
def native_sample_frame(monkeypatch):
    # A registered, code-bound CPU stack fixture, not a real Transformers run.
    module = ModuleType("transformers.generation.utils")
    module.__file__ = "D:/PolicyAgent-PostTrain/tests/_native_sample_stack_fixture.py"
    source = (
        "class GenerationMixin:\n"
        "    def _sample(self, input_ids, unfinished_sequences, forward_kwargs):\n"
        "        secret = 'PRIVATE LOCAL MUST NOT BE DUMPED'\n"
        "        return self(input_ids=input_ids[:, -1:], **forward_kwargs)\n"
    )
    exec(compile(source, module.__file__, "exec"), module.__dict__)
    monkeypatch.setitem(sys.modules, module.__name__, module)
    return module


@pytest.mark.parametrize("invalid_binding", [None, "filename", "method_code"])
def test_failure_stack_prefix_is_code_bound_and_finished_rows_remain_visible(
    native_sample_frame, invalid_binding
):
    import torch

    native_sample = native_sample_frame.GenerationMixin._sample
    if invalid_binding == "filename":
        native_sample_frame.__file__ += ".unverified"
    elif invalid_binding == "method_code":
        native_sample_frame.GenerationMixin._sample = lambda *args: None

    class PrefixTrainer(FakeTrainer):
        def _generate_single_turn(self, prompts, images, fields):
            prefix = torch.tensor([row + [44, 45] for row in prompts])
            return native_sample(
                self.model.get_base_model(), prefix, torch.tensor([0, 1]), {}
            )

    trainer = make_sampling_trainer(PrefixTrainer)()
    core = trainer.model.get_base_model()
    core.nonfinite_value = float("nan")
    events = []
    with pytest.raises(FloatingPointError, match="NONFINITE_RAW_MODEL_LOGITS"):
        trainer.sample_group([{"task_id": "43", "prompt": []}] * 2, events.append)
    context = events[1]["failure_context"]
    assert context["forward_inputs"]["input_ids"]["values"] == [[45], [45]]
    state = context["generation_state"]
    if invalid_binding:
        assert state == {
            "status": "unavailable",
            "reason": "no_verified_native_sample_frame",
        }
    else:
        assert state["status"] == "verified_frame"
        assert state["rows"] == [0, 1]
        assert state["full_prefix"]["values"] == [[1, 2, 44, 45], [1, 2, 44, 45]]
        assert state["unfinished_sequences"]["values"] == [0, 1]
        assert state["source"]["function"] == "_sample"
    assert "PRIVATE" not in str(events)
    assert not core._forward_hooks and not core._forward_pre_hooks


def test_failure_stack_binds_single_continuing_candidate_to_original_row(
    native_sample_frame,
):
    import torch

    class ContinueOneTrainer(FakeTrainer):
        def _generate(self, prompts):
            initial = [[1, 2], [1, 2]]
            first, _ = self._generate_single_turn(initial, None, {})
            self._budget_trace.tool_called(1)
            self._budget_trace.suffix([30], [{"role": "tool", "content": "observed"}])
            self.model.get_base_model().nonfinite_value = float("nan")
            self._generate_single_turn([initial[1] + first[1] + [30]], None, {})

        def _generate_single_turn(self, prompts, images, fields):
            if self.model.get_base_model().nonfinite_value is None:
                return super()._generate_single_turn(prompts, images, fields)
            prefix = torch.tensor([prompts[0] + [55]])
            return native_sample_frame.GenerationMixin._sample(
                self.model.get_base_model(), prefix, torch.tensor([1]), {}
            )

    trainer = make_sampling_trainer(ContinueOneTrainer)()
    events = []
    with pytest.raises(FloatingPointError, match="NONFINITE_RAW_MODEL_LOGITS"):
        trainer.sample_group([{"task_id": "43", "prompt": []}] * 2, events.append)
    failure = next(e for e in events if e["event"] == "nonfinite_raw_logits")
    assert failure["rows"] == [1]
    state = failure["failure_context"]["generation_state"]
    assert state["rows"] == [1]
    assert state["full_prefix"]["values"] == [[1, 2, 11, 99, 30, 55]]
    assert state["unfinished_sequences"]["values"] == [1]
    assert len(trainer.sampling_draws) == 1


@pytest.mark.parametrize("fail_logging", [False, True])
def test_failure_evidence_errors_never_replace_numerical_exception(fail_logging):
    trainer = make_sampling_trainer(FakeTrainer)()
    core = trainer.model.get_base_model()
    core.nonfinite_value = float("nan")
    events = []

    def emit(event):
        if fail_logging and event["event"] != "generation_start":
            raise OSError("PRIVATE disk failure payload")
        events.append(event)

    with (
        patch(
            "src.training.rollout_diagnostics._nonfinite_forward_context",
            side_effect=RuntimeError("PRIVATE payload"),
        ),
        pytest.raises(
            FloatingPointError, match="^NONFINITE_RAW_MODEL_LOGITS"
        ) as caught,
    ):
        trainer.sample_group([{"task_id": "43", "prompt": []}] * 2, emit)
    assert caught.value.args == ("NONFINITE_RAW_MODEL_LOGITS",)
    assert not core._forward_hooks and not core._forward_pre_hooks
    assert trainer.sampling_draws == []
    if fail_logging:
        assert (
            len(caught.value.__notes__) == 3
        )  # unverified native state + two log failures
        assert "PRIVATE" not in str(caught.value.__notes__)
    else:
        assert events[1]["failure_context"] == {
            "status": "unavailable",
            "exception_type": "RuntimeError",
        }
    assert "PRIVATE" not in str(events)


def test_native_generation_without_forward_is_rejected_and_hook_removed():
    class BypassForwardTrainer(FakeTrainer):
        def _generate_single_turn(self, prompts, images, fields):
            self.native_generation_calls += 1
            return [[10 + i, 99] for i in range(len(prompts))], None

    trainer = make_sampling_trainer(BypassForwardTrainer)()
    events = []
    with pytest.raises(RuntimeError, match="guard did not observe model forward"):
        trainer.sample_group([{"task_id": "43", "prompt": []}] * 2, events.append)
    assert trainer.native_generation_calls == 1
    assert trainer.model.get_base_model().forward_calls == 0
    assert not trainer.model.get_base_model()._forward_hooks
    assert events[-1]["event"] == "generation_error"
    assert events[-1]["observed_forward_count"] == 0
    assert not trainer.scoring_reached


def test_generation_with_forward_but_without_terminal_processor_is_rejected():
    import torch

    class BypassProcessorTrainer(FakeTrainer):
        def _generate_single_turn(self, prompts, images, fields):
            self.model.get_base_model()(torch.tensor(prompts))
            return [[10, 99] for p in prompts], None

    trainer = make_sampling_trainer(BypassProcessorTrainer)()
    core = trainer.model.get_base_model()
    events = []
    with pytest.raises(RuntimeError, match="processor was not installed"):
        trainer.sample_group([{"task_id": "43", "prompt": []}] * 2, events.append)
    assert not core._forward_hooks and not core._forward_pre_hooks
    assert "_get_logits_processor" not in core.__dict__
    assert "_policyagent_finished_row_owner" not in core.__dict__
    assert events[-1]["event"] == "generation_error"


def test_cpu_generation_records_entry_rng_without_cuda_access_or_rng_reset():
    import torch

    trainer = make_sampling_trainer(FakeTrainer)()
    rng_before = torch.get_rng_state().clone()
    events = []
    with (
        patch.object(torch.cuda, "is_available", return_value=True),
        patch.object(
            torch.cuda, "get_rng_state", side_effect=AssertionError("CUDA read")
        ),
        patch.object(
            torch.cuda, "get_rng_state_all", side_effect=AssertionError("CUDA read")
        ),
        patch.object(torch, "set_rng_state", side_effect=AssertionError("RNG reset")),
        patch.object(torch, "manual_seed", side_effect=AssertionError("RNG reset")),
        patch.object(
            torch.cuda, "set_rng_state", side_effect=AssertionError("CUDA reset")
        ),
        patch.object(
            torch.cuda, "set_rng_state_all", side_effect=AssertionError("CUDA reset")
        ),
    ):
        trainer.sample_group([{"task_id": "43", "prompt": []}] * 2, events.append)
    start = events[0]
    assert start["event"] == "generation_start"
    assert start["generation_index"] == 0
    assert start["rows"] == [0, 1]
    assert start["input_ids"] == [[1, 2], [1, 2]]
    assert start["raw_logits_guard"] is True
    assert start["rng_before"]["encoding"] == "base64_uint8"
    assert start["rng_before"]["cuda"] == {}
    assert base64.b64decode(start["rng_before"]["cpu"]) == bytes(rng_before.tolist())
    assert not torch.equal(torch.get_rng_state(), rng_before)


@pytest.mark.parametrize("fail_continuation", [False, True])
def test_second_candidate_continuation_keeps_original_row_and_fresh_rng(
    fail_continuation,
):
    import torch

    class ContinueCandidateOneTrainer(FakeTrainer):
        def _generate(self, prompts):
            initial = [[1, 2], [1, 2]]
            first, _ = self._generate_single_turn(initial, None, {})
            self._budget_trace.tool_called(1)
            self._budget_trace.suffix([30], [{"role": "tool", "content": "observed"}])
            continuation = initial[1] + first[1] + [30]
            self.rng_before_continuation = bytes(torch.get_rng_state().tolist())
            if fail_continuation:
                self.model.get_base_model().nonfinite_value = float("nan")
            second, _ = self._generate_single_turn([continuation], None, {})
            return (
                initial,
                [first[0], first[1] + [30] + second[0]],
                [[1, 1], [1, 1, 0, 1, 1]],
                [[{"role": "assistant", "content": "done"}]] * 2,
            )

    trainer = make_sampling_trainer(ContinueCandidateOneTrainer)()
    events = []
    if fail_continuation:
        with pytest.raises(FloatingPointError, match="NONFINITE_RAW_MODEL_LOGITS"):
            trainer.sample_group([{"task_id": "43", "prompt": []}] * 2, events.append)
        failure = next(e for e in events if e["event"] == "nonfinite_raw_logits")
        assert failure["rows"] == [1]
        assert failure["generation_index"] == 1
        assert failure["logits_shape"] == [1, 1, 3]
        assert len(trainer.sampling_draws) == 1
    else:
        _, report = trainer.sample_group(
            [{"task_id": "43", "prompt": []}] * 2, events.append
        )
        assert [row["completion_tokens"] for row in report] == [2, 5]
        assert len(trainer.sampling_draws) == 2
    starts = [e for e in events if e["event"] == "generation_start"]
    assert len(starts) == 2
    assert starts[0]["rows"] == [0, 1]
    assert starts[1]["rows"] == [1]
    assert starts[1]["input_ids"] == [[1, 2, 11, 99, 30]]
    assert [e["generation_index"] for e in starts] == [0, 1]
    assert (
        base64.b64decode(starts[1]["rng_before"]["cpu"])
        == trainer.rng_before_continuation
    )
    assert starts[0]["rng_before"]["cpu"] != starts[1]["rng_before"]["cpu"]
    assert trainer.native_generation_calls == 2
    assert not trainer.model.get_base_model()._forward_hooks
    assert not trainer.scoring_reached


def native_trainer(*, suffix_length=2, unknown=False, fail_api=False):
    # Optional CPU contract suite: the test process injects ONLY the verified
    # upstream _tool_call_loop AST, never installs/runs an unaudited TRL package.
    native = pytest.importorskip("_policyagent_audited_trl_loop")

    class LoopTrainer(FakeTrainer):
        _tool_call_loop = native.loop

        def __init__(self):
            super().__init__()
            self.model.config.max_position_embeddings = 100
            self._async_tool_dicts = [{}, {}]
            self.writes = [0, 0]
            self.environments = [
                SimpleNamespace(
                    _user_stopped=True,
                    _runtime_stop_signals=[],
                    _customer_turns=1,
                    _max_customer_turns=8,
                    _tool_counter=0,
                    _max_tool_calls=32,
                    completion_telemetry=None,
                )
                for _ in range(2)
            ]
            for environment in self.environments:
                environment._set_trainer_completion_telemetry = (
                    lambda payload, environment=environment: setattr(
                        environment, "completion_telemetry", deepcopy(payload)
                    )
                )

            def write(i):
                def call():
                    if fail_api:
                        from src.rl.user_simulator_fail_fast import (
                            UserSimulatorSystemFailure,
                        )

                        raise UserSimulatorSystemFailure(
                            category="TEST", message="test", attempts=1, abort_run=True
                        )
                    self.writes[i] += 1
                    return "executed"

                return call

            self._sync_tool_dicts = [{"write": write(i)} for i in range(2)]

        def _get_tool_suffix_ids(self, messages):
            return [20] * suffix_length

        def _generate(self, prompts):
            initial_ids = [[1, 2]] * 2
            ids, _ = self._generate_single_turn(initial_ids, None, {})
            completions = [
                [
                    {
                        "role": "assistant",
                        "tool_calls": [
                            {
                                "type": "function",
                                "function": {
                                    "name": "unknown" if unknown else "write",
                                    "arguments": {},
                                },
                            }
                        ],
                    }
                ]
                for _ in range(2)
            ]
            mask, messages, ids, logps, _, _, images = self._tool_call_loop(
                deepcopy(prompts), initial_ids, ids, completions, None, None, {}
            )
            return initial_ids, ids, mask, messages, logps, {}, None, images

    return make_sampling_trainer(LoopTrainer)()


def test_native_loop_preserves_write_even_when_tool_result_is_rolled_back():
    trainer = native_trainer(suffix_length=7)
    events = []
    _, report = trainer.sample_group(
        [{"task_id": "43", "prompt": []}] * 2, events.append
    )
    assert trainer.writes == [1, 1]
    assert all(row["stop_reason"] == "TOOL_RESULT_BUDGET_EXCEEDED" for row in report)
    suffixes = [e for e in events if e["event"] == "tool_suffix_observed"]
    assert len(suffixes) == 2
    assert all(e["suffix_tokens"] == 7 and e["would_rollback"] for e in suffixes)
    assert not trainer.scoring_reached


def test_native_loop_can_continue_after_observation_without_backward():
    trainer = native_trainer()
    _, report = trainer.sample_group(
        [{"task_id": "43", "prompt": []}] * 2, lambda e: None
    )
    assert trainer.writes == [1, 1]
    assert all(row["stop_reason"] == "USER_STOP_AND_MODEL_EOS" for row in report)
    assert all(row["completion_tokens"] == 6 for row in report)
    assert all(row["observation_tokens_retained"] == 2 for row in report)


def test_native_loop_unknown_tool_is_bound_and_not_silently_successful():
    trainer = native_trainer(unknown=True)
    events = []
    _, report = trainer.sample_group(
        [{"task_id": "43", "prompt": []}] * 2, events.append
    )
    assert trainer.writes == [0, 0]
    assert all("TOOL_EXCEPTION" not in row["stop_flags"] for row in report)
    errors = [event for event in events if event["event"] == "tool_error"]
    assert len(errors) == 2
    assert all(event["code"] == "TOOL_EXCEPTION" for event in errors)


def test_native_loop_api_failure_escapes_trl_exception_to_error_conversion():
    trainer = native_trainer(fail_api=True)
    from src.rl.user_simulator_fail_fast import UserSimulatorSystemFailure

    with pytest.raises(UserSimulatorSystemFailure):
        trainer.sample_group([{"task_id": "43", "prompt": []}] * 2, lambda e: None)


def test_failed_group_keeps_artifacts_and_does_not_score_or_retry(tmp_path):
    import json
    import torch
    from src.training.rollout_diagnostics import run_pure_sampling

    calls = []

    class Environment:
        _environment = object()

        def _persist_rollout(self, payload):
            calls.append(payload)

    class BrokenSampler:
        environments = [Environment(), Environment()]

        def sample_group(self, inputs, emit):
            emit(
                {
                    "event": "tool_result",
                    "row": 0,
                    "messages": [{"content": "write happened"}],
                }
            )
            raise RuntimeError("do not persist raw exception text")

    with patch.object(torch.cuda, "reset_peak_memory_stats"):
        with pytest.raises(RuntimeError):
            run_pure_sampling(
                BrokenSampler(),
                [{"task_id": "43", "user_seed": 1}],
                {
                    "grpo": {"num_generations": 2},
                    "sampling": {
                        "mode": "STOCHASTIC_GROUP_SAMPLING",
                        "do_sample": True,
                    },
                },
                {},
                tmp_path,
                1,
                {},
            )
    assert len(calls) == 2
    assert all(x["reward"] is None and not x["training_eligible"] for x in calls)
    rows = [
        json.loads(x)
        for x in (tmp_path / "generation_events.jsonl").read_text().splitlines()
    ]
    assert rows[0]["messages"][0]["content"] == "write happened"
    assert rows[-1]["event"] == "group_failed"
    assert (
        "do not persist raw exception text"
        not in (tmp_path / "generation_events.jsonl").read_text()
    )
    assert not (tmp_path / "run_manifest.json").exists()


@pytest.mark.parametrize("missing_terminal_evaluator", [False, True])
def test_sampling_manifest_requires_complete_hash_bound_evidence(
    tmp_path, missing_terminal_evaluator
):
    import hashlib
    import json
    import torch
    from src.training.rollout_diagnostics import run_pure_sampling

    completion = {
        "stop_reason": "USER_STOP_AND_MODEL_EOS",
        "stop_reason_source": "guarded_grpo_trainer_v1",
        "stop_flags": [],
        "model_ended": True,
        "model_eos_observed": True,
        "completion_token_budget_exhausted": False,
        "context_limit_reached": False,
        "tool_iteration_limit_reached": False,
        "unresolved_tool_call": False,
        "framework_loop_abnormal_end": False,
        "prompt_tokens": 2,
        "completion_tokens": 1,
        "model_tokens_retained": 1,
        "observation_tokens_retained": 0,
        "model_completion_truncated": False,
        "model_completion_truncation_source": "guarded_grpo_trainer_v1",
    }

    def append(path, row):
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    class Environment:
        def get_reward(self):
            reward_payload = {
                "reward": 0.0,
                "terminal_environment_reward": 0.0,
                "complete_success": False,
                "components": {"required_write_progress": {"value": 0.0}},
            }
            evidence = {
                "schema_version": "test-evidence-v1",
                "task_id": "43",
                "user_seed": 350291,
                "hidden_user_scenario_persisted": False,
                "initial_state": {},
                "final_state": {},
                "state_diff": [],
                "state_hashes": {},
                "tool_trace": [],
                "terminal_evaluator": {"reward": 0},
                "completion": completion,
            }
            if missing_terminal_evaluator:
                evidence.pop("terminal_evaluator")
            digest = hashlib.sha256(
                json.dumps(
                    evidence,
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                    default=str,
                ).encode("utf-8")
            ).hexdigest().upper()
            evidence["evidence_sha256"] = digest
            raw = {
                "schema_version": "test-raw-v1",
                "task_id": "43",
                "user_seed": 350291,
                "hidden_user_scenario_persisted": False,
                "messages": [
                    {"role": "user", "content": "initial"},
                    {"role": "assistant", "content": "done"},
                ],
                "tool_calls": 0,
                "reward": reward_payload,
                "completion": completion,
                "evidence_sha256": digest,
            }
            append(tmp_path / "rollout_evidence.jsonl", evidence)
            append(tmp_path / "raw_rollouts.jsonl", raw)
            return 0.0

    class Sampler:
        environments = []

        def sample_group(self, inputs, emit):
            assert len(inputs) == 1
            emit({"event": "generation_stop_observed", "row": 0})
            self.environments = [Environment()]
            diagnostics = [
                {
                    **completion,
                    "trajectory_transport_complete": True,
                    "training_eligible": False,
                }
            ]
            return (
                [[1, 2]],
                [[99]],
                [[1]],
                [[{"role": "assistant", "content": "done"}]],
            ), diagnostics

    frozen_config = {
        "model": {"expected_sha256": "B" * 64},
        "grpo": {"num_generations": 1, "temperature": 1.0},
        "sampling": {"mode": "TRUE_GREEDY", "do_sample": False},
        "data": {"task_ids": ["43"]},
        "diagnostic": {
            "expected_tasks": 1,
            "expected_rollouts": 1,
            "expected_rollouts_per_task": 1,
            "groups_per_task": 1,
            "group_size": 1,
            "trainer_max_steps_unused": True,
        },
    }
    config = {
        **frozen_config,
        "sampling": {
            **frozen_config["sampling"],
            "actual_num_generations": 1,
            "trl_constructor_num_generations": 2,
            "groups_per_task": 1,
            "trainer_max_steps_unused": True,
            "configured_temperature": 1.0,
        },
    }
    config_path = tmp_path / "frozen_config.json"
    config_path.write_text(
        json.dumps(frozen_config, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    preflight = {
        "config_sha256": hashlib.sha256(config_path.read_bytes()).hexdigest().upper(),
        "model_sha256": "B" * 64,
        "split_sha256": "C" * 64,
        "openings_sha256": "D" * 64,
        "upstream_checkout": {},
        "git_commit": "deadbeef",
        "git_dirty_at_start": True,
    }
    (tmp_path / "user_simulator_preflight.json").write_text(
        json.dumps(
            {
                "status": "PASSED",
                "model": "test-user",
                "external_api_called": True,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    sampling_runtime = {
        "sampling_adapter": {
            "direct_merged_checkpoint_inference": True,
                "decode_contract": {
                    **config["sampling"],
                    "effective_do_sample": False,
                    "effective_num_generations": 1,
                    "temperature_is_decode_authority": False,
                }
        },
        "user_simulator": {
            "model": "test-user",
            "llm_args": {"temperature": 0.0},
            "llm_args_sha256": "E" * 64,
            "preflight_status": "PASSED",
            "external_api_called": True,
        },
        "model_loading": {
            "mode": "qwen3_bf16_inference_v1",
            "quantized": False,
            "peft_adapter_applied": False,
        },
    }
    cuda_patches = (
        patch.object(torch.cuda, "reset_peak_memory_stats"),
        patch.object(torch.cuda, "max_memory_allocated", return_value=0),
        patch.object(torch.cuda, "max_memory_reserved", return_value=0),
    )
    with cuda_patches[0], cuda_patches[1], cuda_patches[2]:
        if missing_terminal_evaluator:
            with pytest.raises(RuntimeError, match="sidecar is incomplete"):
                run_pure_sampling(
                    Sampler(),
                    [
                        {
                            "task_id": "43",
                            "user_seed": 350291,
                            "prompt": [{"role": "user", "content": "initial"}],
                        }
                    ],
                    config,
                    preflight,
                    tmp_path,
                    1,
                    sampling_runtime,
                )
            assert not (tmp_path / "run_manifest.json").exists()
        else:
            manifest = run_pure_sampling(
                Sampler(),
                [
                    {
                        "task_id": "43",
                        "user_seed": 350291,
                        "prompt": [{"role": "user", "content": "initial"}],
                    }
                ],
                config,
                preflight,
                tmp_path,
                1,
                sampling_runtime,
            )
            assert manifest["status"] == "COMPLETED"
            assert manifest["rollouts"] == 1
            assert manifest["artifacts"]["raw_rollouts.jsonl"]["rows"] == 1
            assert manifest["artifacts"]["rollout_evidence.jsonl"]["rows"] == 1
            from src.evaluation.grpo_training_audit import audit_run

            audit = audit_run(tmp_path, config_path)
            assert audit["status"] == "PASSED"
            assert audit["capability"]["greedy_pass_at_1_assessed"] is True


def test_recovered_tool_exception_is_not_missing_transport_evidence():
    trace, _ = new_trace()
    trace.flag(0, "TOOL_EXCEPTION")
    output = (
        [[1, 2]] * 2,
        [[10, 99], [11, 99]],
        [[1, 1]] * 2,
        [[{"role": "assistant", "content": "done"}]] * 2,
    )
    tokenizer = SimpleNamespace(decode=lambda ids, **kw: str(ids))
    report = trace.summarize(
        output, [SimpleNamespace(_user_stopped=True)] * 2, tokenizer, 40
    )
    assert report[0]["tool_exception_observed"]
    assert report[0]["trajectory_transport_complete"]
    assert report[0]["stop_reason"] == "USER_STOP_AND_MODEL_END"


# Frozen Transformers 5.14.1 CPU contract tests. The three generation methods
# below are executed from their unmodified, hash-verified upstream AST. Model
# forward, KV storage and stopping criteria are doubles: passing this suite does
# NOT establish that real Qwen/NF4 CUDA generation is numerically repaired.
@pytest.fixture
def native_5141_sampling(monkeypatch):
    import ast
    from contextlib import nullcontext
    import hashlib
    import importlib.util
    import os
    from pathlib import Path

    import torch

    source_path = (
        Path(__file__).resolve().parents[1]
        / "_local_private_runs/cloud_numerics_20260827_v1"
        / "20260827-failure-step-v1/transformers_generation_utils.py"
    )
    configured = os.environ.get("POLICYAGENT_TF_GENERATION_SOURCE")
    if configured:
        source_path = Path(configured)
        assert source_path.is_file(), "Explicit audited source path is missing"
    elif not source_path.is_file():
        # A clean cloud checkout can use its installed, exactly hash-matching
        # runtime source. No download, dependency install or code substitution.
        spec = importlib.util.find_spec("transformers")
        if spec is None or spec.origin is None:
            pytest.skip("Audited Transformers 5.14.1 source is not available")
        source_path = Path(spec.origin).parent / "generation/utils.py"
        if not source_path.is_file() or hashlib.sha256(
            source_path.read_bytes()
        ).hexdigest().upper() != (
            "EC9E5BCE8B654D5EF8169DDF595BE18A06A95C87568FCD909D37D88255FFB8F7"
        ):
            pytest.skip("Installed generation source is not the audited 5.14.1 source")
    raw = source_path.read_bytes()
    assert hashlib.sha256(raw).hexdigest().upper() == (
        "EC9E5BCE8B654D5EF8169DDF595BE18A06A95C87568FCD909D37D88255FFB8F7"
    )
    source_tree = ast.parse(raw.decode("utf-8"), filename=str(source_path))
    original_class = next(
        node
        for node in source_tree.body
        if isinstance(node, ast.ClassDef) and node.name == "GenerationMixin"
    )
    method_names = {
        "_sample",
        "_get_logits_processor",
        "_merge_criteria_processor_list",
    }
    methods = [
        node
        for node in original_class.body
        if isinstance(node, ast.FunctionDef) and node.name in method_names
    ]
    assert {method.name for method in methods} == method_names
    isolated_class = ast.parse("class GenerationMixin:\n    pass\n").body[0]
    ast.copy_location(isolated_class, original_class)
    isolated_class.body = methods
    tree = ast.Module(
        body=[
            ast.parse("from __future__ import annotations").body[0],
            isolated_class,
        ],
        type_ignores=[],
    )

    class ProcessorList(list):
        def __call__(self, input_ids, scores):
            for processor in self:
                scores = processor(input_ids, scores)
            return scores

    class TemperatureLogitsWarper:
        def __init__(self, temperature):
            self.temperature = temperature

        def __call__(self, input_ids, scores):
            return scores / self.temperature

    module = ModuleType("transformers.generation.utils")
    module.__file__ = str(source_path)
    module.__dict__.update(
        torch=torch,
        nn=torch.nn,
        LogitsProcessorList=ProcessorList,
        TemperatureLogitsWarper=TemperatureLogitsWarper,
    )
    exec(compile(tree, str(source_path), "exec"), module.__dict__)
    monkeypatch.setitem(sys.modules, module.__name__, module)

    class EOSCriteria:
        eos_token_id = torch.tensor([6])

        def __call__(self, input_ids, scores):
            return input_ids[:, -1] == 6

    class StopList(list):
        def __call__(self, input_ids, scores):
            # Hard safety bound for the CPU double; never a success assertion.
            if input_ids.shape[1] > 12:
                raise AssertionError("CPU fixture exceeded its generation bound")
            stopped = torch.zeros(len(input_ids), dtype=torch.bool)
            for criterion in self:
                stopped |= criterion(input_ids, scores)
            return stopped

    config_attributes = {
        node.attr
        for method in methods
        for node in ast.walk(method)
        if isinstance(node, ast.Attribute)
        and isinstance(node.value, ast.Name)
        and node.value.id == "generation_config"
    }

    class Core(torch.nn.Module, module.GenerationMixin):
        def __init__(self, eos_steps, bad_step, bad_row, bad_value):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.ones(1))
            self.config = SimpleNamespace(
                is_encoder_decoder=False, max_position_embeddings=100
            )
            self.eos_steps = eos_steps
            self.bad_step, self.bad_row, self.bad_value = bad_step, bad_row, bad_value
            self.inputs, self.cache_inputs, self.raw_outputs = [], [], []
            self.cache = SimpleNamespace(batch_size=None, sequence_length=0)
            self.extra_processors, self.extra_stops = [], []

        def forward(self, input_ids, past_key_values=None, **kwargs):
            step = len(self.inputs)
            self.inputs.append(input_ids.clone())
            self.cache_inputs.append(past_key_values)
            if past_key_values is not None:
                assert past_key_values is self.cache
                assert self.cache.batch_size == len(input_ids)
            self.cache.batch_size = len(input_ids)
            self.cache.sequence_length += input_ids.shape[1]
            logits = torch.full((len(input_ids), 1, 7), -1000.0)
            for row in range(len(input_ids)):
                if step >= self.eos_steps[row]:
                    logits[row, 0, 6] = 0.0
                else:
                    logits[row, 0, 2:4] = torch.tensor([0.0, 0.25])
            if self.bad_step is not None and step >= self.bad_step:
                logits[self.bad_row] = self.bad_value
            self.raw_outputs.append(logits)
            return SimpleNamespace(logits=logits, past_key_values=self.cache)

        def _valid_auto_compile_criteria(self, model_kwargs, generation_config):
            return False

        def _prefill(self, input_ids, generation_config, model_kwargs, **kwargs):
            return self(input_ids=input_ids, return_dict=True)

        def _optimize_model_for_decode(self):
            return nullcontext()

        def _has_unfinished_sequences(self, this_peer_finished, *args, **kwargs):
            return not bool(this_peer_finished)

        def prepare_inputs_for_generation(self, input_ids, **kwargs):
            return {
                "input_ids": input_ids[:, -1:],
                "past_key_values": kwargs["past_key_values"],
            }

        def _update_model_kwargs_for_generation(self, outputs, model_kwargs, **kwargs):
            return dict(model_kwargs, past_key_values=outputs.past_key_values)

        def generate(self, prompts, generation_config):
            width = max(map(len, prompts))
            pad = int(generation_config._pad_token_tensor.item())
            padded = torch.tensor([[pad] * (width - len(p)) + p for p in prompts])
            self.processors = self._get_logits_processor(
                generation_config=generation_config,
                input_ids_seq_length=width,
                logits_processor=ProcessorList(self.extra_processors),
                device="cpu",
            )
            self.output = self._sample(
                padded,
                logits_processor=self.processors,
                stopping_criteria=StopList([EOSCriteria(), *self.extra_stops]),
                generation_config=generation_config,
                use_cache=True,
            )
            return self.output

    def new_case(
        *,
        prompts=None,
        eos_steps=(2, 0),
        bad_step=None,
        bad_row=1,
        bad_value=float("nan"),
        do_sample=True,
        temperature=0.8,
    ):
        core = Core(eos_steps, bad_step, bad_row, bad_value)
        values = dict.fromkeys(config_attributes)
        values.update(
            _pad_token_tensor=torch.tensor(0),
            _eos_token_tensor=torch.tensor([6]),
            pad_token_id=0,
            eos_token_id=6,
            num_beams=1,
            num_return_sequences=1,
            do_sample=do_sample,
            temperature=temperature,
            top_k=0,
            top_p=1.0,
            repetition_penalty=1.0,
            output_attentions=False,
            output_hidden_states=False,
            output_scores=False,
            output_logits=False,
            return_dict_in_generate=False,
            is_assistant=False,
            remove_invalid_values=False,
            renormalize_logits=False,
            max_new_tokens=8,
            disable_compile=True,
        )
        config = SimpleNamespace(**values)
        return SimpleNamespace(
            core=core,
            generation_config=config,
            prompts=prompts if prompts is not None else [[1, 2], [2]],
            events=[],
        )

    def wrapped_trainer(case):
        class NativeSamplerTrainer(FakeTrainer):
            def __init__(self):
                super().__init__()
                self.model.core = case.core
                self.model.config = case.core.config
                self.model.eval()
                self.generation_config = case.generation_config
                self._tokenizer.eos_token_id = 6
                self._tokenizer.pad_token_id = 0
                self.generation_kwargs = {
                    "do_sample": case.generation_config.do_sample,
                }
                self._budget_trace = GuardedTrajectoryTrace(32, 100, case.events.append)

            def _generate_single_turn(self, prompts, images, fields):
                output = self.model.get_base_model().generate(
                    prompts, self.generation_config
                )
                suffixes = output[:, max(map(len, prompts)) :].tolist()
                return [s[: s.index(6) + 1] if 6 in s else s for s in suffixes], None

        return make_sampling_trainer(
            NativeSamplerTrainer,
            expected_do_sample=case.generation_config.do_sample,
        )()

    return SimpleNamespace(module=module, new_case=new_case, trainer=wrapped_trainer)


def _native_5141_call(fixture, case):
    import torch

    trainer = fixture.trainer(case)
    with torch.inference_mode():
        result = trainer._generate_single_turn(case.prompts, None, {})
    return trainer, result


def test_native_5141_true_greedy_uses_argmax_without_multinomial_or_rng(
    native_5141_sampling,
):
    import torch

    case = native_5141_sampling.new_case(
        do_sample=False,
        # A tiny temperature is deliberately present to prove it has no decode
        # authority when sampling is explicitly disabled.
        temperature=1e-8,
    )
    before = torch.get_rng_state().clone()
    with patch.object(
        torch,
        "multinomial",
        side_effect=AssertionError("true greedy called multinomial"),
    ) as draw:
        _, (outputs, _) = _native_5141_call(native_5141_sampling, case)
    draw.assert_not_called()
    assert outputs == [[3, 3, 6], [6]]
    assert torch.equal(torch.get_rng_state(), before)
    assert not any(
        type(processor).__name__ == "TemperatureLogitsWarper"
        for processor in case.core.processors
    )
    _assert_native_5141_clean(native_5141_sampling, case.core)


def test_native_5141_near_zero_temperature_remains_stochastic(
    native_5141_sampling,
):
    import torch

    case = native_5141_sampling.new_case(
        eos_steps=(0, 0),
        do_sample=True,
        temperature=1e-8,
    )
    with patch.object(torch, "multinomial", wraps=torch.multinomial) as draw:
        _, (outputs, _) = _native_5141_call(native_5141_sampling, case)
    assert outputs == [[6], [6]]
    assert draw.call_count == 1
    assert any(
        type(processor).__name__ == "TemperatureLogitsWarper"
        for processor in case.core.processors
    )
    _assert_native_5141_clean(native_5141_sampling, case.core)


def _assert_native_5141_clean(fixture, core):
    assert not core._forward_hooks and not core._forward_pre_hooks
    assert "_get_logits_processor" not in core.__dict__
    assert "_policyagent_finished_row_owner" not in core.__dict__
    assert (
        core._get_logits_processor.__func__
        is fixture.module.GenerationMixin._get_logits_processor
    )


def _guarded_training_double(fixture, case):
    """CPU training double: checks integration, not a real GRPO experiment."""
    import torch
    from src.training.rollout_diagnostics import make_guarded_grpo_trainer

    native_class = type(fixture.trainer(case)).__bases__[0]

    class NativeTrainingDouble(native_class):
        def __init__(self):
            super().__init__()
            # The reused sampler fixture sets this; real training has no trace.
            del self._budget_trace
            self.order = []
            self.beta = 0.02
            self.state = SimpleNamespace(global_step=0)
            self.native_logprobs = torch.tensor([0.5], requires_grad=True)

            class RewardEnvironment:
                _user_stopped = True
                _runtime_stop_signals = []

                def _set_trainer_completion_telemetry(environment, payload):
                    environment.telemetry = payload

                def get_reward(environment):
                    assert environment.telemetry["stop_reason"] == (
                        "USER_STOP_AND_MODEL_EOS"
                    )
                    return 1.0

            self.environments = [RewardEnvironment(), RewardEnvironment()]

        def _generate_single_turn(self, prompts, images, fields):
            # The underlying implementation owns its generation grad context.
            with torch.no_grad():
                output, _ = super()._generate_single_turn(prompts, images, fields)
            self.native_result = (output, self.native_logprobs)
            return self.native_result

        def _generate(self, prompts):
            prompt_ids = [[1, 2]] * len(prompts)
            ids, _ = self._generate_single_turn(prompt_ids, None, {})
            return (
                prompt_ids,
                ids,
                [[1] * len(row) for row in ids],
                [[{"role": "assistant", "content": "done"}]] * len(prompts),
            )

        def create_optimizer(self):
            self.order.append("optimizer")
            self.optimizer = torch.optim.SGD(self.model.parameters(), lr=0.1)
            return self.optimizer

        def _generate_and_score_completions(self, inputs):
            self.order.append("generate")
            self._generate(inputs)
            _assert_native_5141_clean(fixture, case.core)
            assert torch.is_grad_enabled() and not torch.is_inference_mode_enabled()
            assert self.model.training
            assert all(
                hasattr(environment, "telemetry") for environment in self.environments
            )
            self.order.append("reward")
            assert [environment.get_reward() for environment in self.environments] == [
                1.0,
                1.0,
            ]
            self.order.append("score")
            # A regular scoring graph must remain possible after generation.
            return torch.nn.functional.linear(
                torch.ones(1, 1), case.core.weight.view(1, 1)
            ), self.native_result

        def compute_loss(self, scored):
            self.order.append("loss")
            score, output = scored
            assert output[0] is self.native_result[0]
            assert output[1] is self.native_logprobs  # not detached or replaced
            return score.square().sum() + self.beta * (score - 1).square().sum()

        def train(self):
            self.order.append("train")
            self.model.train()
            optimizer = self.create_optimizer()
            loss = self.compute_loss(self._generate_and_score_completions(case.prompts))
            loss.backward()
            self.order.append("backward")
            assert torch.isfinite(case.core.weight.grad).all()
            optimizer.step()
            self.order.append("step")
            self.state.global_step += 1
            return loss

    wrapped = make_guarded_grpo_trainer(NativeTrainingDouble, case.events.append)
    for name in (
        "train",
        "create_optimizer",
        "compute_loss",
        "_generate_and_score_completions",
    ):
        assert getattr(wrapped, name) is getattr(NativeTrainingDouble, name)
    return wrapped()


def test_guarded_training_keeps_native_methods_and_allows_backward(
    native_5141_sampling,
):
    import torch

    # One row finishes early, then produces NaNs. The active row still generates.
    case = native_5141_sampling.new_case(bad_step=1, bad_row=1)
    trainer = _guarded_training_double(native_5141_sampling, case)
    before = case.core.weight.detach().clone()
    flags = [parameter.requires_grad for parameter in trainer.model.parameters()]
    loss = trainer.train()
    assert torch.isfinite(loss)
    assert trainer.order == [
        "train",
        "optimizer",
        "generate",
        "reward",
        "score",
        "loss",
        "backward",
        "step",
    ]
    assert not torch.equal(before, case.core.weight)
    assert trainer.state.global_step == 1 and trainer.beta == 0.02
    assert flags == [
        parameter.requires_grad for parameter in trainer.model.parameters()
    ]
    assert not hasattr(trainer, "_budget_trace")
    assert any(
        event["event"] == "finished_row_nonfinite_logits" for event in case.events
    )
    generation_events = [
        event for event in case.events if event["row_scope"] == "generation_local"
    ]
    stop_events = [event for event in case.events if event["event"] == "rollout_stop"]
    assert generation_events and len(stop_events) == 2
    assert all(event["row_scope"] == "rollout_group" for event in stop_events)
    assert all(event["optimizer_step"] == 0 for event in case.events)
    _assert_native_5141_clean(native_5141_sampling, case.core)


@pytest.mark.parametrize("bad_value", [float("nan"), float("inf"), -float("inf")])
def test_guarded_training_active_fault_stops_before_score_or_step(
    native_5141_sampling, bad_value
):
    import torch

    case = native_5141_sampling.new_case(bad_step=1, bad_row=0, bad_value=bad_value)
    trainer = _guarded_training_double(native_5141_sampling, case)
    before = case.core.weight.detach().clone()
    with pytest.raises(FloatingPointError, match="NONFINITE_RAW_MODEL_LOGITS"):
        trainer.train()
    assert trainer.order == ["train", "optimizer", "generate"]
    assert trainer.state.global_step == 0
    assert torch.equal(before, case.core.weight) and case.core.weight.grad is None
    assert case.core.weight.requires_grad and trainer.model.training
    _assert_native_5141_clean(native_5141_sampling, case.core)


@pytest.mark.parametrize("unsupported", ["processes", "vllm", "vlm", "continuous"])
def test_guarded_training_rejects_unsupported_runtime_before_generation(
    native_5141_sampling, unsupported
):
    case = native_5141_sampling.new_case()
    trainer = _guarded_training_double(native_5141_sampling, case)
    if unsupported == "processes":
        trainer.accelerator.num_processes = 2
    else:
        setattr(
            trainer,
            {
                "vllm": "use_vllm",
                "vlm": "_is_vlm",
                "continuous": "use_transformers_continuous_batching",
            }[unsupported],
            True,
        )
    with pytest.raises(ValueError, match="single-process"):
        trainer._generate_single_turn(case.prompts, None, {})
    assert not case.events and not case.core.inputs
    _assert_native_5141_clean(native_5141_sampling, case.core)


def test_guarded_training_continuation_indices_are_explicitly_local(
    native_5141_sampling,
):
    case = native_5141_sampling.new_case()
    trainer = _guarded_training_double(native_5141_sampling, case)
    trainer._generate_single_turn(case.prompts, None, {})
    # A later native call contains one continuing candidate, not a complete group.
    trainer._generate_single_turn([[1, 2, 3]], None, {})
    entries = [event for event in case.events if event["event"] == "generation_start"]
    assert [event["generation_index"] for event in entries] == [0, 1]
    assert [event["rows"] for event in entries] == [[0, 1], [0]]
    assert all(event["row_scope"] == "generation_local" for event in entries)
    assert not hasattr(trainer, "_budget_trace")
    _assert_native_5141_clean(native_5141_sampling, case.core)


def test_native_5141_finished_nan_isolated_without_compacting_batch_or_cache(
    native_5141_sampling,
):
    import torch
    from src.training.rollout_diagnostics import _FinishedRowPad

    reference = native_5141_sampling.new_case(bad_step=1)
    with torch.inference_mode(), pytest.raises(RuntimeError, match="probability"):
        reference.core.generate(reference.prompts, reference.generation_config)

    case = native_5141_sampling.new_case(bad_step=1)
    probabilities = []
    native_multinomial = torch.multinomial

    def observe(probs, *args, **kwargs):
        probabilities.append(probs.clone())
        return native_multinomial(probs, *args, **kwargs)

    with patch.object(torch, "multinomial", side_effect=observe):
        _, (outputs, _) = _native_5141_call(native_5141_sampling, case)
    assert outputs[0][-1] == 6 and outputs[1] == [6]
    assert len(probabilities) == 3
    assert all(probs.shape == (2, 7) for probs in probabilities)
    for probs in probabilities[1:]:
        assert probs[1].tolist() == [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    assert all(len(inputs) == 2 for inputs in case.core.inputs)
    assert case.core.cache_inputs[0] is None
    assert all(cache is case.core.cache for cache in case.core.cache_inputs[1:])
    assert case.core.output[1, -3:].tolist() == [6, 0, 0]
    assert torch.isnan(case.core.raw_outputs[1][1]).all()
    assert isinstance(case.core.processors[-1], _FinishedRowPad)
    assert type(case.core.processors[-2]).__name__ == "TemperatureLogitsWarper"
    _assert_native_5141_clean(native_5141_sampling, case.core)


@pytest.mark.parametrize("bad_value", [float("nan"), float("inf"), -float("inf")])
def test_native_5141_active_nonfinite_stops_before_next_draw_and_cleans_up(
    native_5141_sampling, bad_value
):
    import torch

    case = native_5141_sampling.new_case(bad_step=1, bad_row=0, bad_value=bad_value)
    with (
        patch.object(torch, "multinomial", wraps=torch.multinomial) as draw,
        pytest.raises(FloatingPointError, match="NONFINITE_RAW_MODEL_LOGITS"),
    ):
        _native_5141_call(native_5141_sampling, case)
    assert draw.call_count == 1
    assert any(event["event"] == "generation_error" for event in case.events)
    _assert_native_5141_clean(native_5141_sampling, case.core)


@pytest.mark.parametrize("bad_row", [0, 1])
def test_native_5141_prompt_eos_and_left_padding_never_exempt_active_nan(
    native_5141_sampling, bad_row
):
    import torch

    case = native_5141_sampling.new_case(
        prompts=[[6, 2], [6]], bad_step=0, bad_row=bad_row
    )
    before = torch.get_rng_state().clone()
    with (
        patch.object(torch, "multinomial", wraps=torch.multinomial) as draw,
        pytest.raises(FloatingPointError, match="NONFINITE_RAW_MODEL_LOGITS"),
    ):
        _native_5141_call(native_5141_sampling, case)
    draw.assert_not_called()
    assert torch.equal(before, torch.get_rng_state())
    _assert_native_5141_clean(native_5141_sampling, case.core)


def test_native_5141_unfinished_false_without_generated_eos_cannot_exempt_nan(
    native_5141_sampling,
):
    import torch

    case = native_5141_sampling.new_case(eos_steps=(2, 2), bad_step=1)
    case.core.extra_stops = [lambda ids, scores: torch.tensor([False, True])]
    with pytest.raises((FloatingPointError, RuntimeError)):
        _native_5141_call(native_5141_sampling, case)
    assert len(case.core.inputs) <= 2
    _assert_native_5141_clean(native_5141_sampling, case.core)


@pytest.mark.parametrize("eos_steps", [(2, 2), (2, 0)])
def test_native_5141_finite_active_probabilities_tokens_and_cpu_rng_unchanged(
    native_5141_sampling, eos_steps
):
    import torch

    original_rng = torch.get_rng_state().clone()
    reference = native_5141_sampling.new_case(eos_steps=eos_steps)
    candidate = native_5141_sampling.new_case(eos_steps=eos_steps)
    native_multinomial = torch.multinomial
    observed = [[], []]

    def record(index):
        def draw(probs, *args, **kwargs):
            observed[index].append(probs.clone())
            return native_multinomial(probs, *args, **kwargs)

        return draw

    try:
        torch.manual_seed(20260827)
        with (
            torch.inference_mode(),
            patch.object(torch, "multinomial", side_effect=record(0)),
        ):
            expected = reference.core.generate(
                reference.prompts, reference.generation_config
            )
        expected_rng = torch.get_rng_state().clone()
        torch.manual_seed(20260827)
        with patch.object(torch, "multinomial", side_effect=record(1)):
            _native_5141_call(native_5141_sampling, candidate)
        assert torch.equal(candidate.core.output, expected)
        assert torch.equal(torch.get_rng_state(), expected_rng)
        assert len(observed[0]) == len(observed[1]) == 3
        for step, (before, after) in enumerate(zip(*observed)):
            assert before.shape == after.shape == (2, 7)
            assert torch.equal(before[0], after[0])
            if step <= eos_steps[1]:
                assert torch.equal(before[1], after[1])
        _assert_native_5141_clean(native_5141_sampling, candidate.core)
    finally:
        torch.set_rng_state(original_rng)


@pytest.mark.parametrize("bad_value", [float("nan"), float("inf"), -float("inf")])
def test_native_5141_processed_active_nonfinite_is_not_sanitized(
    native_5141_sampling, bad_value
):
    import torch

    def corrupt_active(input_ids, scores):
        result = scores.clone()
        result[0] = bad_value
        return result

    case = native_5141_sampling.new_case()
    case.core.extra_processors = [corrupt_active]
    with (
        patch.object(torch, "multinomial", wraps=torch.multinomial) as draw,
        pytest.raises(FloatingPointError),
    ):
        _native_5141_call(native_5141_sampling, case)
    draw.assert_not_called()
    assert all(torch.isfinite(logits).all() for logits in case.core.raw_outputs)
    _assert_native_5141_clean(native_5141_sampling, case.core)


def test_native_5141_continuation_uses_original_candidate_row_not_local_batch_row(
    native_5141_sampling,
):
    import torch

    case = native_5141_sampling.new_case(
        prompts=[[2, 6, 3]], eos_steps=(2,), bad_step=0, bad_row=0
    )
    trainer = native_5141_sampling.trainer(case)
    trainer._budget_trace.start([[1, 2], [2]])
    trainer._budget_trace.next_generation = [(1, case.prompts[0])]
    trainer._budget_trace.generation_count = 1
    with (
        torch.inference_mode(),
        pytest.raises(FloatingPointError, match="NONFINITE_RAW_MODEL_LOGITS"),
    ):
        trainer._generate_single_turn(case.prompts, None, {})
    failure = next(e for e in case.events if e["event"] == "nonfinite_raw_logits")
    assert failure["rows"] == [1] and failure["generation_index"] == 1
    _assert_native_5141_clean(native_5141_sampling, case.core)


def test_native_5141_processed_partial_negative_inf_is_valid(native_5141_sampling):
    import torch

    def exclude_one_token(input_ids, scores):
        result = scores.clone()
        result[:, 0] = -torch.inf
        return result

    case = native_5141_sampling.new_case(bad_step=1)
    case.core.extra_processors = [exclude_one_token]
    _, (outputs, _) = _native_5141_call(native_5141_sampling, case)
    assert outputs[0][-1] == 6 and outputs[1] == [6]
    assert type(case.core.processors[-2]).__name__ == "TemperatureLogitsWarper"
    _assert_native_5141_clean(native_5141_sampling, case.core)


def test_native_5141_pad_equal_eos_is_supported(native_5141_sampling):
    import torch

    case = native_5141_sampling.new_case(bad_step=1)
    case.generation_config.pad_token_id = 6
    case.generation_config._pad_token_tensor = torch.tensor(6)
    _, (outputs, _) = _native_5141_call(native_5141_sampling, case)
    assert outputs[0][-1] == 6 and outputs[1] == [6]
    assert case.core.inputs[0][1].tolist() == [6, 2]
    assert case.core.output[1, -3:].tolist() == [6, 6, 6]
    assert torch.isnan(case.core.raw_outputs[1][1]).all()
    padding_event = next(
        e for e in case.events if e["event"] == "finished_rows_padding"
    )
    assert padding_event["pad_token_id"] == 6
    assert padding_event["business_success_implied"] is False
    _assert_native_5141_clean(native_5141_sampling, case.core)


@pytest.mark.parametrize("builder_raises", [False, True])
def test_native_5141_existing_instance_builder_restored_even_on_builder_error(
    native_5141_sampling, builder_raises
):
    from functools import wraps

    case = native_5141_sampling.new_case()
    other = native_5141_sampling.new_case()
    native_builder = case.core._get_logits_processor

    @wraps(native_builder)
    def instance_builder(*args, **kwargs):
        if builder_raises:
            raise OSError("CPU fixture builder failed")
        return native_builder(*args, **kwargs)

    case.core._get_logits_processor = instance_builder
    if builder_raises:
        with pytest.raises(OSError, match="CPU fixture builder failed"):
            _native_5141_call(native_5141_sampling, case)
        assert case.core.inputs == []
    else:
        _, (outputs, _) = _native_5141_call(native_5141_sampling, case)
        assert outputs[0][-1] == 6 and outputs[1] == [6]
    assert case.core._get_logits_processor is instance_builder
    assert not case.core._forward_hooks and not case.core._forward_pre_hooks
    assert "_policyagent_finished_row_owner" not in case.core.__dict__
    _assert_native_5141_clean(native_5141_sampling, other.core)


def test_native_5141_nested_generation_rejected_without_consuming_binding(
    native_5141_sampling,
):
    import torch

    case = native_5141_sampling.new_case()
    trainer = native_5141_sampling.trainer(case)

    def recurse(input_ids, scores):
        before = deepcopy(trainer._budget_trace.next_generation)
        with pytest.raises(RuntimeError, match="Concurrent or nested"):
            trainer._generate_single_turn(case.prompts, None, {})
        assert trainer._budget_trace.next_generation == before
        return scores

    case.core.extra_processors = [recurse]
    with torch.inference_mode():
        outputs, _ = trainer._generate_single_turn(case.prompts, None, {})
    assert outputs[0][-1] == 6 and outputs[1] == [6]
    starts = [e for e in case.events if e["event"] == "generation_start"]
    assert len(starts) == 1
    assert trainer._budget_trace.generation_count == 1
    _assert_native_5141_clean(native_5141_sampling, case.core)


def test_native_5141_exclusive_entry_rejects_parallel_before_getter_marker(
    native_5141_sampling,
):
    import threading
    from src.training.rollout_diagnostics import _exclusive_sampling_model

    core = native_5141_sampling.new_case().core
    other = native_5141_sampling.new_case().core
    start = threading.Barrier(3)
    acquired, rejected, release = (threading.Event() for _ in range(3))
    outcomes, errors = [], []

    def compete():
        try:
            start.wait(timeout=5)
            try:
                with _exclusive_sampling_model(core):
                    outcomes.append("acquired")
                    acquired.set()
                    if not release.wait(timeout=5):
                        raise AssertionError("Exclusive-holder release timed out")
            except RuntimeError as exc:
                if "Concurrent or nested" not in str(exc):
                    raise
                outcomes.append("rejected")
                rejected.set()
        except BaseException as exc:
            errors.append(exc)

    workers = [threading.Thread(target=compete, daemon=True) for _ in range(2)]
    for worker in workers:
        worker.start()
    try:
        start.wait(timeout=5)
        assert acquired.wait(timeout=5)
        assert rejected.wait(timeout=5)
        # No builder/forward hook has been installed: this specifically tests
        # the earlier check-to-install race, not the existing marker check.
        assert "_policyagent_finished_row_owner" not in core.__dict__
        with _exclusive_sampling_model(other):
            pass
    finally:
        release.set()
        for worker in workers:
            worker.join(timeout=5)
    assert all(not worker.is_alive() for worker in workers)
    assert errors == []
    assert sorted(outcomes) == ["acquired", "rejected"]
    with pytest.raises(ValueError, match="fixture exit"):
        with _exclusive_sampling_model(core):
            raise ValueError("fixture exit")
    with _exclusive_sampling_model(core):
        pass  # Successful and exceptional exits both release model ownership.
