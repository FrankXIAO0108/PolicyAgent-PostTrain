"""Startup contract tests. No real model, GPU, network or training is used."""

import json
from pathlib import Path
import os
import sys
from copy import deepcopy
from itertools import combinations
from types import SimpleNamespace as NS
from unittest.mock import Mock, patch

import pytest

from src.training import run_retail_agentic_grpo as runner


@pytest.fixture(autouse=True)
def preserve_environment():
    with patch.dict(os.environ):
        yield


def preflight():
    config = runner.load_json(
        runner.REPO_ROOT
        / "configs"
        / "retail_agentic_qwen3_4b_staged_reward_write_coverage_tasks43_72_v1.json"
    )
    return {
        "config": config,
        "config_sha256": "config",
        "split_sha256": "split",
        "openings_sha256": "openings",
        "model_sha256": "model",
        "model_path": "MODEL_NOT_LOADED",
        "openings": [],
    }


def cli(monkeypatch, output, *flags):
    monkeypatch.setattr(sys, "argv", ["runner", "--output-dir", str(output), *flags])
    validate = Mock(return_value=preflight())
    monkeypatch.setattr(runner, "validate_inputs", validate)
    return validate


@pytest.mark.parametrize("kind", ["nonempty", "empty", "file"])
@pytest.mark.parametrize("sampling", [False, True])
def test_existing_output_untouched(tmp_path, monkeypatch, kind, sampling):
    output = tmp_path / "existing"
    if kind == "file":
        output.write_bytes(b"preserved")
    else:
        output.mkdir()
        if kind == "nonempty":
            (output / "failure_manifest.json").write_bytes(b"original failure")
    flags = ["--sample-only", "--completion-budget", "2048"] if sampling else []
    cli(monkeypatch, output, *flags)
    run = Mock(side_effect=FileExistsError("existing directory"))
    monkeypatch.setattr(runner, "run", run)
    with pytest.raises(FileExistsError):
        runner.main()
    run.assert_not_called()
    if kind == "file":
        assert output.read_bytes() == b"preserved"
    elif kind == "empty":
        assert list(output.iterdir()) == []
    else:
        assert (output / "failure_manifest.json").read_bytes() == b"original failure"
        assert len(list(output.iterdir())) == 1


MODES = [
    "--sample-only",
    "--preflight-only",
    "--environment-only-preflight",
    "--user-simulator-api-preflight-only",
]


@pytest.mark.parametrize("modes", list(combinations(MODES, 2)))
def test_exclusive_modes_fail_before_any_preflight(tmp_path, monkeypatch, modes):
    flags = list(modes)
    if "--sample-only" in modes:
        flags += ["--completion-budget", "2048"]
    validate = cli(monkeypatch, tmp_path / "unused", *flags)
    env = Mock()
    api = Mock()
    monkeypatch.setattr(runner, "environment_only_preflight", env)
    monkeypatch.setattr(runner, "probe_user_simulator_api", api)
    with pytest.raises(SystemExit):
        runner.main()
    for mock in (validate, env, api):
        mock.assert_not_called()
    assert not (tmp_path / "unused").exists()


def test_new_run_failure_gets_manifest_only_in_new_directory(tmp_path, monkeypatch):
    output = tmp_path / "new"
    cli(monkeypatch, output)

    def fail(*args, **kwargs):
        assert output.is_dir()
        assert not list(output.iterdir())
        raise RuntimeError("local dependency failure")

    monkeypatch.setattr(runner, "run", fail)
    with pytest.raises(RuntimeError, match="local dependency failure"):
        runner.main()
    saved = json.loads((output / "failure_manifest.json").read_text(encoding="utf-8"))
    assert saved["status"] == "FAILED"
    assert saved["training_eligible"] is False
    assert saved["reward_eligible"] is False
    state = json.loads((output / "run_state.json").read_text(encoding="utf-8"))
    assert state["status"] == "FAILED"
    assert state["failure_manifest_sha256"] == runner.sha256(
        output / "failure_manifest.json"
    )


def test_failed_run_hash_binds_partial_sampling_artifacts(tmp_path, monkeypatch):
    output = tmp_path / "partial"
    cli(monkeypatch, output)

    def fail(*args, **kwargs):
        (output / "raw_rollouts.jsonl").write_text(
            '{"task_id":"43"}\n', encoding="utf-8"
        )
        (output / "generation_events.jsonl").write_text(
            '{"event":"started"}\n', encoding="utf-8"
        )
        raise RuntimeError("interrupted rollout")

    monkeypatch.setattr(runner, "run", fail)
    with pytest.raises(RuntimeError, match="interrupted rollout"):
        runner.main()

    manifest = json.loads(
        (output / "failure_manifest.json").read_text(encoding="utf-8")
    )
    artifacts = manifest["partial_artifacts"]
    assert set(artifacts) == {"raw_rollouts.jsonl", "generation_events.jsonl"}
    assert artifacts["raw_rollouts.jsonl"]["rows"] == 1
    assert artifacts["raw_rollouts.jsonl"]["sha256"] == runner.sha256(
        output / "raw_rollouts.jsonl"
    )


def test_user_simulator_binding_redacts_secrets_and_binds_behavioral_args():
    binding = runner.user_simulator_binding(
        "deepseek/user",
        '{"temperature":0.0,"api_key":"secret","nested":{"access_token":"x"}}',
    )

    assert binding["model"] == "deepseek/user"
    assert binding["llm_args"] == {
        "temperature": 0.0,
        "api_key": "<REDACTED>",
        "nested": {"access_token": "<REDACTED>"},
    }
    assert len(binding["llm_args_sha256"]) == 64
    assert "secret" not in json.dumps(binding)
    with pytest.raises(ValueError, match="JSON object"):
        runner.user_simulator_binding("deepseek/user", "[]")


@pytest.mark.parametrize("failure", ["runtime", "template"])
def test_local_failure_precedes_api(tmp_path, monkeypatch, failure):
    api = Mock()
    monkeypatch.setattr(runner, "probe_user_simulator_api", api)
    runtime = Mock(return_value={"bf16_supported": True})
    template = Mock(return_value={"tool_schema_rendered": True})
    (runtime if failure == "runtime" else template).side_effect = RuntimeError(failure)
    monkeypatch.setattr(runner, "check_runtime", runtime)
    monkeypatch.setattr(runner, "check_tool_template", template)
    with pytest.raises(RuntimeError, match=failure):
        runner.run(preflight(), tmp_path / "run")
    api.assert_not_called()


@pytest.mark.parametrize(
    "rendered,valid",
    [
        ("ordinary chat, no tool schema", False),
        ("", False),
        (
            '{"name":"probe_tool","parameters":{"properties":{"value":{"type":"string"}}}}',
            True,
        ),
    ],
)
def test_template_must_include_probe_schema(monkeypatch, rendered, valid):
    tokenizer = NS(
        chat_template="template", apply_chat_template=Mock(return_value=rendered)
    )
    fake = NS(AutoTokenizer=NS(from_pretrained=Mock(return_value=tokenizer)))
    monkeypatch.setitem(sys.modules, "transformers", fake)
    if valid:
        assert runner.check_tool_template("MODEL_NOT_LOADED")["tool_schema_rendered"]
    else:
        with pytest.raises(RuntimeError):
            runner.check_tool_template("MODEL_NOT_LOADED")


@pytest.mark.parametrize("failure", ["grpo", "lora", None])
def test_argument_construction_before_api(tmp_path, monkeypatch, failure):
    order = []

    def construct(name):
        def factory(**kwargs):
            order.append(name)
            if failure == name:
                raise ValueError(name)
            return NS(**kwargs)

        return factory

    def probe(**kwargs):
        order.append("api")
        return {"status": "MOCK_ONLY"}

    def trainer(**kwargs):
        order.append("trainer")
        raise RuntimeError("STOP_BEFORE_MODEL_LOADING")

    monkeypatch.setattr(runner, "check_runtime", lambda: {"bf16_supported": True})
    monkeypatch.setattr(runner, "check_tool_template", lambda path: {})
    monkeypatch.setattr(runner, "build_dataset", lambda data: [])
    api = Mock(side_effect=probe)
    monkeypatch.setattr(runner, "probe_user_simulator_api", api)
    monkeypatch.setitem(sys.modules, "torch", NS(bfloat16="bf16", float16="fp16"))
    monkeypatch.setitem(
        sys.modules, "peft", NS(LoraConfig=construct("lora"), PeftModel=object)
    )
    monkeypatch.setitem(
        sys.modules,
        "transformers",
        NS(
            AutoTokenizer=object,
            AutoModelForCausalLM=object,
            BitsAndBytesConfig=lambda **kwargs: NS(**kwargs),
            set_seed=lambda seed: None,
        ),
    )
    monkeypatch.setitem(
        sys.modules, "trl", NS(GRPOConfig=construct("grpo"), GRPOTrainer=trainer)
    )
    with pytest.raises((ValueError, RuntimeError)):
        runner.run(preflight(), tmp_path / "new")
    if failure:
        api.assert_not_called()
        assert order == (["grpo"] if failure == "grpo" else ["grpo", "lora"])
    else:
        assert order == ["grpo", "lora", "api", "trainer"]


def test_environment_preflight_uses_selected_task(monkeypatch):
    from src.rl import retail_agentic_env

    config = preflight()["config"]
    validated = {
        "config": config,
        "split": {"splits": {"rl_train": ["0"], "development_audit": ["43", "72"]}},
        "config_sha256": "config",
        "split_sha256": "split",
        "upstream_checkout": {},
    }
    monkeypatch.setattr(runner, "validate_config_and_split", lambda path: validated)
    monkeypatch.setitem(sys.modules, "tau2.data_model.message", NS(UserMessage=object))
    environment = Mock()
    environment.list_all_product_types.return_value = '["Backpack"]'
    environment.get_reward.return_value = 0.0
    environment._last_reward_info = {}
    monkeypatch.setattr(
        retail_agentic_env, "RetailAgenticEnvironment", lambda **kwargs: environment
    )
    result = runner.environment_only_preflight(None)
    assert result["task_id"] == "43"
    assert environment.reset.call_args.kwargs["task_id"] == "43"
    assert result["external_api_called"] is False


# These tests exercise launch wiring only. Native generation/backprop are tested
# separately; none of the doubles below loads a model or calls an external API.
MINIMAL_GRPO_CONFIG = (
    runner.REPO_ROOT
    / "configs"
    / "retail_agentic_qwen3_4b_staged_reward_minimal_grpo_tasks43_72_v1.json"
)
BF16_CLOSURE_CONFIG = (
    runner.REPO_ROOT
    / "configs"
    / "retail_agentic_qwen3_4b_staged_reward_minimal_grpo_bf16_lora_2step_v3.json"
)
TASK43_TERMINAL_CLOSURE_CONFIG = (
    runner.REPO_ROOT
    / "configs"
    / "retail_agentic_qwen3_4b_terminal_grpo_closure_task43_v1.json"
)
TASK43_TERMINAL_CLOSURE_N4_CONFIG = (
    runner.REPO_ROOT
    / "configs"
    / "retail_agentic_qwen3_4b_terminal_grpo_closure_task43_n4_v1.json"
)
TASK43_TERMINAL_SAMPLING_CONFIG = (
    runner.REPO_ROOT
    / "configs"
    / "retail_agentic_qwen3_4b_terminal_sampling_gate_task43_v1.json"
)
TASK43_TERMINAL_POST_GRPO_EVAL_CONFIG = (
    runner.REPO_ROOT
    / "configs"
    / "retail_agentic_qwen3_4b_terminal_post_grpo_eval_task43_v1.json"
)
S6_GREEDY_CONFIG = (
    runner.REPO_ROOT
    / "configs"
    / "retail_agentic_qwen3_4b_exact_checkpoint_greedy_tasks43_72_v1.json"
)
S7_STOCHASTIC_CONFIG = (
    runner.REPO_ROOT
    / "configs"
    / "retail_agentic_qwen3_4b_exact_checkpoint_stochastic_passk_tasks43_72_v1.json"
)
TASK95_PRESCREEN_CONFIG = (
    runner.REPO_ROOT
    / "configs"
    / "retail_agentic_qwen3_4b_exact_checkpoint_terminal_prescreen_task95_v1.json"
)
TASK43_ADAPTIVE_QUALITY_N4_CONFIG = (
    runner.REPO_ROOT
    / "configs"
    / "retail_agentic_qwen3_4b_adaptive_quality_sampling_task43_n4_v1.json"
)
TASK43_TERMINAL_N4_POST_EVAL_CONFIG = (
    runner.REPO_ROOT
    / "configs"
    / "retail_agentic_qwen3_4b_terminal_n4_post_eval_task43_n4_v1.json"
)
TASK43_STAGED_V4_POST_EVAL_CONFIG = (
    runner.REPO_ROOT
    / "configs"
    / "retail_agentic_qwen3_4b_staged_v4_post_eval_task43_n4_v1.json"
)

STOCHASTIC_SAMPLING_CONTRACT = {
    "mode": "STOCHASTIC_GROUP_SAMPLING",
    "do_sample": True,
    "actual_num_generations": 2,
    "trl_constructor_num_generations": 2,
    "groups_per_task": 1,
    "trainer_max_steps_unused": True,
}

TRUE_GREEDY_SAMPLING_CONTRACT = {
    "mode": "TRUE_GREEDY",
    "do_sample": False,
    "actual_num_generations": 1,
    "trl_constructor_num_generations": 2,
    "trl_constructor_steps_per_generation": 2,
    "groups_per_task": 1,
    "trainer_max_steps_unused": True,
}
S7_STOCHASTIC_SAMPLING_CONTRACT = {
    "mode": "STOCHASTIC_GROUP_SAMPLING",
    "do_sample": True,
    "actual_num_generations": 2,
    "trl_constructor_num_generations": 2,
    "groups_per_task": 2,
    "trainer_max_steps_unused": True,
    "temperature": 0.8,
    "top_p": 1.0,
    "top_k": 0,
}
ADAPTIVE_QUALITY_N4_SAMPLING_CONTRACT = {
    "mode": "STOCHASTIC_GROUP_SAMPLING",
    "do_sample": True,
    "contract_version": "adaptive-quality-gate-v1",
    "sample_schedule": [4, 6, 8],
    "minimum_quality_positive": 1,
    "actual_num_generations": 4,
    "trl_constructor_num_generations": 4,
    "groups_per_task": 1,
    "trainer_max_steps_unused": True,
    "temperature": 0.8,
    "top_p": 1.0,
    "top_k": 0,
}


def guarded_preflight():
    result = preflight()
    config = result["config"]
    config["execution_mode"] = "OPTIMIZE"
    config["generation_safety"] = {"mode": "eos_finished_rows_v1"}
    config["grpo"].update(
        max_steps=2,
        learning_rate=5e-6,
        beta=0.02,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=2,
        max_completion_length=13312,
    )
    config.pop("diagnostic")
    return result


def bf16_closure_preflight():
    result = preflight()
    result["config"] = runner.load_json(BF16_CLOSURE_CONFIG)
    result["openings"] = [{"task_id": "43"}, {"task_id": "72"}]
    return result


def task43_terminal_closure_preflight():
    result = preflight()
    result["config"] = runner.load_json(TASK43_TERMINAL_CLOSURE_CONFIG)
    result["openings"] = [{"task_id": "43"}]
    return result


def s6_greedy_preflight():
    result = preflight()
    result["config"] = runner.load_json(S6_GREEDY_CONFIG)
    result["model_path"] = result["config"]["model"]["name_or_path"]
    result["model_sha256"] = result["config"]["model"]["expected_sha256"]
    result["openings"] = [{"task_id": "43"}, {"task_id": "72"}]
    return result


def s7_stochastic_preflight():
    result = preflight()
    result["config"] = runner.load_json(S7_STOCHASTIC_CONFIG)
    result["model_path"] = result["config"]["model"]["name_or_path"]
    result["model_sha256"] = result["config"]["model"]["expected_sha256"]
    result["openings"] = [{"task_id": "43"}, {"task_id": "72"}]
    return result


def test_task95_prescreen_config_and_opening_manifest_are_strictly_bound():
    config = runner.load_json(TASK95_PRESCREEN_CONFIG)
    data = config["data"]
    split_path = runner.REPO_ROOT / data["task_split"]
    openings_path = runner.REPO_ROOT / data["openings"]
    manifest = runner.load_json(runner.REPO_ROOT / data["openings_manifest"])
    openings = runner.load_jsonl(openings_path)

    runner.validate_opening_contract(
        data=data,
        split_sha256=runner.sha256(split_path),
        openings_path=openings_path,
        openings_manifest=manifest,
        openings=openings,
        expected_ordered=["95"],
        strict=True,
    )
    assert config["diagnostic"]["expected_tasks"] == 1
    assert config["diagnostic"]["expected_rollouts"] == 4
    assert openings[0]["task_id"] == "95"
    assert openings[0]["user_seed"] == 350291
    assert openings[0]["hidden_user_scenario_persisted"] is False

    manifest_mutations = {
        "task_ids": ["72"],
        "rows": 2,
        "subset": "rl_train",
        "task_split_path": "data/other_split.json",
        "output_path": "data/other_opening.jsonl",
    }
    for field, value in manifest_mutations.items():
        changed = deepcopy(manifest)
        changed[field] = value
        with pytest.raises(ValueError):
            runner.validate_opening_contract(
                data=data,
                split_sha256=runner.sha256(split_path),
                openings_path=openings_path,
                openings_manifest=changed,
                openings=openings,
                expected_ordered=["95"],
                strict=True,
            )

    duplicate_openings = [openings[0], deepcopy(openings[0])]
    duplicate_manifest = deepcopy(manifest)
    duplicate_manifest["rows"] = 2
    with pytest.raises(ValueError, match="Opening coverage mismatch"):
        runner.validate_opening_contract(
            data=data,
            split_sha256=runner.sha256(split_path),
            openings_path=openings_path,
            openings_manifest=duplicate_manifest,
            openings=duplicate_openings,
            expected_ordered=["95"],
            strict=True,
        )


class ReachedMockTrainer(RuntimeError):
    """Stop launch verification before model loading or training."""


def guarded_launch_doubles(monkeypatch):
    from src.training import rollout_diagnostics as diagnostics

    order = []
    received = {}
    source_binding = {
        "version": "1.9.0",
        "source_sha256": "A" * 64,
        "transformers_version": "5.14.1",
        "transformers_source_sha256": "B" * 64,
    }

    class NativeTrainer:
        def __init__(self, **kwargs):
            received.update(kind="native", kwargs=kwargs)
            order.append("native_trainer")
            raise ReachedMockTrainer("NATIVE_CONSTRUCTOR_ONLY")

    class GuardedTrainer(NativeTrainer):
        def __init__(self, **kwargs):
            received.update(kind="guarded", kwargs=kwargs)
            order.append("guarded_trainer")
            raise ReachedMockTrainer("GUARDED_CONSTRUCTOR_ONLY")

    def verify(trainer_class):
        assert trainer_class is NativeTrainer
        order.append("source_verified")
        return source_binding

    def guard_factory(base_class, emit):
        assert base_class is NativeTrainer
        assert callable(emit)
        order.append("guard_factory")
        emit({"event": "test_guard_setup", "test_double_only": True})
        return GuardedTrainer

    def api_probe(**kwargs):
        order.append("api_probe_mock")
        return {"status": "MOCK_ONLY", "external_api_called": False}

    def grpo_config(**kwargs):
        received["grpo_arguments"] = kwargs
        return NS(**kwargs)

    monkeypatch.setattr(runner, "check_runtime", lambda: {"bf16_supported": True})
    monkeypatch.setattr(runner, "check_tool_template", lambda path: {})
    monkeypatch.setattr(runner, "build_dataset", lambda data: [])
    api = Mock(side_effect=api_probe)
    verify_mock = Mock(side_effect=verify)
    factory_mock = Mock(side_effect=guard_factory)
    monkeypatch.setattr(runner, "probe_user_simulator_api", api)
    monkeypatch.setattr(diagnostics, "verify_trl_source", verify_mock)
    monkeypatch.setattr(diagnostics, "make_guarded_grpo_trainer", factory_mock)
    monkeypatch.setitem(sys.modules, "torch", NS(bfloat16="bf16", float16="fp16"))
    monkeypatch.setitem(
        sys.modules,
        "peft",
        NS(LoraConfig=lambda **kwargs: NS(**kwargs), PeftModel=object),
    )
    bnb = Mock(side_effect=lambda **kwargs: NS(**kwargs))
    monkeypatch.setitem(
        sys.modules,
        "transformers",
        NS(
            AutoTokenizer=object,
            AutoModelForCausalLM=object,
            BitsAndBytesConfig=bnb,
            set_seed=lambda seed: None,
        ),
    )
    monkeypatch.setitem(
        sys.modules, "trl", NS(GRPOConfig=grpo_config, GRPOTrainer=NativeTrainer)
    )
    return NS(
        order=order,
        received=received,
        api=api,
        verify=verify_mock,
        factory=factory_mock,
        bnb=bnb,
        source_binding=source_binding,
    )


def test_generation_safety_is_explicit_and_does_not_mutate_config():
    from src.training.rollout_diagnostics import validate_generation_safety

    legacy = preflight()["config"]
    assert validate_generation_safety(legacy) is False
    enabled = guarded_preflight()["config"]
    before = json.dumps(enabled, sort_keys=True)
    assert validate_generation_safety(enabled) is True
    assert json.dumps(enabled, sort_keys=True) == before


@pytest.mark.parametrize(
    "setting",
    [
        None,
        {},
        "eos_finished_rows_v1",
        {"mode": "unknown"},
        {"mode": "eos_finished_rows_v1", "repair_active_logits": True},
    ],
)
def test_unknown_generation_safety_fails_before_api(tmp_path, monkeypatch, setting):
    doubles = guarded_launch_doubles(monkeypatch)
    inputs = guarded_preflight()
    inputs["config"]["generation_safety"] = setting
    with pytest.raises(ValueError):
        runner.run(inputs, tmp_path / "rejected")
    doubles.api.assert_not_called()
    doubles.verify.assert_not_called()
    doubles.factory.assert_not_called()


@pytest.mark.parametrize(
    "mode,use_vllm",
    [("ROLLOUT_DIAGNOSTIC", True), ("OPTIMIZE", True), ("unknown", False)],
)
def test_generation_safety_rejects_unsupported_runtime(mode, use_vllm):
    from src.training.rollout_diagnostics import validate_generation_safety

    config = guarded_preflight()["config"]
    config["execution_mode"] = mode
    config["grpo"]["use_vllm"] = use_vllm
    with pytest.raises(ValueError):
        validate_generation_safety(config)


def test_guard_source_hash_failure_stops_before_api(tmp_path, monkeypatch):
    doubles = guarded_launch_doubles(monkeypatch)
    doubles.verify.side_effect = RuntimeError("AUDITED_SOURCE_HASH_MISMATCH")
    with pytest.raises(RuntimeError, match="AUDITED_SOURCE_HASH_MISMATCH"):
        runner.run(guarded_preflight(), tmp_path / "source_rejected")
    doubles.verify.assert_called_once()
    doubles.api.assert_not_called()
    doubles.factory.assert_not_called()
    assert "kind" not in doubles.received


def test_optimize_selects_guard_and_preserves_kl_arguments(tmp_path, monkeypatch):
    doubles = guarded_launch_doubles(monkeypatch)
    output = tmp_path / "guarded"
    inputs = guarded_preflight()
    with pytest.raises(ReachedMockTrainer, match="GUARDED_CONSTRUCTOR_ONLY"):
        runner.run(inputs, output)
    assert doubles.received["kind"] == "guarded"
    doubles.verify.assert_called_once()
    doubles.factory.assert_called_once()
    doubles.api.assert_called_once()
    assert doubles.order.index("source_verified") < doubles.order.index(
        "api_probe_mock"
    )
    grpo_args = doubles.received["grpo_arguments"]
    assert grpo_args["beta"] == 0.02
    assert grpo_args["learning_rate"] == 5e-6
    assert grpo_args["max_steps"] == 2
    assert grpo_args["num_generations"] == 2
    assert grpo_args["per_device_train_batch_size"] == 1
    assert grpo_args["gradient_accumulation_steps"] == 2
    assert grpo_args["steps_per_generation"] == 2
    assert (
        grpo_args["max_completion_length"]
        == inputs["config"]["grpo"]["max_completion_length"]
    )
    assert grpo_args["save_strategy"] == "steps"
    assert doubles.received["kwargs"]["args"].beta == 0.02
    assert doubles.received["kwargs"]["reward_funcs"] == []
    events = [
        json.loads(line)
        for line in (output / "generation_events.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert {"event": "test_guard_setup", "test_double_only": True} in events
    runtime = runner.load_json(output / "environment.json")
    runtime_text = json.dumps(runtime)
    assert doubles.source_binding["source_sha256"] in runtime_text
    assert doubles.source_binding["transformers_source_sha256"] in runtime_text


def test_bf16_closure_loads_explicit_dtype_without_bitsandbytes(
    tmp_path, monkeypatch
):
    doubles = guarded_launch_doubles(monkeypatch)
    inputs = bf16_closure_preflight()
    with pytest.raises(ReachedMockTrainer, match="GUARDED_CONSTRUCTOR_ONLY"):
        runner.run(inputs, tmp_path / "bf16_closure")
    grpo_args = doubles.received["grpo_arguments"]
    assert grpo_args["model_init_kwargs"] == {"dtype": "bf16"}
    assert grpo_args["steps_per_generation"] == 2
    assert grpo_args["max_steps"] == 2
    assert grpo_args["max_completion_length"] == 1024
    assert grpo_args["beta"] == 0.02
    doubles.bnb.assert_not_called()


def test_bf16_closure_contract_is_bound_before_launch():
    config = runner.load_json(BF16_CLOSURE_CONFIG)
    contract = runner.validate_optimization_contract(
        config, selected_task_count=2
    )
    assert contract == {
        "status": "VALIDATED",
        "generation_batch_size": 2,
        "steps_per_generation": 2,
        "expected_optimizer_steps": 2,
        "expected_groups": 2,
        "expected_rollouts": 4,
        "configured_task_pool_size": 2,
        "kl_reference_required": True,
    }


def test_task43_terminal_closure_binds_single_task_reward_kl_and_length():
    from src.rl.retail_agentic_env import TERMINAL_ONLY_REWARD_CONFIG

    config = runner.load_json(TASK43_TERMINAL_CLOSURE_CONFIG)
    assert config["data"]["task_ids"] == ["43"]
    assert config["reward"] == TERMINAL_ONLY_REWARD_CONFIG
    assert config["grpo"]["max_completion_length"] == 4096
    assert config["grpo"]["beta"] == 0.02
    assert config["claims"]["generalization_claim_allowed"] is False
    assert config["claims"]["terminal_reward_is_db_state_only"] is True
    assert runner.validate_optimization_contract(config, selected_task_count=1) == {
        "status": "VALIDATED",
        "generation_batch_size": 2,
        "steps_per_generation": 2,
        "expected_optimizer_steps": 2,
        "expected_groups": 2,
        "expected_rollouts": 4,
        "configured_task_pool_size": 1,
        "kl_reference_required": True,
    }


def test_task43_terminal_closure_n4_preserves_one_group_per_optimizer_step():
    baseline = runner.load_json(TASK43_TERMINAL_CLOSURE_CONFIG)
    config = runner.load_json(TASK43_TERMINAL_CLOSURE_N4_CONFIG)
    for section in ("claims", "upstream", "model", "data", "lora", "rollout", "reward"):
        assert config[section] == baseline[section], section
    for field in (
        "max_steps",
        "learning_rate",
        "per_device_train_batch_size",
        "max_completion_length",
        "temperature",
        "beta",
        "loss_type",
        "use_vllm",
    ):
        assert config["grpo"][field] == baseline["grpo"][field], field
    assert config["grpo"]["num_generations"] == 4
    assert config["grpo"]["steps_per_generation"] == 4
    assert config["grpo"]["gradient_accumulation_steps"] == 4
    assert runner.validate_optimization_contract(config, selected_task_count=1) == {
        "status": "VALIDATED",
        "generation_batch_size": 4,
        "steps_per_generation": 4,
        "expected_optimizer_steps": 2,
        "expected_groups": 2,
        "expected_rollouts": 8,
        "configured_task_pool_size": 1,
        "kl_reference_required": True,
    }


def test_task43_terminal_sampling_gate_matches_rollout_contract():
    from src.training.rollout_diagnostics import validate_sampling_request

    optimize = runner.load_json(TASK43_TERMINAL_CLOSURE_CONFIG)
    sampling = runner.load_json(TASK43_TERMINAL_SAMPLING_CONFIG)
    for section in ("model", "data", "rollout", "reward", "upstream"):
        assert sampling[section] == optimize[section], section
    for field in (
        "num_generations",
        "max_completion_length",
        "temperature",
        "use_vllm",
    ):
        assert sampling["grpo"][field] == optimize["grpo"][field], field
    assert sampling["execution_mode"] == "ROLLOUT_DIAGNOSTIC"
    assert sampling["grpo"]["max_steps"] == 1
    assert sampling["grpo"]["learning_rate"] == 0.0
    assert sampling["grpo"]["beta"] == 0.0
    assert validate_sampling_request(sampling, 4096, 1) == {
        "mode": "STOCHASTIC_GROUP_SAMPLING",
        "do_sample": True,
        "actual_num_generations": 2,
        "trl_constructor_num_generations": 2,
        "groups_per_task": 1,
        "trainer_max_steps_unused": True,
    }


def test_task43_adaptive_quality_n4_is_single_group_pure_sampling():
    from src.training.rollout_diagnostics import validate_sampling_request

    config = runner.load_json(TASK43_ADAPTIVE_QUALITY_N4_CONFIG)
    assert config["execution_mode"] == "ROLLOUT_DIAGNOSTIC"
    assert config["claims"]["parameter_update_allowed"] is False
    assert config["diagnostic"]["weight_update_expected"] is False
    assert config["diagnostic"]["expected_rollouts"] == 4
    assert config["diagnostic"]["groups_per_task"] == 1
    assert config["grpo"]["num_generations"] == 4
    assert config["grpo"]["learning_rate"] == 0.0
    assert config["grpo"]["beta"] == 0.0
    assert runner.validate_model_loading(config) == "qwen3_bf16_inference_v1"
    assert validate_sampling_request(config, 4096, 1) == (
        ADAPTIVE_QUALITY_N4_SAMPLING_CONTRACT
    )
    with pytest.raises(ValueError, match="differs from frozen task shape"):
        validate_sampling_request(config, 4096, 2)


def test_task43_three_arm_n4_evaluation_changes_only_model_identity_and_claims():
    from src.training.rollout_diagnostics import validate_sampling_request

    paths = (
        TASK43_ADAPTIVE_QUALITY_N4_CONFIG,
        TASK43_TERMINAL_N4_POST_EVAL_CONFIG,
        TASK43_STAGED_V4_POST_EVAL_CONFIG,
    )
    configs = [runner.load_json(path) for path in paths]
    frozen_sections = (
        "upstream",
        "data",
        "diagnostic",
        "sampling",
        "seed",
        "precision",
        "quantization",
        "model_loading",
        "lora",
        "grpo",
        "rollout",
        "reward",
        "generation_safety",
        "engineering_acceptance",
    )
    for candidate in configs[1:]:
        for section in frozen_sections:
            assert candidate[section] == configs[0][section], section
        assert validate_sampling_request(candidate, 4096, 1) == (
            ADAPTIVE_QUALITY_N4_SAMPLING_CONTRACT
        )
    assert len({config["model"]["name_or_path"] for config in configs}) == 3
    assert len({config["model"]["expected_sha256"] for config in configs}) == 3
    assert configs[1]["claims"]["comparison_arm"] == "TERMINAL_N4"
    assert (
        configs[2]["claims"]["comparison_arm"]
        == "STAGED_CONFIRMATION_V4_N4"
    )


def test_task43_post_grpo_eval_changes_only_model_identity():
    from src.training.rollout_diagnostics import validate_sampling_request

    before = runner.load_json(TASK43_TERMINAL_SAMPLING_CONFIG)
    after = runner.load_json(TASK43_TERMINAL_POST_GRPO_EVAL_CONFIG)
    for section in (
        "upstream",
        "data",
        "diagnostic",
        "seed",
        "precision",
        "quantization",
        "model_loading",
        "lora",
        "grpo",
        "rollout",
        "reward",
        "generation_safety",
    ):
        assert after[section] == before[section], section
    assert before["model"]["name_or_path"] != after["model"]["name_or_path"]
    assert before["model"]["expected_sha256"] != after["model"]["expected_sha256"]
    assert after["model"]["source_stage"] == "TERMINAL_GRPO_2STEP_TASK43"
    assert after["claims"]["post_grpo_evaluation"] is True
    assert validate_sampling_request(after, 4096, 1)["actual_num_generations"] == 2


def test_task43_terminal_closure_launch_wiring(tmp_path, monkeypatch):
    doubles = guarded_launch_doubles(monkeypatch)
    with pytest.raises(ReachedMockTrainer, match="GUARDED_CONSTRUCTOR_ONLY"):
        runner.run(
            task43_terminal_closure_preflight(),
            tmp_path / "task43_terminal_closure",
        )
    grpo_args = doubles.received["grpo_arguments"]
    assert grpo_args["model_init_kwargs"] == {"dtype": "bf16"}
    assert grpo_args["max_steps"] == 2
    assert grpo_args["num_generations"] == 2
    assert grpo_args["max_completion_length"] == 4096
    assert grpo_args["beta"] == 0.02
    doubles.bnb.assert_not_called()


@pytest.mark.parametrize(
    "mutate,match",
    [
        (
            lambda config: config["grpo"].update(steps_per_generation=1),
            "must be divisible by num_generations",
        ),
        (
            lambda config: config["engineering_acceptance"].update(
                expected_rollouts=2
            ),
            "expected_rollouts",
        ),
        (
            lambda config: config["grpo"].update(beta=0.0),
            "beta > 0",
        ),
    ],
)
def test_bf16_closure_rejects_contract_drift_before_api(mutate, match):
    config = runner.load_json(BF16_CLOSURE_CONFIG)
    mutate(config)
    with pytest.raises(ValueError, match=match):
        runner.validate_optimization_contract(config, selected_task_count=2)


def test_bf16_closure_changes_only_stability_variables():
    original = runner.load_json(
        runner.REPO_ROOT
        / "configs"
        / "retail_agentic_qwen3_4b_staged_reward_minimal_grpo_tasks43_72_v2.json"
    )
    closure = runner.load_json(BF16_CLOSURE_CONFIG)
    for section in ("upstream", "model", "data", "lora", "rollout", "reward"):
        assert closure[section] == original[section], section
    assert closure["seed"] == original["seed"]
    assert closure["precision"] == original["precision"]
    assert closure["generation_safety"] == original["generation_safety"]
    original_grpo = dict(original["grpo"])
    closure_grpo = dict(closure["grpo"])
    assert closure_grpo.pop("steps_per_generation") == 2
    assert closure_grpo.pop("max_completion_length") == 1024
    assert original_grpo.pop("max_completion_length") == 13312
    assert closure_grpo == original_grpo
    assert original["quantization"]["enabled"] is True
    assert closure["quantization"] == {"enabled": False}
    assert "activation_precision" in original
    assert "activation_precision" not in closure


def test_legacy_launch_does_not_silently_enable_generation_guard(tmp_path, monkeypatch):
    doubles = guarded_launch_doubles(monkeypatch)
    with pytest.raises(ReachedMockTrainer, match="NATIVE_CONSTRUCTOR_ONLY"):
        runner.run(preflight(), tmp_path / "legacy")
    doubles.verify.assert_not_called()
    doubles.factory.assert_not_called()
    doubles.api.assert_called_once()
    assert doubles.received["kind"] == "native"


def precision_preflight():
    inputs = guarded_preflight()
    inputs["config"]["activation_precision"] = {"mode": "qwen3_nf4_bf16_outputs_v1"}
    inputs["config"]["quantization"] = {"enabled": True, "mode": "4bit_nf4"}
    inputs["config"]["precision"] = "bf16"
    return inputs


@pytest.mark.parametrize(
    "setting",
    [
        None,
        {},
        "bf16",
        {"mode": "unknown"},
        {"mode": "qwen3_nf4_bf16_outputs_v1", "detach": True},
    ],
)
def test_precision_rejects_invalid_opt_in_before_api(tmp_path, monkeypatch, setting):
    doubles = guarded_launch_doubles(monkeypatch)
    inputs = precision_preflight()
    inputs["config"]["activation_precision"] = setting
    with pytest.raises(ValueError, match="activation_precision"):
        runner.run(inputs, tmp_path / "rejected_precision")
    doubles.api.assert_not_called()


def test_precision_requires_real_bf16_support_before_api(tmp_path, monkeypatch):
    doubles = guarded_launch_doubles(monkeypatch)
    monkeypatch.setattr(runner, "check_runtime", lambda: {"bf16_supported": False})
    with pytest.raises(RuntimeError, match="actual BF16 support"):
        runner.run(precision_preflight(), tmp_path / "no_bf16")
    doubles.api.assert_not_called()


def test_precision_wraps_entire_train_and_reference_and_cleans_up(
    tmp_path, monkeypatch
):
    from contextlib import contextmanager
    from src.training import rollout_diagnostics as diagnostics

    doubles = guarded_launch_doubles(monkeypatch)
    phases = []
    actor, reference = object(), object()

    class TrainingSentinel:
        def __init__(self, **kwargs):
            self.model, self.ref_model = actor, reference

        def save_model(self, path):
            phases.append("starting_adapter")

        def train(self):
            assert phases[-1] == "precision_enter"
            phases.append("native_train_including_backward")
            raise ReachedMockTrainer("NATIVE_TRAIN_SENTINEL")

    @contextmanager
    def precision_context(models, emit):
        assert models == (actor, reference)
        assert callable(emit)
        phases.append("precision_enter")
        try:
            yield
        finally:
            phases.append("precision_exit")

    factory = Mock(return_value=TrainingSentinel)
    monkeypatch.setattr(
        runner,
        "adapter_weights_artifact",
        lambda path: {"path": str(path / "adapter_model.safetensors"), "sha256": "A" * 64},
    )
    monkeypatch.setattr(
        runner,
        "trainable_parameter_fingerprint",
        lambda model: {
            "sha256": "B" * 64,
            "tensor_count": 1,
            "numel": 1,
            "all_finite": True,
            "nonfinite_count": 0,
        },
    )
    monkeypatch.setattr(diagnostics, "make_guarded_grpo_trainer", factory)
    monkeypatch.setattr(
        diagnostics, "qwen3_nf4_bf16_activation_context", precision_context
    )
    with pytest.raises(ReachedMockTrainer, match="NATIVE_TRAIN_SENTINEL"):
        runner.run(precision_preflight(), tmp_path / "train_precision")
    assert factory.call_args.kwargs == {"bf16_generation": True}
    assert phases == [
        "starting_adapter",
        "precision_enter",
        "native_train_including_backward",
        "precision_exit",
    ]
    assert doubles.received["grpo_arguments"]["beta"] == 0.02


def test_precision_retry_changes_only_activation_policy():
    root = Path(__file__).resolve().parents[1]
    names = [
        root
        / f"configs/retail_agentic_qwen3_4b_staged_reward_minimal_grpo_tasks43_72_v{v}.json"
        for v in (1, 2)
    ]
    original, retry = [runner.load_json(path) for path in names]
    assert retry.pop("activation_precision") == {"mode": "qwen3_nf4_bf16_outputs_v1"}
    assert retry == original


def test_sampling_precision_does_not_precreate_generation_log(tmp_path, monkeypatch):
    from contextlib import contextmanager
    from src.training import rollout_diagnostics as diagnostics

    guarded_launch_doubles(monkeypatch)
    inputs = precision_preflight()
    config = inputs["config"]
    config.pop("generation_safety")
    config["execution_mode"] = "ROLLOUT_DIAGNOSTIC"
    config["grpo"].update(learning_rate=0, beta=0)
    config["diagnostic"] = {}
    validate_sampling = Mock(return_value=dict(STOCHASTIC_SAMPLING_CONTRACT))
    monkeypatch.setattr(diagnostics, "validate_sampling_request", validate_sampling)
    trainer = NS(
        model=object(),
        ref_model=None,
        generation_config=NS(do_sample=True),
        generation_kwargs={"do_sample": True},
        num_generations=2,
    )
    monkeypatch.setattr(
        diagnostics, "make_sampling_trainer", lambda *a, **kw: lambda **k: trainer
    )

    @contextmanager
    def precision(models, emit):
        emit({"event": "enter"})
        try:
            yield
        finally:
            emit({"event": "exit"})

    def sample(trainer, dataset, config, preflight, output, *args):
        assert not (output / "generation_events.jsonl").exists()
        runner.save_json(output / "run_state.json", {"status": "COMPLETED"})
        return {"status": "COMPLETED", "artifacts": {}}

    monkeypatch.setattr(diagnostics, "qwen3_nf4_bf16_activation_context", precision)
    monkeypatch.setattr(diagnostics, "run_pure_sampling", sample)
    output = tmp_path / "precision_sampling"
    manifest = runner.run(inputs, output, sample_only=True, completion_budget=13312)
    validate_sampling.assert_called_once()
    events = [
        json.loads(s)
        for s in (output / "precision_events.jsonl").read_text().splitlines()
    ]
    assert events == [{"event": "enter"}, {"event": "exit"}]
    assert manifest["artifacts"]["precision_events.jsonl"]["sha256"] == runner.sha256(
        output / "precision_events.jsonl"
    )
    assert runner.load_json(output / "run_state.json")[
        "run_manifest_sha256"
    ] == runner.sha256(output / "run_manifest.json")


def test_s6_greedy_config_binds_exact_checkpoint_and_pass_at_one():
    from src.training.rollout_diagnostics import validate_sampling_request

    config = runner.load_json(S6_GREEDY_CONFIG)
    baseline = preflight()["config"]
    assert config["model"] == baseline["model"]
    assert config["model"]["name_or_path"].endswith("/teacher_sft_merged")
    assert config["execution_mode"] == "ROLLOUT_DIAGNOSTIC"
    assert config["claims"]["parameter_update_allowed"] is False
    assert config["diagnostic"]["weight_update_expected"] is False
    assert config["model_loading"] == {"mode": "qwen3_bf16_inference_v1"}
    assert config["quantization"] == {"enabled": False}
    assert config["sampling"] == {"mode": "TRUE_GREEDY", "do_sample": False}
    assert config["grpo"]["num_generations"] == 1
    assert config["grpo"]["learning_rate"] == 0
    assert config["grpo"]["beta"] == 0
    assert runner.validate_model_loading(config) == "qwen3_bf16_inference_v1"
    assert validate_sampling_request(config, 13312, 1) == (
        TRUE_GREEDY_SAMPLING_CONTRACT
    )


def test_s7_stochastic_config_changes_only_decode_and_sample_count_from_s6():
    from src.training.rollout_diagnostics import validate_sampling_request

    greedy = runner.load_json(S6_GREEDY_CONFIG)
    stochastic = runner.load_json(S7_STOCHASTIC_CONFIG)
    for section in (
        "model",
        "data",
        "rollout",
        "reward",
        "upstream",
        "lora",
        "quantization",
        "model_loading",
        "generation_safety",
    ):
        assert stochastic[section] == greedy[section], section
    assert stochastic["seed"] == greedy["seed"]
    assert stochastic["precision"] == greedy["precision"]
    assert stochastic["grpo"]["learning_rate"] == 0
    assert stochastic["grpo"]["beta"] == 0
    assert stochastic["grpo"]["max_completion_length"] == 13312
    assert stochastic["diagnostic"]["expected_rollouts"] == 8
    assert stochastic["diagnostic"]["expected_rollouts_per_task"] == 4
    assert validate_sampling_request(stochastic, 13312, 2) == (
        S7_STOCHASTIC_SAMPLING_CONTRACT
    )


@pytest.mark.parametrize(
    (
        "inputs_factory",
        "groups_per_task",
        "expected_contract",
        "expected_generation_kwargs",
        "effective_num_generations",
        "expected_do_sample",
        "output_name",
    ),
    [
        (
            s6_greedy_preflight,
            1,
            TRUE_GREEDY_SAMPLING_CONTRACT,
            {
                "do_sample": False,
                "num_beams": 1,
                "num_return_sequences": 1,
            },
            1,
            False,
            "s6_greedy",
        ),
        (
            s7_stochastic_preflight,
            2,
            S7_STOCHASTIC_SAMPLING_CONTRACT,
            {
                "do_sample": True,
                "temperature": 0.8,
                "top_p": 1.0,
                "top_k": 0,
                "num_beams": 1,
                "num_return_sequences": 1,
            },
            2,
            True,
            "s7_stochastic",
        ),
    ],
)
def test_sampling_runner_uses_exact_checkpoint_without_peft_and_never_trains(
    tmp_path,
    monkeypatch,
    inputs_factory,
    groups_per_task,
    expected_contract,
    expected_generation_kwargs,
    effective_num_generations,
    expected_do_sample,
    output_name,
):
    from src.training import rollout_diagnostics as diagnostics

    inputs = inputs_factory()
    received = {}
    source_binding = {
        "version": "1.9.0",
        "source_sha256": "A" * 64,
        "transformers_version": "5.14.1",
        "transformers_source_sha256": "B" * 64,
    }

    class NativeTrainer:
        pass

    class MergedCheckpointModel:
        pass

    class SamplingTrainer:
        def __init__(self, **kwargs):
            received["trainer_kwargs"] = kwargs
            self.model = MergedCheckpointModel()
            self.ref_model = None
            self.generation_config = NS(**kwargs["args"].generation_kwargs)
            self.generation_kwargs = dict(kwargs["args"].generation_kwargs)
            self.num_generations = kwargs["args"].num_generations
            self.args = kwargs["args"]
            self.train = Mock(side_effect=AssertionError("training is forbidden"))
            self.save_model = Mock()
            received["trainer"] = self

    def grpo_config(**kwargs):
        received["grpo_arguments"] = kwargs
        return NS(**kwargs)

    def sampling_factory(base_class, **kwargs):
        assert base_class is NativeTrainer
        received["sampling_factory_kwargs"] = kwargs
        return SamplingTrainer

    def pure_sampling(trainer, dataset, config, preflight, output, groups, runtime):
        received["runtime"] = runtime
        assert trainer.num_generations == effective_num_generations
        assert trainer.generation_config.do_sample is expected_do_sample
        assert trainer.generation_kwargs == expected_generation_kwargs
        assert config["diagnostic"]["expected_rollouts"] == (
            2 * groups_per_task * effective_num_generations
        )
        return {"status": "COMPLETED", "artifacts": {}}

    lora_config = Mock(side_effect=lambda **kwargs: NS(**kwargs))
    bitsandbytes_config = Mock(side_effect=lambda **kwargs: NS(**kwargs))
    verify_source = Mock(return_value=source_binding)
    monkeypatch.setattr(runner, "check_runtime", lambda: {"bf16_supported": True})
    monkeypatch.setattr(runner, "check_tool_template", lambda path: {})
    monkeypatch.setattr(runner, "build_dataset", lambda data: data["openings"])
    monkeypatch.setattr(
        runner,
        "probe_user_simulator_api",
        lambda **kwargs: {"status": "MOCK_ONLY", "external_api_called": False},
    )
    monkeypatch.setattr(
        runner,
        "adapter_weights_artifact",
        lambda path: {"path": str(path / "unused"), "sha256": "C" * 64},
    )
    monkeypatch.setattr(
        runner,
        "trainable_parameter_fingerprint",
        lambda model: {
            "sha256": "D" * 64,
            "tensor_count": 0,
            "numel": 0,
            "all_finite": True,
            "nonfinite_count": 0,
        },
    )
    monkeypatch.setattr(diagnostics, "verify_trl_source", verify_source)
    monkeypatch.setattr(diagnostics, "make_sampling_trainer", sampling_factory)
    monkeypatch.setattr(diagnostics, "run_pure_sampling", pure_sampling)
    monkeypatch.setitem(sys.modules, "torch", NS(bfloat16="bf16", float16="fp16"))
    monkeypatch.setitem(
        sys.modules,
        "peft",
        NS(LoraConfig=lora_config, PeftModel=object),
    )
    monkeypatch.setitem(
        sys.modules,
        "transformers",
        NS(
            AutoTokenizer=object,
            AutoModelForCausalLM=object,
            BitsAndBytesConfig=bitsandbytes_config,
            set_seed=lambda seed: None,
        ),
    )
    monkeypatch.setitem(
        sys.modules,
        "trl",
        NS(GRPOConfig=grpo_config, GRPOTrainer=NativeTrainer),
    )

    result = runner.run(
        inputs,
        tmp_path / output_name,
        sample_only=True,
        completion_budget=13312,
        groups_per_task=groups_per_task,
    )

    assert result["status"] == "COMPLETED"
    assert received["sampling_factory_kwargs"] == {
        "expected_do_sample": expected_do_sample
    }
    assert received["grpo_arguments"]["num_generations"] == 2
    assert received["grpo_arguments"]["steps_per_generation"] == (
        2
        if not expected_do_sample
        else inputs["config"]["grpo"]["steps_per_generation"]
    )
    assert received["grpo_arguments"]["generation_kwargs"] == (
        expected_generation_kwargs
    )
    assert received["grpo_arguments"]["save_strategy"] == "no"
    assert received["trainer_kwargs"]["model"] == inputs["model_path"]
    assert received["trainer_kwargs"]["peft_config"] is None
    assert not hasattr(received["trainer"].model, "peft_config")
    assert received["trainer"].num_generations == effective_num_generations
    assert received["trainer"].args.num_generations == effective_num_generations
    received["trainer"].train.assert_not_called()
    received["trainer"].save_model.assert_not_called()
    lora_config.assert_not_called()
    bitsandbytes_config.assert_not_called()
    sampling_adapter = received["runtime"]["sampling_adapter"]
    assert sampling_adapter["direct_merged_checkpoint_inference"] is True
    assert sampling_adapter["decode_contract"] == {
        **expected_contract,
        "effective_do_sample": expected_do_sample,
        "effective_num_generations": effective_num_generations,
        **(
            {
                "effective_temperature": expected_contract["temperature"],
                "effective_top_p": expected_contract["top_p"],
                "effective_top_k": expected_contract["top_k"],
            }
            if expected_do_sample and "temperature" in expected_contract
            else {}
        ),
        "temperature_is_decode_authority": expected_do_sample,
    }
    assert received["runtime"]["model_loading"] == {
        "mode": "qwen3_bf16_inference_v1",
        "base_weight_dtype": "torch.bfloat16",
        "quantized": False,
        "peft_adapter_applied": False,
    }


def test_minimal_grpo_keeps_frozen_task_reward_and_checkpoint():
    config = runner.load_json(MINIMAL_GRPO_CONFIG)
    baseline = preflight()["config"]
    for section in (
        "model",
        "data",
        "rollout",
        "reward",
        "upstream",
        "lora",
        "quantization",
    ):
        assert config[section] == baseline[section], section
    assert config["seed"] == baseline["seed"]
    assert config["precision"] == baseline["precision"]
    assert config["execution_mode"] == "OPTIMIZE"
    assert config["generation_safety"] == {"mode": "eos_finished_rows_v1"}
    assert "diagnostic" not in config
    assert config["grpo"]["max_steps"] == 2
    assert config["grpo"]["num_generations"] == 2
    assert config["grpo"]["learning_rate"] == 5e-6
    assert config["grpo"]["beta"] == 0.02
    # Completion budget is separately chosen by the project owner. This test
    # freezes the experiment identity, not an unapproved budget assumption.
    assert type(config["grpo"]["max_completion_length"]) is int
    assert config["grpo"]["max_completion_length"] > 0


def test_minimal_grpo_uses_real_derived_manifest_binding(monkeypatch):
    monkeypatch.setattr(runner, "validate_upstream_checkout", lambda *args: {})
    result = runner.validate_config_and_split(MINIMAL_GRPO_CONFIG)
    binding = result["sft_manifest_binding"]
    model = result["config"]["model"]
    assert binding["binding_type"] == "DERIVED"
    assert (
        binding["training_data_manifest_sha256"]
        == model["training_data_manifest_sha256"]
    )
    assert (
        binding["source_manifest_sha256"]
        == result["split"]["source"]["sft_data_manifest_sha256"]
    )
    assert runner.selected_task_ids(result["config"], result["split"]) == ["43", "72"]


def test_minimal_grpo_rejects_tampered_derived_manifest_binding():
    config = runner.load_json(MINIMAL_GRPO_CONFIG)
    split = runner.load_json(runner.REPO_ROOT / config["data"]["task_split"])
    config["model"]["training_data_manifest_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="Derived SFT training manifest hash mismatch"):
        runner.validate_sft_manifest_binding(config["model"], split)
