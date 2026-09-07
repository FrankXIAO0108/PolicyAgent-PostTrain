"""Frozen Task 88 prescreen and 50-step GRPO contract tests."""

from pathlib import Path

from src.training import run_retail_agentic_grpo as runner
from src.training.rollout_diagnostics import validate_sampling_request


ROOT = Path(__file__).resolve().parents[1]
PRESCREEN = (
    ROOT / "configs" / "retail_agentic_qwen3_4b_task88_terminal_prescreen_n4_v1.json"
)
PRESCREEN_V2 = (
    ROOT / "configs" / "retail_agentic_qwen3_4b_task88_terminal_prescreen_n4_v2.json"
)
LONGRUN = (
    ROOT / "configs" / "retail_agentic_qwen3_4b_task88_terminal_grpo_50step_n4_v1.json"
)


def test_task88_prescreen_is_one_non_updating_n4_group():
    config = runner.load_json(PRESCREEN)
    assert config["data"]["task_ids"] == ["88"]
    assert config["claims"]["single_sft_exposed_train_task"] is True
    assert config["grpo"]["learning_rate"] == 0.0
    assert config["grpo"]["beta"] == 0.0
    assert validate_sampling_request(config, 5120, 1) == {
        "mode": "STOCHASTIC_GROUP_SAMPLING",
        "contract_version": "fixed-n4-single-group-v1",
        "do_sample": True,
        "temperature": 0.8,
        "top_p": 1.0,
        "top_k": 0,
        "actual_num_generations": 4,
        "trl_constructor_num_generations": 4,
        "groups_per_task": 1,
        "trainer_max_steps_unused": True,
    }


def test_task88_longrun_changes_training_fields_not_task_protocol():
    prescreen = runner.load_json(PRESCREEN)
    longrun = runner.load_json(LONGRUN)
    for section in (
        "upstream",
        "model",
        "data",
        "lora",
        "rollout",
        "reward",
        "generation_safety",
    ):
        assert longrun[section] == prescreen[section], section
    assert longrun["grpo"]["max_steps"] == 50
    assert longrun["grpo"]["num_generations"] == 4
    assert longrun["grpo"]["beta"] == 0.02
    assert longrun["grpo"]["max_completion_length"] == 5120
    assert runner.validate_optimization_contract(longrun, selected_task_count=1) == {
        "status": "VALIDATED",
        "generation_batch_size": 4,
        "steps_per_generation": 4,
        "expected_optimizer_steps": 50,
        "expected_groups": 50,
        "expected_rollouts": 200,
        "configured_task_pool_size": 1,
        "kl_reference_required": True,
    }


def test_task88_prescreen_v2_changes_only_completion_budget_and_claim():
    v1 = runner.load_json(PRESCREEN)
    v2 = runner.load_json(PRESCREEN_V2)
    for section in (
        "upstream",
        "model",
        "data",
        "seed",
        "precision",
        "quantization",
        "model_loading",
        "lora",
        "sampling",
        "diagnostic",
        "rollout",
        "reward",
        "generation_safety",
    ):
        assert v2[section] == v1[section], section
    assert v1["grpo"]["max_completion_length"] == 5120
    assert v2["grpo"]["max_completion_length"] == 8192
    assert v2["claims"]["completion_budget_increased_after_v1_transport_failure"]
    assert validate_sampling_request(v2, 8192, 1)["actual_num_generations"] == 4
