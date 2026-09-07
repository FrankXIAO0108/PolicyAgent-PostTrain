"""Contract checks for the Task 44 Terminal-GRPO 50-step extension."""

from pathlib import Path

import pytest

from src.training import run_retail_agentic_grpo as runner


ROOT = Path(__file__).resolve().parents[1]
BASELINE = (
    ROOT
    / "configs"
    / "retail_agentic_qwen3_4b_task44_reward_ab_terminal_n4_10step_v1.json"
)
LONGRUN = (
    ROOT
    / "configs"
    / "retail_agentic_qwen3_4b_task44_reward_ab_terminal_n4_50step_v1.json"
)
RESUME = (
    ROOT
    / "configs"
    / "retail_agentic_qwen3_4b_task44_reward_ab_terminal_n4_50step_c8192_resume_v1.json"
)
RESUME_TOOL_LIMIT = (
    ROOT
    / "configs"
    / "retail_agentic_qwen3_4b_task44_reward_ab_terminal_n4_50step_c8192_resume20_toollimit0_v1.json"
)


def _diff_paths(left, right, prefix=""):
    if isinstance(left, dict) and isinstance(right, dict):
        paths = set()
        for key in left.keys() | right.keys():
            child = f"{prefix}.{key}" if prefix else key
            paths.update(_diff_paths(left.get(key), right.get(key), child))
        return paths
    return set() if left == right else {prefix}


def test_task44_50step_changes_only_step_accounting_fields():
    baseline = runner.load_json(BASELINE)
    longrun = runner.load_json(LONGRUN)

    assert _diff_paths(baseline, longrun) == {
        "grpo.max_steps",
        "grpo.save_steps",
        "engineering_acceptance.expected_optimizer_steps",
        "engineering_acceptance.expected_rollouts",
        "engineering_acceptance.expected_groups",
    }

    assert longrun["grpo"]["max_steps"] == 50
    assert longrun["grpo"]["num_generations"] == 4
    assert longrun["grpo"]["temperature"] == 0.8
    assert longrun["grpo"]["learning_rate"] == 5e-6
    assert longrun["grpo"]["beta"] == 0.02
    assert longrun["reward"] == baseline["reward"]
    assert longrun["model"] == baseline["model"]
    assert longrun["data"] == baseline["data"]


def test_task44_50step_optimization_contract():
    config = runner.load_json(LONGRUN)
    validated = runner.validate_config_and_split(LONGRUN)
    contract = runner.validate_optimization_contract(
        validated["config"], selected_task_count=1
    )

    assert config["data"]["task_ids"] == ["44"]
    assert contract == {
        "status": "VALIDATED",
        "generation_batch_size": 4,
        "steps_per_generation": 4,
        "expected_optimizer_steps": 50,
        "expected_groups": 50,
        "expected_rollouts": 200,
        "configured_task_pool_size": 1,
        "kl_reference_required": True,
    }


def test_task44_resume_changes_only_completion_budget():
    longrun = runner.load_json(LONGRUN)
    resumed = runner.load_json(RESUME)

    assert _diff_paths(longrun, resumed) == {"grpo.max_completion_length"}
    assert longrun["grpo"]["max_completion_length"] == 5120
    assert resumed["grpo"]["max_completion_length"] == 8192
    assert resumed["reward"] == longrun["reward"]
    assert resumed["grpo"]["beta"] == 0.02
    assert runner.validate_optimization_contract(
        resumed, selected_task_count=1
    )["expected_rollouts"] == 200


def test_resume_checkpoint_rejects_incomplete_directory(tmp_path):
    checkpoint = tmp_path / "run" / "trainer" / "checkpoint-10"
    checkpoint.mkdir(parents=True)

    try:
        runner.validate_resume_checkpoint(checkpoint, runner.load_json(RESUME))
    except ValueError as exc:
        assert "incomplete" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("Incomplete resume checkpoint was accepted")


def test_task44_resume20_changes_only_tool_iteration_failure_policy():
    resumed = runner.load_json(RESUME)
    resume20 = runner.load_json(RESUME_TOOL_LIMIT)

    assert _diff_paths(resumed, resume20) == {
        "engineering_acceptance.tool_iteration_limit_as_terminal_failure"
    }
    assert resume20["engineering_acceptance"][
        "tool_iteration_limit_as_terminal_failure"
    ] is True
    assert resume20["rollout"] == resumed["rollout"]
    assert resume20["reward"] == resumed["reward"]


def test_resume_inherits_only_checkpoint_backed_rows_and_completions(tmp_path):
    parent, output = tmp_path / "parent", tmp_path / "resumed"
    completions = parent / "trainer" / "completions"
    completions.mkdir(parents=True)
    output.mkdir()
    for name in ("raw_rollouts.jsonl", "rollout_evidence.jsonl"):
        (parent / name).write_text('"saved"\n"saved"\n"unsaved"\n', encoding="utf-8")
    for step in (1, 2):
        (completions / f"completions_{step:05d}.parquet").write_bytes(b"fixture")
    (parent / "rejected_rollouts.jsonl").write_text("quarantine", encoding="utf-8")
    resume = {"parent_run": str(parent), "prior_rollouts": 2, "completed_steps": 1}
    runner.seed_resume_artifacts(output, resume)
    assert len((output / "raw_rollouts.jsonl").read_text().splitlines()) == 2
    assert [p.name for p in (output / "trainer/completions").iterdir()] == [
        "completions_00001.parquet"
    ]
    assert not (output / "rejected_rollouts.jsonl").exists()


def test_resume_rejects_unrecognized_completion_filename(tmp_path):
    parent, output = tmp_path / "parent", tmp_path / "resumed"
    completions = parent / "trainer/completions"
    completions.mkdir(parents=True)
    output.mkdir()
    for name in ("raw_rollouts.jsonl", "rollout_evidence.jsonl"):
        (parent / name).write_text('{}\n', encoding="utf-8")
    (completions / "unknown.parquet").write_bytes(b"fixture")
    with pytest.raises(ValueError, match="Unknown completion"):
        runner.seed_resume_artifacts(output, {
            "parent_run": str(parent), "prior_rollouts": 1, "completed_steps": 1,
        })
