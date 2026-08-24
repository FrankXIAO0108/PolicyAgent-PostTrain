from __future__ import annotations

import argparse
import hashlib
import json
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def analyze(
    path: Path,
    expected_rollouts: int,
    expected_tasks: int,
    baseline_path: Path | None = None,
    system_failure_path: Path | None = None,
) -> dict[str, Any]:
    rows = load_jsonl(path)
    rewards = [float(row["reward"]["reward"]) for row in rows]
    task_counts = Counter(str(row["task_id"]) for row in rows)
    tool_calls = [int(row["tool_calls"]) for row in rows]
    customer_turns = [int(row["customer_turns"]) for row in rows]
    tool_errors = [int(row["reward"].get("tool_error_count", 0)) for row in rows]
    duplicate_excess = [
        int(row["reward"].get("action_progress", {}).get("duplicate_excess_count", 0))
        for row in rows
    ]
    unfinished = [
        float(row["reward"].get("unfinished_interaction_penalty", 0.0)) > 0
        for row in rows
    ]
    rollout_stages = [
        str(
            row.get("rollout_stage")
            or row["reward"].get("rollout_stage")
            or "FULL_TASK"
        )
        for row in rows
    ]
    full_task_completed = [
        stage == "FULL_TASK" and not is_unfinished
        for stage, is_unfinished in zip(
            rollout_stages, unfinished, strict=True
        )
    ]
    staged_completed = [
        stage != "FULL_TASK" and bool(row["reward"].get("stage_complete"))
        for row, stage in zip(rows, rollout_stages, strict=True)
    ]
    system_failures = (
        load_jsonl(system_failure_path)
        if system_failure_path is not None and system_failure_path.is_file()
        else []
    )
    system_failure_categories = Counter(
        str(row.get("category") or "UNKNOWN") for row in system_failures
    )
    action_recalls = [
        float(row["reward"]["action_progress"]["recall"] or 0.0) for row in rows
    ]
    positive_without_tool = sum(
        reward > 0 and tool_count == 0
        for reward, tool_count in zip(rewards, tool_calls, strict=True)
    )
    positive_without_action_progress = sum(
        reward > 0 and action_recall == 0
        for reward, action_recall in zip(rewards, action_recalls, strict=True)
    )
    reward_variance = statistics.pvariance(rewards) if rewards else 0.0
    distinct_rewards = sorted(set(rewards))
    grouped_rewards: dict[str, list[float]] = defaultdict(list)
    grouped_action_recalls: dict[str, list[float]] = defaultdict(list)
    grouped_stage_completions: dict[str, list[bool]] = defaultdict(list)
    for row, reward, action_recall, stage_completed in zip(
        rows, rewards, action_recalls, staged_completed, strict=True
    ):
        task_id = str(row["task_id"])
        grouped_rewards[task_id].append(reward)
        grouped_action_recalls[task_id].append(action_recall)
        grouped_stage_completions[task_id].append(stage_completed)
    task_group_diagnostics = {}
    reward_variance_task_count = 0
    action_variance_task_count = 0
    joint_variance_task_count = 0
    stage_completion_variance_task_count = 0
    stage_target_variance_task_count = 0
    for task_id in sorted(grouped_rewards, key=int):
        task_rewards = grouped_rewards[task_id]
        task_action_recalls = grouped_action_recalls[task_id]
        task_stage_completions = grouped_stage_completions[task_id]
        task_reward_variance = (
            statistics.pvariance(task_rewards) if len(task_rewards) > 1 else 0.0
        )
        task_action_variance = (
            statistics.pvariance(task_action_recalls)
            if len(task_action_recalls) > 1
            else 0.0
        )
        task_stage_completion_variance = (
            statistics.pvariance(int(value) for value in task_stage_completions)
            if len(task_stage_completions) > 1
            else 0.0
        )
        reward_has_variance = task_reward_variance > 0.0
        action_has_variance = task_action_variance > 0.0
        stage_completion_has_variance = task_stage_completion_variance > 0.0
        reward_variance_task_count += int(reward_has_variance)
        action_variance_task_count += int(action_has_variance)
        joint_variance_task_count += int(
            reward_has_variance and action_has_variance
        )
        stage_completion_variance_task_count += int(stage_completion_has_variance)
        stage_target_variance_task_count += int(
            reward_has_variance and stage_completion_has_variance
        )
        task_group_diagnostics[task_id] = {
            "rollouts": len(task_rewards),
            "reward_values": task_rewards,
            "reward_population_variance": task_reward_variance,
            "action_recall_values": task_action_recalls,
            "action_recall_population_variance": task_action_variance,
            "stage_complete_values": task_stage_completions,
            "stage_complete_population_variance": task_stage_completion_variance,
            "reward_has_variance": reward_has_variance,
            "action_progress_has_variance": action_has_variance,
            "stage_completion_has_variance": stage_completion_has_variance,
        }
    observed_task_groups = len(grouped_rewards)
    minimum_signal_task_count = min(
        expected_tasks, max(2, (expected_tasks + 3) // 4)
    )
    staged_run = bool(rollout_stages) and all(
        stage != "FULL_TASK" for stage in rollout_stages
    )
    variance_gate_mode = (
        "REWARD_AND_STAGE_COMPLETION"
        if staged_run
        else "REWARD_AND_ACTION_PROGRESS"
    )
    selected_variance_task_count = (
        stage_target_variance_task_count
        if staged_run
        else joint_variance_task_count
    )
    group_variance_gate = selected_variance_task_count >= minimum_signal_task_count
    gates = {
        "expected_rollout_count_met": len(rows) == expected_rollouts,
        "all_expected_tasks_observed": len(task_counts) == expected_tasks,
        "tool_call_rate_positive": any(value > 0 for value in tool_calls),
        "customer_continuation_observed": any(value > 0 for value in customer_turns),
        "normal_termination_observed": (
            any(full_task_completed)
            if any(stage == "FULL_TASK" for stage in rollout_stages)
            else True
        ),
        "stage_completion_observed": (
            any(staged_completed)
            if any(stage != "FULL_TASK" for stage in rollout_stages)
            else True
        ),
        "single_rollout_stage_observed": len(set(rollout_stages)) <= 1,
        "reward_has_variance": reward_variance > 0.0,
        "no_positive_reward_without_tool": positive_without_tool == 0,
        "no_positive_reward_without_action_progress": (
            positive_without_action_progress == 0
        ),
        "no_system_failures": len(system_failures) == 0,
    }
    if staged_run:
        gates.update(
            {
                "stage_completion_has_variance": len(set(staged_completed)) > 1,
                "sufficient_task_groups_have_stage_target_variance": (
                    group_variance_gate
                ),
            }
        )
    else:
        gates.update(
            {
                "action_progress_has_variance": len(set(action_recalls)) > 1,
                "sufficient_task_groups_have_joint_variance": group_variance_gate,
            }
        )
    regression = None
    if baseline_path is not None:
        baseline_rows = load_jsonl(baseline_path)
        baseline_task_ids = {str(row["task_id"]) for row in baseline_rows}
        candidate_task_ids = set(task_counts)
        baseline_recalls = [
            float(row["reward"]["action_progress"]["recall"] or 0.0)
            for row in baseline_rows
        ]
        baseline_errors = sum(
            int(row["reward"].get("tool_error_count", 0)) for row in baseline_rows
        )
        baseline_repeats = sum(
            int(row["reward"].get("action_progress", {}).get("duplicate_excess_count", 0))
            for row in baseline_rows
        )
        baseline_positive = sum(float(row["reward"]["reward"]) > 0 for row in baseline_rows)
        baseline_unfinished = sum(
            float(row["reward"].get("unfinished_interaction_penalty", 0.0)) > 0
            for row in baseline_rows
        )
        regression = {
            "baseline": {"path": str(baseline_path), "sha256": sha256(baseline_path)},
            "candidate_rollouts": len(rows),
            "baseline_rollouts": len(baseline_rows),
            "candidate_task_ids": sorted(candidate_task_ids, key=int),
            "baseline_task_ids": sorted(baseline_task_ids, key=int),
            "candidate_mean_action_recall": statistics.fmean(action_recalls),
            "baseline_mean_action_recall": statistics.fmean(baseline_recalls),
            "candidate_tool_error_count": sum(tool_errors),
            "baseline_tool_error_count": baseline_errors,
            "candidate_duplicate_excess_count": sum(duplicate_excess),
            "baseline_duplicate_excess_count": baseline_repeats,
            "candidate_positive_reward_count": sum(value > 0 for value in rewards),
            "baseline_positive_reward_count": baseline_positive,
            "candidate_unfinished_count": sum(unfinished),
            "baseline_unfinished_count": baseline_unfinished,
        }
        gates.update(
            {
                "baseline_protocol_comparable": (
                    len(baseline_rows) == len(rows)
                    and baseline_task_ids == candidate_task_ids
                ),
                "mean_action_recall_not_regressed": regression["candidate_mean_action_recall"] >= regression["baseline_mean_action_recall"],
                "tool_error_count_not_increased": sum(tool_errors) <= baseline_errors,
                "duplicate_excess_count_not_increased": sum(duplicate_excess) <= baseline_repeats,
                "positive_reward_count_not_decreased": regression["candidate_positive_reward_count"] >= baseline_positive,
                "unfinished_count_not_increased": sum(unfinished) <= baseline_unfinished,
            }
        )
    gates["ready_to_consider_optimization"] = all(gates.values())
    return {
        "schema_version": "retail-agentic-rollout-diagnostic-v4",
        "source": {"path": str(path), "sha256": sha256(path)},
        "expected_rollouts": expected_rollouts,
        "expected_tasks": expected_tasks,
        "observed_rollouts": len(rows),
        "unique_tasks": len(task_counts),
        "task_rollout_counts": dict(sorted(task_counts.items(), key=lambda item: int(item[0]))),
        "reward": {
            "mean": statistics.fmean(rewards) if rewards else None,
            "population_variance": reward_variance,
            "distinct_values": distinct_rewards,
            "positive_count": sum(value > 0 for value in rewards),
        },
        "behavior": {
            "rollout_stages": sorted(set(rollout_stages)),
            "tool_call_rollout_count": sum(value > 0 for value in tool_calls),
            "customer_continuation_rollout_count": sum(value > 0 for value in customer_turns),
            "tool_error_rollout_count": sum(value > 0 for value in tool_errors),
            "tool_error_count": sum(tool_errors),
            "duplicate_excess_count": sum(duplicate_excess),
            "unfinished_rollout_count": sum(unfinished),
            "normal_termination_rollout_count": sum(full_task_completed),
            "stage_complete_rollout_count": sum(staged_completed),
            "completion_target_rollout_count": sum(
                completed or stage_completed
                for completed, stage_completed in zip(
                    full_task_completed, staged_completed, strict=True
                )
            ),
            "mean_tool_calls": statistics.fmean(tool_calls) if tool_calls else None,
            "mean_action_recall": statistics.fmean(action_recalls) if action_recalls else None,
            "distinct_action_recalls": sorted(set(action_recalls)),
            "positive_without_tool_count": positive_without_tool,
            "positive_without_action_progress_count": (
                positive_without_action_progress
            ),
        },
        "system_failures": {
            "source": (
                {
                    "path": str(system_failure_path),
                    "sha256": sha256(system_failure_path),
                }
                if system_failure_path is not None and system_failure_path.is_file()
                else None
            ),
            "count": len(system_failures),
            "categories": dict(sorted(system_failure_categories.items())),
        },
        "group_variance": {
            "definition": "within-task population variance across repeated rollouts",
            "minimum_signal_task_count": minimum_signal_task_count,
            "observed_task_groups": observed_task_groups,
            "reward_variance_task_count": reward_variance_task_count,
            "action_progress_variance_task_count": action_variance_task_count,
            "joint_variance_task_count": joint_variance_task_count,
            "joint_variance_task_ratio": (
                joint_variance_task_count / observed_task_groups
                if observed_task_groups
                else 0.0
            ),
            "stage_completion_variance_task_count": (
                stage_completion_variance_task_count
            ),
            "stage_target_variance_task_count": stage_target_variance_task_count,
            "variance_gate_mode": variance_gate_mode,
            "selected_variance_task_count": selected_variance_task_count,
            "threshold_is_diagnostic_heuristic": True,
            "tasks": task_group_diagnostics,
        },
        "baseline_regression": regression,
        "gates": gates,
        "formal_retail_readiness_gate_opened": False,
        "business_improvement_claim_allowed": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rollouts", type=Path, required=True)
    parser.add_argument("--expected-rollouts", type=int, default=32)
    parser.add_argument("--expected-tasks", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--baseline-rollouts", type=Path)
    parser.add_argument("--system-failures", type=Path)
    args = parser.parse_args()
    report = analyze(
        args.rollouts.resolve(),
        args.expected_rollouts,
        args.expected_tasks,
        args.baseline_rollouts.resolve() if args.baseline_rollouts else None,
        args.system_failures.resolve() if args.system_failures else None,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
