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
    path: Path, expected_rollouts: int, expected_tasks: int
) -> dict[str, Any]:
    rows = load_jsonl(path)
    rewards = [float(row["reward"]["reward"]) for row in rows]
    task_counts = Counter(str(row["task_id"]) for row in rows)
    tool_calls = [int(row["tool_calls"]) for row in rows]
    customer_turns = [int(row["customer_turns"]) for row in rows]
    tool_errors = [int(row["reward"].get("tool_error_count", 0)) for row in rows]
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
    for row, reward, action_recall in zip(
        rows, rewards, action_recalls, strict=True
    ):
        task_id = str(row["task_id"])
        grouped_rewards[task_id].append(reward)
        grouped_action_recalls[task_id].append(action_recall)
    task_group_diagnostics = {}
    reward_variance_task_count = 0
    action_variance_task_count = 0
    joint_variance_task_count = 0
    for task_id in sorted(grouped_rewards, key=int):
        task_rewards = grouped_rewards[task_id]
        task_action_recalls = grouped_action_recalls[task_id]
        task_reward_variance = (
            statistics.pvariance(task_rewards) if len(task_rewards) > 1 else 0.0
        )
        task_action_variance = (
            statistics.pvariance(task_action_recalls)
            if len(task_action_recalls) > 1
            else 0.0
        )
        reward_has_variance = task_reward_variance > 0.0
        action_has_variance = task_action_variance > 0.0
        reward_variance_task_count += int(reward_has_variance)
        action_variance_task_count += int(action_has_variance)
        joint_variance_task_count += int(
            reward_has_variance and action_has_variance
        )
        task_group_diagnostics[task_id] = {
            "rollouts": len(task_rewards),
            "reward_values": task_rewards,
            "reward_population_variance": task_reward_variance,
            "action_recall_values": task_action_recalls,
            "action_recall_population_variance": task_action_variance,
            "reward_has_variance": reward_has_variance,
            "action_progress_has_variance": action_has_variance,
        }
    observed_task_groups = len(grouped_rewards)
    minimum_signal_task_count = min(
        expected_tasks, max(2, (expected_tasks + 3) // 4)
    )
    group_variance_gate = (
        joint_variance_task_count >= minimum_signal_task_count
    )
    gates = {
        "expected_rollout_count_met": len(rows) == expected_rollouts,
        "all_expected_tasks_observed": len(task_counts) == expected_tasks,
        "tool_call_rate_positive": any(value > 0 for value in tool_calls),
        "customer_continuation_observed": any(value > 0 for value in customer_turns),
        "reward_has_variance": reward_variance > 0.0,
        "action_progress_has_variance": len(set(action_recalls)) > 1,
        "sufficient_task_groups_have_joint_variance": group_variance_gate,
        "no_positive_reward_without_tool": positive_without_tool == 0,
        "no_positive_reward_without_action_progress": (
            positive_without_action_progress == 0
        ),
    }
    gates["ready_to_consider_optimization"] = all(gates.values())
    return {
        "schema_version": "retail-agentic-rollout-diagnostic-v2",
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
            "tool_call_rollout_count": sum(value > 0 for value in tool_calls),
            "customer_continuation_rollout_count": sum(value > 0 for value in customer_turns),
            "tool_error_rollout_count": sum(value > 0 for value in tool_errors),
            "mean_tool_calls": statistics.fmean(tool_calls) if tool_calls else None,
            "mean_action_recall": statistics.fmean(action_recalls) if action_recalls else None,
            "distinct_action_recalls": sorted(set(action_recalls)),
            "positive_without_tool_count": positive_without_tool,
            "positive_without_action_progress_count": (
                positive_without_action_progress
            ),
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
            "threshold_is_diagnostic_heuristic": True,
            "tasks": task_group_diagnostics,
        },
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
    args = parser.parse_args()
    report = analyze(
        args.rollouts.resolve(), args.expected_rollouts, args.expected_tasks
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
