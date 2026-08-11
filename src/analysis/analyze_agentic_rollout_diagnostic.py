from __future__ import annotations

import argparse
import hashlib
import json
import statistics
from collections import Counter
from pathlib import Path
from typing import Any


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def analyze(path: Path, expected_rollouts: int) -> dict[str, Any]:
    rows = load_jsonl(path)
    rewards = [float(row["reward"]["reward"]) for row in rows]
    task_counts = Counter(str(row["task_id"]) for row in rows)
    tool_calls = [int(row["tool_calls"]) for row in rows]
    customer_turns = [int(row["customer_turns"]) for row in rows]
    tool_errors = [int(row["reward"].get("tool_error_count", 0)) for row in rows]
    action_recalls = [
        float(row["reward"]["action_progress"]["recall"] or 0.0) for row in rows
    ]
    reward_variance = statistics.pvariance(rewards) if rewards else 0.0
    distinct_rewards = sorted(set(rewards))
    gates = {
        "expected_rollout_count_met": len(rows) == expected_rollouts,
        "all_expected_tasks_observed": len(task_counts) == expected_rollouts // 4,
        "tool_call_rate_positive": any(value > 0 for value in tool_calls),
        "customer_continuation_observed": any(value > 0 for value in customer_turns),
        "reward_has_variance": reward_variance > 0.0,
        "action_progress_has_variance": len(set(action_recalls)) > 1,
    }
    gates["ready_to_consider_optimization"] = all(gates.values())
    return {
        "schema_version": "retail-agentic-rollout-diagnostic-v1",
        "source": {"path": str(path), "sha256": sha256(path)},
        "expected_rollouts": expected_rollouts,
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
        },
        "gates": gates,
        "formal_retail_readiness_gate_opened": False,
        "business_improvement_claim_allowed": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rollouts", type=Path, required=True)
    parser.add_argument("--expected-rollouts", type=int, default=32)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = analyze(args.rollouts.resolve(), args.expected_rollouts)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
