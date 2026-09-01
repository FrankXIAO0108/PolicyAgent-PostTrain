from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from src.evaluation.staged_reward_shadow import (
    _canonical_arguments,
    _read_json,
    _sha256,
    score_rollout,
)
from src.guards.retail_pre_action import WRITE_TOOLS


SCHEMA_VERSION = "retail-historical-staged-reward-shadow-v1"


def _normalized_arguments(arguments: dict[str, Any]) -> dict[str, Any]:
    normalized = dict(arguments)
    item_ids = normalized.get("item_ids")
    new_item_ids = normalized.get("new_item_ids")
    if isinstance(item_ids, list) and isinstance(new_item_ids, list):
        if len(item_ids) == len(new_item_ids):
            normalized.pop("item_ids")
            normalized.pop("new_item_ids")
            normalized["item_pairs"] = sorted(
                [str(old), str(new)]
                for old, new in zip(item_ids, new_item_ids, strict=True)
            )
    elif isinstance(item_ids, list):
        normalized["item_ids"] = sorted(str(value) for value in item_ids)
    return normalized


def _signature(name: str, arguments: dict[str, Any]) -> str:
    return f"{name}:{_canonical_arguments(_normalized_arguments(arguments))}"


def _tool_trace(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    tool_results = {
        str(message.get("id")): message
        for message in messages
        if message.get("role") == "tool" and message.get("id")
    }
    trace = []
    for message_index, message in enumerate(messages):
        if message.get("role") != "assistant":
            continue
        for call in message.get("tool_calls") or []:
            result = tool_results.get(str(call.get("id")))
            trace.append(
                {
                    "message_index": message_index,
                    "call_id": str(call.get("id")),
                    "name": str(call.get("name")),
                    "arguments": dict(call.get("arguments") or {}),
                    "result": (
                        {
                            "content": result.get("content"),
                            "error": bool(result.get("error")),
                        }
                        if result is not None
                        else {"content": None, "error": True}
                    ),
                }
            )
    return trace


def _write_progress_and_unexpected(
    reward_info: dict[str, Any],
    trace: list[dict[str, Any]],
    required_action_ids: set[str],
) -> dict[str, Any]:
    action_checks = list(reward_info.get("action_checks") or [])
    matches = [
        {
            "action_id": str((row.get("action") or {}).get("action_id")),
            "name": str((row.get("action") or {}).get("name")),
            "matched": bool(row.get("action_match")),
        }
        for row in action_checks
    ]
    expected_writes = [
        row.get("action") or {}
        for row in action_checks
        if str((row.get("action") or {}).get("action_id")) in required_action_ids
    ]
    remaining = Counter(
        _signature(str(action["name"]), dict(action.get("arguments") or {}))
        for action in expected_writes
    )
    unexpected = []
    for call in trace:
        if call["name"] not in WRITE_TOOLS:
            continue
        signature = _signature(call["name"], dict(call["arguments"]))
        if remaining[signature] > 0:
            remaining[signature] -= 1
        else:
            unexpected.append(
                {"name": call["name"], "arguments": dict(call["arguments"])}
            )
    return {
        "matches": matches,
        "unexpected_write_count": len(unexpected),
        "unexpected_writes": unexpected,
    }


def simulation_to_inputs(
    simulation: dict[str, Any], spec: dict[str, Any]
) -> tuple[dict[str, Any], dict[str, Any]]:
    task_id = str(simulation["task_id"])
    task_spec = dict(spec["tasks"][task_id])
    messages = list(simulation.get("messages") or [])
    trace = _tool_trace(messages)
    reward_info = dict(simulation.get("reward_info") or {})
    progress = _write_progress_and_unexpected(
        reward_info,
        trace,
        {str(value) for value in task_spec["required_write_action_ids"]},
    )
    binding = f"historical:{simulation['id']}"
    termination = str(simulation.get("termination_reason") or "")
    raw = {
        "task_id": task_id,
        "messages": messages,
        "evidence_sha256": binding,
    }
    evidence = {
        "task_id": task_id,
        "evidence_sha256": binding,
        "tool_trace": trace,
        "completion": {
            "customer_turn_limit_reached": termination == "max_steps",
            "tool_call_limit_reached": termination == "max_errors",
        },
        "terminal_evaluator": {
            "reward": float(
                ((reward_info.get("db_check") or {}).get("db_reward") or 0.0)
            ),
            "user_stopped": termination == "user_stop",
            "action_progress": progress,
            "tau2": {
                "communication": {
                    "communicate_checks": reward_info.get("communicate_checks")
                }
            },
        },
    }
    return raw, evidence


def build_report(
    *,
    task_result_paths: list[Path],
    spec_path: Path,
    c3_report_path: Path,
) -> dict[str, Any]:
    spec = _read_json(spec_path)
    c3_report = _read_json(c3_report_path)
    simulations: list[dict[str, Any]] = []
    sources = []
    for path in task_result_paths:
        payload = _read_json(path)
        simulations.extend(payload.get("simulations") or [])
        sources.append({"path": str(path), "sha256": _sha256(path)})

    rows = []
    for simulation in simulations:
        raw, evidence = simulation_to_inputs(simulation, spec)
        score = score_rollout(raw, evidence, spec)
        score.update(
            {
                "simulation_id": str(simulation["id"]),
                "trial": int(simulation["trial"]),
                "seed": int(simulation["seed"]),
                "termination_reason": simulation.get("termination_reason"),
                "benchmark_reward": float(
                    (simulation.get("reward_info") or {}).get("reward") or 0.0
                ),
                "benchmark_db_reward": float(
                    (((simulation.get("reward_info") or {}).get("db_check") or {}).get(
                        "db_reward"
                    )
                    or 0.0)
                ),
            }
        )
        rows.append(score)

    benchmark_success = [row for row in rows if row["benchmark_reward"] == 1.0]
    benchmark_failure = [row for row in rows if row["benchmark_reward"] == 0.0]
    write_rows = [
        row
        for row in rows
        if row["components"]["required_write_progress"]["value"] > 0.0
    ]
    full_write_rows = [
        row
        for row in rows
        if row["components"]["required_write_progress"]["value"] == 1.0
    ]
    c3_no_write_scores = [
        float(row["staged_reward"])
        for row in c3_report["trajectories"]
        if row["components"]["required_write_progress"]["value"] == 0.0
    ]
    collisions = []
    for task_id in sorted({row["task_id"] for row in rows}):
        task_rows = [row for row in rows if row["task_id"] == task_id]
        for success in [row for row in task_rows if row["benchmark_reward"] == 1.0]:
            for failure in [row for row in task_rows if row["benchmark_reward"] == 0.0]:
                if success["staged_reward"] <= failure["staged_reward"]:
                    collisions.append(
                        {
                            "task_id": task_id,
                            "success_simulation_id": success["simulation_id"],
                            "failure_simulation_id": failure["simulation_id"],
                            "success_staged_reward": success["staged_reward"],
                            "failure_staged_reward": failure["staged_reward"],
                        }
                    )

    checks = {
        "contains_real_correct_write": bool(write_rows),
        "contains_real_full_write": bool(full_write_rows),
        "full_write_above_c3_no_write_max": bool(full_write_rows)
        and min(row["staged_reward"] for row in full_write_rows)
        > max(c3_no_write_scores),
        "benchmark_success_strictly_above_failure_within_task": not collisions,
    }
    return {
        "schema_version": SCHEMA_VERSION,
        "scope": {
            "offline_shadow_only": True,
            "changes_online_reward": False,
            "calls_external_api": False,
            "uses_cloud_gpu": False,
            "historical_checkpoint_sha256": (
                "94AE57492834A6FF19C782B6B2C30CE3255F83F63925E6399F104021815E726E"
            ),
            "c3_checkpoint_sha256": (
                "0A2E06C9BCA6082F3FE6723EC54A46D4CE1D37A8C6DD6B9BC116BBA4BAB16576"
            ),
            "same_checkpoint_as_c3": False,
            "valid_use": "component ordering and collision audit only",
        },
        "sources": {
            "spec": {"path": str(spec_path), "sha256": _sha256(spec_path)},
            "c3_shadow_report": {
                "path": str(c3_report_path),
                "sha256": _sha256(c3_report_path),
            },
            "historical_task_results": sources,
        },
        "summary": {
            "trajectory_count": len(rows),
            "benchmark_success_count": len(benchmark_success),
            "benchmark_failure_count": len(benchmark_failure),
            "correct_write_trajectory_count": len(write_rows),
            "full_write_trajectory_count": len(full_write_rows),
            "c3_no_write_max_reward": max(c3_no_write_scores),
            "historical_full_write_min_reward": min(
                row["staged_reward"] for row in full_write_rows
            ),
            "ranking_collision_count": len(collisions),
        },
        "checks": checks,
        "online_promotion_ready": all(checks.values()),
        "ranking_collisions": collisions,
        "trajectories": rows,
        "limitations": [
            "Historical trajectories and C3 trajectories come from different SFT checkpoints.",
            "The comparison validates reward ordering only and is not a model-effect comparison.",
            "Historical artifacts do not contain C3-style initial/final state sidecars; DB reward and evaluator action checks are used as recorded.",
            "Any benchmark-success collision blocks online promotion until the reward formula is revised and re-audited.",
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Audit staged reward against historical write trajectories"
    )
    parser.add_argument("--task-results", type=Path, nargs="+", required=True)
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--c3-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = build_report(
        task_result_paths=[path.resolve() for path in args.task_results],
        spec_path=args.spec.resolve(),
        c3_report_path=args.c3_report.resolve(),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report["summary"], ensure_ascii=False, indent=2))
    print(json.dumps(report["checks"], ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
