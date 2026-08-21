from __future__ import annotations

import argparse
import hashlib
import json
import statistics
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.evaluation.replay_evaluator import Tau2Runtime
from src.rl.retail_agentic_env import (
    DEFAULT_REWARD_CONFIG,
    gate_environment_state_reward,
    one_to_one_action_progress,
)
from src.training.teacher_evidence_pack import claim_state_consistency


SCHEMA_VERSION = "retail-process-reward-offline-audit-v1.1.0"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def _signature(name: str, arguments: dict[str, Any]) -> str:
    return f"{name}:{json.dumps(arguments, ensure_ascii=False, sort_keys=True)}"


def _single_result(raw: dict[str, Any], path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    tasks = raw.get("tasks") or []
    simulations = raw.get("simulations") or []
    if len(tasks) != 1 or len(simulations) != 1:
        raise ValueError(
            f"Expected one task and one simulation in {path}; "
            f"found tasks={len(tasks)}, simulations={len(simulations)}"
        )
    return tasks[0], simulations[0]


def _call_records(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    results = {
        str(message.get("id") or message.get("tool_call_id") or ""): message
        for message in messages
        if message.get("role") == "tool"
    }
    records: list[dict[str, Any]] = []
    for message_index, message in enumerate(messages):
        if message.get("role") != "assistant":
            continue
        for call in message.get("tool_calls") or []:
            call_id = str(call.get("id") or "")
            name = str(call.get("name") or "")
            arguments = dict(call.get("arguments") or {})
            result = results.get(call_id) or {}
            records.append(
                {
                    "message_index": message_index,
                    "call_id": call_id,
                    "name": name,
                    "arguments": arguments,
                    "signature": _signature(name, arguments),
                    "result_error": bool(result.get("error", False)),
                }
            )
    return records


def error_recovery_diagnostics(
    messages: list[dict[str, Any]], *, outcome_success: bool
) -> dict[str, Any]:
    """Describe error/retry behavior without assigning a new reward weight."""

    calls = _call_records(messages)
    failed = [call for call in calls if call["result_error"]]
    failed_counts = Counter(call["signature"] for call in failed)
    repeated_failed = sum(max(0, count - 1) for count in failed_counts.values())
    changed_after_error = 0
    for index, call in enumerate(calls):
        if not call["result_error"]:
            continue
        if any(later["signature"] != call["signature"] for later in calls[index + 1 :]):
            changed_after_error += 1
    return {
        "tool_error_count": len(failed),
        "repeated_failed_exact_call_count": repeated_failed,
        "failed_signature_counts": dict(sorted(failed_counts.items())),
        "error_events_followed_by_changed_call": changed_after_error,
        "successful_despite_tool_error": outcome_success and bool(failed),
        "interpretation": (
            "Diagnostic only: a changed later call is evidence of strategy change, "
            "not proof that the error was causally recovered."
        ),
    }


def _communication_recall(task: Any, reward_info: dict[str, Any]) -> float | None:
    criteria = task.evaluation_criteria
    required = list(criteria.communicate_info or []) if criteria is not None else []
    if not required:
        return None
    checks = reward_info.get("communicate_checks") or []
    if not checks:
        return 0.0
    return sum(float(bool(check.get("met"))) for check in checks) / len(required)


def _compose_v1_proxy(
    *,
    environment_state_raw: float,
    action_progress: dict[str, Any],
    communication_recall: float | None,
    tool_errors: int,
    user_stopped: bool,
    reward_config: dict[str, Any],
) -> dict[str, Any]:
    action_recall = action_progress["recall"]
    environment_value, environment_gate = gate_environment_state_reward(
        environment_state_raw, action_recall
    )
    weighted: list[tuple[str, float, float]] = [
        (
            "environment_state",
            float(reward_config["environment_state_weight"]),
            environment_value,
        )
    ]
    if action_recall is not None:
        weighted.append(
            (
                "required_action_recall",
                float(reward_config["required_action_weight"]),
                float(action_recall),
            )
        )
    if communication_recall is not None:
        weighted.append(
            (
                "communication_recall",
                float(reward_config["communication_weight"]),
                communication_recall,
            )
        )
    weighted = [item for item in weighted if item[1] > 0]
    weight_sum = sum(weight for _, weight, _ in weighted)
    raw_reward = sum(weight * value for _, weight, value in weighted) / weight_sum
    error_penalty = min(
        float(reward_config["tool_error_penalty_cap"]),
        float(reward_config["tool_error_penalty_each"]) * tool_errors,
    )
    repeat_penalty = min(
        float(reward_config["repeated_call_penalty_cap"]),
        float(reward_config["repeated_call_penalty_each"])
        * int(action_progress["duplicate_excess_count"]),
    )
    unexpected_write_penalty = min(
        float(reward_config["unexpected_write_penalty_cap"]),
        float(reward_config["unexpected_write_penalty_each"])
        * int(action_progress["unexpected_write_count"]),
    )
    unfinished_penalty = (
        0.0 if user_stopped else float(reward_config["unfinished_interaction_penalty"])
    )
    score = max(
        0.0,
        raw_reward
        - error_penalty
        - repeat_penalty
        - unexpected_write_penalty
        - unfinished_penalty,
    )
    return {
        "score": score,
        "raw_reward": raw_reward,
        "components": {
            name: {"weight": weight / weight_sum, "value": value}
            for name, weight, value in weighted
        },
        "penalties": {
            "tool_error": error_penalty,
            "repeated_call": repeat_penalty,
            "unexpected_write": unexpected_write_penalty,
            "unfinished_interaction": unfinished_penalty,
        },
        "environment_state_gate": environment_gate,
    }


def audit_trajectory(
    result_path: str | Path,
    *,
    tau2_root: str | Path,
    run_name: str,
    reward_config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    path = Path(result_path).resolve()
    runtime = Tau2Runtime(tau2_root)
    results, raw = runtime.load_results(path)
    raw_task, raw_simulation = _single_result(raw, path)
    if len(results.tasks) != 1 or len(results.simulations) != 1:
        raise ValueError(f"Expected one parsed task and simulation in {path}")
    task = results.tasks[0]
    simulation = results.simulations[0]
    reward_info = raw_simulation.get("reward_info")
    if reward_info is None:
        raise ValueError(f"Infrastructure-invalid trajectory has no reward_info: {path}")
    outcome_reward = float(reward_info.get("reward") or 0.0)
    db_check = reward_info.get("db_check") or {}
    environment_state_raw = float(
        db_check.get("db_reward")
        if db_check.get("db_reward") is not None
        else bool(db_check.get("db_match"))
    )
    action_progress = one_to_one_action_progress(task, list(simulation.messages or []))
    raw_messages = list(raw_simulation.get("messages") or [])
    recovery = error_recovery_diagnostics(
        raw_messages, outcome_success=outcome_reward == 1.0
    )
    communication_recall = _communication_recall(task, reward_info)
    config = dict(reward_config or DEFAULT_REWARD_CONFIG)
    proxy = _compose_v1_proxy(
        environment_state_raw=environment_state_raw,
        action_progress=action_progress,
        communication_recall=communication_recall,
        tool_errors=recovery["tool_error_count"],
        user_stopped=str(raw_simulation.get("termination_reason")) == "user_stop",
        reward_config=config,
    )
    nl_checks = reward_info.get("nl_assertions") or []
    claim_check = claim_state_consistency(raw_messages, {"agent": {"orders": {}}})
    return {
        "task_id": str(raw_task.get("id")),
        "run_name": run_name,
        "artifact": {"path": str(path), "sha256": _sha256(path)},
        "benchmark": {
            "reward": outcome_reward,
            "success": outcome_reward == 1.0,
            "db_match": bool(db_check.get("db_match")),
            "nl_assertions_met": sum(bool(check.get("met")) for check in nl_checks),
            "nl_assertions_total": len(nl_checks),
            "termination_reason": raw_simulation.get("termination_reason"),
        },
        "v1_reward_proxy": proxy,
        "action_progress": action_progress,
        "error_recovery": recovery,
        "claim_state_consistency": claim_check,
        "efficiency_bonus_eligible": outcome_reward == 1.0
        and bool(db_check.get("db_match")),
        "validity_notes": [
            "This is an offline proxy for the implemented Agentic GRPO v1 reward, not a regenerated rollout.",
            "DB reward and frozen messages come from returned_results.json; no LLM or environment mutation is performed.",
            "Tau2 reference actions are benchmark-only diagnostics and are not deployable runtime knowledge.",
            "NL assertions are reported but are not part of the implemented v1 Agentic GRPO reward.",
            "Claim-state consistency is diagnostic only and does not change the v1 proxy score.",
            "No new reward weights are introduced by this audit.",
        ],
    }


def _result_paths(run_dir: Path) -> dict[str, Path]:
    paths: dict[str, Path] = {}
    for path in sorted(run_dir.rglob("returned_results.json")):
        raw = json.loads(path.read_text(encoding="utf-8"))
        _, simulation = _single_result(raw, path)
        task_id = str(simulation.get("task_id"))
        if task_id in paths:
            raise ValueError(f"Duplicate task {task_id} below {run_dir}")
        paths[task_id] = path.resolve()
    return paths


def _apply_replacements(
    paths: dict[str, Path], replacements: dict[str, str | Path] | None
) -> dict[str, Path]:
    result = dict(paths)
    for task_id, root in (replacements or {}).items():
        replacement_paths = _result_paths(Path(root).resolve())
        if str(task_id) not in replacement_paths:
            raise FileNotFoundError(f"Replacement for task {task_id} not found in {root}")
        result[str(task_id)] = replacement_paths[str(task_id)]
    return result


def _mean(rows: list[dict[str, Any]], key: str) -> float | None:
    values = [float(row["v1_reward_proxy"][key]) for row in rows]
    return statistics.fmean(values) if values else None


def _claim_verdict_rank(verdict: str) -> int | None:
    return {"FAIL": 0, "REVIEW": 1, "PASS": 2}.get(verdict)


def build_audit(
    *,
    run_a: str | Path,
    run_b: str | Path,
    tau2_root: str | Path,
    output_dir: str | Path,
    replacements_a: dict[str, str | Path] | None = None,
    replacements_b: dict[str, str | Path] | None = None,
) -> dict[str, Any]:
    output = Path(output_dir).resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty output: {output}")
    paths_a = _apply_replacements(_result_paths(Path(run_a).resolve()), replacements_a)
    paths_b = _apply_replacements(_result_paths(Path(run_b).resolve()), replacements_b)
    if set(paths_a) != set(paths_b):
        raise ValueError("Run task IDs differ after replacements")
    rows_a = {
        task_id: audit_trajectory(
            path,
            tau2_root=tau2_root,
            run_name="run_a",
        )
        for task_id, path in paths_a.items()
    }
    rows_b = {
        task_id: audit_trajectory(
            path,
            tau2_root=tau2_root,
            run_name="run_b",
        )
        for task_id, path in paths_b.items()
    }
    pairs: list[dict[str, Any]] = []
    for task_id in sorted(rows_a, key=int):
        a = rows_a[task_id]
        b = rows_b[task_id]
        a_success = a["benchmark"]["success"]
        b_success = b["benchmark"]["success"]
        cohort = (
            "common_success"
            if a_success and b_success
            else "flip"
            if a_success != b_success
            else "common_failure"
        )
        successful_run = "run_a" if a_success and not b_success else "run_b" if b_success and not a_success else None
        successful_score = (
            a["v1_reward_proxy"]["score"]
            if successful_run == "run_a"
            else b["v1_reward_proxy"]["score"]
            if successful_run == "run_b"
            else None
        )
        failed_score = (
            b["v1_reward_proxy"]["score"]
            if successful_run == "run_a"
            else a["v1_reward_proxy"]["score"]
            if successful_run == "run_b"
            else None
        )
        successful_claim_rank = (
            _claim_verdict_rank(
                a["claim_state_consistency"]["verdict"]
                if successful_run == "run_a"
                else b["claim_state_consistency"]["verdict"]
            )
            if successful_run is not None
            else None
        )
        failed_claim_rank = (
            _claim_verdict_rank(
                b["claim_state_consistency"]["verdict"]
                if successful_run == "run_a"
                else a["claim_state_consistency"]["verdict"]
            )
            if successful_run is not None
            else None
        )
        pairs.append(
            {
                "task_id": task_id,
                "cohort": cohort,
                "run_a": a,
                "run_b": b,
                "flip_ranking": (
                    {
                        "successful_run": successful_run,
                        "successful_score": successful_score,
                        "failed_score": failed_score,
                        "success_ranked_higher": successful_score > failed_score,
                        "score_tied": successful_score == failed_score,
                    }
                    if successful_run is not None
                    else None
                ),
                "claim_state_diagnostic": (
                    {
                        "successful_verdict": (
                            a["claim_state_consistency"]["verdict"]
                            if successful_run == "run_a"
                            else b["claim_state_consistency"]["verdict"]
                        ),
                        "failed_verdict": (
                            b["claim_state_consistency"]["verdict"]
                            if successful_run == "run_a"
                            else a["claim_state_consistency"]["verdict"]
                        ),
                        "evaluable": successful_claim_rank is not None
                        and failed_claim_rank is not None,
                        "successful_run_preferred": (
                            successful_claim_rank > failed_claim_rank
                            if successful_claim_rank is not None
                            and failed_claim_rank is not None
                            else None
                        ),
                    }
                    if successful_run is not None
                    else None
                ),
            }
        )
    common_success = [pair for pair in pairs if pair["cohort"] == "common_success"]
    flips = [pair for pair in pairs if pair["cohort"] == "flip"]
    common_success_rows = [
        pair[run]
        for pair in common_success
        for run in ("run_a", "run_b")
    ]
    all_success_rows = [
        pair[run]
        for pair in pairs
        for run in ("run_a", "run_b")
        if pair[run]["benchmark"]["success"]
    ]
    flip_ranked = [
        pair for pair in flips if pair["flip_ranking"]["success_ranked_higher"]
    ]
    flip_tied = [pair for pair in flips if pair["flip_ranking"]["score_tied"]]
    successful_with_errors = [
        row
        for row in all_success_rows
        if row["error_recovery"]["successful_despite_tool_error"]
    ]
    claim_failures_on_success = [
        row
        for row in all_success_rows
        if row["claim_state_consistency"]["verdict"] == "FAIL"
    ]
    evaluable_flip_claims = [
        pair
        for pair in flips
        if pair["claim_state_diagnostic"]["evaluable"]
    ]
    preferred_flip_claims = [
        pair
        for pair in evaluable_flip_claims
        if pair["claim_state_diagnostic"]["successful_run_preferred"]
    ]
    report = {
        "schema_version": SCHEMA_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "inputs": {
            "run_a": str(Path(run_a).resolve()),
            "run_b": str(Path(run_b).resolve()),
            "tau2_root": str(Path(tau2_root).resolve()),
            "replacements_a": {
                str(key): str(Path(value).resolve())
                for key, value in (replacements_a or {}).items()
            },
            "replacements_b": {
                str(key): str(Path(value).resolve())
                for key, value in (replacements_b or {}).items()
            },
            "reward_config": DEFAULT_REWARD_CONFIG,
        },
        "summary": {
            "task_count": len(pairs),
            "common_success_task_count": len(common_success),
            "common_success_trajectory_count": len(common_success_rows),
            "all_success_trajectory_count": len(all_success_rows),
            "common_failure_task_count": sum(
                pair["cohort"] == "common_failure" for pair in pairs
            ),
            "flip_task_count": len(flips),
            "flip_success_ranked_higher_count": len(flip_ranked),
            "flip_score_tied_count": len(flip_tied),
            "flip_success_not_ranked_higher_task_ids": [
                pair["task_id"] for pair in flips if pair not in flip_ranked
            ],
            "common_success_mean_proxy_score": _mean(common_success_rows, "score"),
            "common_success_min_proxy_score": (
                min(row["v1_reward_proxy"]["score"] for row in common_success_rows)
                if common_success_rows
                else None
            ),
            "successful_with_tool_error_count": len(
                successful_with_errors
            ),
            "successful_with_tool_error_tasks": [
                {"task_id": row["task_id"], "run_name": row["run_name"]}
                for row in successful_with_errors
            ],
            "claim_state_failure_on_success_count": len(
                claim_failures_on_success
            ),
            "claim_state_failure_on_success_tasks": [
                {"task_id": row["task_id"], "run_name": row["run_name"]}
                for row in claim_failures_on_success
            ],
            "claim_state_evaluable_flip_count": len(evaluable_flip_claims),
            "claim_state_success_preferred_flip_count": len(
                preferred_flip_claims
            ),
            "claim_state_success_preferred_flip_task_ids": [
                pair["task_id"] for pair in preferred_flip_claims
            ],
        },
        "pairs": pairs,
        "gates": {
            "positive_validation_executed": bool(common_success_rows),
            "all_flip_successes_ranked_higher": len(flip_ranked) == len(flips),
            "claim_state_has_zero_failures_on_frozen_successes": not claim_failures_on_success,
            "ready_to_use_v1_reward_for_grpo": False,
        },
        "validity_notes": [
            "Common-success trajectories are positive sanity checks, not human-gold clean-process labels.",
            "Flip tasks were selected after observing instability and cannot estimate generalization alone.",
            "A separate untouched holdout is required after reward rules and weights are frozen.",
            "Claim-state diagnostics use only explicit tool observations/final text and remain outside the scalar reward.",
            "The audit must not overwrite frozen source artifacts or be reported as a new rollout evaluation.",
        ],
    }
    output.mkdir(parents=True, exist_ok=True)
    (output / "process_reward_audit.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (output / "summary.json").write_text(
        json.dumps(report["summary"], ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return report


def _replacement(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("replacement must be TASK_ID=RUN_DIR")
    task_id, path = value.split("=", 1)
    return task_id, Path(path)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Offline-audit the implemented Agentic GRPO v1 reward on frozen trajectories."
    )
    parser.add_argument("--run-a", type=Path, required=True)
    parser.add_argument("--run-b", type=Path, required=True)
    parser.add_argument("--replacement-a", action="append", type=_replacement, default=[])
    parser.add_argument("--replacement-b", action="append", type=_replacement, default=[])
    parser.add_argument("--tau2-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = build_audit(
        run_a=args.run_a,
        run_b=args.run_b,
        replacements_a=dict(args.replacement_a),
        replacements_b=dict(args.replacement_b),
        tau2_root=args.tau2_root,
        output_dir=args.output,
    )
    print(json.dumps(report["summary"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
