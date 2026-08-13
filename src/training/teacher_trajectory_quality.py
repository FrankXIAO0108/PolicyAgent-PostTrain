from __future__ import annotations

import json
from collections import Counter
from types import SimpleNamespace
from typing import Any

from src.guards.retail_pre_action import WRITE_TOOLS
from src.rl.retail_agentic_env import confirmation_diagnostics


AUTO_PASS = "AUTO_PASS_CANDIDATE"
REVIEW = "REVIEW_REQUIRED"
REJECT = "REJECTED"
SYSTEM_FAILURE = "SYSTEM_FAILURE"


def _value(obj: Any, key: str, default: Any = None) -> Any:
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def _tool_calls(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        call
        for message in messages
        if message.get("role") == "assistant"
        for call in (message.get("tool_calls") or [])
    ]


def _signature(call: dict[str, Any]) -> str:
    return f"{call.get('name')}:{json.dumps(call.get('arguments') or {}, sort_keys=True, ensure_ascii=False)}"


def _message_objects(messages: list[dict[str, Any]]) -> list[SimpleNamespace]:
    converted: list[SimpleNamespace] = []
    for message in messages:
        calls = [
            SimpleNamespace(
                id=str(call.get("id", "")),
                name=str(call.get("name", "")),
                arguments=dict(call.get("arguments") or {}),
            )
            for call in (message.get("tool_calls") or [])
        ]
        converted.append(
            SimpleNamespace(
                role=message.get("role"),
                content=message.get("content") or "",
                tool_calls=calls,
                error=bool(message.get("error", False)),
            )
        )
    return converted


def audit_simulation(simulation: dict[str, Any]) -> dict[str, Any]:
    """Assign a conservative automatic triage label to one tau2 rollout.

    This function never emits human GOLD. Expected actions are consumed only
    from the post-run evaluator output, not from the teacher prompt.
    """

    messages = list(simulation.get("messages") or [])
    reward = dict(simulation.get("reward_info") or {})
    calls = _tool_calls(messages)
    call_counts = Counter(_signature(call) for call in calls)
    duplicate_excess = sum(max(0, count - 1) for count in call_counts.values())
    tool_errors = sum(
        message.get("role") == "tool" and bool(message.get("error", False))
        for message in messages
    )
    action_checks = list(reward.get("action_checks") or [])
    unmatched_actions = [
        check
        for check in action_checks
        if not bool(check.get("action_match", False))
    ]
    expected_write_signatures = {
        _signature(check["action"])
        for check in action_checks
        if (check.get("action") or {}).get("name") in WRITE_TOOLS
    }
    actual_writes = [call for call in calls if call.get("name") in WRITE_TOOLS]
    unexpected_writes = [
        call for call in actual_writes if _signature(call) not in expected_write_signatures
    ]
    communication_checks = list(reward.get("communicate_checks") or [])
    unmet_communication = [
        check for check in communication_checks if not bool(check.get("met", False))
    ]
    confirmation = confirmation_diagnostics(_message_objects(messages))
    termination = str(simulation.get("termination_reason") or "")
    terminal_ok = termination in {"agent_stop", "user_stop"}
    db_check = reward.get("db_check") or {}
    db_match = bool(db_check.get("db_match", False))

    hard_reasons: list[str] = []
    review_reasons: list[str] = []
    if not terminal_ok:
        hard_reasons.append("non_terminal_termination")
    if not db_match:
        hard_reasons.append("final_database_state_mismatch")
    if unmatched_actions:
        review_reasons.append("expected_action_mismatch_requires_semantic_review")
    if unmet_communication:
        hard_reasons.append("required_communication_missing")
    if tool_errors:
        hard_reasons.append("tool_error_present")
    if unexpected_writes:
        review_reasons.append("non_reference_write_requires_semantic_review")
    if confirmation["missing_confirmation_count"]:
        review_reasons.append("confirmation_not_detected_by_provisional_rule")
    if duplicate_excess:
        review_reasons.append("duplicate_exact_tool_call")
    if str(simulation.get("task_id") or "") == "57":
        review_reasons.append("task_57_requires_task_specific_semantic_review")
    assistant_content_tool_call_turn_count = sum(
        bool(
            message.get("role") == "assistant"
            and message.get("content")
            and message.get("tool_calls")
        )
        for message in messages
    )
    if not messages or not reward:
        label = SYSTEM_FAILURE
        hard_reasons.append("missing_trajectory_or_evaluation")
    elif hard_reasons:
        label = REJECT
    elif review_reasons:
        label = REVIEW
    else:
        label = AUTO_PASS

    return {
        "candidate_id": str(simulation.get("id") or ""),
        "task_id": str(simulation.get("task_id") or ""),
        "trial": simulation.get("trial"),
        "seed": simulation.get("seed"),
        "automatic_label": label,
        "sft_release_allowed": False,
        "human_review_required": True,
        "hard_rejection_reasons": hard_reasons,
        "review_reasons": review_reasons,
        "metrics": {
            "termination_reason": termination,
            "terminal_ok": terminal_ok,
            "tau2_reward": reward.get("reward"),
            "db_match": db_match,
            "tool_call_count": len(calls),
            "tool_error_count": tool_errors,
            "required_action_count": len(action_checks),
            "unmatched_required_action_count": len(unmatched_actions),
            "unexpected_write_count": len(unexpected_writes),
            "duplicate_exact_call_excess": duplicate_excess,
            "assistant_content_tool_call_turn_count": assistant_content_tool_call_turn_count,
            "required_communication_count": len(communication_checks),
            "unmet_communication_count": len(unmet_communication),
            "confirmation": confirmation,
        },
        "provenance": {
            "teacher_was_blind_to_gold": True,
            "expected_actions_used_post_generation_only": True,
            "human_adjudicated": False,
        },
    }


def summarize_audits(rows: list[dict[str, Any]]) -> dict[str, Any]:
    labels = Counter(row["automatic_label"] for row in rows)
    tasks = Counter(row["task_id"] for row in rows)
    return {
        "candidate_count": len(rows),
        "task_candidate_counts": dict(sorted(tasks.items(), key=lambda item: int(item[0]))),
        "automatic_label_counts": dict(sorted(labels.items())),
        "sft_released_count": 0,
        "human_review_required_count": len(rows),
        "teacher_qualified": False,
        "interpretation": (
            "This smoke can validate the data-production plumbing and expose failure "
            "modes. It cannot qualify the teacher or release SFT data."
        ),
    }
