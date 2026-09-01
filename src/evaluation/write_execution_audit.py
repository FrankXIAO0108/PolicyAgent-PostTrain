"""Local write-result audit; deliberately not connected to a training reward.

PASS means a required call has a uniquely bound, non-error tool response.
It does not establish final database correctness, authorization or policy safety.
Inputs use the same task/message objects as the Retail environment. No tools,
models or evaluators are executed by this module.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any

from src.guards.retail_pre_action import WRITE_TOOLS
from src.rl.retail_agentic_env import _action_matches


def audit_write_execution(task: Any, messages: list[Any]) -> dict[str, Any]:
    criteria = getattr(task, "evaluation_criteria", None)
    if criteria is None or getattr(criteria, "actions", None) is None:
        raise ValueError("Task action criteria are required; missing is not empty")
    expected = [
        a
        for a in criteria.actions
        if a.requestor == "assistant" and a.name in WRITE_TOOLS
    ]
    action_ids = [a.action_id for a in expected]
    if any(not isinstance(x, str) or not x for x in action_ids):
        raise ValueError("Required write action IDs must be nonempty strings")
    if len(set(action_ids)) != len(action_ids):
        raise ValueError("Required write action IDs must be unique")

    calls = []
    results: dict[str, list[tuple[int, Any]]] = defaultdict(list)
    for position, message in enumerate(messages):
        if getattr(message, "role", None) in {"assistant", "user"}:
            for call in getattr(message, "tool_calls", None) or []:
                calls.append((position, call))
        if getattr(message, "role", None) == "tool":
            results[getattr(message, "id", None)].append((position, message))
    id_counts = Counter(getattr(call, "id", None) for _, call in calls)

    checks = []
    for index, (position, call) in enumerate(calls):
        if (
            getattr(messages[position], "role", None) != "assistant"
            or call.name not in WRITE_TOOLS
            or getattr(call, "requestor", None) != "assistant"
        ):
            continue
        call_id = getattr(call, "id", None)
        bound = results.get(call_id, [])
        status, reason = "ERROR", "MISSING_RESULT"
        if not isinstance(call_id, str) or not call_id:
            reason = "MISSING_CALL_ID"
        elif id_counts[call_id] != 1:
            reason = "DUPLICATE_CALL_ID"
        elif len(bound) > 1:
            reason = "DUPLICATE_RESULT_ID"
        elif bound:
            response_position, response = bound[0]
            error = getattr(response, "error", None)
            content = getattr(response, "content", None)
            response_name = getattr(response, "name", None)
            if response_position <= position:
                reason = "RESULT_BEFORE_CALL"
            elif getattr(response, "requestor", "assistant") != "assistant":
                reason = "RESULT_REQUESTOR_MISMATCH"
            elif response_name is not None and response_name != call.name:
                reason = "RESULT_TOOL_NAME_MISMATCH"
            elif not isinstance(error, bool):
                reason = "INVALID_ERROR_FLAG"
            elif error:
                status, reason = "FAIL", "TOOL_REPORTED_ERROR"
            elif not isinstance(content, str) or not content.strip():
                reason = "MISSING_RESULT_CONTENT"
            else:
                status, reason = "PASS", "BOUND_NONERROR_RESPONSE"
        checks.append(
            {
                "call_index": index,
                "call_id": call_id,
                "name": call.name,
                "message_index": position,
                "status": status,
                "reason": reason,
            }
        )

    candidate_groups = [
        [
            check
            for check in checks
            if _action_matches(action, calls[check["call_index"]][1])
        ]
        for action in expected
    ]
    # Maximal one-to-one coverage, not first-match greediness: a broad
    # compare_args requirement must not consume the only match for a stricter one.
    assigned: dict[int, int] = {}

    def allocate(action_index: int, seen: set[int]) -> bool:
        for check in candidate_groups[action_index]:
            index = check["call_index"]
            if check["status"] != "PASS" or index in seen:
                continue
            seen.add(index)
            if index not in assigned or allocate(assigned[index], seen):
                assigned[index] = action_index
                return True
        return False

    for action_index in range(len(expected)):
        allocate(action_index, set())
    selected = {action_index: index for index, action_index in assigned.items()}
    required = []
    for action_index, action in enumerate(expected):
        candidates = candidate_groups[action_index]
        successful = next(
            (c for c in candidates if c["call_index"] == selected.get(action_index)),
            None,
        )
        if successful is not None:
            status, reason = "PASS", "BOUND_NONERROR_RESPONSE"
        elif any(c["status"] == "ERROR" for c in candidates):
            status, reason = "ERROR", "INCOMPLETE_OR_AMBIGUOUS_EVIDENCE"
        else:
            status, reason = "FAIL", "NO_UNUSED_SUCCESSFUL_MATCH"
        required.append(
            {
                "action_id": action.action_id,
                "name": action.name,
                "status": status,
                "reason": reason,
                "candidate_call_indices": [c["call_index"] for c in candidates],
                "successful_call_index": successful["call_index"]
                if successful
                else None,
                "successful_call_id": successful["call_id"] if successful else None,
            }
        )
    evidence_valid = not any(c["status"] == "ERROR" for c in checks)
    count = sum(row["status"] == "PASS" for row in required)
    status = (
        "ERROR"
        if not evidence_valid
        else "NOT_APPLICABLE"
        if not required
        else "PASS"
        if count == len(required)
        else "FAIL"
    )
    return {
        "schema_version": "write-execution-audit-v1",
        "status": status,
        "evidence_valid": evidence_valid,
        "required_write_count": len(required),
        "verified_write_count": count,
        "verified_write_fraction": (
            count / len(required) if required and evidence_valid else None
        ),
        "required_writes": required,
        "write_calls": checks,
        "final_state_verified": False,
        "policy_verified": False,
        "used_as_training_reward": False,
    }
