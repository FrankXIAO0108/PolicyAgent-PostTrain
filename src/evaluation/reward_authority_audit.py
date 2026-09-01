from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any


SCHEMA_VERSION = "retail-reward-authority-audit-v1"
IDENTITY_AUTHENTICATION_STAGE = "IDENTITY_AUTHENTICATION"
IDENTITY_AUTHENTICATION_ACTIONS = {
    "find_user_id_by_email",
    "find_user_id_by_name_zip",
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not rows:
        raise ValueError(f"No JSONL rows found in {path}")
    return rows


def _tool_events(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    pending: list[dict[str, Any]] = []
    for message_index, message in enumerate(messages):
        for call in message.get("tool_calls") or []:
            event = {
                "message_index": message_index,
                "name": str(call.get("name") or ""),
                "arguments": dict(call.get("arguments") or {}),
                "result_observed": False,
                "result_error": None,
                "result_content_nonempty": None,
                "result_content": None,
            }
            events.append(event)
            pending.append(event)
        if message.get("role") == "tool" and pending:
            event = pending.pop(0)
            event["result_observed"] = True
            event["result_error"] = bool(message.get("error", False))
            event["result_content_nonempty"] = bool(
                str(message.get("content") or "").strip()
            )
            event["result_content"] = str(message.get("content") or "").strip()
    return events


def _final_response_observation(messages: list[dict[str, Any]]) -> dict[str, Any]:
    tool_indices = [
        index
        for index, message in enumerate(messages)
        if message.get("role") == "tool"
    ]
    if not tool_indices:
        return {
            "verdict": "UNKNOWN",
            "assistant_text_after_last_tool": [],
            "note": "No tool result exists, so a post-tool final response cannot be assessed.",
        }
    last_tool_index = max(tool_indices)
    later_assistant_text = [
        str(message.get("content") or "").strip()
        for index, message in enumerate(messages)
        if index > last_tool_index
        and message.get("role") == "assistant"
        and str(message.get("content") or "").strip()
    ]
    return {
        "verdict": "OBSERVED" if later_assistant_text else "UNKNOWN",
        "assistant_text_after_last_tool": later_assistant_text,
        "note": (
            "The environment rollout log does not persist the trainer-side final "
            "non-tool completion when no assistant text follows the final tool result."
            if not later_assistant_text
            else "Assistant text after the final tool result is present in the environment log."
        ),
    }


def _high_confidence_policy_failures(reward: dict[str, Any]) -> list[dict[str, Any]]:
    failures: list[dict[str, Any]] = []
    for finding in reward.get("diagnostic_policy_findings") or []:
        severity = str(finding.get("severity") or "").upper()
        verdict = str(finding.get("verdict") or "").upper()
        if severity in {"MAJOR", "CRITICAL"} or verdict == "FAIL":
            failures.append(dict(finding))
    return failures


def audit_rollout(
    row: dict[str, Any],
    *,
    rollout_index: int,
    expected_user_id: str | None = None,
) -> dict[str, Any]:
    stage = str(row.get("rollout_stage") or "")
    if stage != IDENTITY_AUTHENTICATION_STAGE:
        raise ValueError(
            "Reward authority audit v1 supports only IDENTITY_AUTHENTICATION rollouts"
        )
    reward = row.get("reward")
    messages = row.get("messages")
    if not isinstance(reward, dict) or "reward" not in reward:
        raise ValueError(f"Rollout {rollout_index} is missing reward.reward")
    if not isinstance(messages, list):
        raise ValueError(f"Rollout {rollout_index} is missing messages")

    events = _tool_events(messages)
    auth_indices = [
        index
        for index, event in enumerate(events)
        if event["name"] in IDENTITY_AUTHENTICATION_ACTIONS
    ]
    executable_auth_indices = [
        index
        for index in auth_indices
        if events[index]["result_observed"]
        and events[index]["result_error"] is False
        and events[index]["result_content_nonempty"] is True
    ]
    first_auth_index = auth_indices[0] if auth_indices else None
    post_auth_events = (
        events[first_auth_index + 1 :] if first_auth_index is not None else []
    )
    stage_boundary_events = [
        event
        for event in post_auth_events
        if event["name"] not in IDENTITY_AUTHENTICATION_ACTIONS
    ]
    tool_error_count = sum(event["result_error"] is True for event in events)
    auth_result_matches_expected = (
        events[executable_auth_indices[0]]["result_content"] == expected_user_id
        if len(executable_auth_indices) == 1 and expected_user_id is not None
        else None
    )
    if (
        len(auth_indices) != 1
        or len(executable_auth_indices) != 1
        or stage_boundary_events
        or tool_error_count
    ):
        observable_core_success: bool | None = False
    elif auth_result_matches_expected is None:
        observable_core_success = None
    else:
        observable_core_success = auth_result_matches_expected
    current_reward = float(reward["reward"])
    current_positive = current_reward > 0.0
    conflicts: list[dict[str, Any]] = []
    if current_positive and observable_core_success is False:
        conflicts.append(
            {
                "code": "POSITIVE_REWARD_CONTRADICTS_OBSERVABLE_STAGE_CORE",
                "severity": "HIGH",
                "reason": (
                    "Positive reward was assigned although the observable stage core "
                    "contains a missing/failed authentication action, a post-authentication "
                    "boundary violation, or a tool error."
                ),
            }
        )
    if not current_positive and observable_core_success is True:
        conflicts.append(
            {
                "code": "ZERO_REWARD_WITH_SUCCESSFUL_AUTH_RESULT_REQUIRES_GROUNDING_REVIEW",
                "severity": "REVIEW",
                "reason": (
                    "Exactly one authentication tool returned the expected user ID, no "
                    "later tool crossed the stage boundary, and no tool error was observed. "
                    "The hidden action matcher nevertheless assigned zero reward. This may "
                    "be an exact-match false negative, but argument grounding against the "
                    "customer's explicit disclosures still requires review."
                ),
            }
        )

    policy_failures = _high_confidence_policy_failures(reward)
    if current_positive and policy_failures:
        conflicts.append(
            {
                "code": "POSITIVE_REWARD_WITH_HIGH_CONFIDENCE_POLICY_FAILURE",
                "severity": "HIGH",
                "reason": "A high-confidence policy failure coexists with positive reward.",
            }
        )

    action_progress = dict(reward.get("action_progress") or {})
    return {
        "rollout_index": rollout_index,
        "task_id": str(row.get("task_id")),
        "rollout_stage": stage,
        "current_reward": current_reward,
        "current_stage_complete": reward.get("stage_complete"),
        "observable_stage_core_success": observable_core_success,
        "authority_verdict": (
            "REVIEW_REQUIRED"
            if any(item["severity"] == "REVIEW" for item in conflicts)
            else "OBSERVABLE_CORE_CONTRADICTED"
            if conflicts
            else (
                "OBSERVABLE_CORE_ALIGNED"
                if observable_core_success is not None
                else "NOT_EVALUATED"
            )
        ),
        "tool_sequence": [event["name"] for event in events],
        "authentication": {
            "call_count": len(auth_indices),
            "executable_call_count": len(executable_auth_indices),
            "expected_user_id": expected_user_id,
            "result_matches_expected_user_id": auth_result_matches_expected,
            "hidden_action_match_recall": action_progress.get("recall"),
            "hidden_action_matched_count": action_progress.get("matched_count"),
            "events": [events[index] for index in auth_indices],
        },
        "stage_boundary": {
            "verdict": "FAIL" if stage_boundary_events else "PASS",
            "post_authentication_non_stage_calls": stage_boundary_events,
        },
        "tool_error_count": tool_error_count,
        "high_confidence_policy_failures": policy_failures,
        "final_response_observation": _final_response_observation(messages),
        "conflicts": conflicts,
        "claim_limits": {
            "full_task_success_assessed": False,
            "reasoning_correctness_assessed": False,
            "final_response_correctness_assessed": False,
        },
    }


def build_reward_authority_audit(
    rows: list[dict[str, Any]],
    *,
    source: dict[str, Any],
    expected_user_ids: dict[str, str] | None = None,
) -> dict[str, Any]:
    expected_user_ids = expected_user_ids or {}
    audited = [
        audit_rollout(
            row,
            rollout_index=index,
            expected_user_id=expected_user_ids.get(str(row.get("task_id"))),
        )
        for index, row in enumerate(rows, start=1)
    ]
    conflicts = [item for row in audited for item in row["conflicts"]]
    conflict_counts = Counter(item["code"] for item in conflicts)
    tool_sequences = Counter(
        " -> ".join(row["tool_sequence"]) or "<none>" for row in audited
    )
    boundary_failures = [
        row for row in audited if row["stage_boundary"]["verdict"] == "FAIL"
    ]
    return {
        "schema_version": SCHEMA_VERSION,
        "scope": "IDENTITY_AUTHENTICATION_REWARD_AUTHORITY_SHADOW_AUDIT",
        "source": source,
        "summary": {
            "rollout_count": len(audited),
            "positive_reward_count": sum(row["current_reward"] > 0 for row in audited),
            "observable_core_aligned_count": sum(
                row["authority_verdict"] == "OBSERVABLE_CORE_ALIGNED"
                for row in audited
            ),
            "observable_core_contradicted_count": sum(
                row["authority_verdict"] == "OBSERVABLE_CORE_CONTRADICTED"
                for row in audited
            ),
            "review_required_count": sum(
                row["authority_verdict"] == "REVIEW_REQUIRED" for row in audited
            ),
            "not_evaluated_count": sum(
                row["authority_verdict"] == "NOT_EVALUATED" for row in audited
            ),
            "conflict_counts": dict(sorted(conflict_counts.items())),
            "stage_boundary_failure_count": len(boundary_failures),
            "stage_boundary_failure_task_ids": sorted(
                {row["task_id"] for row in boundary_failures}, key=int
            ),
            "final_response_unknown_count": sum(
                row["final_response_observation"]["verdict"] == "UNKNOWN"
                for row in audited
            ),
            "tool_sequence_counts": dict(tool_sequences.most_common()),
        },
        "rows": audited,
        "conclusion_limits": [
            "This audit checks the observable identity-authentication stage only.",
            "A successful tool result can expose hidden-action matcher false negatives, but does not prove full-task success.",
            "A successful authentication result is marked REVIEW when tool arguments may have been inferred rather than explicitly grounded in user disclosures.",
            "Final response correctness is UNKNOWN when trainer-side non-tool text is absent from the environment log.",
            "Policy checks include only high-confidence findings already persisted in the rollout reward payload.",
            "No shadow signal in this report changes the online training reward.",
        ],
    }


def _load_expected_user_ids(tau2_root: Path) -> dict[str, str]:
    task_path = (
        tau2_root.resolve()
        / "data"
        / "tau2"
        / "domains"
        / "retail"
        / "tasks.json"
    )
    if not task_path.is_file():
        raise FileNotFoundError(task_path)
    tasks = json.loads(task_path.read_text(encoding="utf-8"))
    expected: dict[str, str] = {}
    for task in tasks:
        criteria = task.get("evaluation_criteria") or {}
        candidate_user_ids: set[str] = set()
        for action in criteria.get("actions") or []:
            user_id = str((action.get("arguments") or {}).get("user_id") or "")
            if user_id:
                candidate_user_ids.add(user_id)
        if len(candidate_user_ids) == 1:
            expected[str(task.get("id"))] = next(iter(candidate_user_ids))
    return expected


def audit_run(raw_rollouts_path: Path, *, tau2_root: Path) -> dict[str, Any]:
    path = raw_rollouts_path.resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    return build_reward_authority_audit(
        _read_jsonl(path),
        source={
            "raw_rollouts_path": str(path),
            "raw_rollouts_sha256": _sha256(path),
            "tau2_root": str(tau2_root.resolve()),
        },
        expected_user_ids=_load_expected_user_ids(tau2_root),
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Audit observable authority conflicts in staged Agentic RL reward"
    )
    parser.add_argument("--raw-rollouts", type=Path, required=True)
    parser.add_argument("--tau2-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    report = audit_run(args.raw_rollouts, tau2_root=args.tau2_root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
