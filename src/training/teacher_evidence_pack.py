from __future__ import annotations

import json
import re
from typing import Any


ORDER_ID = re.compile(r"#?W\d{7}", re.IGNORECASE)
STATUS_PATTERNS = {
    "cancelled": re.compile(
        r"\b(?:has been|was successfully|is now|successfully) "
        r"(?:cancelled|canceled)\b",
        re.IGNORECASE,
    ),
    "return requested": re.compile(
        r"\b(?:return has been requested|return request (?:was|is) submitted)\b",
        re.IGNORECASE,
    ),
    "exchange requested": re.compile(
        r"\b(?:exchange has been requested|exchange request (?:was|is) submitted|"
        r"updated to\s+\*{0,2}[\"']?exchange requested[\"']?\*{0,2}|"
        r"exchange requested for order)\b",
        re.IGNORECASE,
    ),
}
AMBIGUOUS_STATUS_LANGUAGE = re.compile(
    r"\b(cancel(?:led|ed)?|return request(?:ed)?|exchange request(?:ed)?)\b",
    re.IGNORECASE,
)


def _calls(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        call
        for message in messages
        if message.get("role") == "assistant"
        for call in (message.get("tool_calls") or [])
    ]


def final_answer(messages: list[dict[str, Any]]) -> str:
    for message in reversed(messages):
        if message.get("role") == "assistant" and str(message.get("content") or "").strip():
            return str(message["content"]).strip()
    return ""


def state_diff(before: Any, after: Any, path: str = "") -> list[dict[str, Any]]:
    if isinstance(before, dict) and isinstance(after, dict):
        rows: list[dict[str, Any]] = []
        for key in sorted(set(before) | set(after)):
            child = f"{path}.{key}" if path else str(key)
            if key not in before:
                rows.append({"path": child, "before": None, "after": after[key]})
            elif key not in after:
                rows.append({"path": child, "before": before[key], "after": None})
            else:
                rows.extend(state_diff(before[key], after[key], child))
        return rows
    if isinstance(before, list) and isinstance(after, list):
        return [] if before == after else [{"path": path, "before": before, "after": after}]
    return [] if before == after else [{"path": path, "before": before, "after": after}]


def _referenced_values(messages: list[dict[str, Any]]) -> set[str]:
    values: set[str] = set()
    for call in _calls(messages):
        stack = [call.get("arguments") or {}]
        while stack:
            value = stack.pop()
            if isinstance(value, dict):
                stack.extend(value.values())
            elif isinstance(value, list):
                stack.extend(value)
            elif isinstance(value, str):
                values.add(value)
    for message in messages:
        values.update(match.group(0).upper().replace("W", "#W", 1) if not match.group(0).startswith("#") else match.group(0).upper() for match in ORDER_ID.finditer(str(message.get("content") or "")))
    return values


def relevant_state(
    before: dict[str, Any], after: dict[str, Any], messages: list[dict[str, Any]]
) -> dict[str, Any]:
    refs = _referenced_values(messages)
    changed_paths = {row["path"] for row in state_diff(before, after)}
    output: dict[str, Any] = {"initial": {}, "final": {}, "selection": {"referenced_values": sorted(refs), "changed_paths": sorted(changed_paths)}}
    for side, source in (("initial", before), ("final", after)):
        agent = source.get("agent") or {}
        selected: dict[str, Any] = {}
        for collection in ("users", "orders", "products"):
            values = agent.get(collection) or {}
            picked = {
                key: value
                for key, value in values.items()
                if key in refs
                or any(path.startswith(f"agent.{collection}.{key}") for path in changed_paths)
            }
            if picked:
                selected[collection] = picked
        output[side] = selected
    return output


def claim_state_consistency(
    messages: list[dict[str, Any]], final_state: dict[str, Any]
) -> dict[str, Any]:
    answer = final_answer(messages)
    orders = ((final_state.get("agent") or {}).get("orders") or {})
    findings: list[dict[str, Any]] = []
    explicit_order_ids = {
        (value if value.startswith("#") else f"#{value}").upper()
        for value in ORDER_ID.findall(answer)
    }
    for order_id in sorted(explicit_order_ids):
        order = orders.get(order_id)
        if order is None:
            findings.append({"claim": {"order_id": order_id}, "verdict": "UNVERIFIED", "reason_code": "ORDER_NOT_FOUND_IN_FINAL_STATE"})
            continue
        window_start = max(0, answer.upper().find(order_id) - 120)
        window_end = min(len(answer), answer.upper().find(order_id) + len(order_id) + 120)
        window = answer[window_start:window_end]
        for claimed_status, pattern in STATUS_PATTERNS.items():
            if not pattern.search(window):
                continue
            actual_status = str(order.get("status") or "")
            findings.append(
                {
                    "claim": {"order_id": order_id, "status": claimed_status},
                    "actual": {"status": actual_status},
                    "verdict": "SUPPORTED" if actual_status == claimed_status else "CONTRADICTED",
                    "reason_code": "CLAIM_MATCHES_FINAL_STATE" if actual_status == claimed_status else "CLAIM_CONTRADICTS_FINAL_STATE",
                }
            )
    status_language = bool(AMBIGUOUS_STATUS_LANGUAGE.search(answer))
    if status_language and not findings:
        findings.append({"claim": {"text": answer}, "verdict": "UNVERIFIED", "reason_code": "BROAD_STATUS_CLAIM_WITHOUT_ENTITY_BINDING"})
    verdicts = {row["verdict"] for row in findings}
    overall = "FAIL" if "CONTRADICTED" in verdicts else "REVIEW" if "UNVERIFIED" in verdicts else "PASS" if findings else "NOT_APPLICABLE"
    return {"verdict": overall, "final_answer": answer, "findings": findings}


def compact_trajectory(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    for index, message in enumerate(messages):
        role = str(message.get("role") or "")
        row: dict[str, Any] = {"event_id": f"E{index:03d}", "role": role}
        if role == "assistant":
            if message.get("content"):
                row["content"] = str(message["content"])
            if message.get("tool_calls"):
                row["tool_calls"] = message["tool_calls"]
        elif role == "tool":
            row.update({"tool_call_id": message.get("id"), "result": message.get("content"), "error": bool(message.get("error", False))})
        elif role == "user":
            row["content"] = str(message.get("content") or "")
        events.append(row)
    return events


def build_evidence_pack(
    *,
    simulation: dict[str, Any],
    task: dict[str, Any],
    initial_state: dict[str, Any],
    final_state: dict[str, Any],
    automatic_audit: dict[str, Any],
    prompt_audit: dict[str, Any] | None,
) -> dict[str, Any]:
    messages = list(simulation.get("messages") or [])
    state = relevant_state(initial_state, final_state, messages)
    diff = state_diff(state["initial"], state["final"])
    claim_check = claim_state_consistency(messages, final_state)
    return {
        "schema_version": "retail-teacher-evidence-pack-v1",
        "candidate_id": str(simulation.get("id") or ""),
        "task_id": str(simulation.get("task_id") or task.get("id") or ""),
        "review_state": {"status": "PENDING", "human_quality_label": None, "reviewer_id": None, "rationale": None},
        "task_goal": (task.get("user_scenario") or {}).get("instructions"),
        "initial_state": state["initial"],
        "final_state": state["final"],
        "state_diff": diff,
        "trajectory": compact_trajectory(messages),
        "claim_state_consistency": claim_check,
        "automatic_verification": automatic_audit,
        "teacher_prompt_audit": prompt_audit,
        "recommended_label": automatic_audit["automatic_label"],
        "training_release_allowed": False,
        "review_instructions": [
            "Inspect referenced events and state changes before assigning a human label.",
            "Automatic and model recommendations are routing signals, not gold labels.",
        ],
    }
