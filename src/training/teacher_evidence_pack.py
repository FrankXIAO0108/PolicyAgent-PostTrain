from __future__ import annotations

import json
import re
from typing import Any


ORDER_ID = re.compile(r"#?W\d{7}", re.IGNORECASE)
AMOUNT_CLAIM = re.compile(
    r"(?:(?:total(?:\s+paid)?|paid|amount)(?:\s+(?:was|is|of))?|which\s+was)"
    r"\s*[:=-]?\s*\*{0,2}"
    r"\$\s*([0-9][0-9,]*(?:\.\d{2})?)",
    re.IGNORECASE,
)
UNSUPPORTED_SELECTION_CLAIM = re.compile(
    r"\b(?:most recent|latest)\b[\s\S]{0,160}?"
    r"(?:is|was|would be|appears to be)\s+\*{0,2}(?:order\s+)?"
    r"\*{0,2}#?W\d{7}\b",
    re.IGNORECASE,
)
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


def _normalise_order_id(value: str) -> str:
    result = value.upper()
    return result if result.startswith("#") else f"#{result}"


def _single_payment_amount(order: dict[str, Any]) -> float | None:
    """Return only an unambiguous single payment amount.

    Exchange/refund histories can contain multiple transactions whose business
    meaning cannot be recovered safely by blindly summing them.
    """

    payments = [
        row.get("amount")
        for row in (order.get("payment_history") or [])
        if str(row.get("transaction_type") or "").lower() == "payment"
        and row.get("amount") is not None
    ]
    return float(payments[0]) if len(payments) == 1 else None


def _order_facts(
    messages: list[dict[str, Any]], final_state: dict[str, Any]
) -> dict[str, dict[str, Any]]:
    facts: dict[str, dict[str, Any]] = {}
    state_orders = ((final_state.get("agent") or {}).get("orders") or {})
    for key, value in state_orders.items():
        if not isinstance(value, dict):
            continue
        order_id = _normalise_order_id(str(value.get("order_id") or key))
        facts[order_id] = {
            "status": value.get("status"),
            "single_payment_amount": _single_payment_amount(value),
            "sources": ["final_state"],
        }

    for message in messages:
        if message.get("role") != "tool" or bool(message.get("error", False)):
            continue
        content = message.get("content")
        if not isinstance(content, str):
            continue
        try:
            value = json.loads(content)
        except (TypeError, json.JSONDecodeError):
            continue
        if not isinstance(value, dict) or not value.get("order_id"):
            continue
        order_id = _normalise_order_id(str(value["order_id"]))
        fact = facts.setdefault(order_id, {"sources": []})
        if value.get("status") is not None:
            fact["status"] = value["status"]
        amount = _single_payment_amount(value)
        if amount is not None:
            fact["single_payment_amount"] = amount
        fact["sources"].append("tool_observation")
    return facts


def _order_mentions(answer: str) -> list[tuple[str, str]]:
    """Split an answer into local order-bound segments.

    The segment ends at the next order ID so an amount listed for one order is
    not accidentally bound to a neighbouring order.
    """

    matches = list(ORDER_ID.finditer(answer))
    return [
        (
            _normalise_order_id(match.group(0)),
            answer[match.start() : matches[index + 1].start()]
            if index + 1 < len(matches)
            else answer[match.start() :],
        )
        for index, match in enumerate(matches)
    ]


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
    orders = _order_facts(messages, final_state)
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
    for order_id, segment in _order_mentions(answer):
        order = orders.get(order_id)
        for match in AMOUNT_CLAIM.finditer(segment):
            claimed_amount = float(match.group(1).replace(",", ""))
            actual_amount = (
                order.get("single_payment_amount") if order is not None else None
            )
            if actual_amount is None:
                findings.append(
                    {
                        "claim": {
                            "order_id": order_id,
                            "single_payment_amount": claimed_amount,
                        },
                        "verdict": "UNVERIFIED",
                        "reason_code": "ORDER_PAYMENT_AMOUNT_NOT_UNAMBIGUOUSLY_AVAILABLE",
                    }
                )
                continue
            matches_amount = abs(float(actual_amount) - claimed_amount) < 0.005
            findings.append(
                {
                    "claim": {
                        "order_id": order_id,
                        "single_payment_amount": claimed_amount,
                    },
                    "actual": {"single_payment_amount": float(actual_amount)},
                    "verdict": "SUPPORTED" if matches_amount else "CONTRADICTED",
                    "reason_code": (
                        "CLAIM_MATCHES_TOOL_OR_FINAL_STATE"
                        if matches_amount
                        else "CLAIM_CONTRADICTS_TOOL_OR_FINAL_STATE"
                    ),
                }
            )
    selection_match = UNSUPPORTED_SELECTION_CLAIM.search(answer)
    if selection_match:
        findings.append(
            {
                "claim": {"text": selection_match.group(0)},
                "verdict": "UNVERIFIED",
                "reason_code": "COMPARATIVE_SELECTION_OUTSIDE_PROGRAMMATIC_CHECKER_SCOPE",
            }
        )
    status_language = bool(AMBIGUOUS_STATUS_LANGUAGE.search(answer))
    if status_language and not findings:
        findings.append({"claim": {"text": answer}, "verdict": "UNVERIFIED", "reason_code": "BROAD_STATUS_CLAIM_WITHOUT_ENTITY_BINDING"})
    verdicts = {row["verdict"] for row in findings}
    overall = "FAIL" if "CONTRADICTED" in verdicts else "REVIEW" if "UNVERIFIED" in verdicts else "PASS" if findings else "NOT_APPLICABLE"
    return {
        "verdict": overall,
        "final_answer": answer,
        "findings": findings,
        "scope_notes": [
            "Only explicit order-bound status and unambiguous single-payment claims are checked.",
            "Comparative selections such as latest/most recent are outside the current checker scope and are routed to review.",
            "This checker does not use Tau2 hidden reference actions or NL assertions.",
        ],
    }


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
