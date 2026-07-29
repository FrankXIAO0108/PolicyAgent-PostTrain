from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from typing import Any

from src.rules.scope_checker import check_scope_confirmation
from src.rules.variant_checker import check_variant_understanding

from .db_diff import StateDiff
from .nl_checker import NLCheckResult
from .replay_evaluator import ReplayResult


@dataclass(slots=True)
class RootCause:
    code: str
    explanation: str
    evidence: list[Any] = field(default_factory=list)
    caused_official_failure: bool = True
    confidence: str = "high"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _messages(simulation: Any, role: str | None = None) -> list[str]:
    return [
        str(message.content or "")
        for message in simulation.messages or []
        if role is None or message.role == role
    ]


def _tool_calls(simulation: Any) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for message in simulation.messages or []:
        calls = getattr(message, "tool_calls", None) or []
        for call in calls:
            output.append(
                {
                    "name": call.name,
                    "arguments": call.arguments,
                    "message_index": getattr(message, "turn_idx", None),
                }
            )
    return output


def _legacy_evidence(replay: ReplayResult) -> dict[str, list[str]]:
    task_text = str(replay.task.user_scenario)
    assistant = _messages(replay.simulation, "assistant")
    calls = _tool_calls(replay.simulation)
    return {
        "user_intent": [task_text],
        "constraints": [task_text],
        "agent_actions": assistant,
        "tool_actions": [
            f"{call['name']}({call['arguments']})" for call in calls
        ],
    }


def _same_item_exchanges(calls: list[dict[str, Any]]) -> list[dict[str, Any]]:
    violations = []
    for call in calls:
        if call["name"] != "exchange_delivered_order_items":
            continue
        old_items = call["arguments"].get("item_ids") or []
        new_items = call["arguments"].get("new_item_ids") or []
        if any(old == new for old, new in zip(old_items, new_items)):
            violations.append(call)
    return violations


def _multi_call_messages(simulation: Any) -> list[int | None]:
    return [
        getattr(message, "turn_idx", None)
        for message in simulation.messages or []
        if len(getattr(message, "tool_calls", None) or []) > 1
    ]


def _gold_reuses_variant(replay: ReplayResult) -> bool:
    variants: list[str] = []
    for action in replay.task.evaluation_criteria.actions or []:
        if action.name == "exchange_delivered_order_items":
            variants.extend(action.arguments.get("new_item_ids") or [])
    return len(variants) != len(set(variants))


def _confirmed_actual_cancel_conflicts_with_gold(
    replay: ReplayResult, calls: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    gold_cancel_orders = {
        action.arguments.get("order_id")
        for action in replay.task.evaluation_criteria.actions or []
        if action.name == "cancel_pending_order"
    }
    user_text = "\n".join(_messages(replay.simulation, "user")).lower()
    conflicts = []
    for call in calls:
        if call["name"] != "cancel_pending_order":
            continue
        order_id = call["arguments"].get("order_id")
        if (
            order_id not in gold_cancel_orders
            and order_id
            and order_id.lower() in user_text
            and re.search(
                rf"(confirm|yes|proceed)[\s\S]{{0,180}}{re.escape(order_id.lower())}"
                rf"|{re.escape(order_id.lower())}[\s\S]{{0,180}}(confirm|yes|proceed)",
                user_text,
            )
        ):
            conflicts.append(call)
    return conflicts


def attribute_failure(
    replay: ReplayResult,
    state_diff: StateDiff,
    nl_result: NLCheckResult,
) -> dict[str, Any]:
    """Map official signals to evidence-backed root causes."""

    calls = _tool_calls(replay.simulation)
    assistant_text = "\n".join(_messages(replay.simulation, "assistant")).lower()
    causes: list[RootCause] = []
    flags = state_diff.flags

    conflicts = _confirmed_actual_cancel_conflicts_with_gold(replay, calls)
    if conflicts:
        causes.append(
            RootCause(
                code="golden_mismatch",
                explanation=(
                    "The executed cancellation follows the simulator's explicit "
                    "confirmed order, while the static gold cancels a different order."
                ),
                evidence=conflicts,
            )
        )
        causes.append(
            RootCause(
                code="dataset_issue",
                explanation=(
                    "Static gold and the latest simulated user intent are inconsistent; "
                    "the trajectory should not be used as an ordinary negative."
                ),
                evidence=["latest_user_confirmation_vs_gold_actions"],
                confidence="high",
            )
        )

    missing_exchanges = flags["missing_exchange"]
    transferred = any(call["name"] == "transfer_to_human_agents" for call in calls)
    inventory_misread = (
        missing_exchanges
        and transferred
        and "only one available" in assistant_text
        and _gold_reuses_variant(replay)
    )
    if inventory_misread:
        causes.append(
            RootCause(
                code="variant_understanding_failure",
                explanation=(
                    "The agent interpreted a variant's availability boolean as a "
                    "single inventory unit, although the gold reuses that variant "
                    "across multiple orders."
                ),
                evidence=["only one available", "gold_reuses_same_new_item_id"],
            )
        )

    if flags["wrong_variant"] and not conflicts:
        causes.append(
            RootCause(
                code="wrong_variant",
                explanation="The reconstructed exchange_new_items differ from gold.",
                evidence=[
                    {
                        order_id: value["differing_fields"].get("exchange_new_items")
                        for order_id, value in state_diff.order_diffs.items()
                        if "exchange_new_items" in value["differing_fields"]
                    }
                ],
            )
        )

    if flags["wrong_payment"]:
        causes.append(
            RootCause(
                code="wrong_payment_method",
                explanation=(
                    "The exchange/return payment method persisted in DB differs "
                    "from the gold action."
                ),
                evidence=[
                    {
                        order_id: value["differing_fields"].get(
                            "exchange_payment_method_id"
                        )
                        for order_id, value in state_diff.order_diffs.items()
                        if "exchange_payment_method_id" in value["differing_fields"]
                    }
                ],
            )
        )

    if flags["partial_scope"]:
        causes.append(
            RootCause(
                code="scope_confirmation_failure",
                explanation=(
                    "The items affected by the exchange are only a subset/superset "
                    "of the gold scope."
                ),
                evidence=["exchange_items_diff"],
            )
        )

    cancel_calls = [
        call for call in calls if call["name"] == "cancel_pending_order"
    ]
    if cancel_calls and "skateboard cancelled" in assistant_text:
        affected_orders = {
            call["arguments"].get("order_id") for call in cancel_calls
        }
        multi_item_orders = []
        initial_orders = replay.initial_state.get("agent", {}).get("orders", {})
        for order_id in affected_orders:
            if len(initial_orders.get(order_id, {}).get("items", [])) > 1:
                multi_item_orders.append(order_id)
        if multi_item_orders:
            causes.append(
                RootCause(
                    code="scope_confirmation_failure",
                    explanation=(
                        "The tool cancelled a multi-item order, but the final claim "
                        "described an item-only cancellation."
                    ),
                    evidence=multi_item_orders,
                    caused_official_failure=False,
                )
            )

    same_item = _same_item_exchanges(calls)
    if same_item:
        causes.append(
            RootCause(
                code="policy_violation",
                explanation=(
                    "Retail policy requires exchange to a different product option, "
                    "but the tool call used the same old and new item ID. The tool "
                    "accepted a business-rule violation."
                ),
                evidence=same_item,
            )
        )

    multi_call_indices = _multi_call_messages(replay.simulation)
    if multi_call_indices:
        causes.append(
            RootCause(
                code="policy_violation",
                explanation=(
                    "Retail policy permits at most one tool call per assistant turn."
                ),
                evidence=multi_call_indices,
                caused_official_failure=False,
            )
        )

    if missing_exchanges:
        causes.append(
            RootCause(
                code="missing_action",
                explanation="One or more gold exchanges were not reflected in final DB.",
                evidence=[
                    order_id
                    for order_id, value in state_diff.order_diffs.items()
                    if "exchange_items" in value["differing_fields"]
                ],
            )
        )
    if flags["missing_cancel"]:
        causes.append(
            RootCause(
                code="missing_action",
                explanation="A gold cancellation was not reflected in final DB.",
                evidence=["missing_cancel"],
            )
        )
    if nl_result.nl_match is False:
        causes.append(
            RootCause(
                code="communication_omission",
                explanation="At least one frozen official NL assertion failed.",
                evidence=[
                    item
                    for item in nl_result.assertions
                    if not item.get("met", False)
                ],
            )
        )

    # Preserve existing v6-era rules as secondary diagnostics. They are not
    # permitted to override the reconstructed official signal.
    legacy_evidence = _legacy_evidence(replay)
    legacy_checks = {
        "scope_checker": check_scope_confirmation(legacy_evidence),
        "variant_checker": check_variant_understanding(legacy_evidence),
    }

    unique: list[RootCause] = []
    seen: set[tuple[str, bool, str]] = set()
    for cause in causes:
        key = (cause.code, cause.caused_official_failure, cause.explanation)
        if key not in seen:
            seen.add(key)
            unique.append(cause)

    suggestions = []
    codes = {cause.code for cause in unique}
    if "golden_mismatch" in codes:
        suggestions.append("Reconcile latest user intent with static gold; quarantine case.")
    if "variant_understanding_failure" in codes or "wrong_variant" in codes:
        suggestions.append("Resolve variants structurally from product options and availability.")
    if "scope_confirmation_failure" in codes:
        suggestions.append("Confirm and report the complete tool effect scope.")
    if "wrong_payment_method" in codes:
        suggestions.append("Bind the confirmed payment method to every exchange action.")
    if "policy_violation" in codes:
        suggestions.append("Run deterministic policy checks before write tool execution.")
    if "missing_action" in codes:
        suggestions.append("Track every requested goal until completed, denied, or transferred.")
    if "communication_omission" in codes:
        suggestions.append("Generate final communication from verified tool results.")

    return {
        "root_causes": [cause.to_dict() for cause in unique],
        "legacy_rule_checks": legacy_checks,
        "improvement_suggestions": suggestions,
    }
