from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


_ORDER_EFFECT_FIELDS = (
    "status",
    "address",
    "cancel_reason",
    "exchange_items",
    "exchange_new_items",
    "exchange_payment_method_id",
    "exchange_price_difference",
    "return_items",
    "return_payment_method_id",
    "payment_history",
)


@dataclass(slots=True)
class StateDiff:
    changed_paths: list[dict[str, Any]] = field(default_factory=list)
    order_diffs: dict[str, dict[str, Any]] = field(default_factory=dict)
    flags: dict[str, bool] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "changed_paths": self.changed_paths,
            "order_diffs": self.order_diffs,
            "flags": self.flags,
        }


def _recursive_diff(
    agent: Any,
    gold: Any,
    *,
    path: str = "",
    output: list[dict[str, Any]],
) -> None:
    if isinstance(agent, dict) and isinstance(gold, dict):
        for key in sorted(set(agent) | set(gold)):
            child = f"{path}.{key}" if path else str(key)
            if key not in agent:
                output.append({"path": child, "agent": None, "gold": gold[key]})
            elif key not in gold:
                output.append({"path": child, "agent": agent[key], "gold": None})
            else:
                _recursive_diff(agent[key], gold[key], path=child, output=output)
        return
    if agent != gold:
        output.append({"path": path, "agent": agent, "gold": gold})


def _changed_order_fields(initial: dict[str, Any], final: dict[str, Any]) -> set[str]:
    return {
        field
        for field in _ORDER_EFFECT_FIELDS
        if initial.get(field) != final.get(field)
    }


def _has_refund(order: dict[str, Any]) -> bool:
    return any(
        payment.get("transaction_type") == "refund"
        for payment in order.get("payment_history") or []
        if isinstance(payment, dict)
    )


def analyze_db_diff(
    initial_state: dict[str, Any],
    agent_state: dict[str, Any],
    gold_state: dict[str, Any],
) -> StateDiff:
    """Explain a Tau2 DB mismatch with Retail-aware order effect flags."""

    changed_paths: list[dict[str, Any]] = []
    _recursive_diff(agent_state, gold_state, output=changed_paths)

    initial_db = initial_state.get("agent", {})
    agent_db = agent_state.get("agent", {})
    gold_db = gold_state.get("agent", {})
    initial_orders = initial_db.get("orders", {})
    agent_orders = agent_db.get("orders", {})
    gold_orders = gold_db.get("orders", {})
    order_diffs: dict[str, dict[str, Any]] = {}

    flags = {
        "extra_cancel": False,
        "missing_cancel": False,
        "missing_exchange": False,
        "extra_exchange": False,
        "wrong_variant": False,
        "wrong_payment": False,
        "wrong_refund": False,
        "wrong_address": False,
        "wrong_status": False,
        "partial_scope": False,
        "missing_mutation": False,
        "extra_mutation": False,
    }

    for order_id in sorted(set(initial_orders) | set(agent_orders) | set(gold_orders)):
        initial = initial_orders.get(order_id, {})
        agent = agent_orders.get(order_id, {})
        gold = gold_orders.get(order_id, {})
        differing_fields = {
            field: {"agent": agent.get(field), "gold": gold.get(field)}
            for field in _ORDER_EFFECT_FIELDS
            if agent.get(field) != gold.get(field)
        }
        if not differing_fields:
            continue

        agent_changes = _changed_order_fields(initial, agent)
        gold_changes = _changed_order_fields(initial, gold)
        order_diffs[order_id] = {
            "differing_fields": differing_fields,
            "agent_mutations": sorted(agent_changes),
            "gold_mutations": sorted(gold_changes),
        }

        agent_cancelled = agent.get("status") == "cancelled"
        gold_cancelled = gold.get("status") == "cancelled"
        agent_exchange = bool(agent.get("exchange_items"))
        gold_exchange = bool(gold.get("exchange_items"))
        flags["extra_cancel"] |= agent_cancelled and not gold_cancelled
        flags["missing_cancel"] |= gold_cancelled and not agent_cancelled
        flags["missing_exchange"] |= gold_exchange and not agent_exchange
        flags["extra_exchange"] |= agent_exchange and not gold_exchange
        flags["wrong_status"] |= agent.get("status") != gold.get("status")
        flags["wrong_address"] |= agent.get("address") != gold.get("address")
        flags["wrong_refund"] |= _has_refund(agent) != _has_refund(gold)
        flags["wrong_payment"] |= (
            (agent_exchange and gold_exchange)
            and agent.get("exchange_payment_method_id")
            != gold.get("exchange_payment_method_id")
        ) or (
            bool(agent.get("return_items"))
            and bool(gold.get("return_items"))
            and agent.get("return_payment_method_id")
            != gold.get("return_payment_method_id")
        )

        agent_items = set(agent.get("exchange_items") or [])
        gold_items = set(gold.get("exchange_items") or [])
        agent_variants = agent.get("exchange_new_items") or []
        gold_variants = gold.get("exchange_new_items") or []
        flags["partial_scope"] |= bool(agent_items & gold_items) and agent_items != gold_items
        flags["wrong_variant"] |= (
            agent_exchange
            and gold_exchange
            and agent_variants != gold_variants
        )
        flags["missing_mutation"] |= bool(gold_changes - agent_changes)
        flags["extra_mutation"] |= bool(agent_changes - gold_changes)

    return StateDiff(
        changed_paths=changed_paths,
        order_diffs=order_diffs,
        flags=flags,
    )
