from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any


WRITE_TOOLS = {
    "cancel_pending_order",
    "exchange_delivered_order_items",
    "modify_pending_order_address",
    "modify_pending_order_items",
    "modify_pending_order_payment",
    "return_delivered_order_items",
}

ONE_SHOT_ORDER_MUTATION_TOOLS = {
    "exchange_delivered_order_items",
    "modify_pending_order_address",
    "modify_pending_order_items",
    "modify_pending_order_payment",
}


class GuardDecision(str, Enum):
    ALLOW = "ALLOW"
    REQUIRE_CONFIRMATION = "REQUIRE_CONFIRMATION"
    REGENERATE = "REGENERATE"
    BLOCK = "BLOCK"
    TRANSFER = "TRANSFER"


@dataclass(slots=True)
class ToolProposal:
    name: str
    arguments: dict[str, Any]
    id: str = ""


@dataclass(slots=True)
class GuardFinding:
    rule_id: str
    category: str
    severity: str
    blocking: bool
    message: str
    evidence: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class GuardResult:
    allowed: bool
    findings: list[GuardFinding] = field(default_factory=list)

    @property
    def decision(self) -> GuardDecision:
        return resolve_guard_decision(self.findings)

    def to_dict(self) -> dict[str, Any]:
        return {
            "allowed": self.allowed,
            "decision": self.decision.value,
            "findings": [finding.to_dict() for finding in self.findings],
        }


@dataclass(slots=True)
class GuardContext:
    orders: dict[str, dict[str, Any]] = field(default_factory=dict)
    products: dict[str, dict[str, Any]] = field(default_factory=dict)
    item_catalog: dict[str, dict[str, Any]] = field(default_factory=dict)
    payment_method_ids: set[str] = field(default_factory=set)
    user_texts: list[str] = field(default_factory=list)
    completed_writes: list[ToolProposal] = field(default_factory=list)
    reference_actions: list[ToolProposal] = field(default_factory=list)
    enforce_reference: bool = False


_CONFIRMATION_RULES = {
    "scope.item_request_would_cancel_whole_order",
}

_TERMINAL_BLOCK_RULES = {
    "policy.cancel_requires_pending",
    "policy.delivered_action_requires_delivered",
    "policy.one_shot_order_mutation",
}


def resolve_guard_decision(
    findings: list[GuardFinding],
) -> GuardDecision:
    blocking = [finding for finding in findings if finding.blocking]
    if not blocking:
        return GuardDecision.ALLOW
    rule_ids = {finding.rule_id for finding in blocking}
    if rule_ids & _CONFIRMATION_RULES:
        return GuardDecision.REQUIRE_CONFIRMATION
    if rule_ids & _TERMINAL_BLOCK_RULES:
        return GuardDecision.BLOCK
    return GuardDecision.REGENERATE


def _finding(
    rule_id: str,
    category: str,
    message: str,
    *,
    blocking: bool,
    severity: str = "major",
    evidence: dict[str, Any] | None = None,
) -> GuardFinding:
    return GuardFinding(
        rule_id=rule_id,
        category=category,
        severity=severity,
        blocking=blocking,
        message=message,
        evidence=evidence or {},
    )


def _normal(text: str) -> str:
    return re.sub(r"\s+", " ", text.lower()).strip()


def _mentioned_cancel_items(
    context: GuardContext,
    order: dict[str, Any],
) -> list[str]:
    text = _normal("\n".join(context.user_texts))
    mentioned: list[str] = []
    for item in order.get("items") or []:
        name = _normal(str(item.get("name", "")))
        if not name:
            continue
        cancel_near_name = re.search(
            rf"\bcancel(?:lation of|ling|led)?"
            rf"(?:\s+(?:the|my|this))?\s+{re.escape(name)}\b"
            rf"|\b{re.escape(name)}\b\s+cancell?ation\b",
            text,
        )
        if cancel_near_name:
            mentioned.append(str(item.get("item_id")))
    return mentioned


def _whole_order_confirmed(context: GuardContext, order_id: str) -> bool:
    text = _normal("\n".join(context.user_texts))
    order = re.escape(order_id.lower())
    return bool(
        re.search(
            rf"(whole|entire|full)\s+order.{{0,100}}{order}"
            rf"|{order}.{{0,100}}(whole|entire|full)\s+order",
            text,
        )
    )


def _known_variant_checks(
    proposal: ToolProposal,
    context: GuardContext,
) -> list[GuardFinding]:
    if proposal.name not in {
        "exchange_delivered_order_items",
        "modify_pending_order_items",
    }:
        return []

    old_items = list(proposal.arguments.get("item_ids") or [])
    new_items = list(proposal.arguments.get("new_item_ids") or [])
    findings: list[GuardFinding] = []
    if len(old_items) != len(new_items) or not old_items:
        findings.append(
            _finding(
                "variant.parallel_list_shape",
                "variant_error",
                "Old and new item lists must be non-empty and have equal length.",
                blocking=True,
                evidence={"item_ids": old_items, "new_item_ids": new_items},
            )
        )
        return findings

    if any(old == new for old, new in zip(old_items, new_items)):
        findings.append(
            _finding(
                "policy.exchange_requires_different_option",
                "policy_error",
                "An exchange must select a different product option.",
                blocking=True,
                evidence={"item_ids": old_items, "new_item_ids": new_items},
            )
        )

    for old_item, new_item in zip(old_items, new_items):
        old = context.item_catalog.get(str(old_item))
        new = context.item_catalog.get(str(new_item))
        if new and new.get("available") is False:
            findings.append(
                _finding(
                    "variant.new_item_unavailable",
                    "variant_error",
                    "The selected replacement variant is unavailable.",
                    blocking=True,
                    evidence={"new_item_id": new_item},
                )
            )
        if (
            old
            and new
            and old.get("product_id")
            and new.get("product_id")
            and old["product_id"] != new["product_id"]
        ):
            findings.append(
                _finding(
                    "variant.product_type_mismatch",
                    "variant_error",
                    "Old and replacement items belong to different product types.",
                    blocking=True,
                    evidence={
                        "old_item_id": old_item,
                        "new_item_id": new_item,
                        "old_product_id": old["product_id"],
                        "new_product_id": new["product_id"],
                    },
                )
            )
    return findings


def _payment_checks(
    proposal: ToolProposal,
    context: GuardContext,
) -> list[GuardFinding]:
    payment_id = proposal.arguments.get("payment_method_id")
    if not payment_id or not context.payment_method_ids:
        return []
    if payment_id in context.payment_method_ids:
        return []
    return [
        _finding(
            "payment.method_not_owned",
            "payment_error",
            "The selected payment method is not in the authenticated user's profile.",
            blocking=True,
            evidence={
                "payment_method_id": payment_id,
                "known_payment_method_ids": sorted(context.payment_method_ids),
            },
        )
    ]


def _order_checks(
    proposal: ToolProposal,
    context: GuardContext,
) -> list[GuardFinding]:
    order_id = str(proposal.arguments.get("order_id", ""))
    order = context.orders.get(order_id)
    if not order:
        return []

    findings: list[GuardFinding] = []
    status = order.get("status")
    if proposal.name == "cancel_pending_order" and status != "pending":
        findings.append(
            _finding(
                "policy.cancel_requires_pending",
                "policy_error",
                "Only pending orders can be cancelled.",
                blocking=True,
                evidence={"order_id": order_id, "status": status},
            )
        )
    if proposal.name in {
        "exchange_delivered_order_items",
        "return_delivered_order_items",
    } and status != "delivered":
        findings.append(
            _finding(
                "policy.delivered_action_requires_delivered",
                "policy_error",
                "Return and exchange actions require a delivered order.",
                blocking=True,
                evidence={"order_id": order_id, "status": status},
            )
        )

    if proposal.name == "cancel_pending_order":
        mentioned = _mentioned_cancel_items(context, order)
        order_items = [
            str(item.get("item_id"))
            for item in order.get("items") or []
            if item.get("item_id")
        ]
        if (
            mentioned
            and len(mentioned) < len(order_items)
            and not _whole_order_confirmed(context, order_id)
        ):
            findings.append(
                _finding(
                    "scope.item_request_would_cancel_whole_order",
                    "scope_error",
                    "The user named only item(s), but this tool cancels the whole order. "
                    "Explicit whole-order confirmation is required.",
                    blocking=True,
                    evidence={
                        "order_id": order_id,
                        "mentioned_item_ids": mentioned,
                        "all_order_item_ids": order_items,
                    },
                )
            )
    return findings


def _completed_write_checks(
    proposal: ToolProposal,
    context: GuardContext,
) -> list[GuardFinding]:
    if proposal.name not in WRITE_TOOLS:
        return []
    order_id = proposal.arguments.get("order_id")
    previous = [
        call
        for call in context.completed_writes
        if call.arguments.get("order_id") == order_id
        and call.name in ONE_SHOT_ORDER_MUTATION_TOOLS
    ]
    if not previous:
        return []
    if proposal.name in ONE_SHOT_ORDER_MUTATION_TOOLS:
        return [
            _finding(
                "policy.one_shot_order_mutation",
                "policy_error",
                "Exchange/modify tools can only be called once per order.",
                blocking=True,
                evidence={
                    "order_id": order_id,
                    "previous_tools": [call.name for call in previous],
                },
            )
        ]
    return []


def _target_values_for_product(
    context: GuardContext,
    product: dict[str, Any],
) -> set[str]:
    text = _normal("\n".join(context.user_texts))
    values = {
        _normal(str(value))
        for variant in (product.get("variants") or {}).values()
        for value in (variant.get("options") or {}).values()
    }
    requested: set[str] = set()
    for value in values:
        if not value:
            continue
        for match in re.finditer(re.escape(value), text):
            prefix = text[max(0, match.start() - 35) : match.start()]
            if re.search(
                r"(not interested in|do not want|don't want|not)"
                r"(?:\s+the)?\s*$",
                prefix,
            ):
                continue
            requested.add(value)
            break
    return requested


def _requested_product_quantity(context: GuardContext, product_name: str) -> int:
    text = _normal("\n".join(context.user_texts))
    name = _normal(product_name)
    plural = f"{name}s"
    patterns = {
        2: (
            rf"\b(two|2|both|a couple of|couple of)\s+"
            rf"{re.escape(plural)}\b"
        ),
        3: rf"\b(three|3)\s+{re.escape(plural)}\b",
    }
    for quantity, pattern in patterns.items():
        if re.search(pattern, text):
            return quantity
    return 1 if re.search(rf"\b{name}s?\b", text) else 0


def _actionable_transfer_findings(
    proposal: ToolProposal,
    context: GuardContext,
) -> list[GuardFinding]:
    if proposal.name != "transfer_to_human_agents":
        return []

    completed_exchange_count = sum(
        call.name == "exchange_delivered_order_items"
        for call in context.completed_writes
    )
    for product in context.products.values():
        name = str(product.get("name", ""))
        quantity = _requested_product_quantity(context, name)
        if quantity <= completed_exchange_count:
            continue
        target_values = _target_values_for_product(context, product)
        if len(target_values) < 2:
            continue
        candidates = [
            item_id
            for item_id, variant in (product.get("variants") or {}).items()
            if variant.get("available") is True
            and target_values.issubset(
                {_normal(str(value)) for value in variant.get("options", {}).values()}
            )
        ]
        matching_owned_items = [
            item
            for order in context.orders.values()
            if order.get("status") == "delivered"
            for item in order.get("items") or []
            if item.get("product_id") == product.get("product_id")
        ]
        if candidates and len(matching_owned_items) >= quantity:
            return [
                _finding(
                    "goal.transfer_with_actionable_variant",
                    "variant_error",
                    "Transfer is premature: an available variant can satisfy the "
                    "requested product attributes, and availability is boolean rather "
                    "than a one-unit inventory count.",
                    blocking=True,
                    evidence={
                        "product": name,
                        "requested_quantity": quantity,
                        "candidate_item_ids": candidates,
                        "matching_owned_item_count": len(matching_owned_items),
                    },
                )
            ]
    return []


def _reference_checks(
    proposal: ToolProposal,
    context: GuardContext,
) -> list[GuardFinding]:
    if not context.enforce_reference or proposal.name not in WRITE_TOOLS:
        return []
    order_id = proposal.arguments.get("order_id")
    candidates = [
        action
        for action in context.reference_actions
        if action.name == proposal.name
        and action.arguments.get("order_id") == order_id
    ]
    if not candidates:
        return [
            _finding(
                "reference.unexpected_action",
                "reference_mismatch",
                "The proposed write action has no matching reference action.",
                blocking=True,
                evidence={"proposal": asdict(proposal)},
            )
        ]
    if any(action.arguments == proposal.arguments for action in candidates):
        return []
    differing_keys = sorted(
        {
            key
            for action in candidates
            for key in set(action.arguments) | set(proposal.arguments)
            if action.arguments.get(key) != proposal.arguments.get(key)
        }
    )
    if "payment_method_id" in differing_keys:
        category = "payment_error"
    elif "new_item_ids" in differing_keys:
        category = "variant_error"
    elif "item_ids" in differing_keys:
        category = "scope_error"
    else:
        category = "reference_mismatch"
    return [
        _finding(
            "reference.argument_mismatch",
            category,
            "The proposed write arguments differ from the frozen reference.",
            blocking=True,
            evidence={
                "proposal": asdict(proposal),
                "reference_candidates": [asdict(action) for action in candidates],
                "differing_keys": differing_keys,
            },
        )
    ]


def evaluate_retail_actions(
    proposals: list[ToolProposal],
    context: GuardContext,
    *,
    assistant_content: str = "",
) -> GuardResult:
    findings: list[GuardFinding] = []
    writes = [proposal for proposal in proposals if proposal.name in WRITE_TOOLS]
    if len(proposals) > 1:
        findings.append(
            _finding(
                "protocol.one_tool_call_per_turn",
                "policy_error",
                "Retail policy permits at most one tool call per assistant turn.",
                blocking=len(writes) > 1,
                severity="major" if len(writes) > 1 else "minor",
                evidence={
                    "tool_names": [proposal.name for proposal in proposals],
                    "write_count": len(writes),
                },
            )
        )
    if proposals and assistant_content.strip():
        findings.append(
            _finding(
                "protocol.no_content_with_tool_call",
                "policy_error",
                "An assistant turn cannot contain both user-facing text and a tool call.",
                blocking=False,
                severity="minor",
            )
        )

    for proposal in proposals:
        findings.extend(_known_variant_checks(proposal, context))
        findings.extend(_payment_checks(proposal, context))
        findings.extend(_order_checks(proposal, context))
        findings.extend(_completed_write_checks(proposal, context))
        findings.extend(_actionable_transfer_findings(proposal, context))
        findings.extend(_reference_checks(proposal, context))

    return GuardResult(
        allowed=not any(finding.blocking for finding in findings),
        findings=findings,
    )


def observe_tool_result(
    context: GuardContext,
    proposal: ToolProposal,
    content: str | None,
    *,
    error: bool = False,
) -> None:
    if error:
        return
    try:
        payload = json.loads(content or "")
    except (json.JSONDecodeError, TypeError):
        payload = None

    if proposal.name == "get_user_details" and isinstance(payload, dict):
        context.payment_method_ids.update(
            str(key) for key in (payload.get("payment_methods") or {})
        )
    elif proposal.name == "get_order_details" and isinstance(payload, dict):
        order_id = str(payload.get("order_id", ""))
        if order_id:
            context.orders[order_id] = payload
        for item in payload.get("items") or []:
            item_id = str(item.get("item_id", ""))
            if item_id:
                context.item_catalog[item_id] = {
                    **item,
                    "available": True,
                }
    elif proposal.name == "get_product_details" and isinstance(payload, dict):
        product_id = str(payload.get("product_id", ""))
        if product_id:
            context.products[product_id] = payload
        for item_id, variant in (payload.get("variants") or {}).items():
            context.item_catalog[str(item_id)] = {
                **variant,
                "product_id": product_id,
            }

    if proposal.name in WRITE_TOOLS:
        context.completed_writes.append(proposal)


def context_from_messages(messages: list[Any]) -> GuardContext:
    """Hydrate guard context from Tau2 messages or compatible message objects."""

    context = GuardContext()
    pending: dict[str, ToolProposal] = {}
    for message in messages:
        role = getattr(message, "role", None)
        if role == "user":
            context.user_texts.append(str(getattr(message, "content", "") or ""))
            continue
        if role == "assistant":
            for call in getattr(message, "tool_calls", None) or []:
                proposal = ToolProposal(
                    id=str(getattr(call, "id", "") or ""),
                    name=str(getattr(call, "name", "") or ""),
                    arguments=dict(getattr(call, "arguments", {}) or {}),
                )
                if proposal.id:
                    pending[proposal.id] = proposal
            continue
        if role == "tool":
            call_id = str(getattr(message, "id", "") or "")
            proposal = pending.get(call_id)
            if proposal is not None:
                observe_tool_result(
                    context,
                    proposal,
                    getattr(message, "content", None),
                    error=bool(getattr(message, "error", False)),
                )
    return context
