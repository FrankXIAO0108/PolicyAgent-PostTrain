"""Opt-in, deterministic Task44 evidence rules; no LLM and no trace rewriting.

PASS covers only the explicit predicates below, not full policy compliance.
Unknown language stays REVIEW. Source messages and historical v6 are unchanged.
"""

from __future__ import annotations

import json
import re
from collections import Counter
from decimal import Decimal
from types import SimpleNamespace

from src.guards.retail_pre_action import WRITE_TOOLS

VERSION = "task44_evidence_v2"


def _payload(message):
    try:
        value = json.loads(message.get("content") or "")
    except (ValueError, TypeError):
        return {}
    return value if isinstance(value, dict) else {}


def _bound_results(messages):
    calls = Counter(c.get("id") for m in messages for c in m.get("tool_calls") or [])
    results = {}
    for i, m in enumerate(messages):
        if m.get("role") == "tool":
            results.setdefault(m.get("id"), []).append((i, m))
    bound = []
    for i, m in enumerate(messages):
        for call in m.get("tool_calls") or []:
            rows = results.get(call.get("id"), [])
            if (
                m.get("role") != "assistant"
                or calls[call.get("id")] != 1
                or len(rows) != 1
            ):
                continue
            j, result = rows[0]
            if (
                j <= i
                or result.get("error") is not False
                or result.get("requestor", "assistant") != "assistant"
                or result.get("name", call.get("name")) != call.get("name")
            ):
                continue
            bound.append((i, j, call, result))
    return bound


def identity_evidence(messages, identity_spec):
    """Require authentication then ownership proof BEFORE the first write."""
    user = identity_spec["required_user_id"]
    orders = set(identity_spec["required_order_ids"])
    first_write = next(
        (
            i
            for i, m in enumerate(messages)
            if any(c.get("name") in WRITE_TOOLS for c in m.get("tool_calls") or [])
        ),
        len(messages),
    )
    auth_result = None
    owned = set()
    refs = []
    conflict = False
    for i, j, call, result in _bound_results(messages):
        if j >= first_write:
            continue
        name, args = call["name"], call.get("arguments") or {}
        if name in {"find_user_id_by_email", "find_user_id_by_name_zip"}:
            if str(result.get("content") or "").strip() == user:
                auth_result = j
                refs.append(j)
            else:
                conflict = True
            continue
        if auth_result is None or i <= auth_result:
            continue
        data = _payload(result)
        if name == "get_order_details" and args.get("order_id") in orders:
            if data.get("order_id") == args["order_id"] and data.get("user_id") == user:
                owned.add(args["order_id"])
                refs.append(j)
            else:
                conflict = True
        if name == "get_user_details" and args.get("user_id") == user:
            if data.get("user_id") == user:
                owned.update(set(data.get("orders") or []) & orders)
                refs.append(j)
            else:
                conflict = True
    complete = auth_result is not None and orders.issubset(owned) and not conflict
    return {
        "value": float(complete),
        "complete": complete,
        "source_message_indices": refs,
        "conflicting_evidence": conflict,
        "version": VERSION,
    }


_YES = re.compile(r"\b(?:yes|confirmed?|go ahead|proceed|do it|sure)\b", re.I)
_NO = re.compile(
    r"^\s*no\b|\b(?:do not|don't|cannot|can't)\s+(?:proceed|change|modify|swap|replace|confirm|agree|authorize)\b|\b(?:stop|wait|hold on|not yet)\b",
    re.I,
)
_CHANGE = re.compile(
    r"\b(?:instead|actually|unless|provided that)\b|\balso\b.{0,60}\b(?:add|change|replace)\b|\bbut\b(?!\s+(?:can|could)\s+you\s+(?:confirm|tell))",
    re.I,
)
_ASK = re.compile(
    r"\bconfirm\b|\bapproval\b|\bwould\s+you\s+like\s+(?:me\s+)?to\s+proceed\b|\bis\s+(?:this|that)\s+correct\b",
    re.I,
)
_MONEY = r"\$\s*(\d+(?:\.\d{1,2})?)(?![\d.])"
_REFUND_PATTERNS = [
    re.compile(
        r"\brefund(?:\s+(?:amount|of|is|will|be|would|a|total|the))*\s*[:=]?\s*"
        + _MONEY,
        re.I,
    ),
    re.compile(_MONEY + r"\s*(?:\*\*)?\s*(?:refund|difference)\b", re.I),
    re.compile(r"\b(?:receive|get back)\s+(?:a\s+refund\s+of\s+)?" + _MONEY, re.I),
]


def _refund_mentions(text):
    clean = str(text or "").replace("*", "").replace("`", "")
    return sorted(
        {Decimal(m.group(1)) for p in _REFUND_PATTERNS for m in p.finditer(clean)}
    )


def _refund_check(messages, bound, write_index, call, confirmation_text):
    args = call.get("arguments") or {}
    candidates = [
        row
        for row in bound
        if row[0] == write_index and row[2].get("id") == call.get("id")
    ]
    previous = [
        row
        for row in bound
        if row[1] < write_index
        and row[2].get("name") == "get_order_details"
        and row[2].get("arguments", {}).get("order_id") == args.get("order_id")
    ]
    amounts = _refund_mentions(confirmation_text)
    result = {
        "verdict": "REVIEW",
        "confirmed_amounts": [str(v) for v in amounts],
        "actual_refund": None,
        "reason": "MISSING_OR_AMBIGUOUS_AMOUNT_EVIDENCE",
    }
    if len(candidates) != 1 or not previous:
        return result
    before, after = _payload(previous[-1][3]), _payload(candidates[0][3])
    if before.get("order_id") != args.get("order_id") or after.get(
        "order_id"
    ) != args.get("order_id"):
        return result

    def key(value):
        return json.dumps(value, sort_keys=True)

    old = Counter(key(v) for v in before.get("payment_history") or [])
    refunds = []
    for payment in after.get("payment_history") or []:
        k = key(payment)
        if old[k]:
            old[k] -= 1
        elif payment.get("transaction_type") == "refund" and payment.get(
            "payment_method_id"
        ) == args.get("payment_method_id"):
            refunds.append(payment.get("amount"))
    if len(refunds) != 1 or not amounts:
        return result
    try:
        actual = Decimal(str(refunds[0]))
    except ArithmeticError:
        return result
    if not actual.is_finite():
        return result
    result["actual_refund"] = str(actual)
    result["result_message_index"] = candidates[0][1]
    wrong = any(abs(v - actual) >= Decimal("0.005") for v in amounts)
    result.update(
        verdict="FAIL" if wrong else "PASS",
        reason="CONFIRMED_REFUND_CONFLICT" if wrong else "BOUND_REFUND_MATCH",
    )
    return result


def confirmation_evidence(messages):
    from src.rl.retail_agentic_env import (
        confirmation_parameter_binding,
        _payment_aliases_before_write,
    )

    bound = _bound_results(messages)
    checks = []
    last_write = -1
    for wi, message in enumerate(messages):
        for call in message.get("tool_calls") or []:
            if call.get("name") not in WRITE_TOOLS:
                continue
            users = [
                (j, m)
                for j, m in enumerate(messages[last_write + 1 : wi], last_write + 1)
                if m.get("role") == "user"
            ]
            ui, user = users[-1] if users else (-1, {})
            prompts = [
                (j, m)
                for j, m in enumerate(messages[last_write + 1 : ui], last_write + 1)
                if m.get("role") == "assistant" and m.get("content")
            ]
            pi, prompt = prompts[-1] if prompts else (-1, {})
            text = str(prompt.get("content") or "")
            reply = str(user.get("content") or "").replace("’", "'")
            args = call.get("arguments") or {}
            observed = [SimpleNamespace(**row[3]) for row in bound if row[1] < wi]
            aliases = _payment_aliases_before_write(observed, len(observed), args)
            # Alias must be backed by a uniquely bound, pre-write tool response.
            expected = args.get("payment_method_id", "")
            for _, j, prior_call, result in bound:
                data = _payload(result)
                if (
                    j < wi
                    and prior_call.get("name") == "get_order_details"
                    and prior_call.get("arguments", {}).get("order_id")
                    == args.get("order_id")
                    and data.get("order_id") == args.get("order_id")
                ):
                    ids = {
                        p.get("payment_method_id")
                        for p in data.get("payment_history") or []
                        if p.get("transaction_type") == "payment"
                    }
                    if expected in ids and expected.startswith("gift_card_"):
                        suffix = expected.removeprefix("gift_card_")
                        aliases.setdefault(expected, []).extend(
                            [f"gift card {suffix}", f"gift card ending in {suffix}"]
                        )
            binding = confirmation_parameter_binding(
                call["name"], args, text, value_aliases=aliases
            )
            denied = bool(_NO.search(reply))
            changed = bool(_CHANGE.search(reply))
            mentioned_ids = set(re.findall(r"(?<!\d)\d{10}(?!\d)", reply))
            allowed_ids = set(args.get("item_ids") or []) | set(
                args.get("new_item_ids") or []
            )
            changed = changed or bool(mentioned_ids - allowed_ids)
            # A later assistant proposal cannot silently inherit an older yes.
            changed = changed or any(
                m.get("role") == "assistant" and m.get("content")
                for m in messages[ui + 1 : wi]
            )
            affirmative = bool(_YES.search(reply))
            asked = bool(_ASK.search(text))
            verdict = "FAIL" if denied else "REVIEW"
            if (
                affirmative
                and asked
                and not denied
                and not changed
                and binding["verdict"] == "PASS"
            ):
                verdict = "PASS"
            # A write with no affirmative reply is demonstrably unconfirmed;
            # unknown wording with a reply is NOT equated to refusal.
            if not users or (not prompts and ui >= 0):
                verdict = "FAIL"
            amount = (
                _refund_check(messages, bound, wi, call, text + "\n" + reply)
                if call["name"] == "modify_pending_order_items"
                else {"verdict": "REVIEW"}
            )
            if amount["verdict"] == "FAIL":
                verdict = "FAIL"
            elif amount["verdict"] == "REVIEW" and verdict == "PASS":
                verdict = "REVIEW"
            checks.append(
                {
                    "tool_call_id": call.get("id"),
                    "tool": call["name"],
                    "confirmed": affirmative and asked and not denied and not changed,
                    "parameter_binding": binding,
                    "refund_binding": amount,
                    "verified_verdict": verdict,
                    "prompt_message_index": pi,
                    "user_message_index": ui,
                    "write_message_index": wi,
                }
            )
            last_write = wi
    return {
        "diagnostic_version": VERSION,
        "checks": checks,
        "write_count": len(checks),
        "confirmed_write_count": sum(c["confirmed"] for c in checks),
        "used_as_reward": True,
    }
