"""Bounded English refund-timing diagnostic; opt-in, not a general NL judge."""
from __future__ import annotations

import json
import re
from typing import Any

GIFT = r"gift[\s_-]*cards?"
CARD = r"(?:credit[\s_-]*cards?|visa|mastercard|american express)"
TIMING_V2 = r"\b(?:immediate(?:ly)?|instant(?:ly)?)\b"
# Split independent assertions, not payment-type enumerations ('gift or credit').
SCOPE_BREAK_V2 = re.compile(
    r";|\b(?:but|however|otherwise)\b|,\s*(?:and|or)\s+"
    r"|\band\s+(?=(?:your|the|this|that|both|these|those|it|refunds?|funds?)\b)"
    r"|\bor\s+(?=(?:immediate|instant|within)\b)", re.I,
)


def _scope_v2(sentence: str, start: int, end: int) -> tuple[str, str]:
    """Use full-sentence boundary offsets so lookahead is not lost at the anchor."""
    boundaries = list(SCOPE_BREAK_V2.finditer(sentence))
    left = max((m.end() for m in boundaries if m.end() <= start), default=0)
    right = min((m.start() for m in boundaries if m.start() >= end), default=len(sentence))
    return sentence[left:start], sentence[end:right]


def _classify_v2(before: str, after: str, sources: list[str]) -> tuple[str, str]:
    """Bounded local semantics; REVIEW is not treated as a verified violation."""
    local = before + " immediately " + after
    if re.search(TIMING_V2, before + after, re.I):
        return "REVIEW", "multiple timing assertions without a supported clause boundary"
    # The if-clause may be parenthesized or follow a payment-method phrase.
    conditions = re.findall(r"\bif\b([^,;()]{0,160})", local, re.I)
    gift_conditions = [c for c in conditions if re.search(GIFT, c, re.I)]
    ambiguous_condition = any(re.search(rf"\b(?:not|unless)\b|{CARD}|\bpaypal\b", c, re.I)
                              for c in gift_conditions)
    generic_gift = bool(re.search(rf"\bfor\s+(?:(?:the|any|all)\s+)?{GIFT}\b", before, re.I)
                        or re.match(rf"\s+for\s+(?:(?:the|any|all)\s+)?{GIFT}\b", after, re.I))
    negated = bool(re.search(
        r"\b(?:not|never)\s+(?:(?:be|processed|credited|returned|refunded|received)\s+){0,3}$",
        before, re.I))
    if negated:
        return "PASS", "explicitly negated immediate-refund assertion"
    if re.search(rf"\byour\s+{GIFT}\b", local, re.I) and any(s != "gift_card" for s in sources):
        return "REVIEW", "gift-card ownership assertion conflicts or has unclear scope"
    if ambiguous_condition:
        return "REVIEW", "gift condition is negated or includes non-gift payment types"
    if generic_gift and re.search(rf"{GIFT}\s+(?:or|and)\s+{CARD}", local, re.I):
        return "REVIEW", "generic immediate-refund branch mixes gift and credit cards"
    if gift_conditions or generic_gift:
        return "PASS", "local gift-card condition; independent assertions checked separately"
    if re.search(r"\b(?:if|unless|not|never|cannot|can't)\b", local, re.I):
        return "REVIEW", "conditional or negated scope is outside the supported grammar"
    # Processing a REQUEST does not assert that money has arrived. This narrow
    # exemption cannot mask a later credited/arrival assertion in another clause.
    if re.search(r"\brefund\s+request\b", before, re.I) and re.search(r"\bprocessed\s*$", before, re.I):
        return "PASS", "immediate request processing, not an immediate funds-arrival claim"
    if len(set(sources)) > 1:
        return "REVIEW", "statement does not disambiguate mixed payment sources"
    if sources[0] == "gift_card":
        if re.search(CARD, local, re.I):
            return "REVIEW", "claimed card type differs from bound gift-card payment"
        return "PASS", "bound original payments are gift cards"
    # 'Processed immediately' alone does not resolve initiation versus arrival.
    # A causal statement tying immediate refunds to credit-card payment is an
    # explicit false policy explanation, not merely processing-time language.
    causal_card = bool(re.search(rf"\b(?:since|because)\b[^;.!?]{{0,100}}{CARD}", local, re.I))
    if re.search(r"\bprocessed\s*$", before, re.I) and not causal_card:
        return "REVIEW", "refund processing versus funds arrival is not explicit"
    if re.search(r"\([^)]*gift[\s_-]*card", local, re.I):
        return "REVIEW", "parenthetical payment-type assertion conflicts or has unclear scope"
    if re.search(CARD, local, re.I) or re.search(r"\b(?:credited|returned|received|refunded)\b", before, re.I):
        return "FAIL", "immediate refund asserted for bound non-gift-card payment"
    if re.search(r"\brefund\w*\b.*\b(?:is|are|be)\s*$", before, re.I):
        return "FAIL", "immediate refund timing asserted for bound non-gift-card payment"
    return "REVIEW", "immediate processing versus funds arrival is not explicit"


def _payload(trace: list[dict[str, Any]], expected: dict[str, Any]) -> dict[str, Any]:
    matches = [c for c in trace if c.get("name") == expected.get("name")
               and c.get("arguments") == expected.get("arguments")
               and not (c.get("result") or {}).get("error")]
    if not matches:
        raise ValueError("missing successful payment evidence call")
    value = json.loads(matches[-1]["result"]["content"])
    if not isinstance(value, dict):
        raise ValueError("payment evidence is not an object")
    return value


def refund_timing_diagnostic(
    messages: list[dict[str, Any]], trace: list[dict[str, Any]], rule: dict[str, Any]
) -> dict[str, Any]:
    """Return local anchors and REVIEW for unsupported/ambiguous grammar.

    PASS only means no immediate-refund conflict in this bounded detector.
    Sources establish actual payment state, not whether the agent knew it at
    the moment of its statement; authorization remains a separate predicate.
    """
    findings: list[dict[str, Any]] = []
    use_v2 = rule.get("rule_type") == "refund_timing_by_payment_v2"
    candidates = []
    for mi, message in enumerate(messages):
        if message.get("role") != "assistant":
            continue
        text = str(message.get("content") or "").replace("*", "").replace("’", "'")
        if not re.search(r"\brefund\w*\b", text, re.I):
            continue
        for sentence in re.split(r"[.!?](?:\s+|$)|[\r\n]+", text):
            timing = TIMING_V2 if use_v2 else r"\bimmediate(?:ly)?\b"
            for match in re.finditer(timing, sentence, re.I):
                # No cross-message or cross-sentence trigger concatenation.
                if re.search(r"\b(?:refund\w*|credit(?:ed)?|return(?:ed)?|receive\w*)\b", sentence, re.I):
                    candidates.append((mi, sentence, match.start(), match.end()))
    if not candidates:
        return {"verdict": "PASS", "findings": [], "errors": []}
    try:
        user = _payload(trace, rule["source_call"])
        if not isinstance(user.get("user_id"), str) or not user["user_id"]:
            raise ValueError("payment owner identity is missing")
        if use_v2 and user["user_id"] != rule["source_call"]["arguments"].get("user_id"):
            raise ValueError("payment owner does not match requested user")
        methods = user["payment_methods"]
        if not isinstance(methods, dict):
            raise ValueError("payment_methods must be an object")
        bindings = {}
        for call in rule["order_source_calls"]:
            order = _payload(trace, call)
            oid = call["arguments"]["order_id"]
            if order.get("order_id") != oid or order.get("user_id") != user.get("user_id"):
                raise ValueError("order identity does not match payment owner")
            sources = []
            for payment in order["payment_history"]:
                if payment.get("transaction_type") != "payment":
                    continue
                method = methods[payment["payment_method_id"]]
                kind = method["source"]
                if kind not in {"credit_card", "gift_card", "paypal"}:
                    raise ValueError("unsupported payment source")
                sources.append(kind)
            if not sources:
                raise ValueError("original order payments are missing")
            bindings[oid] = sources
        if not bindings:
            raise ValueError("no bound order source calls")
    except (KeyError, TypeError, ValueError) as exc:
        return {"verdict": "ERROR", "findings": [], "errors": [{"error": str(exc)}]}

    for mi, sentence, start, end in candidates:
        if use_v2:
            before, after = _scope_v2(sentence, start, end)
            explicit_ids = [oid for oid in bindings if oid in before + after]
            sources = [s for oid in (explicit_ids or list(bindings)) for s in bindings[oid]]
            verdict, reason = _classify_v2(before, after, sources)
            findings.append({"message_index": mi, "text": sentence, "immediate_offset": start,
                             "timing_surface": sentence[start:end], "assertion_before": before,
                             "assertion_after": after, "verdict": verdict, "reason": reason,
                             "bound_order_ids": explicit_ids or list(bindings), "payment_sources": sources})
            continue
        # A contrast starts a new assertion, but an if-condition may span a comma.
        before = re.split(r";|\bbut\b|\bhowever\b|\botherwise\b", sentence[:start], flags=re.I)[-1]
        after = re.split(r";|\bbut\b|\bhowever\b|\botherwise\b", sentence[end:], flags=re.I)[0]
        local = before[-240:] + "immediately" + after[:200]
        explicit_ids = [oid for oid in bindings if oid in local]
        sources = [s for oid in (explicit_ids or list(bindings)) for s in bindings[oid]]
        gift_conditional = bool(
            re.search(rf"\bif\b[^;.!?]{{0,110}}\b{GIFT}\b", before, re.I)
            or re.match(rf"\s+if\b[^;.!?]{{0,110}}\b{GIFT}\b", after, re.I)
            or re.search(rf"\bfor\s+(?:the\s+)?{GIFT}\b", before, re.I)
            or re.match(rf"\s+for\s+(?:the\s+)?{GIFT}\b", after, re.I)
        )
        possessive_gift = bool(re.search(rf"\byour\s+{GIFT}\b", local, re.I))
        conditional_unclear = bool(re.search(r"\b(?:if|unless|not|never|cannot|can't)\b", local, re.I))
        simple_negation = bool(re.search(r"\b(?:not|never)\s+(?:(?:be|processed|credited|returned|refunded|received)\s+){0,3}$", before, re.I))
        if simple_negation:
            verdict, reason = "PASS", "explicitly negated immediate-refund assertion"
        elif possessive_gift and any(s != "gift_card" for s in sources):
            verdict, reason = "REVIEW", "gift-card ownership assertion conflicts or has unclear scope"
        elif gift_conditional and not re.search(r"\b(?:not|unless)\b", local, re.I):
            verdict, reason = "PASS", "explicit gift-card conditional or generic payment-type branch"
        elif conditional_unclear:
            verdict, reason = "REVIEW", "conditional or negated scope is outside the supported grammar"
        elif len(set(sources)) > 1:
            verdict, reason = "REVIEW", "statement does not disambiguate mixed payment sources"
        elif sources[0] == "gift_card":
            if re.search(CARD, local, re.I):
                verdict, reason = "REVIEW", "claimed card type differs from bound gift-card payment"
            else:
                verdict, reason = "PASS", "bound original payments are gift cards"
        elif re.search(CARD, local, re.I) or re.search(r"\b(?:credited|returned|received|refunded)\b", before, re.I):
            verdict, reason = "FAIL", "immediate refund asserted for bound non-gift-card payment"
        else:
            verdict, reason = "REVIEW", "immediate processing versus funds arrival is not explicit"
        findings.append({"message_index": mi, "text": sentence, "immediate_offset": start,
                         "verdict": verdict, "reason": reason, "bound_order_ids": explicit_ids or list(bindings),
                         "payment_sources": sources})
    aggregate = "FAIL" if any(f["verdict"] == "FAIL" for f in findings) else "REVIEW" if any(f["verdict"] == "REVIEW" for f in findings) else "PASS"
    return {"verdict": aggregate, "findings": findings, "errors": []}
