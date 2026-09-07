"""Synthetic diagnostic fixtures only; no model rollout or API calls."""
import json
from copy import deepcopy

import pytest

from src.evaluation.refund_timing import refund_timing_diagnostic
from src.evaluation.staged_reward_shadow import _claim_evidence_diagnostic


RULE = {
    "rule_id": "timing_v2", "rule_type": "refund_timing_by_payment_v2", "verdict": "FAIL",
    "source_call": {"name": "get_user_details", "arguments": {"user_id": "u"}},
    "order_source_calls": [{"name": "get_order_details", "arguments": {"order_id": "#O"}}],
}


def evidence(kind="credit_card"):
    user = {"user_id": "u", "payment_methods": {"p": {"source": kind}}}
    order = {"order_id": "#O", "user_id": "u", "payment_history": [
        {"transaction_type": "payment", "payment_method_id": "p"}]}
    return [{**c, "result": {"content": json.dumps(v), "error": False}}
            for c, v in zip([RULE["source_call"], *RULE["order_source_calls"]], [user, order])]


@pytest.mark.parametrize("text,expected", [
    ("If paid by gift card, the refund is credited immediately, and your Visa refund will be credited immediately.", "FAIL"),
    ("Your Visa refund request will be processed immediately, but the funds will reach your credit card in 5 to 7 business days.", "PASS"),
    ("The refund is immediate (if it was paid by gift card), otherwise it takes 5 to 7 business days.", "PASS"),
    ("The refund will be immediate for the original payment method if it's a gift card, otherwise in 5 to 7 business days.", "PASS"),
    ("Your Visa refund will be credited instantly.", "FAIL"),
    ("Your refund will be immediate since it was paid by credit card.", "FAIL"),
    ("If paid by gift card, the refund is credited immediately. Your Visa refund will be credited immediately.", "FAIL"),
    ("Your Visa refund will not be credited immediately; it takes 5 to 7 business days.", "PASS"),
    ("The refund takes 5-7 days if it's not a gift card, or immediately if it is a gift card.", "PASS"),
    ("Your credit card refund is immediate if NOT paid with a gift card.", "REVIEW"),
    ("Your refund is immediate if paid by gift card or credit card.", "REVIEW"),
    ("If paid by gift card, the refund is immediate and your Visa refund is immediate too.", "FAIL"),
    ("If paid by gift card, the refund is immediate, but your Visa refund is immediate too.", "FAIL"),
    ("Refunds arrive immediately for gift cards and within 5-7 business days for credit cards.", "PASS"),
    ("Your refund is credited instantly if it's a gift card, otherwise in 5-7 days.", "PASS"),
    ("Refunds are processed immediately (if it was a gift card) or within 5-7 days (if it was a credit card).", "PASS"),
    ("Your Visa refund request will be processed immediately, and your refund will be credited instantly.", "FAIL"),
    ("Your Visa refund request will be credited immediately.", "FAIL"),
    ("Your refund will be processed immediately.", "REVIEW"),
    ("Your Visa refund will be processed immediately.", "REVIEW"),
    ("Your refund will be processed immediately since it was paid with a credit card.", "FAIL"),
    ("The refund will be immediate for the original payment method (gift card) or within 5-7 days for other methods.", "REVIEW"),
    ("For your gift card the refund is immediate, otherwise in 5-7 days.", "REVIEW"),
    ("If paid by gift card, the refund is immediate and credited instantly to your Visa.", "REVIEW"),
    ("The refunds will be processed immediately for any gift card payments, or within 5-7 business days for credit card payments.", "PASS"),
    ("Refunds are immediate for all gift cards and within 5-7 days for credit cards.", "PASS"),
    ("Refunds are immediate for gift cards or credit cards.", "REVIEW"),
])
def test_scope_and_refund_semantics(text, expected):
    actual = refund_timing_diagnostic([{"role": "assistant", "content": text}], evidence(), RULE)
    assert actual["verdict"] == expected


@pytest.mark.parametrize("separator", [", and ", "; ", ". ", ", but ", "\n"])
def test_correct_condition_cannot_mask_later_error(separator):
    text = "If paid by gift card, the refund is immediate" + separator + "your Visa refund is credited instantly."
    assert refund_timing_diagnostic([{"role": "assistant", "content": text}], evidence(), RULE)["verdict"] == "FAIL"


def test_v2_dispatch_is_opt_in_and_does_not_mutate_inputs():
    messages = [{"role": "assistant", "content": "Your Visa refund will be credited instantly."}]
    inputs = evidence()
    saved = deepcopy(inputs)
    v1 = {**RULE, "rule_type": "refund_timing_by_payment_v1"}
    assert _claim_evidence_diagnostic(messages, inputs, {"claim_evidence_rules": [v1]})["verdict"] == "PASS"
    assert _claim_evidence_diagnostic(messages, inputs, {"claim_evidence_rules": [RULE]})["verdict"] == "FAIL"
    assert inputs == saved


def test_bound_gift_card_and_missing_evidence():
    messages = [{"role": "assistant", "content": "Your refund will be credited instantly."}]
    assert refund_timing_diagnostic(messages, evidence("gift_card"), RULE)["verdict"] == "PASS"
    assert refund_timing_diagnostic(messages, [], RULE)["verdict"] == "ERROR"


def test_separate_messages_do_not_share_conditions():
    messages = [{"role": "assistant", "content": "If paid by gift card, refunds are immediate."},
                {"role": "assistant", "content": "Your Visa refund is credited instantly."}]
    assert refund_timing_diagnostic(messages, evidence(), RULE)["verdict"] == "FAIL"


def test_tool_payload_identity_must_match_requested_user():
    inputs = evidence()
    for call in inputs:
        payload = json.loads(call["result"]["content"])
        payload["user_id"] = "wrong-user"
        call["result"]["content"] = json.dumps(payload)
    messages = [{"role": "assistant", "content": "Your refund is credited instantly."}]
    assert refund_timing_diagnostic(messages, inputs, RULE)["verdict"] == "ERROR"
