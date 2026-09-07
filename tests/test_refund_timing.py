import json
from copy import deepcopy

import pytest

from src.evaluation.refund_timing import refund_timing_diagnostic
from src.evaluation.staged_reward_shadow import _claim_evidence_diagnostic


RULE = {"rule_id": "timing", "rule_type": "refund_timing_by_payment_v1", "verdict": "FAIL",
        "source_call": {"name": "get_user_details", "arguments": {"user_id": "u1"}},
        "order_source_calls": [{"name": "get_order_details", "arguments": {"order_id": "#O1"}}]}


def trace(kind="credit_card"):
    user = {"user_id": "u1", "payment_methods": {"p1": {"source": kind}, "unused_gift": {"source": "gift_card"}}}
    order = {"order_id": "#O1", "user_id": "u1", "payment_history": [{"transaction_type": "payment", "payment_method_id": "p1"}]}
    return [{**c, "result": {"content": json.dumps(p), "error": False}}
            for c, p in zip([RULE["source_call"], *RULE["order_source_calls"]], [user, order])]


@pytest.mark.parametrize("text,expected", [
    ("Your refund will be credited to your Visa immediately.", "FAIL"),
    ("The refund will be processed immediately since it was paid with a credit card.", "FAIL"),
    ("Refunds arrive immediately for gift cards and within 5-7 business days for credit cards.", "PASS"),
    ("For **gift card** payments: refunded immediately. Credit cards: 5-7 days.", "PASS"),
    ("The refund is returned immediately if it's a gift card, otherwise in 5 to 7 days.", "PASS"),
    ("If the original payment was made via a gift card, the refund is processed immediately.", "PASS"),
    ("The refund arrives immediately if they were paid with a gift card, but since it was Visa, it takes 5-7 days.", "PASS"),
    ("Your refund will not be credited immediately.", "PASS"),
    ("Refunds will be credited immediately unless it was a credit card.", "REVIEW"),
    ("For your gift card, the refund is processed immediately, otherwise in 5 to 7 days.", "REVIEW"),
    ("The refund will be processed immediately.", "REVIEW"),
    ("Refunds are immediate if NOT paid with a gift card.", "REVIEW"),
    ("The refund is credited immediately if not paid with a gift card.", "REVIEW"),
    ("If paid with a gift card, the refund is immediate. Your Visa refund is credited immediately.", "FAIL"),
    ("If gift card, refund credited immediately; but your Visa refund is credited immediately.", "FAIL"),
])
def test_local_scope(text, expected):
    assert refund_timing_diagnostic([{"role": "assistant", "content": text}], trace(), RULE)["verdict"] == expected


def test_unused_gift_card_cannot_justify_credit_card_immediate_refund():
    text = [{"role": "assistant", "content": "Your Visa refund is credited immediately."}]
    assert refund_timing_diagnostic(text, trace(), RULE)["verdict"] == "FAIL"


def test_actual_gift_card_payment_and_missing_evidence():
    messages = [{"role": "assistant", "content": "Your refund is credited immediately."}]
    assert refund_timing_diagnostic(messages, trace("gift_card"), RULE)["verdict"] == "PASS"
    assert refund_timing_diagnostic(messages, [], RULE)["verdict"] == "ERROR"
    wrong = trace()
    wrong[1]["result"]["content"] = '{}'
    assert refund_timing_diagnostic(messages, wrong, RULE)["verdict"] == "ERROR"


def test_messages_never_concatenated_and_user_text_ignored():
    messages = [{"role": "assistant", "content": "Your refund will take 5-7 days."},
                {"role": "user", "content": "My Visa refund should arrive immediately."},
                {"role": "assistant", "content": "Contact support immediately."}]
    assert refund_timing_diagnostic(messages, trace(), RULE)["findings"] == []


def test_mixed_payment_requires_local_order_binding():
    evidence = trace()
    order = json.loads(evidence[-1]["result"]["content"])
    order["payment_history"].append({"transaction_type": "payment", "payment_method_id": "unused_gift"})
    evidence[-1]["result"]["content"] = json.dumps(order)
    messages = [{"role": "assistant", "content": "Your refund is credited immediately."}]
    assert refund_timing_diagnostic(messages, evidence, RULE)["verdict"] == "REVIEW"


def test_new_rule_is_opt_in_and_no_input_mutation():
    messages = [{"role": "assistant", "content": "Your Visa refund is credited immediately."}]
    evidence = trace()
    original = deepcopy(evidence)
    assert _claim_evidence_diagnostic(messages, evidence, {}) is None
    d = _claim_evidence_diagnostic(messages, evidence, {"claim_evidence_rules": [RULE]})
    assert d["verdict"] == "FAIL" and evidence == original
