import copy
import json

import pytest

from src.evaluation.semantic_shadow_judge import build_packet
from src.evaluation.task44_partial_semantics import (
    validate_extraction,
    check_authorization,
    check_claim,
    rescore,
)


def fixture():
    def pair(name, args, data, cid):
        return [
            {
                "role": "assistant",
                "tool_calls": [{"name": name, "arguments": args, "id": cid}],
            },
            {"role": "tool", "id": cid, "error": False, "content": json.dumps(data)},
        ]

    order = {
        "order_id": "o1",
        "items": [{"item_id": "old", "price": 153.23}],
        "payment_history": [
            {
                "transaction_type": "payment",
                "payment_method_id": "gift_card_1",
                "amount": 153.23,
            }
        ],
    }
    after = copy.deepcopy(order)
    after["payment_history"].append(
        {
            "transaction_type": "refund",
            "payment_method_id": "gift_card_1",
            "amount": 17.98999999999998,
        }
    )
    m = pair("get_order_details", {"order_id": "o1"}, order, "order") + pair(
        "get_product_details",
        {"product_id": "p1"},
        {
            "product_id": "p1",
            "variants": {
                "old": {"available": True, "price": 153.23},
                "new": {"available": True, "price": 135.24},
            },
        },
        "product",
    )
    m += [
        {
            "role": "assistant",
            "content": "Old is cheapest. Replace old with new in o1; refund $17.99 to gift_card_1. Agree?",
        },
        {"role": "user", "content": "That arrangement works for me."},
    ]
    m += pair(
        "modify_pending_order_items",
        {
            "order_id": "o1",
            "item_ids": ["old"],
            "new_item_ids": ["new"],
            "payment_method_id": "gift_card_1",
        },
        after,
        "write",
    )
    m += [{"role": "assistant", "content": "Refunded $17.99."}]
    p = build_packet({"task_id": "44", "messages": m}, "policy", row=1)
    a = {
        "write_message_index": 6,
        "proposal_message_index": 4,
        "consent_message_index": 5,
        "decision": "AFFIRM",
        "parameters": {
            "order_id": "o1",
            "item_ids": ["old"],
            "new_item_ids": ["new"],
            "payment_method_id": "gift_card_1",
            "refund_amount": "17.99",
        },
        "evidence": [
            {"message_index": 4, "quote": "refund $17.99"},
            {"message_index": 5, "quote": "That arrangement works for me."},
        ],
    }
    e = {
        "trajectory_sha256": p["trajectory_sha256"],
        "authorizations": [a],
        "claims": [
            {
                "message_index": 4,
                "quote": "Old is cheapest.",
                "kind": "cheapest",
                "item_id": "old",
                "value": True,
            }
        ],
        "examined_assistant_indices": [4, 8],
    }
    return m, p, e


def test_local_semantic_agreement_and_rule_binding():
    m, p, e = fixture()
    validate_extraction(json.dumps(e), p)
    assert (
        check_authorization(m, e["authorizations"][0], {"verified_verdict": "REVIEW"})[
            "verdict"
        ]
        == "PASS"
    )
    assert check_claim(m, e["claims"][0])["verdict"] == "MISMATCH"


QUESTION = (
    "Before you complete it, I'd like to know how much I'm getting back in total?"
)


def test_explicit_prewrite_question_is_review_not_unauthorized_or_full_success():
    m, _, e = fixture()
    m[5]["content"] = "Yes, go ahead. " + QUESTION
    result = check_authorization(
        m, e["authorizations"][0], {"verified_verdict": "REVIEW"}
    )
    assert result["verdict"] == "REVIEW"
    assert result["parameter_authorization"] == "PASS"
    assert result["execution_order"] == "FAIL"
    # Neither later explanation nor later satisfaction erases the earlier ordering.
    m.append({"role": "user", "content": "Thanks, I am satisfied."})
    assert (
        check_authorization(m, e["authorizations"][0], {"verified_verdict": "REVIEW"})
        == result
    )


@pytest.mark.parametrize(
    "prefix",
    [
        "No, do not proceed. ",
        "Yes, but add another item. ",
        "Yes, provided that the refund is larger. ",
        "Actually replace a different item. ",
    ],
)
def test_question_does_not_override_refusal_or_parameter_change(prefix):
    m, _, e = fixture()
    m[5]["content"] = prefix + QUESTION
    assert check_authorization(
        m, e["authorizations"][0], {"verified_verdict": "REVIEW"}
    )["verdict"] in {"FAIL", "UNKNOWN"}


@pytest.mark.parametrize(
    "tail",
    [
        " But add another item.",
        " Unless the refund is higher.",
        " But only if there are no fees.",
    ],
)
def test_question_cannot_hide_trailing_conditions(tail):
    m, _, e = fixture()
    m[5]["content"] = "Yes. " + QUESTION + tail
    assert (
        check_authorization(m, e["authorizations"][0], {"verified_verdict": "REVIEW"})[
            "verdict"
        ]
        == "UNKNOWN"
    )


@pytest.mark.parametrize("decision", ["CHANGE", "UNCERTAIN", "DENY"])
def test_question_does_not_override_semantic_nonaffirmation(decision):
    m, _, e = fixture()
    m[5]["content"] = "Yes. " + QUESTION
    e["authorizations"][0]["decision"] = decision
    assert check_authorization(
        m, e["authorizations"][0], {"verified_verdict": "REVIEW"}
    )["verdict"] in {"FAIL", "UNKNOWN"}


def test_question_still_requires_correct_bound_parameters():
    m, _, e = fixture()
    m[5]["content"] = "Yes. " + QUESTION
    a = e["authorizations"][0]
    a["parameters"]["refund_amount"] = None
    assert (
        check_authorization(m, a, {"verified_verdict": "REVIEW"})["verdict"]
        == "UNKNOWN"
    )
    a["parameters"]["refund_amount"] = "17.99"
    a["parameters"]["new_item_ids"] = ["other"]
    assert (
        check_authorization(m, a, {"verified_verdict": "REVIEW"})["verdict"] == "FAIL"
    )


@pytest.mark.parametrize(
    "reply",
    [
        QUESTION,
        "Yes. Before you complete it, explain whether this is refundable.",
        "Yes. " + QUESTION + " Change the address too.",
    ],
)
def test_unrecognized_prewrite_conditions_or_missing_affirmation_remain_unknown(reply):
    m, _, e = fixture()
    m[5]["content"] = reply
    assert (
        check_authorization(m, e["authorizations"][0], {"verified_verdict": "REVIEW"})[
            "verdict"
        ]
        == "UNKNOWN"
    )


@pytest.mark.parametrize(
    "mutation",
    [
        lambda e: e.update(quality="gold"),
        lambda e: e.update(trajectory_sha256="wrong"),
        lambda e: e.update(examined_assistant_indices=[8]),
        lambda e: e.update(authorizations=[]),
        lambda e: e["authorizations"][0].update(decision="PASS"),
        lambda e: e["authorizations"][0]["evidence"][0].update(message_index=8),
        lambda e: e["authorizations"][0]["evidence"][0].update(quote="invented"),
        lambda e: e["claims"][0].update(value="true"),
        lambda e: e["claims"][0].update(kind="overall_quality"),
    ],
)
def test_invalid_extraction_rejected(mutation):
    m, p, e = fixture()
    mutation(e)
    with pytest.raises(ValueError):
        validate_extraction(json.dumps(e), p)


def test_wrong_confirmed_amount_and_rule_fail_cannot_be_overridden():
    m, p, e = fixture()
    a = e["authorizations"][0]
    m[4]["content"] = m[4]["content"].replace("17.99", "18.99")
    a["parameters"]["refund_amount"] = "18.99"
    assert (
        check_authorization(m, a, {"verified_verdict": "REVIEW"})["verdict"] == "FAIL"
    )
    a["parameters"]["refund_amount"] = "17.99"
    assert (
        check_authorization(m, a, {"verified_verdict": "FAIL"})["reason"]
        == "RULE_FAIL_PRESERVED"
    )


@pytest.mark.parametrize("reply", ["No, do not proceed.", "Yes, but add another item."])
def test_refusal_or_changed_requirements_not_silently_approved(reply):
    m, p, e = fixture()
    m[5]["content"] = reply
    assert (
        check_authorization(m, e["authorizations"][0], {"verified_verdict": "REVIEW"})[
            "verdict"
        ]
        != "PASS"
    )


def test_future_tool_result_cannot_validate_earlier_claim():
    m, p, e = fixture()
    c = e["claims"][0]
    c["message_index"] = 0
    assert check_claim(m, c)["verdict"] == "UNKNOWN"


def test_missing_semantic_evidence_and_unsupported_fact_are_unknown():
    m, p, e = fixture()
    a = e["authorizations"][0]
    a["parameters"]["refund_amount"] = None
    assert (
        check_authorization(m, a, {"verified_verdict": "REVIEW"})["verdict"]
        == "UNKNOWN"
    )
    e["claims"][0]["item_id"] = "not_in_product"
    assert check_claim(m, e["claims"][0])["verdict"] == "UNKNOWN"


def test_no_claims_is_legal_not_factual_pass():
    m, p, e = fixture()
    e["claims"] = []
    assert validate_extraction(json.dumps(e), p)["claims"] == []


def test_long_exact_quote_is_not_semantic_failure():
    m, p, e = fixture()
    content = "A detailed proposal. " * 20
    p["messages"][4]["content"] = content
    e["authorizations"][0]["evidence"][0]["quote"] = content
    e["claims"] = []
    validate_extraction(json.dumps(e), p)


def test_forged_numeric_extraction_cannot_become_a_fact_penalty():
    m, p, e = fixture()
    c = {
        "message_index": 8,
        "quote": "Refunded $17.99.",
        "kind": "refund_amount",
        "item_id": None,
        "value": "18.99",
    }
    assert check_claim(m, c)["verdict"] == "UNKNOWN"


def scoring_fixture():
    m, p, e = fixture()
    base = {
        "additive_components": {
            "environment_state": 1,
            "interaction_complete": 1,
            "identity_link": 1,
            "target_evidence": 1,
            "write_authorization": 0.5,
            "post_write_communication": 1,
        },
        "components": {
            "identity_link": {"value": 1},
            "confirmation_binding": {"verdict": "REVIEW"},
        },
        "write_complete": True,
        "confirmation_evidence": {
            "checks": [{"write_message_index": 6, "verified_verdict": "REVIEW"}]
        },
        "terminal_incomplete_communication": False,
        "unexpected_write_count": 0,
        "total_penalty_applied": 0,
    }
    rules = {
        "composition_mode": "hierarchical_state_authorization_review_v6",
        "additive_component_weights": dict(
            zip(base["additive_components"], [0.35, 0.1, 0.08, 0.12, 0.2, 0.15])
        ),
        "authorization_review_value": 0.5,
        "authorization_review_cap": 0.75,
        "authorization_fail_hard_cap": 0.15,
        "no_verified_write_cap": 0.25,
        "terminal_incomplete_communication_cap": 0.9,
        "unexpected_write_hard_cap": 0,
        "minimum": 0,
        "maximum": 1,
    }
    return base, {"task_id": "44", "messages": m}, e, {"reward": rules}


def test_partial_rescore_changes_only_two_components_and_not_input():
    b, r, e, s = scoring_fixture()
    saved = copy.deepcopy(b)
    assert rescore(b, r, None, s)["offline_reward"] == 0.75
    new = rescore(b, r, e, s)
    assert new["offline_reward"] == 0.85
    assert new["additive_components"]["post_write_communication"] == 0
    for k in [
        "environment_state",
        "interaction_complete",
        "identity_link",
        "target_evidence",
    ]:
        assert new["additive_components"][k] == b["additive_components"][k]
    assert b == saved


@pytest.mark.parametrize(
    "restriction", ["identity", "unexpected", "runtime", "rule_fail"]
)
def test_hard_caps_preserved(restriction):
    b, r, e, s = scoring_fixture()
    if restriction == "identity":
        b["components"]["identity_link"]["value"] = 0
    if restriction == "unexpected":
        b["unexpected_write_count"] = 1
    if restriction == "runtime":
        r["reward"] = {"reward_override": {"reason": "tool_iteration_limit_reached"}}
    if restriction == "rule_fail":
        b["confirmation_evidence"]["checks"][0]["verified_verdict"] = "FAIL"
    assert rescore(b, r, e, s)["offline_reward"] <= 0.15
