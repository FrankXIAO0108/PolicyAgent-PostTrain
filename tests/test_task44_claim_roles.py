"""Role confusion regressions: deterministic/local, not new model evaluations."""

import copy
import json
from decimal import Decimal
from pathlib import Path

import pytest

from src.evaluation import task44_hybrid_reward as hybrid
from src.evaluation.task44_claim_roles import (
    check_refund_binding,
    check_refund_coverage,
    literal_refund_amounts,
    non_refund_scope,
)
from test_task44_hybrid_reward import availability_case, case


BALANCE_SENTENCES = [
    "Since your gift card has a balance of $17.00, it's available to receive the refund.",
    "This amount will be refunded to your gift card (gift card ending in 123), which currently has a balance of $17.00.",
    "After the refund, the gift card will have a balance of $34.99.",
    "Since **$164.28 > $153.23**, you would actually need to **pay an additional $11.05** for the new lamp, not receive a refund.",
]

REFUND_PARAPHRASES = [
    "To answer your question about the total refund: the $17.99 amount is the "
    "**full price difference** between the original item ($153.23) and the new item ($135.24).",
    "The refund appears to be **$17.99** (the difference of $153.23 - $135.24 = $17.99).",
]


@pytest.mark.parametrize("text", REFUND_PARAPHRASES)
def test_refund_paraphrase_binds_only_asserted_amount(text):
    assert literal_refund_amounts(text) == {Decimal("17.99")}
    check_refund_binding("17.99", text)
    for value in ["153.23", "135.24", "34.99"]:
        with pytest.raises(ValueError):
            check_refund_binding(value, text)
    wrong = text.replace("$17.99", "$18.99")
    check_refund_binding("18.99", wrong)
    with pytest.raises(ValueError):
        check_refund_binding("17.99", wrong)


@pytest.mark.parametrize(
    "text",
    [
        "The refund appears to be a gift card balance of $34.99.",
        "The refund appears to be an additional payment of $11.05.",
        "The refund appears not to be $17.99.",
        "If approved, the refund appears to be $17.99.",
        "About the total refund: the $17.99 amount is the gift card balance.",
        "About the total refund: the $17.99 amount is not the full price difference.",
        "About the total refund: the $17.99 amount is the full price difference, if approved.",
    ],
)
def test_refund_paraphrase_rejects_other_roles_negation_and_conditions(text):
    assert not literal_refund_amounts(text)


@pytest.mark.parametrize("text", REFUND_PARAPHRASES)
def test_refund_paraphrase_cannot_hide_a_second_wrong_refund(text):
    combined = text + " Refund $18.99."
    assert literal_refund_amounts(combined) == {Decimal("17.99"), Decimal("18.99")}
    with pytest.raises(ValueError, match="omitted"):
        check_refund_coverage([{"kind": "refund_amount", "value": "17.99"}], combined)


@pytest.mark.parametrize("text", BALANCE_SENTENCES)
@pytest.mark.parametrize("value", ["17.00", "34.99", "11.05"])
def test_balance_payment_sentences_are_not_positive_refund_assertions(text, value):
    assert non_refund_scope(text)
    assert not literal_refund_amounts(text)
    with pytest.raises(ValueError, match="source-role"):
        check_refund_binding(value, text)


@pytest.mark.parametrize("prefix", BALANCE_SENTENCES)
def test_closed_exclusion_cannot_hide_added_wrong_refund(prefix):
    text = prefix + " Refund $18.99."
    assert non_refund_scope(text) is None
    rows = hybrid.candidates([{"role": "assistant", "content": text}])
    assert rows[-1]["required_kinds"] == ["refund_amount"]
    assert literal_refund_amounts(rows[-1]["text"]) == {Decimal("18.99")}


@pytest.mark.parametrize(
    "suffix", [", refund $18.99.", "; old is cheapest.", " and old is available."]
)
def test_mixed_clause_not_excluded(suffix):
    assert non_refund_scope(BALANCE_SENTENCES[2][:-1] + suffix) is None


@pytest.mark.parametrize(
    "text",
    [
        "Refund $18.99; the current balance is $17.00.",
        "The exact refund amount will be $18.99, leaving a balance of $35.99.",
        "The $18.99 refund is complete.",
        "Price difference: $153.23 - $134.24 = $18.99 to be refunded to your card.",
        "The price difference would be $18.99 (original price $153.23 minus new price $134.24), which would be refunded to your gift card.",
    ],
)
def test_wrong_refund_value_is_preserved_not_repaired_from_tool_truth(text):
    check_refund_binding("18.99", text)
    for value in ["17.99", "17.00", "35.99", "153.23", "134.24"]:
        with pytest.raises(ValueError):
            check_refund_binding(value, text)


@pytest.mark.parametrize(
    "text",
    [
        "You will not receive a refund of $18.99.",
        "No refund of $18.99 will be issued.",
        "If it is accepted, refund $18.99.",
        "The amount is $18.99; perhaps we can handle it.",
    ],
)
def test_uncertain_negated_or_conditional_refund_cannot_become_penalty(text):
    with pytest.raises(ValueError):
        check_refund_binding("18.99", text)


def test_card_cannot_become_product_even_outside_balance_grammar():
    _, _, _, packet, answer = availability_case()
    claim = next(
        c
        for r in answer["candidate_results"]
        for c in r["claims"]
        if c["kind"] == "availability"
    )
    claim["item_id"] = "gift_card_123"
    with pytest.raises(ValueError, match="preceding product variant"):
        hybrid.validate_candidate_extraction(json.dumps(answer), packet)


@pytest.mark.parametrize(
    "refund_text",
    [
        "Refund $18.99, actually refund $17.99.",
        "The refund appears to be $18.99, actually refund $17.99.",
        "About the total refund: the $18.99 amount is the full price difference, "
        "actually refund $17.99.",
    ],
)
def test_wrong_refund_still_deducts_and_omission_stops_scoring(refund_text):
    base, raw, spec, _, answer = case()
    raw["messages"][4]["content"] = raw["messages"][4]["content"].replace(
        "Old is cheapest. ", ""
    )
    raw["messages"][8]["content"] = refund_text
    packet = hybrid.candidate_packet(raw, "policy")
    answer.update(
        trajectory_sha256=packet["trajectory_sha256"],
        candidate_table_sha256=packet["candidate_table_sha256"],
    )
    answer["candidate_results"] = []
    for c in packet["candidates"]:
        values = (
            ["18.99", "17.99"]
            if "18.99" in c["text"]
            else ["17.99"]
            if c["required_kinds"]
            else []
        )
        answer["candidate_results"].append(
            {
                "candidate_id": c["candidate_id"],
                "status": "EXTRACTED" if values else "NOT_IN_SCOPE",
                "claims": [
                    {"kind": "refund_amount", "item_id": None, "value": v}
                    for v in values
                ],
            }
        )
    result = hybrid.score_candidate_response(
        base, raw, spec, packet, json.dumps(answer)
    )
    assert result["additive_components"]["post_write_communication"] == 0
    assert any(
        c["value"] == "18.99" and c["verdict"] == "MISMATCH"
        for c in result["claim_checks"]
    )
    damaged = copy.deepcopy(answer)
    damaged["candidate_results"][-1]["claims"].pop(0)
    with pytest.raises(ValueError, match="omitted"):
        hybrid.score_candidate_response(base, raw, spec, packet, json.dumps(damaged))


@pytest.mark.parametrize("index,expected", [(0, 1.0), (1, 0.2), (2, 0.98), (3, 0.25)])
def test_real_cached_extraction_reinterpretation(index, expected):
    root = (
        Path(__file__).resolve().parents[1]
        / "_local_private_runs/task44_hybrid_semantic_rescore_n4_v1"
    )
    if not (root / "summary.json").exists():
        pytest.skip("Private original API evidence not distributed")
    prior = json.loads((root / "summary.json").read_text(encoding="utf-8"))["results"][
        index
    ]
    cache = root / "semantic_cache" / prior["details"]["cache_key"]
    request = json.loads((cache / "request.json").read_text(encoding="utf-8"))
    response = json.loads((cache / "response.json").read_text(encoding="utf-8"))
    data = request["scoring_input"]
    result = hybrid.score_candidate_response(
        data["base"],
        data["raw"],
        data["spec"],
        request["packet"],
        response["choices"][0]["message"]["content"],
    )
    assert result["offline_reward"] == expected
    if index == 2:
        assert all(c["verdict"] == "MATCH" for c in result["claim_checks"])
        assert len(result["context_decisions"]) == 3
        assert all(c["fact_verified"] is False for c in result["context_decisions"])
    if index == 3:
        assert (
            sum(
                c["kind"] == "cheapest" and c["verdict"] == "MISMATCH"
                for c in result["claim_checks"]
            )
            == 2
        )
    # A new prompt/packet must not transparently reuse the old API cache.
    new_packet = hybrid.candidate_packet(data["raw"], request["packet"]["policy"])
    key = hybrid.digest(
        {"packet": new_packet, "prompt": hybrid.PROMPT, "settings": request["settings"]}
    )
    assert key != prior["details"]["cache_key"]


def test_prescreen_receipt_separates_approved_parameters_and_missed_question():
    cache = (
        Path(__file__).resolve().parents[1]
        / "_local_private_runs/task44_role_v2_prescreen_deploy/run/semantic_cache"
        / "36c642ad5e3560951fc306a790d39b442dec5cea13a7a39256491298a327a8ff"
    )
    if not (cache / "response.json").exists():
        pytest.skip("Private prescreen receipt not distributed")
    request = json.loads((cache / "request.json").read_text(encoding="utf-8"))
    response = json.loads((cache / "response.json").read_text(encoding="utf-8"))
    assert response["model"] == "deepseek-v4-flash"
    data = request["scoring_input"]
    content = response["choices"][0]["message"]["content"]
    extraction, uncertain = hybrid.validate_candidate_extraction(
        content, request["packet"]
    )
    assert not uncertain
    result = hybrid.rescore(data["base"], data["raw"], extraction, data["spec"])
    assert len(result["claim_checks"]) == 15
    assert all(c["verdict"] == "MATCH" for c in result["claim_checks"])
    check = result["authorization_checks"][0]
    assert check["verdict"] == "REVIEW"
    assert check["parameter_authorization"] == "PASS"
    assert check["execution_order"] == "FAIL"
    scored = hybrid.score_candidate_response(
        data["base"], data["raw"], data["spec"], request["packet"], content
    )
    assert scored["offline_reward"] == 0.75
    assert scored["semantic_status"] == "READY"
