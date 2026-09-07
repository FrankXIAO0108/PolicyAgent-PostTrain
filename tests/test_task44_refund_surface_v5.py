"""Closed surface grammar regressions; no model/API invocation."""

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
)


PHRASES = [
    (
        "The **price difference** is **$17.99**, which will be **refunded to your gift card** (gift card ending in 7245904).",
        "17.99",
    ),
    (
        "The $153.23 original desk lamp payment was paid via gift card, so the original gift card is being refunded the amount of the modification ($17.99).",
        "17.99",
    ),
    ("The price difference of $17.99 is refunded to the same gift card.", "17.99"),
    (
        "The refund amount would be the original payment for the item being replaced ($153.23 to the gift card).",
        "153.23",
    ),
    (
        "The original payment of $153.23 to gift card gift card ending in 7245904 will be refunded immediately to that gift card.",
        "153.23",
    ),
    (
        "The price difference of $17.99 to the replacement desk lamp (item 5320792178 at $135.24) will be refunded to the same gift card.",
        "17.99",
    ),
    (
        "The price difference of $17.99 (since the new item is cheaper) will be refunded to the same gift card (gift card ending in 7245904).",
        "17.99",
    ),
    (
        "So the total refund to your gift card would be $153.23 (original item refund) + $17.99 (price difference refund) = $171.22.",
        "171.22",
    ),
    (
        "So yes, the total refund to your gift card would be $153.23 (original payment refund) + $17.99 (price difference refund) = $171.22.",
        "171.22",
    ),
    (
        "- **Price difference**: $17.99 (refunded to gift card ending in 7245904)",
        "17.99",
    ),
    ("- **Total refund to gift card**: $171.22", "171.22"),
    ("- **Price difference:** $18.00 (refund to your gift card)", "18.00"),
]


@pytest.mark.parametrize("text,value", PHRASES)
def test_refund_subject_and_stated_total_not_nearby_prices(text, value):
    assert literal_refund_amounts(text) == {Decimal(value)}
    check_refund_binding(value, text)
    with pytest.raises(ValueError):
        check_refund_binding("99.01", text)
    wrong = text.replace("$" + value, "$99.01")
    check_refund_binding("99.01", wrong)
    with pytest.raises(ValueError, match="omitted"):
        check_refund_coverage([], text)


@pytest.mark.parametrize("text,value", PHRASES)
@pytest.mark.parametrize("prefix", ["If approved, ", "Unless you object, "])
def test_conditional_surface_never_becomes_verified_amount(text, value, prefix):
    with pytest.raises(ValueError):
        check_refund_binding(value, prefix + text)


@pytest.mark.parametrize(
    "text",
    [
        "The price difference is $17.99, which will not be refunded.",
        "The original payment of $153.23 to gift card will not be refunded.",
        "Total refund to gift card balance: $171.22",
        "The gift card balance is $171.22, which can receive a refund.",
        "Price difference: $18.00 (additional payment to your gift card)",
        "The original payment of $153.23 to gift card is not a refund.",
    ],
)
def test_wrong_roles_and_negative_refunds_do_not_bind(text):
    assert not literal_refund_amounts(text)


def test_sum_preserves_wrong_total_and_does_not_silently_recalculate():
    text = PHRASES[7][0].replace("$171.22", "$999.00")
    assert literal_refund_amounts(text) == {Decimal("999.00")}
    with pytest.raises(ValueError):
        check_refund_binding("171.22", text)
    with pytest.raises(ValueError, match="omitted"):
        check_refund_coverage(
            [{"kind": "refund_amount", "value": "999.00"}], text + " Refund $18.99."
        )


CONTEXT = [
    "Let me review the available desk lamp variants to find the **cheapest** one available (regardless of color or power source).",
    "Here are the available desk lamp variants:",
]


@pytest.mark.parametrize("text", CONTEXT)
def test_plans_and_headings_are_not_independent_facts(text):
    assert hybrid.context_only_reason(text)
    assert (
        hybrid.candidates([{"role": "assistant", "content": text}])[0]["required_kinds"]
        == []
    )


@pytest.mark.parametrize("text", CONTEXT)
@pytest.mark.parametrize(
    "suffix", [" Item 123 is cheapest.", " refund $18.00.", " available item 123."]
)
def test_context_exception_cannot_swallow_attached_assertion(text, suffix):
    assert hybrid.context_only_reason(text + suffix) is None
    assert hybrid.candidates([{"role": "assistant", "content": text + suffix}])[-1][
        "required_kinds"
    ]


@pytest.mark.parametrize(
    "key,expected",
    [
        ("bd8f23673c0992309e40c5f685951ad66c2171b4e8664e4e8b85c8311495511c", 0.13),
        ("fb91b629ded2e153b402e7f697f068260c42e7e998458c21be1fba804f3476a5", 0.15),
    ],
)
def test_saved_bad_refunds_are_scored_as_failures_not_parser_errors(key, expected):
    root = Path(__file__).resolve().parents[1]
    cache = (
        root
        / "_local_private_runs/task44_consent_v4_launch/supplemental_scores/semantic_cache"
        / key
    )
    if not cache.exists():
        pytest.skip("Private development responses not distributed")
    request = json.loads((cache / "request.json").read_text(encoding="utf-8"))
    response = json.loads((cache / "response.json").read_text(encoding="utf-8"))
    data = request["scoring_input"]
    text = response["choices"][0]["message"]["content"]
    result = hybrid.score_candidate_response(
        data["base"], data["raw"], data["spec"], request["packet"], text
    )
    assert result["authorization_verdict"] == "FAIL"
    assert result["offline_reward"] == expected
    assert result["additive_components"]["post_write_communication"] == 0
    assert any(
        c["verdict"] == "MISMATCH" and c["kind"] == "refund_amount"
        for c in result["claim_checks"]
    )
    # A missing wrong claim must not become a perfect-score response.
    damaged = copy.deepcopy(json.loads(text))
    for row in damaged["candidate_results"]:
        if any(
            c["kind"] == "refund_amount" and c["value"] in {"171.22", "18.00"}
            for c in row["claims"]
        ):
            row.update(status="NOT_IN_SCOPE", claims=[])
            break
    with pytest.raises(ValueError):
        hybrid.score_candidate_response(
            data["base"],
            data["raw"],
            data["spec"],
            request["packet"],
            json.dumps(damaged),
        )
