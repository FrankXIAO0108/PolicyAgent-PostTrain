"""Separate false-full/false-penalty denominators and evidence binding."""

import copy
import json
from pathlib import Path

import pytest

from scripts.audit_task44_score_direction import audit, error_rates


def test_both_dangerous_directions_and_minor_underrating():
    result = error_rates([1.0, 0.2, 0.98, 0.25], [0.9, 1.0, 0.83, 0.25])
    assert result["full_score_wrongly_penalized"] == {
        "count": 1,
        "denominator": 1,
        "rate": 1.0,
        "unscored": 0,
    }
    assert result["non_full_wrongly_given_full"]["count"] == 1
    assert result["non_full_wrongly_given_full"]["denominator"] == 3
    assert result["any_score_error"]["count"] == 3
    assert result["underrated"]["count"] == 2


def test_missing_scores_are_not_successes_and_empty_denominator_is_null():
    result = error_rates([0.2, 0.5], [None, None])
    assert result["full_score_wrongly_penalized"]["rate"] is None
    assert result["non_full_wrongly_given_full"]["unscored"] == 2
    assert result["any_score_error"]["unscored"] == 2


@pytest.mark.parametrize(
    "expected,predicted", [([1.0], []), ([1.1], [1.0]), ([1.0], [float("nan")])]
)
def test_invalid_pairs_refused(expected, predicted):
    with pytest.raises(ValueError):
        error_rates(expected, predicted)


def private_config():
    config = json.loads(
        (
            Path(__file__).resolve().parents[1]
            / "configs/evaluation/task44_score_direction_reviews_v1.json"
        ).read_text(encoding="utf-8")
    )
    if not Path(config["source_dir"]).exists():
        pytest.skip("Private API evidence unavailable")
    return config


def test_real_four_cached_scores_metrics_and_no_network(monkeypatch):
    import socket

    monkeypatch.setattr(
        socket, "create_connection", lambda *a, **k: pytest.fail("No network permitted")
    )
    report = audit(private_config())
    assert [r["new_reward"] for r in report["results"]] == [1.0, 0.2, 0.98, 0.25]
    assert report["before"]["any_score_error"]["rate"] == 0.25
    assert report["after"]["any_score_error"]["rate"] == 0
    assert report["training_release_allowed"] is False
    assert report["new_llm_outputs_obtained"] is False


@pytest.mark.parametrize("mutation", ["summary", "request", "response", "gold", "omit"])
def test_private_audit_binding_tampering_rejected(mutation):
    config = copy.deepcopy(private_config())
    if mutation == "summary":
        config["summary_sha256"] = "0" * 64
    elif mutation in {"request", "response"}:
        config["reviews"][0][mutation + "_sha256"] = "0" * 64
    elif mutation == "gold":
        config["independent_gold"] = True
    else:
        config["reviews"].pop()
    with pytest.raises(ValueError):
        audit(config)
