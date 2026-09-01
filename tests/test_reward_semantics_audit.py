from __future__ import annotations

from src.evaluation.reward_semantics_audit import (
    _candidate_tier,
    _deep_subset,
    _reference_call_progress,
    _result_evidence_progress,
)


def call(name: str, arguments: dict, content: object, *, error: bool = False) -> dict:
    return {
        "call_id": "c1",
        "name": name,
        "arguments": arguments,
        "result": {"content": content, "error": error},
    }


def task_spec() -> dict:
    return {
        "reference_calls": [
            {
                "evidence_id": "order_call",
                "name": "get_order_details",
                "arguments": {"order_id": "O1"},
            }
        ],
        "result_evidence": [
            {
                "evidence_id": "order_fact",
                "name": "get_order_details",
                "arguments": {"order_id": "O1"},
                "assertions": [
                    {"path": ["status"], "expected": "pending"},
                    {
                        "path": ["items"],
                        "expected": [{"name": "Backpack", "options": {"size": "M"}}],
                    },
                ],
            }
        ],
    }


def test_deep_subset_matches_nested_mapping_and_unordered_list_subset() -> None:
    observed = {
        "items": [
            {"name": "Lamp", "options": {"color": "black"}},
            {"name": "Backpack", "options": {"size": "M", "color": "grey"}},
        ]
    }
    expected = {"items": [{"name": "Backpack", "options": {"size": "M"}}]}
    assert _deep_subset(observed, expected)


def test_reference_call_does_not_imply_result_evidence() -> None:
    trace = [
        call(
            "get_order_details",
            {"order_id": "O1"},
            '{"status":"cancelled","items":[]}',
        )
    ]
    assert _reference_call_progress(trace, task_spec())["value"] == 1.0
    assert _result_evidence_progress(trace, task_spec())["value"] == 0.0


def test_result_evidence_requires_bound_successful_json_result() -> None:
    trace = [
        call(
            "get_order_details",
            {"order_id": "O1"},
            '{"status":"pending","items":[{"name":"Backpack","options":{"size":"M","color":"grey"}}]}',
        )
    ]
    result = _result_evidence_progress(trace, task_spec())
    assert result["value"] == 1.0
    assert result["checks"][0]["matched"] is True


def test_tool_error_cannot_satisfy_reference_or_result_evidence() -> None:
    trace = [
        call(
            "get_order_details",
            {"order_id": "O1"},
            '{"status":"pending","items":[{"name":"Backpack","options":{"size":"M"}}]}',
            error=True,
        )
    ]
    assert _reference_call_progress(trace, task_spec())["value"] == 0.0
    assert _result_evidence_progress(trace, task_spec())["value"] == 0.0


def test_unexpected_write_hard_cap_precedes_progress_tiers() -> None:
    tier, name = _candidate_tier(
        complete_success=False,
        write_value=0.5,
        result_evidence_value=1.0,
        unexpected_write_count=1,
    )
    assert tier == -1
    assert name == "unexpected_write_hard_cap"
