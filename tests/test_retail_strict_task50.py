from __future__ import annotations

import copy
import json
from pathlib import Path

from src.evaluation.retail_predicates import RetailPredicateContext, evaluate_retail_predicate
from src.evaluation.task_rubric import AtomicVerdict, load_task_rubric


ROOT = Path(__file__).resolve().parents[1]
RUBRIC_PATH = ROOT / "configs" / "evaluation" / "retail_strict_v1" / "task_50.json"
FIXTURE_DIR = ROOT / "tests" / "fixtures" / "retail_strict_v1" / "task_50"
FROZEN_PACKET = (
    ROOT
    / "experiments"
    / "20260727_unreviewed_success_audit_queue_v2"
    / "packets"
    / "task_50_review_packet.json"
)


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def merge_patch(target: dict, patch: dict) -> dict:
    output = copy.deepcopy(target)
    for key, value in patch.items():
        if isinstance(value, dict) and isinstance(output.get(key), dict):
            output[key] = merge_patch(output[key], value)
        else:
            output[key] = copy.deepcopy(value)
    return output


def context_from_fixture(name: str) -> RetailPredicateContext:
    state = load_json(FIXTURE_DIR / "task_state.json")["initial_state"]
    fixture = load_json(FIXTURE_DIR / name)
    return RetailPredicateContext(
        initial_state=state,
        final_state=merge_patch(state, fixture.get("final_state_patch", {})),
        messages=tuple(fixture["messages"]),
        stopped=fixture["stopped"],
    )


def results_for(name: str) -> dict[str, object]:
    rubric = load_task_rubric(RUBRIC_PATH)
    context = context_from_fixture(name)
    return {
        predicate.predicate_id: evaluate_retail_predicate(predicate, context)
        for predicate in rubric.predicates
    }


def test_task50_rubric_loads_and_covers_all_required_groups() -> None:
    rubric = load_task_rubric(RUBRIC_PATH)
    assert rubric.task_id == "50"
    assert len(rubric.required_capability_groups) == 7
    assert {predicate.capability_group for predicate in rubric.predicates} >= set(
        rubric.required_capability_groups
    )


def test_task50_explain_then_transfer_path_passes() -> None:
    outcomes = results_for("success_explain_then_transfer.json")
    assert {outcome.verdict for outcome in outcomes.values()} == {AtomicVerdict.PASS}


def test_task50_immediate_transfer_is_an_alternative_valid_path() -> None:
    outcomes = results_for("success_immediate_transfer.json")
    assert {outcome.verdict for outcome in outcomes.values()} == {AtomicVerdict.PASS}


def test_task50_does_not_require_reference_summary_wording() -> None:
    first = load_json(FIXTURE_DIR / "success_explain_then_transfer.json")
    second = load_json(FIXTURE_DIR / "success_immediate_transfer.json")
    summaries = [
        call["arguments"]["summary"]
        for fixture in (first, second)
        for message in fixture["messages"]
        for call in message.get("tool_calls", [])
        if call["name"] == "transfer_to_human_agents"
    ]
    assert len(summaries) == 2
    assert summaries[0] != summaries[1]
    rubric = load_task_rubric(RUBRIC_PATH)
    assert not any(
        predicate.predicate_type == "action_argument_equals"
        and predicate.parameters.get("name") == "transfer_to_human_agents"
        for predicate in rubric.predicates
    )


def test_task50_fabricated_restore_fails_action_and_state_checks() -> None:
    outcomes = results_for("failure_fabricated_restore.json")
    assert outcomes["no_fabricated_restore_tool"].verdict is AtomicVerdict.FAIL
    assert outcomes["database_unchanged"].verdict is AtomicVerdict.FAIL
    assert outcomes["cancelled_order_unchanged"].verdict is AtomicVerdict.FAIL
    assert outcomes["cancelled_status_preserved"].verdict is AtomicVerdict.FAIL


def test_task50_missing_transfer_fails_required_execution() -> None:
    outcomes = results_for("failure_no_transfer.json")
    assert outcomes["human_transfer_required"].verdict is AtomicVerdict.FAIL
    assert outcomes["transfer_terminal_contract"].verdict is AtomicVerdict.FAIL


def test_task50_wrong_terminal_message_fails_protocol() -> None:
    outcomes = results_for("failure_wrong_terminal_message.json")
    assert outcomes["human_transfer_required"].verdict is AtomicVerdict.PASS
    assert outcomes["transfer_terminal_contract"].verdict is AtomicVerdict.FAIL


def test_frozen_reward_one_trajectory_has_sound_transfer_but_protocol_violations() -> None:
    rubric = load_task_rubric(RUBRIC_PATH)
    packet = load_json(FROZEN_PACKET)
    frozen = packet["frozen_evidence"]
    context = RetailPredicateContext(
        messages=tuple(frozen["events"]),
        stopped=frozen["summary"]["termination_reason"] == "user_stop",
    )
    selected = {
        predicate.predicate_id: evaluate_retail_predicate(predicate, context)
        for predicate in rubric.predicates
        if predicate.predicate_id
        in {"transfer_terminal_contract", "one_tool_call_each_turn", "message_tool_separation"}
    }
    assert selected["transfer_terminal_contract"].verdict is AtomicVerdict.PASS
    assert selected["one_tool_call_each_turn"].verdict is AtomicVerdict.PASS
    assert selected["message_tool_separation"].verdict is AtomicVerdict.FAIL
    assert [
        item["message_index"]
        for item in selected["message_tool_separation"].evidence
        if item.get("violates")
    ] == [5, 7, 9, 13]
