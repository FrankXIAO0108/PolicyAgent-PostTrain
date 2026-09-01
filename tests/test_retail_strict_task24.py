from __future__ import annotations

import copy
import json
from pathlib import Path

from src.evaluation.retail_predicates import RetailPredicateContext, evaluate_retail_predicate
from src.evaluation.task_rubric import AtomicVerdict, load_task_rubric


ROOT = Path(__file__).resolve().parents[1]
RUBRIC_PATH = ROOT / "configs" / "evaluation" / "retail_strict_v1" / "task_24.json"
FIXTURE_DIR = ROOT / "tests" / "fixtures" / "retail_strict_v1" / "task_24"
FROZEN_PACKET = (
    ROOT
    / "experiments"
    / "20260727_unreviewed_success_audit_queue_v2"
    / "packets"
    / "task_24_review_packet.json"
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
    final_state = merge_patch(state, fixture.get("final_state_patch", {}))
    return RetailPredicateContext(
        initial_state=state,
        final_state=final_state,
        messages=tuple(fixture["messages"]),
        final_answer=fixture["final_answer"],
        claim_checks=fixture["claim_checks"],
    )


def results_for(name: str) -> dict[str, object]:
    rubric = load_task_rubric(RUBRIC_PATH)
    context = context_from_fixture(name)
    return {
        predicate.predicate_id: evaluate_retail_predicate(predicate, context)
        for predicate in rubric.predicates
    }


def test_task24_rubric_loads_and_covers_all_required_groups() -> None:
    rubric = load_task_rubric(RUBRIC_PATH)
    assert rubric.task_id == "24"
    assert len(rubric.required_capability_groups) == 7
    assert {predicate.capability_group for predicate in rubric.predicates} >= set(
        rubric.required_capability_groups
    )


def test_task24_claim_contract_matches_frozen_state_projection() -> None:
    rubric = load_task_rubric(RUBRIC_PATH)
    state = load_json(FIXTURE_DIR / "task_state.json")["initial_state"]
    items = state["orders"]["#W9609649"]["items"]
    observed_identity = [
        {
            "order_id": "#W9609649",
            "item_id": item["item_id"],
            "color": item["options"]["color"],
            "size": item["options"]["size"],
        }
        for item in items
        if item["name"] == "T-Shirt"
    ]
    observed_materials = [
        {"item_id": item["item_id"], "material": item["options"]["material"]}
        for item in items
        if item["name"] == "T-Shirt"
    ]
    identity = next(
        predicate for predicate in rubric.predicates if predicate.predicate_id == "two_target_tshirts_identified"
    )
    materials = next(
        predicate for predicate in rubric.predicates if predicate.predicate_id == "target_materials_truthful"
    )
    assert identity.parameters["expected_facts"] == observed_identity
    assert materials.parameters["expected_facts"] == observed_materials


def test_task24_targeted_read_path_passes_every_required_predicate() -> None:
    outcomes = results_for("success_targeted_read.json")
    assert outcomes
    assert {outcome.verdict for outcome in outcomes.values()} == {AtomicVerdict.PASS}


def test_task24_sequential_discovery_is_an_alternative_valid_path() -> None:
    outcomes = results_for("success_sequential_discovery.json")
    assert {outcome.verdict for outcome in outcomes.values()} == {AtomicVerdict.PASS}
    fixture = load_json(FIXTURE_DIR / "success_sequential_discovery.json")
    order_reads = [
        call["arguments"]["order_id"]
        for message in fixture["messages"]
        for call in message.get("tool_calls", [])
        if call["name"] == "get_order_details"
    ]
    assert order_reads == ["#W3561391", "#W6876713", "#W9609649"]


def test_task24_does_not_require_a_unique_reference_read_sequence() -> None:
    rubric = load_task_rubric(RUBRIC_PATH)
    required_action_names = {
        predicate.parameters.get("name")
        for predicate in rubric.predicates
        if predicate.predicate_type == "required_action"
    }
    assert "get_product_details" not in required_action_names
    target_binding = next(
        predicate for predicate in rubric.predicates if predicate.predicate_id == "target_order_read_bound"
    )
    assert target_binding.parameters["quantifier"] == "any"


def test_task24_cancellation_after_withdrawal_fails_action_and_state_checks() -> None:
    outcomes = results_for("failure_cancelled_after_withdrawal.json")
    assert outcomes["no_cancel_after_withdrawal"].verdict is AtomicVerdict.FAIL
    assert outcomes["database_unchanged"].verdict is AtomicVerdict.FAIL
    assert outcomes["grill_order_unchanged"].verdict is AtomicVerdict.FAIL


def test_task24_missing_material_answer_fails_claim_checks() -> None:
    outcomes = results_for("failure_missing_materials.json")
    assert outcomes["two_target_tshirts_identified"].verdict is AtomicVerdict.FAIL
    assert outcomes["target_materials_truthful"].verdict is AtomicVerdict.FAIL
    assert outcomes["database_unchanged"].verdict is AtomicVerdict.PASS


def test_frozen_reward_one_trajectory_still_fails_strict_turn_protocol() -> None:
    rubric = load_task_rubric(RUBRIC_PATH)
    packet = load_json(FROZEN_PACKET)
    context = RetailPredicateContext(messages=tuple(packet["frozen_evidence"]["events"]))
    protocol = {
        predicate.predicate_id: evaluate_retail_predicate(predicate, context)
        for predicate in rubric.predicates
        if predicate.predicate_type in {"one_tool_call_per_turn", "message_tool_exclusivity"}
    }
    assert protocol["one_tool_call_each_turn"].verdict is AtomicVerdict.FAIL
    assert [
        item["message_index"]
        for item in protocol["one_tool_call_each_turn"].evidence
        if item.get("violates")
    ] == [9]
    assert protocol["message_tool_separation"].verdict is AtomicVerdict.FAIL
    assert [
        item["message_index"]
        for item in protocol["message_tool_separation"].evidence
        if item.get("violates")
    ] == [5, 7, 9, 16]
