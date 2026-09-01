from __future__ import annotations

import copy
import json
from pathlib import Path

from src.evaluation.retail_predicates import (
    RetailPredicateContext,
    evaluate_retail_predicate,
    final_answer_sha256,
)
from src.evaluation.task_rubric import AtomicVerdict, load_task_rubric


ROOT = Path(__file__).resolve().parents[1]
RUBRIC_PATH = ROOT / "configs" / "evaluation" / "retail_strict_v1" / "task_21.json"
FIXTURE_DIR = ROOT / "tests" / "fixtures" / "retail_strict_v1" / "task_21"
FROZEN_RESULT = (
    ROOT
    / "experiments"
    / "20260722_110504_retail_baseline20_trial1_deepseek"
    / "task_21"
    / "returned_results.json"
)


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def context_from_payload(payload: dict) -> RetailPredicateContext:
    initial_state = load_json(FIXTURE_DIR / "task_state.json")["initial_state"]
    answer = payload["final_answer"]
    claim_checks = {
        claim_id: {
            "verdict": verdict,
            "answer_sha256": final_answer_sha256(answer),
        }
        for claim_id, verdict in payload["claim_verdicts"].items()
    }
    return RetailPredicateContext(
        initial_state=initial_state,
        final_state=payload["final_state"],
        messages=tuple(payload["messages"]),
        final_answer=answer,
        confirmation_message_index_by_call_id=payload[
            "confirmation_message_index_by_call_id"
        ],
        latest_intent_revision_message_index_by_call_id=payload[
            "latest_intent_revision_message_index_by_call_id"
        ],
        claim_checks=claim_checks,
    )


def results_for_payload(payload: dict) -> dict[str, object]:
    rubric = load_task_rubric(RUBRIC_PATH)
    context = context_from_payload(payload)
    return {
        predicate.predicate_id: evaluate_retail_predicate(predicate, context)
        for predicate in rubric.predicates
    }


def frozen_context() -> RetailPredicateContext:
    initial_state = load_json(FIXTURE_DIR / "task_state.json")["initial_state"]
    result = load_json(FROZEN_RESULT)
    messages = result["simulations"][0]["messages"]
    write_message = next(
        message
        for message in messages
        if any(
            call.get("name") == "modify_pending_order_items"
            for call in message.get("tool_calls") or []
        )
    )
    write_call = next(
        call
        for call in write_message["tool_calls"]
        if call["name"] == "modify_pending_order_items"
    )
    write_result = next(
        message
        for message in messages
        if message.get("role") == "tool" and message.get("id") == write_call["id"]
    )
    order = json.loads(write_result["content"])
    relevant_items = {
        item["item_id"]: {
            "name": item["name"],
            "product_id": item["product_id"],
            "price": item["price"],
            "options": item["options"],
        }
        for item in order["items"]
        if item["product_id"] in {"6938111410", "1656367028"}
    }
    payment_history = copy.deepcopy(order["payment_history"])
    for payment in payment_history:
        payment["amount"] = round(payment["amount"], 2)
    user_result = json.loads(messages[31]["content"])
    final_state = {
        "orders": {
            "#W9911714": {
                "status": order["status"],
                "items_by_id": relevant_items,
                "payment_history": payment_history,
            }
        },
        "users": {
            "ethan_garcia_1261": {
                "gift_card_balance": user_result["payment_methods"][
                    "gift_card_4332117"
                ]["balance"]
            }
        },
    }
    final_answer = messages[32]["content"]
    return RetailPredicateContext(
        initial_state=initial_state,
        final_state=final_state,
        messages=tuple(messages),
        final_answer=final_answer,
        confirmation_message_index_by_call_id={write_call["id"]: 27},
        latest_intent_revision_message_index_by_call_id={write_call["id"]: 20},
        claim_checks={
            "modification_summary": {
                "verdict": "FAIL",
                "answer_sha256": final_answer_sha256(final_answer),
                "reason": "The returned shoe item carries keyboard price and options.",
            },
            "gift_card_balance": {
                "verdict": "PASS",
                "answer_sha256": final_answer_sha256(final_answer),
                "observed": 44.08,
            },
        },
    )


def test_task21_rubric_loads_and_covers_all_required_groups() -> None:
    rubric = load_task_rubric(RUBRIC_PATH)
    assert rubric.task_id == "21"
    assert len(rubric.required_capability_groups) == 7
    assert {predicate.capability_group for predicate in rubric.predicates} >= set(
        rubric.required_capability_groups
    )


def test_task21_contract_binds_both_latest_item_pairs_and_balance() -> None:
    rubric = load_task_rubric(RUBRIC_PATH)
    pairs = next(
        predicate
        for predicate in rubric.predicates
        if predicate.predicate_id == "atomic_item_pairs_match_latest_intent"
    )
    transition = next(
        predicate
        for predicate in rubric.predicates
        if predicate.predicate_id == "atomic_transition_is_valid"
    )
    assert pairs.parameters["expected_pairs"] == [
        ["9791469541", "4107812777"],
        ["1340995114", "1421289881"],
    ]
    assert {
        tuple(check["path"]): check["expected"]
        for check in transition.parameters["expected_final_fields"]
    }[("users", "ethan_garcia_1261", "gift_card_balance")] == 44.08


def test_task21_canonical_atomic_path_passes_every_required_predicate() -> None:
    payload = load_json(FIXTURE_DIR / "success_canonical.json")
    outcomes = results_for_payload(payload)
    assert outcomes
    assert {outcome.verdict for outcome in outcomes.values()} == {AtomicVerdict.PASS}


def test_task21_reversed_pair_order_is_an_alternative_valid_path() -> None:
    payload = load_json(FIXTURE_DIR / "success_reversed_pairs.json")
    outcomes = results_for_payload(payload)
    assert {outcome.verdict for outcome in outcomes.values()} == {AtomicVerdict.PASS}


def test_task21_stale_confirmation_after_late_revision_is_model_failure() -> None:
    payload = load_json(FIXTURE_DIR / "success_canonical.json")
    payload["confirmation_message_index_by_call_id"]["write"] = 0
    outcomes = results_for_payload(payload)
    assert outcomes["confirmation_covers_latest_revision"].verdict is AtomicVerdict.FAIL
    assert outcomes["atomic_item_pairs_match_latest_intent"].verdict is AtomicVerdict.PASS


def test_task21_cross_paired_atomic_arguments_are_model_failure() -> None:
    payload = load_json(FIXTURE_DIR / "success_canonical.json")
    write = next(
        call
        for message in payload["messages"]
        for call in message.get("tool_calls", [])
        if call["name"] == "modify_pending_order_items"
    )
    write["arguments"]["new_item_ids"] = ["1421289881", "4107812777"]
    outcomes = results_for_payload(payload)
    assert outcomes["atomic_item_pairs_match_latest_intent"].verdict is AtomicVerdict.FAIL
    assert outcomes["atomic_transition_is_valid"].verdict is AtomicVerdict.FAIL


def test_task21_false_balance_claim_fails_without_corrupting_state_verdict() -> None:
    payload = load_json(FIXTURE_DIR / "success_canonical.json")
    payload["claim_verdicts"]["gift_card_balance"] = "FAIL"
    outcomes = results_for_payload(payload)
    assert outcomes["gift_card_balance_truthful"].verdict is AtomicVerdict.FAIL
    assert outcomes["atomic_transition_is_valid"].verdict is AtomicVerdict.PASS


def test_frozen_reward_one_task21_is_invalid_environment_and_protocol_failure() -> None:
    rubric = load_task_rubric(RUBRIC_PATH)
    context = frozen_context()
    outcomes = {
        predicate.predicate_id: evaluate_retail_predicate(predicate, context)
        for predicate in rubric.predicates
    }
    assert outcomes["atomic_item_pairs_match_latest_intent"].verdict is AtomicVerdict.PASS
    assert outcomes["confirmation_covers_latest_revision"].verdict is AtomicVerdict.PASS
    assert outcomes["atomic_transition_is_valid"].verdict is AtomicVerdict.ERROR
    assert outcomes["atomic_transition_is_valid"].counts_as_model_failure is False
    assert outcomes["modification_summary_truthful"].verdict is AtomicVerdict.FAIL
    assert outcomes["one_tool_call_each_turn"].verdict is AtomicVerdict.FAIL
    assert outcomes["message_tool_separation"].verdict is AtomicVerdict.FAIL
