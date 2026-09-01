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
RUBRIC_PATH = ROOT / "configs" / "evaluation" / "retail_strict_v1" / "task_107.json"
FIXTURE_DIR = ROOT / "tests" / "fixtures" / "retail_strict_v1" / "task_107"
FROZEN_RESULT = (
    ROOT
    / "experiments"
    / "20260722_110504_retail_baseline20_trial1_deepseek"
    / "task_107"
    / "returned_results.json"
)


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def canonical_payload() -> dict:
    return load_json(FIXTURE_DIR / "success_two_order_exchange.json")


def context_from_payload(payload: dict) -> RetailPredicateContext:
    answer = payload["final_answer"]
    claim_checks = {
        claim_id: {
            "verdict": verdict,
            "answer_sha256": final_answer_sha256(answer),
        }
        for claim_id, verdict in payload["claim_verdicts"].items()
    }
    return RetailPredicateContext(
        initial_state=load_json(FIXTURE_DIR / "task_state.json")["initial_state"],
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


def exchange_calls(payload: dict) -> list[dict]:
    return [
        call
        for message in payload["messages"]
        for call in message.get("tool_calls") or []
        if call.get("name") == "exchange_delivered_order_items"
    ]


def test_task107_rubric_loads_and_covers_all_required_groups() -> None:
    rubric = load_task_rubric(RUBRIC_PATH)
    assert rubric.task_id == "107"
    assert len(rubric.required_capability_groups) == 7
    assert {predicate.capability_group for predicate in rubric.predicates} >= set(
        rubric.required_capability_groups
    )


def test_task107_contract_binds_each_order_to_exact_variant_and_payment() -> None:
    rubric = load_task_rubric(RUBRIC_PATH)
    transitions = {
        predicate.predicate_id: predicate
        for predicate in rubric.predicates
        if predicate.predicate_type == "action_state_transition_consistent"
    }
    boots = transitions["boots_exchange_transition"].parameters
    puzzle = transitions["puzzle_exchange_transition"].parameters
    assert boots["selector_checks"] == [
        {"path": ["order_id"], "expected": "#W1304208"}
    ]
    assert boots["paired_arguments"]["expected_pairs"] == [
        ["1615379700", "8106223139"]
    ]
    assert puzzle["selector_checks"] == [
        {"path": ["order_id"], "expected": "#W8353027"}
    ]
    assert puzzle["paired_arguments"]["expected_pairs"] == [
        ["6245746168", "3112842858"]
    ]
    assert all(
        {"path": ["payment_method_id"], "expected": "paypal_1679017"}
        in parameters["argument_checks"]
        for parameters in (boots, puzzle)
    )


def test_task107_canonical_two_order_path_passes_every_required_predicate() -> None:
    outcomes = results_for_payload(canonical_payload())
    assert outcomes
    assert {outcome.verdict for outcome in outcomes.values()} == {AtomicVerdict.PASS}


def test_task107_reversed_exchange_write_order_is_alternative_valid_path() -> None:
    payload = canonical_payload()
    payload["messages"][17:21] = payload["messages"][19:21] + payload["messages"][17:19]
    outcomes = results_for_payload(payload)
    assert {outcome.verdict for outcome in outcomes.values()} == {AtomicVerdict.PASS}


def test_task107_same_item_boots_exchange_fails_even_when_tool_succeeds() -> None:
    payload = canonical_payload()
    boots_call = next(
        call for call in exchange_calls(payload) if call["arguments"]["order_id"] == "#W1304208"
    )
    boots_call["arguments"]["new_item_ids"] = ["1615379700"]
    boots_state = payload["final_state"]["orders"]["#W1304208"]
    boots_state["exchange_new_items"] = ["1615379700"]
    boots_state["exchange_price_difference"] = 0.0
    payload["claim_verdicts"]["boots_exchange_summary"] = "FAIL"
    payload["claim_verdicts"]["payment_summary"] = "FAIL"
    outcomes = results_for_payload(payload)
    assert outcomes["exactly_two_exchange_writes"].verdict is AtomicVerdict.PASS
    assert outcomes["boots_exchange_transition"].verdict is AtomicVerdict.FAIL
    assert outcomes["puzzle_exchange_transition"].verdict is AtomicVerdict.PASS
    assert outcomes["boots_exchange_summary_truthful"].verdict is AtomicVerdict.FAIL


def test_task107_cross_bound_order_arguments_do_not_form_false_success() -> None:
    payload = canonical_payload()
    calls = exchange_calls(payload)
    calls[0]["arguments"]["new_item_ids"] = ["3112842858"]
    calls[1]["arguments"]["new_item_ids"] = ["8106223139"]
    outcomes = results_for_payload(payload)
    assert outcomes["exactly_two_exchange_writes"].verdict is AtomicVerdict.PASS
    assert outcomes["boots_exchange_transition"].verdict is AtomicVerdict.FAIL
    assert outcomes["puzzle_exchange_transition"].verdict is AtomicVerdict.FAIL


def test_task107_duplicate_boots_write_cannot_hide_missing_puzzle_write() -> None:
    payload = canonical_payload()
    calls = exchange_calls(payload)
    calls[1]["arguments"] = copy.deepcopy(calls[0]["arguments"])
    outcomes = results_for_payload(payload)
    assert outcomes["exactly_two_exchange_writes"].verdict is AtomicVerdict.PASS
    assert outcomes["boots_exchange_transition"].verdict is AtomicVerdict.FAIL
    assert outcomes["puzzle_exchange_transition"].verdict is AtomicVerdict.FAIL


def test_task107_stale_confirmation_fails_without_corrupting_transitions() -> None:
    payload = canonical_payload()
    payload["confirmation_message_index_by_call_id"] = {
        "boots_write": 0,
        "puzzle_write": 0,
    }
    outcomes = results_for_payload(payload)
    assert (
        outcomes["both_writes_confirmed_after_final_selection"].verdict
        is AtomicVerdict.FAIL
    )
    assert outcomes["boots_exchange_transition"].verdict is AtomicVerdict.PASS
    assert outcomes["puzzle_exchange_transition"].verdict is AtomicVerdict.PASS


def frozen_context() -> tuple[RetailPredicateContext, float]:
    frozen = load_json(FROZEN_RESULT)
    simulation = frozen["simulations"][0]
    messages = simulation["messages"]
    initial_state = load_json(FIXTURE_DIR / "task_state.json")["initial_state"]
    final_state = copy.deepcopy(initial_state)
    write_calls = [
        call
        for message in messages
        for call in message.get("tool_calls") or []
        if call.get("name") == "exchange_delivered_order_items"
    ]
    for call in write_calls:
        result_message = next(
            message
            for message in messages
            if message.get("role") == "tool" and message.get("id") == call["id"]
        )
        result = json.loads(result_message["content"])
        order = final_state["orders"][call["arguments"]["order_id"]]
        for field in (
            "status",
            "exchange_items",
            "exchange_new_items",
            "exchange_payment_method_id",
            "exchange_price_difference",
        ):
            order[field] = result[field]
    final_answer = next(
        message["content"]
        for message in reversed(messages)
        if message.get("role") == "assistant" and message.get("content")
    )
    write_ids = {call["id"] for call in write_calls}
    return (
        RetailPredicateContext(
            initial_state=initial_state,
            final_state=final_state,
            messages=tuple(messages),
            final_answer=final_answer,
            confirmation_message_index_by_call_id={call_id: 19 for call_id in write_ids},
            latest_intent_revision_message_index_by_call_id={call_id: 17 for call_id in write_ids},
            claim_checks={
                "boots_exchange_summary": {
                    "verdict": "FAIL",
                    "answer_sha256": final_answer_sha256(final_answer),
                },
                "puzzle_exchange_summary": {
                    "verdict": "PASS",
                    "answer_sha256": final_answer_sha256(final_answer),
                },
                "payment_summary": {
                    "verdict": "FAIL",
                    "answer_sha256": final_answer_sha256(final_answer),
                },
            },
        ),
        simulation["reward_info"]["reward"],
    )


def test_frozen_reward_zero_has_policy_failure_despite_successful_tool_results() -> None:
    rubric = load_task_rubric(RUBRIC_PATH)
    context, tau2_reward = frozen_context()
    outcomes = {
        predicate.predicate_id: evaluate_retail_predicate(predicate, context)
        for predicate in rubric.predicates
    }
    assert tau2_reward == 0.0
    assert outcomes["exactly_two_exchange_writes"].verdict is AtomicVerdict.PASS
    assert (
        outcomes["both_writes_confirmed_after_final_selection"].verdict
        is AtomicVerdict.PASS
    )
    assert outcomes["boots_exchange_transition"].verdict is AtomicVerdict.FAIL
    assert outcomes["puzzle_exchange_transition"].verdict is AtomicVerdict.PASS
    assert outcomes["boots_exchange_summary_truthful"].verdict is AtomicVerdict.FAIL
    assert outcomes["puzzle_exchange_summary_truthful"].verdict is AtomicVerdict.PASS
    assert outcomes["payment_summary_truthful"].verdict is AtomicVerdict.FAIL
    assert outcomes["one_tool_call_each_turn"].verdict is AtomicVerdict.FAIL
    assert outcomes["message_tool_separation"].verdict is AtomicVerdict.FAIL
