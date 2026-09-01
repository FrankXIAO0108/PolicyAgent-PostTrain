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
RUBRIC_PATH = ROOT / "configs" / "evaluation" / "retail_strict_v1" / "task_59.json"
FIXTURE_DIR = ROOT / "tests" / "fixtures" / "retail_strict_v1" / "task_59"
FROZEN_RESULT = (
    ROOT
    / "experiments"
    / "20260722_110504_retail_baseline20_trial1_deepseek"
    / "task_59"
    / "returned_results.json"
)


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


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


def canonical_payload() -> dict:
    return load_json(FIXTURE_DIR / "success_final_intent.json")


def test_task59_rubric_loads_and_covers_all_required_groups() -> None:
    rubric = load_task_rubric(RUBRIC_PATH)
    assert rubric.task_id == "59"
    assert len(rubric.required_capability_groups) == 7
    assert {predicate.capability_group for predicate in rubric.predicates} >= set(
        rubric.required_capability_groups
    )


def test_task59_records_latest_intent_authority_and_static_gold_conflict() -> None:
    rubric = load_task_rubric(RUBRIC_PATH)
    assert rubric.metadata["latest_authorized_intent"] == {
        "cancel_order_id": "#W2702727",
        "cancel_reason": "no longer needed",
        "leave_unchanged": ["#W8268610"],
        "withdrawn_branch": "modify #W2702727 shipping address",
    }
    assert rubric.metadata["static_gold_conflict"] == {
        "cancel_order_id": "#W8268610",
        "modify_address_order_id": "#W2702727",
        "refund_amount": 164.28,
    }
    assert "original Tau2 reward remains preserved" in rubric.metadata["authority_rule"]


def test_task59_final_authorized_intent_passes_every_required_predicate() -> None:
    outcomes = results_for_payload(canonical_payload())
    assert outcomes
    assert {outcome.verdict for outcome in outcomes.values()} == {AtomicVerdict.PASS}


def test_task59_read_order_is_not_a_unique_gold_path() -> None:
    payload = canonical_payload()
    payload["messages"][5:9] = payload["messages"][7:9] + payload["messages"][5:7]
    outcomes = results_for_payload(payload)
    assert {outcome.verdict for outcome in outcomes.values()} == {AtomicVerdict.PASS}


def test_task59_stale_confirmation_before_final_choice_is_model_failure() -> None:
    payload = canonical_payload()
    payload["confirmation_message_index_by_call_id"]["cancel"] = 0
    outcomes = results_for_payload(payload)
    assert outcomes["confirmation_covers_final_intent"].verdict is AtomicVerdict.FAIL
    assert outcomes["cancellation_transition_matches_final_intent"].verdict is AtomicVerdict.PASS


def test_task59_static_gold_path_fails_latest_intent_and_state_scope() -> None:
    payload = canonical_payload()
    cancel = next(
        call
        for message in payload["messages"]
        for call in message.get("tool_calls", [])
        if call["name"] == "cancel_pending_order"
    )
    cancel["arguments"]["order_id"] = "#W8268610"
    payload["messages"].insert(
        -1,
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "address",
                    "name": "modify_pending_order_address",
                    "arguments": {
                        "order_id": "#W2702727",
                        "address1": "1234 Elm St",
                        "address2": "",
                        "city": "Springfield",
                        "state": "IL",
                        "country": "USA",
                        "zip": "62701",
                    },
                }
            ],
        },
    )
    payload["messages"].insert(
        -1,
        {
            "role": "tool",
            "tool_call_id": "address",
            "success": True,
            "content": {"status": "pending"},
        },
    )
    payload["final_state"]["orders"]["#W2702727"]["status"] = "pending"
    payload["final_state"]["orders"]["#W2702727"]["address"] = {
        "address1": "1234 Elm St",
        "address2": "",
        "city": "Springfield",
        "country": "USA",
        "state": "IL",
        "zip": "62701",
    }
    payload["final_state"]["orders"]["#W2702727"]["payment_history"] = [
        {
            "transaction_type": "payment",
            "amount": 625.6,
            "payment_method_id": "credit_card_3599838",
        }
    ]
    payload["final_state"]["orders"]["#W2702727"]["cancel_reason"] = None
    other = payload["final_state"]["orders"]["#W8268610"]
    other["status"] = "cancelled"
    other["cancel_reason"] = "no longer needed"
    other["payment_history"].append(
        {
            "transaction_type": "refund",
            "amount": 164.28,
            "payment_method_id": "credit_card_3599838",
        }
    )
    outcomes = results_for_payload(payload)
    assert outcomes["cancellation_transition_matches_final_intent"].verdict is AtomicVerdict.FAIL
    assert outcomes["address_not_modified_after_cancellation_choice"].verdict is AtomicVerdict.FAIL
    assert outcomes["address_write_forbidden_after_cancellation_choice"].verdict is AtomicVerdict.FAIL
    assert outcomes["other_order_unchanged"].verdict is AtomicVerdict.FAIL
    assert outcomes["no_unrelated_mutation"].verdict is AtomicVerdict.FAIL


def frozen_context() -> tuple[RetailPredicateContext, float]:
    frozen = load_json(FROZEN_RESULT)
    simulation = frozen["simulations"][0]
    messages = simulation["messages"]
    cancel_message = next(
        message
        for message in messages
        if any(
            call.get("name") == "cancel_pending_order"
            for call in message.get("tool_calls") or []
        )
    )
    cancel_call = next(
        call
        for call in cancel_message["tool_calls"]
        if call["name"] == "cancel_pending_order"
    )
    cancel_result_message = next(
        message
        for message in messages
        if message.get("role") == "tool" and message.get("id") == cancel_call["id"]
    )
    cancel_result = json.loads(cancel_result_message["content"])
    initial_state = load_json(FIXTURE_DIR / "task_state.json")["initial_state"]
    final_state = copy.deepcopy(initial_state)
    target = final_state["orders"]["#W2702727"]
    target["status"] = cancel_result["status"]
    target["address"] = cancel_result["address"]
    target["payment_history"] = cancel_result["payment_history"]
    target["cancel_reason"] = cancel_result["cancel_reason"]
    final_answer = next(
        message["content"]
        for message in reversed(messages)
        if message.get("role") == "assistant" and message.get("content")
    )
    claim_checks = {
        claim_id: {
            "verdict": "PASS",
            "answer_sha256": final_answer_sha256(final_answer),
        }
        for claim_id in ("cancellation_summary", "other_order_unchanged")
    }
    return (
        RetailPredicateContext(
            initial_state=initial_state,
            final_state=final_state,
            messages=tuple(messages),
            final_answer=final_answer,
            confirmation_message_index_by_call_id={cancel_call["id"]: 13},
            latest_intent_revision_message_index_by_call_id={cancel_call["id"]: 11},
            claim_checks=claim_checks,
        ),
        simulation["reward_info"]["reward"],
    )


def test_frozen_reward_zero_follows_final_intent_but_violates_turn_protocol() -> None:
    rubric = load_task_rubric(RUBRIC_PATH)
    context, tau2_reward = frozen_context()
    outcomes = {
        predicate.predicate_id: evaluate_retail_predicate(predicate, context)
        for predicate in rubric.predicates
    }
    assert tau2_reward == 0.0
    for predicate_id in (
        "single_cancellation_write",
        "confirmation_covers_final_intent",
        "cancellation_transition_matches_final_intent",
        "address_not_modified_after_cancellation_choice",
        "address_write_forbidden_after_cancellation_choice",
        "other_order_unchanged",
        "no_unrelated_mutation",
        "cancellation_summary_truthful",
        "other_order_summary_truthful",
    ):
        assert outcomes[predicate_id].verdict is AtomicVerdict.PASS
    assert outcomes["one_tool_call_each_turn"].verdict is AtomicVerdict.FAIL
    assert outcomes["message_tool_separation"].verdict is AtomicVerdict.FAIL
