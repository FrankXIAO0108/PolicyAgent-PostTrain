from __future__ import annotations

import copy
import json
from pathlib import Path

from src.evaluation.retail_predicates import (
    RetailPredicateContext,
    evaluate_retail_predicate,
    final_answer_sha256,
)
from src.evaluation.strict_task_evaluator import aggregate_strict_task_evaluation
from src.evaluation.task_rubric import AtomicVerdict, CapabilityGroup, PredicateResult, load_task_rubric


ROOT = Path(__file__).resolve().parents[1]
RUBRIC_DIR = ROOT / "configs" / "evaluation" / "retail_strict_v1"
FIXTURE_DIR = ROOT / "tests" / "fixtures" / "retail_strict_v1"
BASELINE_DIR = ROOT / "experiments" / "20260722_110504_retail_baseline20_trial1_deepseek"


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def payload(task_id: int, filename: str) -> dict:
    return load_json(FIXTURE_DIR / f"task_{task_id}" / filename)


def context_from_payload(task_id: int, sample: dict) -> RetailPredicateContext:
    answer = sample["final_answer"]
    return RetailPredicateContext(
        initial_state=load_json(FIXTURE_DIR / f"task_{task_id}" / "task_state.json")[
            "initial_state"
        ],
        final_state=sample["final_state"],
        messages=tuple(sample["messages"]),
        final_answer=answer,
        confirmation_message_index_by_call_id=sample[
            "confirmation_message_index_by_call_id"
        ],
        latest_intent_revision_message_index_by_call_id=sample[
            "latest_intent_revision_message_index_by_call_id"
        ],
        claim_checks={
            claim_id: {
                "verdict": verdict,
                "answer_sha256": final_answer_sha256(answer),
            }
            for claim_id, verdict in sample["claim_verdicts"].items()
        },
    )


def evaluate(task_id: int, context: RetailPredicateContext, tau2_result=None):
    rubric = load_task_rubric(RUBRIC_DIR / f"task_{task_id}.json")
    atomic = tuple(
        evaluate_retail_predicate(predicate, context)
        for predicate in rubric.predicates
    )
    return (
        {result.predicate_id: result for result in atomic},
        aggregate_strict_task_evaluation(
            rubric,
            atomic,
            tau2_result=tau2_result,
        ),
    )


def test_missing_required_action_is_not_false_success() -> None:
    sample = payload(107, "success_two_order_exchange.json")
    sample["messages"] = [
        message
        for message in sample["messages"]
        if not (
            any(
                call.get("id") == "puzzle_write"
                for call in message.get("tool_calls") or []
            )
            or message.get("tool_call_id") == "puzzle_write"
        )
    ]
    atomic, strict = evaluate(107, context_from_payload(107, sample))
    assert atomic["exactly_two_exchange_writes"].verdict is AtomicVerdict.FAIL
    assert atomic["puzzle_exchange_transition"].verdict is AtomicVerdict.FAIL
    assert strict.evaluation_valid is True
    assert strict.strict_pass is False


def test_unrelated_final_state_write_is_rejected() -> None:
    sample = payload(107, "success_two_order_exchange.json")
    sample["final_state"]["users"]["yara_ito_8499"]["address"] = {
        "zip": "00000"
    }
    atomic, strict = evaluate(107, context_from_payload(107, sample))
    assert atomic["no_unrelated_mutation"].verdict is AtomicVerdict.FAIL
    assert strict.evaluation_valid is True
    assert strict.strict_pass is False


def test_policy_violation_with_correct_final_state_is_rejected() -> None:
    sample = payload(107, "success_two_order_exchange.json")
    boots_turn = sample["messages"][17]
    puzzle_turn = sample["messages"][19]
    boots_turn["tool_calls"].extend(puzzle_turn["tool_calls"])
    del sample["messages"][19]
    atomic, strict = evaluate(107, context_from_payload(107, sample))
    assert atomic["boots_exchange_transition"].verdict is AtomicVerdict.PASS
    assert atomic["puzzle_exchange_transition"].verdict is AtomicVerdict.PASS
    assert atomic["one_tool_call_each_turn"].verdict is AtomicVerdict.FAIL
    assert strict.evaluation_valid is True
    assert strict.strict_pass is False


def test_false_success_claim_cannot_override_correct_state() -> None:
    sample = payload(107, "success_two_order_exchange.json")
    sample["claim_verdicts"]["boots_exchange_summary"] = "FAIL"
    atomic, strict = evaluate(107, context_from_payload(107, sample))
    assert atomic["boots_exchange_transition"].verdict is AtomicVerdict.PASS
    assert atomic["boots_exchange_summary_truthful"].verdict is AtomicVerdict.FAIL
    assert strict.strict_pass is False


def test_late_user_intent_change_invalidates_stale_confirmation() -> None:
    sample = payload(59, "success_final_intent.json")
    sample["confirmation_message_index_by_call_id"]["cancel"] = 0
    atomic, strict = evaluate(59, context_from_payload(59, sample))
    assert atomic["confirmation_covers_final_intent"].verdict is AtomicVerdict.FAIL
    assert (
        atomic["cancellation_transition_matches_final_intent"].verdict
        is AtomicVerdict.PASS
    )
    assert strict.strict_pass is False


def test_legitimate_repeat_read_from_frozen_task21_is_not_penalized() -> None:
    frozen = load_json(BASELINE_DIR / "task_21" / "returned_results.json")
    messages = frozen["simulations"][0]["messages"]
    rubric = load_task_rubric(RUBRIC_DIR / "task_21.json")
    selected = {
        predicate.predicate_id: predicate
        for predicate in rubric.predicates
        if predicate.predicate_id in {"gift_card_owner_read", "user_reads_succeeded"}
    }
    context = RetailPredicateContext(messages=tuple(messages))
    calls = [
        call
        for message in messages
        for call in message.get("tool_calls") or []
        if call.get("name") == "get_user_details"
        and call.get("arguments", {}).get("user_id") == "ethan_garcia_1261"
    ]
    assert len(calls) == 2
    assert (
        evaluate_retail_predicate(selected["gift_card_owner_read"], context).verdict
        is AtomicVerdict.PASS
    )
    assert (
        evaluate_retail_predicate(selected["user_reads_succeeded"], context).verdict
        is AtomicVerdict.PASS
    )


def frozen_task59_context() -> tuple[RetailPredicateContext, dict]:
    frozen = load_json(BASELINE_DIR / "task_59" / "returned_results.json")
    simulation = frozen["simulations"][0]
    messages = simulation["messages"]
    cancel_call = next(
        call
        for message in messages
        for call in message.get("tool_calls") or []
        if call.get("name") == "cancel_pending_order"
    )
    cancel_result_message = next(
        message
        for message in messages
        if message.get("role") == "tool" and message.get("id") == cancel_call["id"]
    )
    cancel_result = json.loads(cancel_result_message["content"])
    initial_state = load_json(FIXTURE_DIR / "task_59" / "task_state.json")[
        "initial_state"
    ]
    final_state = copy.deepcopy(initial_state)
    target = final_state["orders"]["#W2702727"]
    for field in ("status", "address", "payment_history", "cancel_reason"):
        target[field] = cancel_result[field]
    final_answer = next(
        message["content"]
        for message in reversed(messages)
        if message.get("role") == "assistant" and message.get("content")
    )
    return (
        RetailPredicateContext(
            initial_state=initial_state,
            final_state=final_state,
            messages=tuple(messages),
            final_answer=final_answer,
            confirmation_message_index_by_call_id={cancel_call["id"]: 13},
            latest_intent_revision_message_index_by_call_id={cancel_call["id"]: 11},
            claim_checks={
                claim_id: {
                    "verdict": "PASS",
                    "answer_sha256": final_answer_sha256(final_answer),
                }
                for claim_id in ("cancellation_summary", "other_order_unchanged")
            },
        ),
        simulation["reward_info"],
    )


def test_static_gold_conflict_preserves_tau2_without_overriding_latest_intent() -> None:
    context, tau2 = frozen_task59_context()
    atomic, strict = evaluate(59, context, tau2_result=tau2)
    assert tau2["reward"] == 0.0
    assert strict.tau2_result is tau2
    assert (
        atomic["cancellation_transition_matches_final_intent"].verdict
        is AtomicVerdict.PASS
    )
    assert atomic["other_order_unchanged"].verdict is AtomicVerdict.PASS
    assert strict.evaluation_valid is True
    assert strict.strict_pass is False


def test_missing_state_evidence_is_evaluation_error_not_model_failure() -> None:
    sample = payload(107, "success_two_order_exchange.json")
    context = context_from_payload(107, sample)
    context = RetailPredicateContext(
        initial_state=context.initial_state,
        final_state=None,
        messages=context.messages,
        final_answer=context.final_answer,
        confirmation_message_index_by_call_id=context.confirmation_message_index_by_call_id,
        latest_intent_revision_message_index_by_call_id=context.latest_intent_revision_message_index_by_call_id,
        claim_checks=context.claim_checks,
    )
    atomic, strict = evaluate(107, context)
    assert atomic["boots_exchange_transition"].verdict is AtomicVerdict.ERROR
    assert atomic["boots_exchange_transition"].counts_as_model_failure is False
    assert strict.evaluation_valid is False
    assert strict.strict_pass is False


def test_explicit_infrastructure_error_cannot_be_counted_as_model_failure() -> None:
    rubric = load_task_rubric(RUBRIC_DIR / "task_107.json")
    sample = payload(107, "success_two_order_exchange.json")
    context = context_from_payload(107, sample)
    results = [
        evaluate_retail_predicate(predicate, context)
        for predicate in rubric.predicates
    ]
    index = next(
        i for i, item in enumerate(results) if item.predicate_id == "identity_lookup_succeeded"
    )
    results[index] = PredicateResult(
        predicate_id="identity_lookup_succeeded",
        capability_group=CapabilityGroup.EVALUATION_INTEGRITY,
        verdict=AtomicVerdict.ERROR,
        reason="The frozen tool result could not be decoded.",
        error="invalid frozen JSON",
    )
    strict = aggregate_strict_task_evaluation(rubric, results, tau2_result=None)
    assert strict.evaluation_valid is False
    assert strict.strict_pass is False
    assert results[index].counts_as_model_failure is False


def test_alternative_valid_write_order_remains_strict_pass() -> None:
    sample = payload(107, "success_two_order_exchange.json")
    sample["messages"][17:21] = sample["messages"][19:21] + sample["messages"][17:19]
    atomic, strict = evaluate(107, context_from_payload(107, sample))
    assert {result.verdict for result in atomic.values()} == {AtomicVerdict.PASS}
    assert strict.evaluation_valid is True
    assert strict.strict_pass is True
