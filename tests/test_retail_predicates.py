from __future__ import annotations

import json
from pathlib import Path

from src.evaluation.retail_predicates import (
    PREDICATE_REGISTRY,
    RetailPredicateContext,
    evaluate_retail_predicate,
    final_answer_sha256,
)
from src.evaluation.task_rubric import AtomicVerdict, CapabilityGroup, PredicateSpec


GROUP = CapabilityGroup.REQUIRED_TASK_EXECUTION
ROOT = Path(__file__).resolve().parents[1]
TRANSFER_MESSAGE = "YOU ARE BEING TRANSFERRED TO A HUMAN AGENT. PLEASE HOLD ON."


def spec(predicate_type: str, **parameters: object) -> PredicateSpec:
    return PredicateSpec(
        predicate_id=f"test_{predicate_type}",
        predicate_type=predicate_type,
        capability_group=GROUP,
        required=True,
        parameters=parameters,
    )


def call(call_id: str, name: str, arguments: dict, *, content: str | None = None) -> dict:
    return {
        "role": "assistant",
        "content": content,
        "tool_calls": [{"id": call_id, "name": name, "arguments": arguments}],
    }


def result(call_id: str, success: bool, content: object = None) -> dict:
    return {
        "role": "tool",
        "tool_call_id": call_id,
        "success": success,
        "content": content,
    }


def test_registry_contains_all_required_predicates() -> None:
    assert set(PREDICATE_REGISTRY) == {
        "field_equals",
        "field_unchanged",
        "required_action",
        "forbidden_action",
        "action_cardinality",
        "action_argument_equals",
        "paired_action_arguments_equal",
        "action_state_transition_consistent",
        "write_after_confirmation",
        "no_unrelated_mutation",
        "tool_result_success",
        "final_claim_matches_state",
        "transfer_and_stop",
        "one_tool_call_per_turn",
        "message_tool_exclusivity",
    }


def test_field_predicates_return_structured_state_evidence() -> None:
    context = RetailPredicateContext(
        initial_state={"orders": {"O1": {"status": "pending", "total": 20}}},
        final_state={"orders": {"O1": {"status": "cancelled", "total": 20}}},
    )
    equals = evaluate_retail_predicate(
        spec("field_equals", state="final", path=["orders", "O1", "status"], expected="cancelled"),
        context,
    )
    unchanged = evaluate_retail_predicate(
        spec("field_unchanged", path=["orders", "O1", "total"]), context
    )
    assert equals.verdict is AtomicVerdict.PASS
    assert unchanged.verdict is AtomicVerdict.PASS
    assert equals.evidence[0]["observed"] == "cancelled"
    assert unchanged.evidence[0]["initial"] == unchanged.evidence[0]["final"]


def test_missing_state_path_is_error_not_model_failure() -> None:
    outcome = evaluate_retail_predicate(
        spec("field_equals", state="final", path=["orders", "missing"], expected="cancelled"),
        RetailPredicateContext(final_state={"orders": {}}),
    )
    assert outcome.verdict is AtomicVerdict.ERROR
    assert outcome.counts_as_model_failure is False
    assert outcome.evidence[0]["found"] is False


def test_invalid_state_parameter_returns_error_instead_of_raising() -> None:
    outcome = evaluate_retail_predicate(
        spec("field_equals", state=["final"], path=["orders"], expected={}),
        RetailPredicateContext(final_state={"orders": {}}),
    )
    assert outcome.verdict is AtomicVerdict.ERROR
    assert outcome.evidence[0]["status"] == "INVALID"


def test_required_forbidden_and_argument_predicates_use_observed_calls() -> None:
    context = RetailPredicateContext(
        messages=(
            call("c1", "get_order_details", {"order_id": "O1"}),
            result("c1", True, {"status": "pending"}),
            call("c2", "cancel_pending_order", {"order_id": "O1", "reason": "requested"}),
            result("c2", True, {"status": "cancelled"}),
        )
    )
    required = evaluate_retail_predicate(spec("required_action", name="cancel_pending_order"), context)
    forbidden = evaluate_retail_predicate(spec("forbidden_action", name="exchange_delivered_order_items"), context)
    argument = evaluate_retail_predicate(
        spec("action_argument_equals", name="cancel_pending_order", argument_path=["order_id"], expected="O1"),
        context,
    )
    assert [required.verdict, forbidden.verdict, argument.verdict] == [
        AtomicVerdict.PASS,
        AtomicVerdict.PASS,
        AtomicVerdict.PASS,
    ]
    assert required.evidence[0]["call_id"] == "c2"


def test_cardinality_separates_read_retry_from_one_shot_write() -> None:
    context = RetailPredicateContext(
        messages=(
            call("r1", "get_order_details", {"order_id": "O1"}),
            result("r1", True),
            call("r2", "get_order_details", {"order_id": "O1"}),
            result("r2", True),
            call("w1", "cancel_pending_order", {"order_id": "O1"}),
            result("w1", True),
            call("w2", "cancel_pending_order", {"order_id": "O1"}),
            result("w2", True),
        )
    )
    legitimate_read_retry = evaluate_retail_predicate(
        spec("action_cardinality", name="get_order_details", action_class="read", min_count=1, max_count=2),
        context,
    )
    repeated_write = evaluate_retail_predicate(
        spec("action_cardinality", name="cancel_pending_order", action_class="write", min_count=1, max_count=1),
        context,
    )
    assert legitimate_read_retry.verdict is AtomicVerdict.PASS
    assert repeated_write.verdict is AtomicVerdict.FAIL
    assert legitimate_read_retry.evidence[0]["action_class"] == "read"
    assert repeated_write.evidence[0]["action_class"] == "write"


def test_paired_arguments_accept_equivalent_reordering_and_reject_mispairing() -> None:
    predicate = spec(
        "paired_action_arguments_equal",
        name="modify_pending_order_items",
        left_argument_path=["item_ids"],
        right_argument_path=["new_item_ids"],
        expected_pairs=[["old_shoe", "new_shoe"], ["old_keyboard", "new_keyboard"]],
    )
    canonical = RetailPredicateContext(
        messages=(
            call(
                "w1",
                "modify_pending_order_items",
                {
                    "item_ids": ["old_shoe", "old_keyboard"],
                    "new_item_ids": ["new_shoe", "new_keyboard"],
                },
            ),
            result("w1", True),
        )
    )
    reversed_pairs = RetailPredicateContext(
        messages=(
            call(
                "w1",
                "modify_pending_order_items",
                {
                    "item_ids": ["old_keyboard", "old_shoe"],
                    "new_item_ids": ["new_keyboard", "new_shoe"],
                },
            ),
            result("w1", True),
        )
    )
    mispaired = RetailPredicateContext(
        messages=(
            call(
                "w1",
                "modify_pending_order_items",
                {
                    "item_ids": ["old_shoe", "old_keyboard"],
                    "new_item_ids": ["new_keyboard", "new_shoe"],
                },
            ),
            result("w1", True),
        )
    )
    assert evaluate_retail_predicate(predicate, canonical).verdict is AtomicVerdict.PASS
    assert evaluate_retail_predicate(predicate, reversed_pairs).verdict is AtomicVerdict.PASS
    failed = evaluate_retail_predicate(predicate, mispaired)
    assert failed.verdict is AtomicVerdict.FAIL
    assert failed.evidence[0]["matched_as_multiset"] is False


def test_paired_arguments_preserve_duplicate_multiplicity() -> None:
    predicate = spec(
        "paired_action_arguments_equal",
        name="modify_pending_order_items",
        left_argument_path=["item_ids"],
        right_argument_path=["new_item_ids"],
        expected_pairs=[["old", "new"], ["old", "new"]],
    )
    context = RetailPredicateContext(
        messages=(
            call(
                "w1",
                "modify_pending_order_items",
                {"item_ids": ["old"], "new_item_ids": ["new"]},
            ),
            result("w1", True),
        )
    )
    assert evaluate_retail_predicate(predicate, context).verdict is AtomicVerdict.FAIL


def transition_spec() -> PredicateSpec:
    return spec(
        "action_state_transition_consistent",
        name="modify_pending_order_items",
        argument_checks=[
            {"path": ["order_id"], "expected": "#W1"},
            {"path": ["payment_method_id"], "expected": "gift_card_1"},
        ],
        paired_arguments={
            "left_argument_path": ["item_ids"],
            "right_argument_path": ["new_item_ids"],
            "expected_pairs": [["old_shoe", "new_shoe"], ["old_keyboard", "new_keyboard"]],
        },
        required_initial_fields=[
            {"path": ["orders", "#W1", "status"], "expected": "pending"},
            {"path": ["users", "U1", "gift_card_balance"], "expected": 86.0},
        ],
        expected_final_fields=[
            {"path": ["orders", "#W1", "items_by_id", "new_shoe", "price"], "expected": 155.33},
            {"path": ["orders", "#W1", "items_by_id", "new_keyboard", "price"], "expected": 268.77},
            {"path": ["users", "U1", "gift_card_balance"], "expected": 44.08},
        ],
    )


def transition_context(
    *,
    mispair: bool = False,
    corrupt_state: bool = False,
    tool_success: bool = True,
    initial_status: str = "pending",
) -> RetailPredicateContext:
    new_items = ["new_keyboard", "new_shoe"] if mispair else ["new_shoe", "new_keyboard"]
    shoe_price = 268.77 if corrupt_state else 155.33
    return RetailPredicateContext(
        messages=(
            call(
                "w1",
                "modify_pending_order_items",
                {
                    "order_id": "#W1",
                    "item_ids": ["old_shoe", "old_keyboard"],
                    "new_item_ids": new_items,
                    "payment_method_id": "gift_card_1",
                },
            ),
            result("w1", tool_success),
        ),
        initial_state={
            "orders": {"#W1": {"status": initial_status}},
            "users": {"U1": {"gift_card_balance": 86.0}},
        },
        final_state={
            "orders": {
                "#W1": {
                    "items_by_id": {
                        "new_shoe": {"price": shoe_price},
                        "new_keyboard": {"price": 268.77},
                    }
                }
            },
            "users": {"U1": {"gift_card_balance": 44.08}},
        },
    )


def test_action_state_transition_passes_correct_call_and_state() -> None:
    outcome = evaluate_retail_predicate(transition_spec(), transition_context())
    assert outcome.verdict is AtomicVerdict.PASS


def test_action_state_transition_attributes_wrong_arguments_to_model() -> None:
    outcome = evaluate_retail_predicate(
        transition_spec(),
        transition_context(mispair=True, corrupt_state=True),
    )
    assert outcome.verdict is AtomicVerdict.FAIL
    assert outcome.counts_as_model_failure is True


def test_action_state_transition_attributes_corrupt_state_to_environment() -> None:
    outcome = evaluate_retail_predicate(
        transition_spec(),
        transition_context(corrupt_state=True),
    )
    assert outcome.verdict is AtomicVerdict.ERROR
    assert outcome.counts_as_model_failure is False
    assert "inconsistent final state" in (outcome.error or "")


def test_action_state_transition_requires_valid_initial_preconditions() -> None:
    outcome = evaluate_retail_predicate(
        transition_spec(),
        transition_context(initial_status="cancelled"),
    )
    assert outcome.verdict is AtomicVerdict.ERROR
    assert outcome.counts_as_model_failure is False
    assert "initial state" in (outcome.error or "")


def test_action_state_transition_attributes_correct_call_rejection_to_environment() -> None:
    outcome = evaluate_retail_predicate(
        transition_spec(),
        transition_context(tool_success=False),
    )
    assert outcome.verdict is AtomicVerdict.ERROR
    assert outcome.counts_as_model_failure is False
    assert "rejected by the environment" in (outcome.error or "")


def test_action_state_transition_repeated_write_is_model_failure() -> None:
    base = transition_context()
    repeated = RetailPredicateContext(
        messages=base.messages
        + (
            call(
                "w2",
                "modify_pending_order_items",
                {
                    "order_id": "#W1",
                    "item_ids": ["old_shoe", "old_keyboard"],
                    "new_item_ids": ["new_shoe", "new_keyboard"],
                    "payment_method_id": "gift_card_1",
                },
            ),
            result("w2", True),
        ),
        initial_state=base.initial_state,
        final_state=base.final_state,
    )
    outcome = evaluate_retail_predicate(transition_spec(), repeated)
    assert outcome.verdict is AtomicVerdict.FAIL
    assert outcome.counts_as_model_failure is True


def selected_transition_spec(order_id: str = "#W1") -> PredicateSpec:
    base = transition_spec()
    parameters = dict(base.parameters)
    parameters["selector_checks"] = [{"path": ["order_id"], "expected": order_id}]
    return spec(base.predicate_type, **parameters)


def multi_order_transition_context(
    *,
    selected_success: bool = True,
    duplicate_selected: bool = False,
) -> RetailPredicateContext:
    base = transition_context(tool_success=selected_success)
    extra_messages = (
        call(
            "other",
            "modify_pending_order_items",
            {
                "order_id": "#W2",
                "item_ids": ["other_old"],
                "new_item_ids": ["other_new"],
                "payment_method_id": "gift_card_2",
            },
        ),
        result("other", True),
    )
    if duplicate_selected:
        extra_messages += (
            call(
                "duplicate",
                "modify_pending_order_items",
                {
                    "order_id": "#W1",
                    "item_ids": ["old_shoe", "old_keyboard"],
                    "new_item_ids": ["new_shoe", "new_keyboard"],
                    "payment_method_id": "gift_card_1",
                },
            ),
            result("duplicate", True),
        )
    return RetailPredicateContext(
        messages=base.messages + extra_messages,
        initial_state=base.initial_state,
        final_state=base.final_state,
    )


def test_action_state_transition_selector_isolates_one_same_name_write() -> None:
    outcome = evaluate_retail_predicate(
        selected_transition_spec(),
        multi_order_transition_context(),
    )
    assert outcome.verdict is AtomicVerdict.PASS
    selected = [item for item in outcome.evidence if item["source"] == "tool_call_selector"]
    assert [(item["call_id"], item["selected"]) for item in selected] == [
        ("w1", True),
        ("other", False),
    ]


def test_action_state_transition_missing_selected_write_is_model_failure() -> None:
    outcome = evaluate_retail_predicate(
        selected_transition_spec("#W3"),
        multi_order_transition_context(),
    )
    assert outcome.verdict is AtomicVerdict.FAIL
    assert outcome.counts_as_model_failure is True


def test_action_state_transition_duplicate_selected_write_is_model_failure() -> None:
    outcome = evaluate_retail_predicate(
        selected_transition_spec(),
        multi_order_transition_context(duplicate_selected=True),
    )
    assert outcome.verdict is AtomicVerdict.FAIL
    assert outcome.counts_as_model_failure is True


def test_selected_correct_write_rejected_by_environment_is_error() -> None:
    outcome = evaluate_retail_predicate(
        selected_transition_spec(),
        multi_order_transition_context(selected_success=False),
    )
    assert outcome.verdict is AtomicVerdict.ERROR
    assert outcome.counts_as_model_failure is False


def test_action_state_transition_rejects_empty_selector_configuration() -> None:
    base = transition_spec()
    parameters = dict(base.parameters)
    parameters["selector_checks"] = []
    outcome = evaluate_retail_predicate(
        spec(base.predicate_type, **parameters),
        multi_order_transition_context(),
    )
    assert outcome.verdict is AtomicVerdict.ERROR
    assert outcome.counts_as_model_failure is False


def scalar_cancellation_spec() -> PredicateSpec:
    return spec(
        "action_state_transition_consistent",
        name="cancel_pending_order",
        argument_checks=[
            {"path": ["order_id"], "expected": "#W1"},
            {"path": ["reason"], "expected": "no longer needed"},
        ],
        required_initial_fields=[
            {"path": ["orders", "#W1", "status"], "expected": "pending"},
        ],
        expected_final_fields=[
            {"path": ["orders", "#W1", "status"], "expected": "cancelled"},
            {"path": ["orders", "#W1", "cancel_reason"], "expected": "no longer needed"},
        ],
    )


def scalar_cancellation_context(
    *,
    order_id: str = "#W1",
    reason: str = "no longer needed",
    tool_success: bool = True,
    final_status: str = "cancelled",
) -> RetailPredicateContext:
    return RetailPredicateContext(
        messages=(
            call(
                "w1",
                "cancel_pending_order",
                {"order_id": order_id, "reason": reason},
            ),
            result("w1", tool_success),
        ),
        initial_state={"orders": {"#W1": {"status": "pending"}}},
        final_state={
            "orders": {
                "#W1": {
                    "status": final_status,
                    "cancel_reason": "no longer needed",
                }
            }
        },
    )


def test_scalar_action_state_transition_passes_without_paired_arguments() -> None:
    outcome = evaluate_retail_predicate(
        scalar_cancellation_spec(),
        scalar_cancellation_context(),
    )
    assert outcome.verdict is AtomicVerdict.PASS
    assert not any(item["source"] == "paired_tool_arguments" for item in outcome.evidence)


def test_scalar_action_state_transition_wrong_argument_is_model_failure() -> None:
    outcome = evaluate_retail_predicate(
        scalar_cancellation_spec(),
        scalar_cancellation_context(reason="ordered by mistake"),
    )
    assert outcome.verdict is AtomicVerdict.FAIL
    assert outcome.counts_as_model_failure is True


def test_scalar_action_state_transition_environment_rejection_is_error() -> None:
    outcome = evaluate_retail_predicate(
        scalar_cancellation_spec(),
        scalar_cancellation_context(tool_success=False),
    )
    assert outcome.verdict is AtomicVerdict.ERROR
    assert outcome.counts_as_model_failure is False


def test_scalar_action_state_transition_corrupt_final_state_is_error() -> None:
    outcome = evaluate_retail_predicate(
        scalar_cancellation_spec(),
        scalar_cancellation_context(final_status="pending"),
    )
    assert outcome.verdict is AtomicVerdict.ERROR
    assert outcome.counts_as_model_failure is False


def test_action_state_transition_rejects_empty_action_semantics() -> None:
    predicate = spec(
        "action_state_transition_consistent",
        name="cancel_pending_order",
        argument_checks=[],
        required_initial_fields=[
            {"path": ["orders", "#W1", "status"], "expected": "pending"},
        ],
        expected_final_fields=[
            {"path": ["orders", "#W1", "status"], "expected": "cancelled"},
        ],
    )
    outcome = evaluate_retail_predicate(predicate, scalar_cancellation_context())
    assert outcome.verdict is AtomicVerdict.ERROR
    assert outcome.counts_as_model_failure is False


def test_write_confirmation_requires_explicit_call_binding_before_write() -> None:
    context = RetailPredicateContext(
        messages=(
            {"role": "user", "content": "确认取消"},
            call("w1", "cancel_pending_order", {"order_id": "O1"}),
            result("w1", True),
        ),
        confirmation_message_index_by_call_id={"w1": 0},
    )
    passed = evaluate_retail_predicate(spec("write_after_confirmation", name="cancel_pending_order"), context)
    missing_binding = evaluate_retail_predicate(
        spec("write_after_confirmation", name="cancel_pending_order"),
        RetailPredicateContext(messages=context.messages),
    )
    assert passed.verdict is AtomicVerdict.PASS
    assert missing_binding.verdict is AtomicVerdict.ERROR


def test_write_confirmation_can_require_latest_revision_before_confirmation() -> None:
    messages = (
        {"role": "user", "content": "先只修改鞋"},
        {"role": "user", "content": "再追加修改键盘"},
        {"role": "user", "content": "确认修改这两件商品"},
        call("w1", "modify_pending_order_items", {"order_id": "O1"}),
        result("w1", True),
    )
    predicate = spec(
        "write_after_confirmation",
        name="modify_pending_order_items",
        require_after_latest_revision=True,
    )
    passed = evaluate_retail_predicate(
        predicate,
        RetailPredicateContext(
            messages=messages,
            confirmation_message_index_by_call_id={"w1": 2},
            latest_intent_revision_message_index_by_call_id={"w1": 1},
        ),
    )
    stale_confirmation = evaluate_retail_predicate(
        predicate,
        RetailPredicateContext(
            messages=messages,
            confirmation_message_index_by_call_id={"w1": 0},
            latest_intent_revision_message_index_by_call_id={"w1": 1},
        ),
    )
    missing_revision_binding = evaluate_retail_predicate(
        predicate,
        RetailPredicateContext(
            messages=messages,
            confirmation_message_index_by_call_id={"w1": 2},
        ),
    )
    assert passed.verdict is AtomicVerdict.PASS
    assert passed.evidence[0]["confirmed_after_latest_revision"] is True
    assert stale_confirmation.verdict is AtomicVerdict.FAIL
    assert stale_confirmation.counts_as_model_failure is True
    assert missing_revision_binding.verdict is AtomicVerdict.ERROR
    assert missing_revision_binding.counts_as_model_failure is False


def test_write_confirmation_rejects_non_user_or_boolean_bindings_as_evaluator_error() -> None:
    messages = (
        {"role": "user", "content": "修改商品"},
        {"role": "assistant", "content": "请确认", "tool_calls": []},
        call("w1", "modify_pending_order_items", {"order_id": "O1"}),
        result("w1", True),
    )
    predicate = spec("write_after_confirmation", name="modify_pending_order_items")
    assistant_binding = evaluate_retail_predicate(
        predicate,
        RetailPredicateContext(
            messages=messages,
            confirmation_message_index_by_call_id={"w1": 1},
        ),
    )
    boolean_binding = evaluate_retail_predicate(
        predicate,
        RetailPredicateContext(
            messages=messages,
            confirmation_message_index_by_call_id={"w1": True},
        ),
    )
    assert assistant_binding.verdict is AtomicVerdict.ERROR
    assert boolean_binding.verdict is AtomicVerdict.ERROR


def test_no_unrelated_mutation_rejects_change_outside_allowed_subtree() -> None:
    context = RetailPredicateContext(
        initial_state={"orders": {"O1": {"status": "pending"}}, "users": {"U1": {"tier": "basic"}}},
        final_state={"orders": {"O1": {"status": "cancelled"}}, "users": {"U1": {"tier": "vip"}}},
    )
    outcome = evaluate_retail_predicate(
        spec("no_unrelated_mutation", allowed_paths=[["orders", "O1", "status"]]), context
    )
    assert outcome.verdict is AtomicVerdict.FAIL
    assert outcome.evidence[0]["unrelated_paths"] == [["users", "U1", "tier"]]


def test_tool_result_success_requires_explicit_result_status() -> None:
    passed = evaluate_retail_predicate(
        spec("tool_result_success", name="get_order_details"),
        RetailPredicateContext(messages=(call("r1", "get_order_details", {"order_id": "O1"}), result("r1", True))),
    )
    missing = evaluate_retail_predicate(
        spec("tool_result_success", name="get_order_details"),
        RetailPredicateContext(messages=(call("r1", "get_order_details", {"order_id": "O1"}),)),
    )
    assert passed.verdict is AtomicVerdict.PASS
    assert missing.verdict is AtomicVerdict.ERROR


def test_tool_result_success_accepts_tau2_tool_error_encoding() -> None:
    passed = evaluate_retail_predicate(
        spec("tool_result_success", name="get_order_details"),
        RetailPredicateContext(
            messages=(
                call("r1", "get_order_details", {"order_id": "O1"}),
                {"role": "tool", "tool_call_id": "r1", "tool_error": False, "content": "{}"},
            )
        ),
    )
    failed = evaluate_retail_predicate(
        spec("tool_result_success", name="get_order_details"),
        RetailPredicateContext(
            messages=(
                call("r1", "get_order_details", {"order_id": "O1"}),
                {"role": "tool", "tool_call_id": "r1", "tool_error": True, "content": "error"},
            )
        ),
    )
    assert passed.verdict is AtomicVerdict.PASS
    assert failed.verdict is AtomicVerdict.FAIL


def test_final_claim_check_must_be_bound_to_exact_final_answer() -> None:
    answer = "订单 O1 已取消。"
    check = {"verdict": "PASS", "answer_sha256": final_answer_sha256(answer), "state_path": ["orders", "O1", "status"], "observed": "cancelled"}
    passed = evaluate_retail_predicate(
        spec("final_claim_matches_state", claim_id="order_status"),
        RetailPredicateContext(final_answer=answer, claim_checks={"order_status": check}),
    )
    stale = evaluate_retail_predicate(
        spec("final_claim_matches_state", claim_id="order_status"),
        RetailPredicateContext(final_answer="不同答复", claim_checks={"order_status": check}),
    )
    assert passed.verdict is AtomicVerdict.PASS
    assert stale.verdict is AtomicVerdict.ERROR


def test_transfer_and_stop_requires_successful_transfer_and_no_later_activity() -> None:
    clean_messages = (
        call("t1", "transfer_to_human_agents", {"summary": "unsupported request"}),
        result("t1", True),
    )
    passed = evaluate_retail_predicate(
        spec("transfer_and_stop"), RetailPredicateContext(messages=clean_messages, stopped=True)
    )
    continued = evaluate_retail_predicate(
        spec("transfer_and_stop"),
        RetailPredicateContext(
            messages=clean_messages + ({"role": "assistant", "content": "我再尝试一下", "tool_calls": []},),
            stopped=True,
        ),
    )
    assert passed.verdict is AtomicVerdict.PASS
    assert continued.verdict is AtomicVerdict.FAIL
    assert [
        item["message_index"] for item in continued.evidence[1]["later_assistant_activity"]
    ] == [2]


def test_transfer_and_stop_allows_exact_policy_terminal_message() -> None:
    messages = (
        call("t1", "transfer_to_human_agents", {"summary": "unsupported request"}),
        {"role": "tool", "tool_call_id": "t1", "tool_error": False, "content": "Transfer successful"},
        {"role": "assistant", "content": TRANSFER_MESSAGE, "tool_calls": []},
    )
    passed = evaluate_retail_predicate(
        spec("transfer_and_stop", required_terminal_message=TRANSFER_MESSAGE),
        RetailPredicateContext(messages=messages, stopped=True),
    )
    wrong_message = evaluate_retail_predicate(
        spec("transfer_and_stop", required_terminal_message=TRANSFER_MESSAGE),
        RetailPredicateContext(
            messages=messages[:-1]
            + ({"role": "assistant", "content": "Transfer complete.", "tool_calls": []},),
            stopped=True,
        ),
    )
    assert passed.verdict is AtomicVerdict.PASS
    assert passed.evidence[1]["terminal_protocol_passed"] is True
    assert wrong_message.verdict is AtomicVerdict.FAIL

    interleaved = evaluate_retail_predicate(
        spec("transfer_and_stop", required_terminal_message=TRANSFER_MESSAGE),
        RetailPredicateContext(
            messages=(
                messages[0],
                {"role": "assistant", "content": "Please wait.", "tool_calls": []},
                messages[1],
                messages[2],
            ),
            stopped=True,
        ),
    )
    assert interleaved.verdict is AtomicVerdict.FAIL


def test_frozen_task50_satisfies_policy_terminal_transfer_contract() -> None:
    packet_path = (
        ROOT
        / "experiments"
        / "20260727_unreviewed_success_audit_queue_v2"
        / "packets"
        / "task_50_review_packet.json"
    )
    packet = json.loads(packet_path.read_text(encoding="utf-8"))
    frozen = packet["frozen_evidence"]
    outcome = evaluate_retail_predicate(
        spec("transfer_and_stop", required_terminal_message=TRANSFER_MESSAGE),
        RetailPredicateContext(
            messages=tuple(frozen["events"]),
            stopped=frozen["summary"]["termination_reason"] == "user_stop",
        ),
    )
    assert outcome.verdict is AtomicVerdict.PASS
    assert outcome.evidence[0]["result_success"] is True
    assert outcome.evidence[1]["later_assistant_activity"] == [
        {"message_index": 15, "content": TRANSFER_MESSAGE, "tool_call_count": 0}
    ]


def test_retail_turn_protocol_rejects_parallel_calls_and_mixed_content() -> None:
    violating_context = RetailPredicateContext(
        messages=(
            {
                "role": "assistant",
                "content": "I will check both orders.",
                "tool_calls": [
                    {"id": "r1", "name": "get_order_details", "arguments": {"order_id": "O1"}},
                    {"id": "r2", "name": "get_order_details", "arguments": {"order_id": "O2"}},
                ],
            },
        )
    )
    cardinality = evaluate_retail_predicate(
        spec("one_tool_call_per_turn", max_calls=1), violating_context
    )
    exclusivity = evaluate_retail_predicate(
        spec("message_tool_exclusivity"), violating_context
    )
    assert cardinality.verdict is AtomicVerdict.FAIL
    assert exclusivity.verdict is AtomicVerdict.FAIL
    assert cardinality.evidence[0]["tool_call_count"] == 2
    assert exclusivity.evidence[0]["has_user_facing_content"] is True


def test_retail_turn_protocol_accepts_separate_single_call_turns() -> None:
    clean_context = RetailPredicateContext(
        messages=(
            call("r1", "get_order_details", {"order_id": "O1"}),
            result("r1", True),
            {"role": "assistant", "content": "The order is pending.", "tool_calls": []},
        )
    )
    assert evaluate_retail_predicate(
        spec("one_tool_call_per_turn"), clean_context
    ).verdict is AtomicVerdict.PASS
    assert evaluate_retail_predicate(
        spec("message_tool_exclusivity"), clean_context
    ).verdict is AtomicVerdict.PASS


def test_retail_turn_protocol_missing_messages_is_error() -> None:
    for predicate_type in ("one_tool_call_per_turn", "message_tool_exclusivity"):
        outcome = evaluate_retail_predicate(
            spec(predicate_type), RetailPredicateContext()
        )
        assert outcome.verdict is AtomicVerdict.ERROR
        assert outcome.counts_as_model_failure is False


def test_unsupported_predicate_returns_error_with_evidence() -> None:
    outcome = evaluate_retail_predicate(spec("not_implemented"), RetailPredicateContext())
    assert outcome.verdict is AtomicVerdict.ERROR
    assert outcome.evidence[0]["status"] == "UNSUPPORTED"
