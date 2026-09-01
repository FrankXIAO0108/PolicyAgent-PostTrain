from copy import deepcopy
from types import SimpleNamespace as NS

import pytest

from src.evaluation.write_execution_audit import audit_write_execution
from src.rl.retail_agentic_env import one_to_one_action_progress


NAME = "modify_pending_order_address"
ARGS = {"order_id": "o1", "city": "Denver"}


class ExpectedAction(NS):
    def compare_with_tool_call(self, call):
        return self.name == call.name and self.arguments == call.arguments


def fixture():
    action = ExpectedAction(
        action_id="a1", name=NAME, arguments=deepcopy(ARGS), requestor="assistant"
    )
    call = NS(id="c1", name=NAME, arguments=deepcopy(ARGS), requestor="assistant")
    task = NS(evaluation_criteria=NS(actions=[action]))
    messages = [
        NS(role="assistant", tool_calls=[call]),
        NS(role="tool", id="c1", error=False, content='{"city":"Denver"}'),
    ]
    return task, messages


def test_success_and_no_state_or_policy_claim():
    task, messages = fixture()
    result = audit_write_execution(task, messages)
    assert result["status"] == "PASS"
    assert result["verified_write_fraction"] == 1
    assert not result["final_state_verified"]
    assert not result["policy_verified"]
    assert not result["used_as_training_reward"]


def test_reproduce_legacy_failed_write_credit_without_changing_legacy():
    task, messages = fixture()
    messages[1].error = True
    messages[1].content = "Order is not pending"
    assert one_to_one_action_progress(task, messages)["matched_count"] == 1
    result = audit_write_execution(task, messages)
    assert result["status"] == "FAIL"
    assert result["verified_write_fraction"] == 0
    assert result["write_calls"][0]["reason"] == "TOOL_REPORTED_ERROR"


@pytest.mark.parametrize(
    "variant,reason",
    [
        ("missing", "MISSING_RESULT"),
        ("wrong_id", "MISSING_RESULT"),
        ("early", "RESULT_BEFORE_CALL"),
        ("duplicate_result", "DUPLICATE_RESULT_ID"),
        ("duplicate_call", "DUPLICATE_CALL_ID"),
        ("missing_id", "MISSING_CALL_ID"),
        ("missing_error", "INVALID_ERROR_FLAG"),
        ("string_error", "INVALID_ERROR_FLAG"),
        ("empty", "MISSING_RESULT_CONTENT"),
        ("wrong_name", "RESULT_TOOL_NAME_MISMATCH"),
    ],
)
def test_invalid_evidence_is_error_not_failure(variant, reason):
    task, messages = fixture()
    if variant == "missing":
        messages.pop()
    elif variant == "wrong_id":
        messages[1].id = "other"
    elif variant == "early":
        messages.reverse()
    elif variant == "duplicate_result":
        messages.append(deepcopy(messages[1]))
    elif variant == "duplicate_call":
        messages.insert(1, deepcopy(messages[0]))
    elif variant == "missing_id":
        messages[0].tool_calls[0].id = ""
    elif variant == "missing_error":
        del messages[1].error
    elif variant == "string_error":
        messages[1].error = "false"
    elif variant == "empty":
        messages[1].content = " "
    elif variant == "wrong_name":
        messages[1].name = "get_order_details"
    result = audit_write_execution(task, messages)
    assert result["status"] == "ERROR"
    assert result["verified_write_fraction"] is None
    assert result["write_calls"][0]["reason"] == reason


def test_retry_can_match_later_success():
    task, messages = fixture()
    retry = deepcopy(messages)
    messages[1].error = True
    retry[0].tool_calls[0].id = retry[1].id = "c2"
    result = audit_write_execution(task, messages + retry)
    assert result["status"] == "PASS"
    assert result["required_writes"][0]["successful_call_id"] == "c2"


def test_one_call_cannot_satisfy_two_actions():
    task, messages = fixture()
    second = deepcopy(task.evaluation_criteria.actions[0])
    second.action_id = "a2"
    task.evaluation_criteria.actions.append(second)
    result = audit_write_execution(task, messages)
    assert result["status"] == "FAIL"
    assert result["verified_write_fraction"] == 0.5


def test_two_calls_can_satisfy_two_actions():
    task, messages = fixture()
    second = deepcopy(task.evaluation_criteria.actions[0])
    second.action_id = "a2"
    task.evaluation_criteria.actions.append(second)
    another = deepcopy(messages)
    another[0].tool_calls[0].id = another[1].id = "c2"
    assert audit_write_execution(task, messages + another)["status"] == "PASS"


def test_item_pair_reordering_but_not_cross_pairing():
    task, messages = fixture()
    action, call = task.evaluation_criteria.actions[0], messages[0].tool_calls[0]
    action.name = call.name = "modify_pending_order_items"
    action.arguments = {"item_ids": ["a", "b"], "new_item_ids": ["x", "y"]}
    call.arguments = {"item_ids": ["b", "a"], "new_item_ids": ["y", "x"]}
    assert audit_write_execution(task, messages)["status"] == "PASS"
    call.arguments["new_item_ids"] = ["x", "y"]
    assert audit_write_execution(task, messages)["verified_write_count"] == 0


def test_no_write_and_no_mutation():
    task, messages = fixture()
    before = deepcopy((task, messages))
    assert audit_write_execution(task, messages[:1])["status"] == "ERROR"
    assert audit_write_execution(task, [])["status"] == "FAIL"
    audit_write_execution(task, messages)
    assert (task, messages) == before


def test_empty_vs_missing_criteria():
    assert (
        audit_write_execution(NS(evaluation_criteria=NS(actions=[])), [])["status"]
        == "NOT_APPLICABLE"
    )
    with pytest.raises(ValueError):
        audit_write_execution(NS(), [])


def test_duplicate_action_ids_rejected():
    task, messages = fixture()
    task.evaluation_criteria.actions *= 2
    with pytest.raises(ValueError):
        audit_write_execution(task, messages)


def test_success_does_not_hide_ambiguous_other_attempt():
    task, messages = fixture()
    other = deepcopy(messages[0])
    other.tool_calls[0].id = "missing"
    result = audit_write_execution(task, messages + [other])
    assert result["verified_write_count"] == 1
    assert result["status"] == "ERROR"
    assert result["verified_write_fraction"] is None


def test_user_tool_result_cannot_satisfy_assistant_write():
    task, messages = fixture()
    messages[1].requestor = "user"
    assert audit_write_execution(task, messages)["status"] == "ERROR"


def test_user_tool_call_cannot_satisfy_assistant_write():
    task, messages = fixture()
    messages[0].role = "user"
    messages[0].tool_calls[0].requestor = "user"
    assert audit_write_execution(task, messages)["verified_write_count"] == 0


def test_same_id_in_user_and_assistant_calls_is_ambiguous():
    task, messages = fixture()
    other = deepcopy(messages[0])
    other.role = "user"
    other.tool_calls[0].requestor = "user"
    assert audit_write_execution(task, messages + [other])["status"] == "ERROR"


def test_native_tau2_objects(monkeypatch):
    monkeypatch.setenv("LITELLM_LOCAL_MODEL_COST_MAP", "True")
    tasks = pytest.importorskip("tau2.data_model.tasks")
    types = pytest.importorskip("tau2.data_model.message")
    task = NS(
        evaluation_criteria=NS(
            actions=[
                tasks.Action(
                    action_id="a1",
                    name=NAME,
                    arguments=ARGS,
                )
            ]
        )
    )
    messages = [
        types.AssistantMessage(
            role="assistant",
            tool_calls=[
                types.ToolCall(id="c1", name=NAME, arguments=ARGS),
            ],
        ),
        types.ToolMessage(
            role="tool", id="c1", content='{"city":"Denver"}', error=False
        ),
    ]
    assert audit_write_execution(task, messages)["status"] == "PASS"
    messages[1].error = True
    assert audit_write_execution(task, messages)["status"] == "FAIL"


def test_overlapping_requirements_do_not_depend_on_action_order():
    class SubsetAction(ExpectedAction):
        def compare_with_tool_call(self, call):
            return self.name == call.name and all(
                call.arguments.get(k) == v for k, v in self.arguments.items()
            )

    task, messages = fixture()
    narrow = task.evaluation_criteria.actions[0]
    broad = SubsetAction(
        action_id="broad",
        name=NAME,
        arguments={"order_id": "o1"},
        requestor="assistant",
    )
    task.evaluation_criteria.actions = [broad, narrow]
    second = deepcopy(messages)
    second[0].tool_calls[0].id = second[1].id = "c2"
    second[0].tool_calls[0].arguments["city"] = "Chicago"
    assert audit_write_execution(task, messages + second)["verified_write_count"] == 2
    task.evaluation_criteria.actions.reverse()
    assert audit_write_execution(task, messages + second)["verified_write_count"] == 2
