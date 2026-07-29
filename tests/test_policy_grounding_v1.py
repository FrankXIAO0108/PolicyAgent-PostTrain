from src.verifiers.policy_grounding_v1 import verify_trajectory
from src.verifiers.schemas import Dimension, MessageEvent, ToolCall, Verdict


def _events(new_item_id: str) -> list[MessageEvent]:
    return [
        MessageEvent(
            index=0,
            role="user",
            content="Actually I want size 9 leather waterproof boots.",
        ),
        MessageEvent(
            index=1,
            role="assistant",
            content=(
                "Confirm exchange order #W1, item 111 to item 222 "
                "(size 9, leather, waterproof), using paypal_1?"
            ),
        ),
        MessageEvent(
            index=2,
            role="user",
            content="Yes, exactly as summarized. Please proceed.",
        ),
        MessageEvent(
            index=3,
            role="assistant",
            tool_calls=[
                ToolCall(
                    id="call-1",
                    name="exchange_delivered_order_items",
                    arguments={
                        "order_id": "#W1",
                        "item_ids": ["111"],
                        "new_item_ids": [new_item_id],
                        "payment_method_id": "paypal_1",
                    },
                )
            ],
        ),
    ]


def test_latest_confirmed_summary_supersedes_initial_request() -> None:
    result = verify_trajectory(_events("222"))
    assert result.dimensions[Dimension.LATEST_INTENT] == Verdict.PASS


def test_unconfirmed_tool_argument_fails_latest_intent() -> None:
    result = verify_trajectory(_events("999"))
    assert result.dimensions[Dimension.LATEST_INTENT] == Verdict.FAIL
    assert any(
        finding.code == "PG_ACTION_ARGUMENT_NOT_CONFIRMED"
        for finding in result.findings
    )
