from __future__ import annotations

import json
import unittest

from src.verifiers.policy_grounding_v1 import verify_trajectory
from src.verifiers.schemas import Dimension, MessageEvent, ToolCall, Verdict


def _return_events(summary: str, item_id: str) -> list[MessageEvent]:
    return [
        MessageEvent(
            index=0,
            role="assistant",
            tool_calls=[
                ToolCall(
                    id="read-1",
                    name="get_item_details",
                    arguments={"item_id": item_id},
                )
            ],
        ),
        MessageEvent(
            index=1,
            role="tool",
            tool_call_id="read-1",
            content=json.dumps(
                {
                    "item": {
                        "item_id": item_id,
                        "name": "Smart Thermostat",
                        "options": {
                            "compatibility": "Google Home",
                            "color": "white",
                        },
                    }
                }
            ),
        ),
        MessageEvent(index=2, role="assistant", content=summary),
        MessageEvent(index=3, role="user", content="Yes, please proceed."),
        MessageEvent(
            index=4,
            role="assistant",
            tool_calls=[
                ToolCall(
                    id="write-1",
                    name="return_delivered_order_items",
                    arguments={
                        "order_id": "#W1",
                        "item_ids": [item_id],
                        "payment_method_id": "paypal_1",
                    },
                )
            ],
        ),
    ]


class EntityAliasGroundingTests(unittest.TestCase):
    def test_visible_item_name_can_ground_internal_item_id(self) -> None:
        result = verify_trajectory(
            _return_events(
                "Confirm returning the Smart Thermostat from order #W1 "
                "with the refund to paypal_1?",
                "4293355847",
            )
        )
        self.assertEqual(
            result.dimensions[Dimension.LATEST_INTENT], Verdict.PASS
        )

    def test_unrelated_visible_name_does_not_ground_item_id(self) -> None:
        result = verify_trajectory(
            _return_events(
                "Confirm returning the skateboard from order #W1 "
                "with the refund to paypal_1?",
                "4293355847",
            )
        )
        self.assertEqual(
            result.dimensions[Dimension.LATEST_INTENT], Verdict.FAIL
        )

    def test_multiple_write_calls_remain_policy_failure(self) -> None:
        events = [
            MessageEvent(
                index=0,
                role="assistant",
                content="Confirm both cancellations for #W1 and #W2?",
            ),
            MessageEvent(index=1, role="user", content="Yes, proceed."),
            MessageEvent(
                index=2,
                role="assistant",
                tool_calls=[
                    ToolCall(
                        id="write-1",
                        name="cancel_pending_order",
                        arguments={"order_id": "#W1"},
                    ),
                    ToolCall(
                        id="write-2",
                        name="cancel_pending_order",
                        arguments={"order_id": "#W2"},
                    ),
                ],
            ),
        ]
        result = verify_trajectory(events)
        self.assertEqual(
            result.dimensions[Dimension.POLICY_COMPLIANCE], Verdict.FAIL
        )


if __name__ == "__main__":
    unittest.main()
