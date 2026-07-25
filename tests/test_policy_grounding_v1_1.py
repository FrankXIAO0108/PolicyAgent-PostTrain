from __future__ import annotations

import unittest

from src.verifiers.policy_grounding_v0 import verify_trajectory
from src.verifiers.schemas import (
    Dimension,
    MessageEvent,
    ToolCall,
    Verdict,
)


def _call(call_id: str, name: str) -> ToolCall:
    return ToolCall(id=call_id, name=name, arguments={})


class SeverityAggregationTests(unittest.TestCase):
    def test_message_with_read_tool_is_review_not_fail(self) -> None:
        events = [
            MessageEvent(
                index=0,
                role="assistant",
                content="Let me check that.",
                tool_calls=[_call("1", "get_order_details")],
            )
        ]

        result = verify_trajectory(events)

        self.assertEqual(result.verdict, Verdict.REVIEW)
        self.assertEqual(
            result.dimensions[Dimension.POLICY_COMPLIANCE],
            Verdict.REVIEW,
        )

    def test_multiple_read_tools_are_review_not_fail(self) -> None:
        events = [
            MessageEvent(
                index=0,
                role="assistant",
                tool_calls=[
                    _call("1", "get_order_details"),
                    _call("2", "get_order_details"),
                ],
            )
        ]

        result = verify_trajectory(events)

        self.assertEqual(result.verdict, Verdict.REVIEW)
        self.assertEqual(result.metrics["major_finding_count"], 0)
        self.assertEqual(result.metrics["minor_finding_count"], 1)

    def test_multiple_write_tools_are_fail(self) -> None:
        events = [
            MessageEvent(index=0, role="user", content="Yes, proceed."),
            MessageEvent(
                index=1,
                role="assistant",
                tool_calls=[
                    _call("1", "exchange_delivered_order_items"),
                    _call("2", "exchange_delivered_order_items"),
                ],
            ),
        ]

        result = verify_trajectory(events)

        self.assertEqual(result.verdict, Verdict.FAIL)
        self.assertEqual(
            result.dimensions[Dimension.POLICY_COMPLIANCE],
            Verdict.FAIL,
        )
        self.assertGreaterEqual(result.metrics["major_finding_count"], 1)

    def test_clean_read_only_trajectory_passes(self) -> None:
        events = [
            MessageEvent(index=0, role="user", content="What is my order status?"),
            MessageEvent(index=1, role="assistant", content="It is delivered."),
        ]

        result = verify_trajectory(events)

        self.assertEqual(result.verdict, Verdict.PASS)


if __name__ == "__main__":
    unittest.main()
