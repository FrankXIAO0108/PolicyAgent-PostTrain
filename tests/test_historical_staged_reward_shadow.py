from __future__ import annotations

import unittest

from src.evaluation.historical_staged_reward_shadow import (
    _normalized_arguments,
    _tool_trace,
)


class HistoricalStagedRewardShadowTests(unittest.TestCase):
    def test_tool_calls_are_bound_to_tool_results(self) -> None:
        messages = [
            {
                "role": "assistant",
                "tool_calls": [
                    {"id": "c1", "name": "get_order_details", "arguments": {"order_id": "o1"}}
                ],
            },
            {"role": "tool", "id": "c1", "content": "ok", "error": False},
        ]
        trace = _tool_trace(messages)
        self.assertEqual(len(trace), 1)
        self.assertEqual(trace[0]["result"], {"content": "ok", "error": False})

    def test_missing_tool_result_is_an_error(self) -> None:
        trace = _tool_trace(
            [
                {
                    "role": "assistant",
                    "tool_calls": [
                        {"id": "missing", "name": "get_order_details", "arguments": {}}
                    ],
                }
            ]
        )
        self.assertTrue(trace[0]["result"]["error"])

    def test_item_pairs_preserve_old_to_new_binding(self) -> None:
        first = _normalized_arguments(
            {"item_ids": ["a", "b"], "new_item_ids": ["x", "y"]}
        )
        reordered = _normalized_arguments(
            {"item_ids": ["b", "a"], "new_item_ids": ["y", "x"]}
        )
        wrong = _normalized_arguments(
            {"item_ids": ["a", "b"], "new_item_ids": ["y", "x"]}
        )
        self.assertEqual(first, reordered)
        self.assertNotEqual(first, wrong)


if __name__ == "__main__":
    unittest.main()
