from __future__ import annotations

import unittest

from src.evaluation.process_reward_audit import (
    _claim_verdict_rank,
    _compose_v1_proxy,
    error_recovery_diagnostics,
)


class ProcessRewardAuditTests(unittest.TestCase):
    def test_claim_diagnostic_rank_does_not_treat_not_applicable_as_pass(self) -> None:
        self.assertGreater(_claim_verdict_rank("PASS"), _claim_verdict_rank("REVIEW"))
        self.assertGreater(_claim_verdict_rank("REVIEW"), _claim_verdict_rank("FAIL"))
        self.assertIsNone(_claim_verdict_rank("NOT_APPLICABLE"))

    def test_repeated_failed_exact_calls_are_separate_from_tool_error_count(self) -> None:
        messages = [
            {
                "role": "assistant",
                "tool_calls": [
                    {
                        "id": "c1",
                        "name": "exchange_delivered_order_items",
                        "arguments": {"order_id": "#1", "item_ids": ["bad"]},
                    }
                ],
            },
            {"id": "c1", "role": "tool", "content": "not found", "error": True},
            {
                "role": "assistant",
                "tool_calls": [
                    {
                        "id": "c2",
                        "name": "exchange_delivered_order_items",
                        "arguments": {"order_id": "#1", "item_ids": ["bad"]},
                    }
                ],
            },
            {"id": "c2", "role": "tool", "content": "not found", "error": True},
        ]

        result = error_recovery_diagnostics(messages, outcome_success=False)

        self.assertEqual(result["tool_error_count"], 2)
        self.assertEqual(result["repeated_failed_exact_call_count"], 1)
        self.assertEqual(result["error_events_followed_by_changed_call"], 0)

    def test_changed_call_after_error_is_not_treated_as_exact_repeat(self) -> None:
        messages = [
            {
                "role": "assistant",
                "tool_calls": [
                    {
                        "id": "c1",
                        "name": "exchange_delivered_order_items",
                        "arguments": {"order_id": "#1", "item_ids": ["bad"]},
                    }
                ],
            },
            {"id": "c1", "role": "tool", "content": "not found", "error": True},
            {
                "role": "assistant",
                "tool_calls": [
                    {
                        "id": "c2",
                        "name": "get_order_details",
                        "arguments": {"order_id": "#1"},
                    }
                ],
            },
            {"id": "c2", "role": "tool", "content": "{}", "error": False},
        ]

        result = error_recovery_diagnostics(messages, outcome_success=True)

        self.assertEqual(result["tool_error_count"], 1)
        self.assertEqual(result["repeated_failed_exact_call_count"], 0)
        self.assertEqual(result["error_events_followed_by_changed_call"], 1)
        self.assertTrue(result["successful_despite_tool_error"])

    def test_v1_proxy_does_not_reward_short_failure_for_efficiency(self) -> None:
        config = {
            "environment_state_weight": 0.7,
            "required_action_weight": 0.2,
            "communication_weight": 0.1,
            "tool_error_penalty_each": 0.05,
            "tool_error_penalty_cap": 0.2,
            "repeated_call_penalty_each": 0.03,
            "repeated_call_penalty_cap": 0.15,
            "unexpected_write_penalty_each": 0.05,
            "unexpected_write_penalty_cap": 0.2,
            "unfinished_interaction_penalty": 0.1,
        }
        progress = {
            "recall": 0.0,
            "duplicate_excess_count": 0,
            "unexpected_write_count": 0,
        }

        result = _compose_v1_proxy(
            environment_state_raw=0.0,
            action_progress=progress,
            communication_recall=None,
            tool_errors=0,
            user_stopped=True,
            reward_config=config,
        )

        self.assertEqual(result["score"], 0.0)

    def test_success_with_one_recovered_error_keeps_positive_score(self) -> None:
        config = {
            "environment_state_weight": 0.7,
            "required_action_weight": 0.2,
            "communication_weight": 0.1,
            "tool_error_penalty_each": 0.05,
            "tool_error_penalty_cap": 0.2,
            "repeated_call_penalty_each": 0.03,
            "repeated_call_penalty_cap": 0.15,
            "unexpected_write_penalty_each": 0.05,
            "unexpected_write_penalty_cap": 0.2,
            "unfinished_interaction_penalty": 0.1,
        }
        progress = {
            "recall": 1.0,
            "duplicate_excess_count": 0,
            "unexpected_write_count": 0,
        }

        result = _compose_v1_proxy(
            environment_state_raw=1.0,
            action_progress=progress,
            communication_recall=None,
            tool_errors=1,
            user_stopped=True,
            reward_config=config,
        )

        self.assertAlmostEqual(result["score"], 0.95)


if __name__ == "__main__":
    unittest.main()
