from __future__ import annotations

import unittest

from src.evaluation.process_reward_audit import (
    _auth_lookup_penalty_sensitivity,
    _claim_verdict_rank,
    _compose_v1_proxy,
    error_recovery_diagnostics,
    stopping_condition_diagnostics,
)


class ProcessRewardAuditTests(unittest.TestCase):
    def test_auth_lookup_penalty_sensitivity_does_not_change_observed_reward(self) -> None:
        proxy = {
            "score": 0.9,
            "raw_reward": 1.0,
            "penalties": {
                "tool_error": 0.1,
                "repeated_call": 0.0,
                "unexpected_write": 0.0,
                "unfinished_interaction": 0.0,
            },
        }
        recovery = {
            "tool_error_count": 2,
            "error_category_counts": {"AUTH_LOOKUP_MISS": 2},
        }
        config = {"tool_error_penalty_each": 0.05, "tool_error_penalty_cap": 0.2}

        result = _auth_lookup_penalty_sensitivity(
            proxy=proxy,
            recovery=recovery,
            reward_config=config,
        )

        self.assertEqual(result["observed_score"], 0.9)
        self.assertEqual(result["counterfactual_score"], 1.0)
        self.assertFalse(result["reward_change_proposed"])

    def test_normal_user_stop_passes_stopping_diagnostic(self) -> None:
        messages = [
            {"role": "assistant", "content": "Anything else?"},
            {"role": "user", "content": "No thanks. ###STOP###"},
        ]

        result = stopping_condition_diagnostics(
            messages, termination_reason="user_stop"
        )

        self.assertEqual(result["verdict"], "PASS")
        self.assertEqual(result["findings"], [])

    def test_activity_after_stop_fails_stopping_diagnostic(self) -> None:
        messages = [
            {"role": "user", "content": "Done. ###STOP###"},
            {
                "role": "assistant",
                "tool_calls": [
                    {"id": "c1", "name": "get_order_details", "arguments": {}}
                ],
            },
        ]

        result = stopping_condition_diagnostics(
            messages, termination_reason="user_stop"
        )

        self.assertEqual(result["verdict"], "FAIL")
        self.assertEqual(
            result["findings"][0]["reason_code"], "ACTIVITY_AFTER_USER_STOP"
        )

    def test_successful_transfer_requires_notice_and_terminal_marker(self) -> None:
        messages = [
            {
                "role": "assistant",
                "tool_calls": [
                    {
                        "id": "t1",
                        "name": "transfer_to_human_agents",
                        "arguments": {"summary": "Escalation requested"},
                    }
                ],
            },
            {
                "id": "t1",
                "role": "tool",
                "content": "Transfer successful",
                "error": False,
            },
            {"role": "assistant", "content": "Someone will contact you later."},
            {"role": "user", "content": "Okay."},
        ]

        result = stopping_condition_diagnostics(
            messages, termination_reason="user_stop"
        )

        codes = {finding["reason_code"] for finding in result["findings"]}
        self.assertEqual(result["verdict"], "FAIL")
        self.assertIn("MISSING_REQUIRED_TRANSFER_NOTICE", codes)
        self.assertIn("MISSING_TRANSFER_TERMINATION_MARKER", codes)

    def test_valid_transfer_sequence_passes_stopping_diagnostic(self) -> None:
        messages = [
            {
                "role": "assistant",
                "tool_calls": [
                    {
                        "id": "t1",
                        "name": "transfer_to_human_agents",
                        "arguments": {"summary": "Escalation requested"},
                    }
                ],
            },
            {
                "id": "t1",
                "role": "tool",
                "content": "Transfer successful",
                "error": False,
            },
            {
                "role": "assistant",
                "content": "YOU ARE BEING TRANSFERRED TO A HUMAN AGENT. PLEASE HOLD ON.",
            },
            {"role": "user", "content": "###TRANSFER###"},
        ]

        result = stopping_condition_diagnostics(
            messages, termination_reason="user_stop"
        )

        self.assertEqual(result["verdict"], "PASS")

    def test_tool_call_after_successful_transfer_fails_stopping_diagnostic(self) -> None:
        messages = [
            {
                "role": "assistant",
                "tool_calls": [
                    {
                        "id": "t1",
                        "name": "transfer_to_human_agents",
                        "arguments": {"summary": "Escalation requested"},
                    }
                ],
            },
            {
                "id": "t1",
                "role": "tool",
                "content": "Transfer successful",
                "error": False,
            },
            {
                "role": "assistant",
                "tool_calls": [
                    {"id": "c2", "name": "get_order_details", "arguments": {}}
                ],
            },
            {"id": "c2", "role": "tool", "content": "{}", "error": False},
            {
                "role": "assistant",
                "content": "YOU ARE BEING TRANSFERRED TO A HUMAN AGENT. PLEASE HOLD ON.",
            },
            {"role": "user", "content": "###TRANSFER###"},
        ]

        result = stopping_condition_diagnostics(
            messages, termination_reason="user_stop"
        )

        codes = {finding["reason_code"] for finding in result["findings"]}
        self.assertEqual(result["verdict"], "FAIL")
        self.assertIn("TOOL_CALL_AFTER_SUCCESSFUL_TRANSFER", codes)

    def test_transfer_without_tool_result_is_not_assumed_successful(self) -> None:
        messages = [
            {
                "role": "assistant",
                "tool_calls": [
                    {
                        "id": "t1",
                        "name": "transfer_to_human_agents",
                        "arguments": {"summary": "Escalation requested"},
                    }
                ],
            },
            {"role": "user", "content": "###STOP###"},
        ]

        result = stopping_condition_diagnostics(
            messages, termination_reason="user_stop"
        )

        self.assertEqual(result["verdict"], "REVIEW")
        self.assertEqual(result["successful_transfer_count"], 0)
        self.assertEqual(
            result["findings"][0]["reason_code"], "TRANSFER_RESULT_MISSING"
        )

    def test_abnormal_termination_without_marker_fails_stopping_diagnostic(self) -> None:
        result = stopping_condition_diagnostics(
            [{"role": "assistant", "content": "Still working"}],
            termination_reason="too_many_errors",
        )

        self.assertEqual(result["verdict"], "FAIL")
        self.assertEqual(
            result["findings"][0]["reason_code"], "NO_EXPLICIT_TERMINAL_MARKER"
        )

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
        self.assertEqual(result["error_category_counts"], {"WRITE_TOOL_ERROR": 2})

    def test_auth_lookup_miss_is_separate_from_write_error(self) -> None:
        messages = [
            {
                "role": "assistant",
                "tool_calls": [
                    {
                        "id": "c1",
                        "name": "find_user_id_by_name_zip",
                        "arguments": {
                            "first_name": "Noah",
                            "last_name": "Ito",
                            "zip": "98178",
                        },
                    }
                ],
            },
            {
                "id": "c1",
                "role": "tool",
                "content": "Error: User not found",
                "error": True,
            },
            {
                "role": "assistant",
                "tool_calls": [
                    {
                        "id": "c2",
                        "name": "return_delivered_order_items",
                        "arguments": {"order_id": "#1"},
                    }
                ],
            },
            {
                "id": "c2",
                "role": "tool",
                "content": "Error: Non-delivered order cannot be returned",
                "error": True,
            },
        ]

        result = error_recovery_diagnostics(messages, outcome_success=False)

        self.assertEqual(
            result["error_category_counts"],
            {"AUTH_LOOKUP_MISS": 1, "WRITE_TOOL_ERROR": 1},
        )

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
