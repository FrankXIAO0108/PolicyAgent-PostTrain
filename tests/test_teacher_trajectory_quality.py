from __future__ import annotations

import json
import unittest
from pathlib import Path

from src.training.run_tau2_teacher_trajectory_smoke import validate_config
from src.training.teacher_evidence_pack import (
    claim_state_consistency,
    relevant_state,
    state_diff,
)
from src.training.teacher_trajectory_quality import AUTO_PASS, REJECT, REVIEW, audit_simulation


PROJECT = Path(__file__).resolve().parents[1]


def simulation(*, messages, db_match=True, action_match=True, termination="user_stop"):
    return {
        "id": "candidate-1",
        "task_id": "113",
        "trial": 0,
        "seed": 1,
        "termination_reason": termination,
        "messages": messages,
        "reward_info": {
            "reward": 1.0 if db_match and action_match else 0.0,
            "db_check": {"db_match": db_match, "db_reward": float(db_match)},
            "action_checks": [
                {
                    "action": {
                        "name": "cancel_pending_order",
                        "arguments": {"order_id": "#1", "reason": "ordered by mistake"},
                    },
                    "action_match": action_match,
                }
            ],
            "communicate_checks": [],
        },
    }


def duplicate_read_simulation():
    call = {"name": "get_order_details", "arguments": {"order_id": "#1"}}
    return {
        "id": "candidate-read-duplicate",
        "task_id": "57",
        "trial": 0,
        "seed": 1,
        "termination_reason": "user_stop",
        "messages": [
            {"role": "assistant", "content": None, "tool_calls": [{**call, "id": "r1"}]},
            {"role": "tool", "id": "r1", "content": "{}", "error": False},
            {"role": "assistant", "content": None, "tool_calls": [{**call, "id": "r2"}]},
            {"role": "tool", "id": "r2", "content": "{}", "error": False},
        ],
        "reward_info": {
            "reward": 1.0,
            "db_check": {"db_match": True, "db_reward": 1.0},
            "action_checks": [],
            "communicate_checks": [],
        },
    }


def confirmed_messages(*, duplicate=False):
    call = {
        "id": "c1",
        "name": "cancel_pending_order",
        "arguments": {"order_id": "#1", "reason": "ordered by mistake"},
    }
    rows = [
        {"role": "assistant", "content": "Do you confirm I should cancel order #1?", "tool_calls": None},
        {"role": "user", "content": "Yes, I confirm.", "tool_calls": None},
        {"role": "assistant", "content": None, "tool_calls": [call]},
        {"role": "tool", "id": "c1", "content": "ok", "error": False},
    ]
    if duplicate:
        rows.extend(
            [
                {"role": "assistant", "content": None, "tool_calls": [{**call, "id": "c2"}]},
                {"role": "tool", "id": "c2", "content": "ok", "error": False},
            ]
        )
    return rows


class TeacherSmokeConfigTests(unittest.TestCase):
    def test_frozen_smoke_uses_only_three_rl_train_tasks_and_hides_gold(self):
        result = validate_config(
            PROJECT / "configs" / "retail_tau2_teacher_trajectory_smoke_v1.json"
        )
        self.assertEqual(result["task_ids"], ["57", "113", "0"])


class TeacherTrajectoryAuditTests(unittest.TestCase):
    def test_clean_candidate_is_not_automatically_gold(self):
        result = audit_simulation(simulation(messages=confirmed_messages()))
        self.assertEqual(result["automatic_label"], AUTO_PASS)
        self.assertFalse(result["sft_release_allowed"])
        self.assertTrue(result["human_review_required"])

    def test_missing_confirmation_requires_review_because_rule_is_provisional(self):
        messages = confirmed_messages()[2:]
        result = audit_simulation(simulation(messages=messages))
        self.assertEqual(result["automatic_label"], REVIEW)
        self.assertIn(
            "confirmation_not_detected_by_provisional_rule",
            result["review_reasons"],
        )

    def test_duplicate_call_requires_review(self):
        result = audit_simulation(duplicate_read_simulation())
        self.assertEqual(result["automatic_label"], REVIEW)
        self.assertIn("duplicate_exact_tool_call", result["review_reasons"])

    def test_repeated_write_without_new_confirmation_requires_review(self):
        result = audit_simulation(simulation(messages=confirmed_messages(duplicate=True)))
        self.assertEqual(result["automatic_label"], REVIEW)
        self.assertIn(
            "confirmation_not_detected_by_provisional_rule",
            result["review_reasons"],
        )

    def test_narration_on_tool_call_turn_requires_policy_review(self):
        messages = confirmed_messages()
        messages[2]["content"] = "I will process the cancellation now."
        result = audit_simulation(simulation(messages=messages))
        self.assertEqual(result["automatic_label"], REVIEW)
        self.assertIn(
            "assistant_content_and_tool_call_policy_violation",
            result["review_reasons"],
        )
        self.assertEqual(
            result["metrics"]["assistant_content_tool_call_turn_count"], 1
        )

    def test_parallel_tool_calls_require_policy_review(self):
        messages = confirmed_messages()
        messages[2]["content"] = None
        messages[2]["tool_calls"].append(
            {
                "id": "c2",
                "name": "get_order_details",
                "arguments": {"order_id": "#2"},
            }
        )
        result = audit_simulation(simulation(messages=messages))
        self.assertEqual(result["automatic_label"], REVIEW)
        self.assertIn(
            "parallel_tool_calls_policy_violation", result["review_reasons"]
        )
        self.assertEqual(result["metrics"]["parallel_tool_call_turn_count"], 1)

    def test_expected_action_mismatch_requires_review_not_automatic_rejection(self):
        result = audit_simulation(
            simulation(messages=confirmed_messages(), action_match=False)
        )
        self.assertEqual(result["automatic_label"], REVIEW)
        self.assertIn(
            "expected_action_mismatch_requires_semantic_review",
            result["review_reasons"],
        )

    def test_database_failure_is_rejected_even_when_action_matches(self):
        result = audit_simulation(
            simulation(messages=confirmed_messages(), db_match=False)
        )
        self.assertEqual(result["automatic_label"], REJECT)
        self.assertIn("final_database_state_mismatch", result["hard_rejection_reasons"])

    def test_task_57_never_auto_passes_without_semantic_review(self):
        result = audit_simulation(duplicate_read_simulation())
        self.assertEqual(result["automatic_label"], REVIEW)
        self.assertIn(
            "task_57_requires_task_specific_semantic_review",
            result["review_reasons"],
        )


class TeacherEvidencePackTests(unittest.TestCase):
    def test_state_diff_and_relevant_state_only_select_referenced_or_changed_entities(self):
        before = {
            "agent": {
                "orders": {
                    "#W0000001": {"order_id": "#W0000001", "status": "pending"},
                    "#W0000002": {"order_id": "#W0000002", "status": "pending"},
                },
                "users": {},
                "products": {},
            }
        }
        after = json.loads(json.dumps(before))
        after["agent"]["orders"]["#W0000001"]["status"] = "cancelled"
        messages = [
            {
                "role": "assistant",
                "tool_calls": [
                    {
                        "id": "c1",
                        "name": "cancel_pending_order",
                        "arguments": {
                            "order_id": "#W0000001",
                            "reason": "ordered by mistake",
                        },
                    }
                ],
            }
        ]
        selected = relevant_state(before, after, messages)
        self.assertIn("#W0000001", selected["initial"]["orders"])
        self.assertNotIn("#W0000002", selected["initial"]["orders"])
        changes = state_diff(selected["initial"], selected["final"])
        self.assertEqual(changes[0]["before"], "pending")
        self.assertEqual(changes[0]["after"], "cancelled")

    def test_claim_state_contradiction_is_deterministically_detected(self):
        messages = [
            {
                "role": "assistant",
                "content": "Order #W0000001 has been cancelled.",
                "tool_calls": None,
            }
        ]
        final = {
            "agent": {
                "orders": {
                    "#W0000001": {"order_id": "#W0000001", "status": "pending"}
                }
            }
        }
        result = claim_state_consistency(messages, final)
        self.assertEqual(result["verdict"], "FAIL")
        self.assertEqual(result["findings"][0]["verdict"], "CONTRADICTED")

    def test_broad_status_claim_without_order_binding_requires_review(self):
        messages = [
            {
                "role": "assistant",
                "content": "Both orders have been cancelled.",
                "tool_calls": None,
            }
        ]
        result = claim_state_consistency(messages, {"agent": {"orders": {}}})
        self.assertEqual(result["verdict"], "REVIEW")

    def test_exchange_requested_wording_bound_to_order_is_supported(self):
        final = {
            "agent": {
                "orders": {
                    "#W0000001": {
                        "order_id": "#W0000001",
                        "status": "exchange requested",
                    }
                }
            }
        }
        for answer in (
            'Order #W0000001 has been updated to "exchange requested" status.',
            'Order #W0000001 has been updated to **"exchange requested"** status.',
            "Exchange Requested for Order #W0000001.",
        ):
            with self.subTest(answer=answer):
                result = claim_state_consistency(
                    [{"role": "assistant", "content": answer}], final
                )
                self.assertEqual(result["verdict"], "PASS")
                self.assertEqual(result["findings"][0]["verdict"], "SUPPORTED")

    def test_negated_cancellation_is_not_misread_as_success_claim(self):
        messages = [
            {
                "role": "assistant",
                "content": "Order #W0000001 could not be cancelled and remains pending.",
                "tool_calls": None,
            }
        ]
        final = {
            "agent": {
                "orders": {
                    "#W0000001": {"order_id": "#W0000001", "status": "pending"}
                }
            }
        }
        result = claim_state_consistency(messages, final)
        self.assertEqual(result["verdict"], "REVIEW")
        self.assertEqual(result["findings"][0]["verdict"], "UNVERIFIED")

    def test_order_bound_amount_is_checked_against_tool_observation(self):
        messages = [
            {
                "role": "tool",
                "content": json.dumps(
                    {
                        "order_id": "#W0000001",
                        "status": "pending",
                        "payment_history": [
                            {"transaction_type": "payment", "amount": 829.43}
                        ],
                    }
                ),
                "error": False,
            },
            {
                "role": "assistant",
                "content": "The total for order #W0000001, which was $829.43.",
            },
        ]

        result = claim_state_consistency(messages, {"agent": {"orders": {}}})

        self.assertEqual(result["verdict"], "PASS")
        self.assertEqual(result["findings"][0]["verdict"], "SUPPORTED")

    def test_wrong_order_bound_amount_is_contradicted(self):
        messages = [
            {
                "role": "tool",
                "content": json.dumps(
                    {
                        "order_id": "#W0000001",
                        "payment_history": [
                            {"transaction_type": "payment", "amount": 829.43}
                        ],
                    }
                ),
                "error": False,
            },
            {
                "role": "assistant",
                "content": "Order #W0000001 had a total of $919.67.",
            },
        ]

        result = claim_state_consistency(messages, {"agent": {"orders": {}}})

        self.assertEqual(result["verdict"], "FAIL")
        self.assertEqual(result["findings"][0]["verdict"], "CONTRADICTED")

    def test_asserted_most_recent_selection_without_dates_requires_review(self):
        messages = [
            {
                "role": "assistant",
                "content": (
                    "I believe the most recent order would be Order #W0000001 "
                    "with a total of $829.43."
                ),
            }
        ]

        result = claim_state_consistency(messages, {"agent": {"orders": {}}})

        self.assertEqual(result["verdict"], "REVIEW")
        self.assertIn(
            "COMPARATIVE_SELECTION_OUTSIDE_PROGRAMMATIC_CHECKER_SCOPE",
            {finding["reason_code"] for finding in result["findings"]},
        )

    def test_markdown_and_explanation_do_not_hide_most_recent_claim(self):
        messages = [
            {
                "role": "assistant",
                "content": (
                    'You asked about your "most recent" order. I believe that '
                    "would be **Order #W0000001** with a total of **$829.43**."
                ),
            }
        ]

        result = claim_state_consistency(messages, {"agent": {"orders": {}}})

        self.assertEqual(result["verdict"], "REVIEW")
        self.assertIn(
            "COMPARATIVE_SELECTION_OUTSIDE_PROGRAMMATIC_CHECKER_SCOPE",
            {finding["reason_code"] for finding in result["findings"]},
        )

    def test_explicit_uncertainty_is_not_misread_as_selection_claim(self):
        messages = [
            {
                "role": "assistant",
                "content": (
                    "The order data has no dates, so I cannot determine the most "
                    "recent order."
                ),
            }
        ]

        result = claim_state_consistency(messages, {"agent": {"orders": {}}})

        self.assertEqual(result["verdict"], "NOT_APPLICABLE")


if __name__ == "__main__":
    unittest.main()
