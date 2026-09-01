from __future__ import annotations

import unittest

from src.evaluation.reward_authority_audit import (
    audit_rollout,
    build_reward_authority_audit,
)


def _call(name: str, arguments: dict | None = None) -> dict:
    return {
        "role": "assistant",
        "content": None,
        "tool_calls": [
            {
                "id": f"call-{name}",
                "name": name,
                "arguments": arguments or {},
            }
        ],
    }


def _result(content: str = "user-1", *, error: bool = False) -> dict:
    return {"role": "tool", "content": content, "error": error}


def _rollout(*, reward: float, messages: list[dict], task_id: str = "7") -> dict:
    return {
        "task_id": task_id,
        "rollout_stage": "IDENTITY_AUTHENTICATION",
        "messages": messages,
        "reward": {
            "reward": reward,
            "stage_complete": reward > 0,
            "action_progress": {
                "matched_count": int(reward > 0),
                "recall": float(reward > 0),
            },
            "diagnostic_policy_findings": [],
        },
    }


class RewardAuthorityAuditTests(unittest.TestCase):
    def test_detects_zero_reward_after_observable_success(self) -> None:
        row = _rollout(
            reward=0.0,
            messages=[
                _call(
                    "find_user_id_by_name_zip",
                    {"first_name": "mei", "last_name": "kovacs", "zip": "28236"},
                ),
                _result("mei_kovacs_8020"),
            ],
        )

        audited = audit_rollout(
            row, rollout_index=1, expected_user_id="mei_kovacs_8020"
        )

        self.assertTrue(audited["observable_stage_core_success"])
        self.assertEqual(
            audited["authority_verdict"], "REVIEW_REQUIRED"
        )
        self.assertEqual(
            audited["conflicts"][0]["code"],
            "ZERO_REWARD_WITH_SUCCESSFUL_AUTH_RESULT_REQUIRES_GROUNDING_REVIEW",
        )

    def test_post_authentication_tool_is_supported_zero_not_false_negative(self) -> None:
        row = _rollout(
            reward=0.0,
            messages=[
                _call("find_user_id_by_email", {"email": "a@example.com"}),
                _result(),
                _call("get_user_details", {"user_id": "user-1"}),
                _result("{}"),
            ],
        )

        audited = audit_rollout(row, rollout_index=1, expected_user_id="user-1")

        self.assertFalse(audited["observable_stage_core_success"])
        self.assertEqual(audited["stage_boundary"]["verdict"], "FAIL")
        self.assertEqual(audited["authority_verdict"], "OBSERVABLE_CORE_ALIGNED")
        self.assertEqual(audited["conflicts"], [])

    def test_positive_reward_with_boundary_violation_is_contradicted(self) -> None:
        row = _rollout(
            reward=1.0,
            messages=[
                _call("find_user_id_by_email", {"email": "a@example.com"}),
                _result(),
                _call("get_user_details", {"user_id": "user-1"}),
                _result("{}"),
            ],
        )

        audited = audit_rollout(row, rollout_index=1, expected_user_id="user-1")

        self.assertEqual(
            audited["authority_verdict"], "OBSERVABLE_CORE_CONTRADICTED"
        )
        self.assertEqual(
            audited["conflicts"][0]["code"],
            "POSITIVE_REWARD_CONTRADICTS_OBSERVABLE_STAGE_CORE",
        )

    def test_summary_keeps_final_response_observability_explicit(self) -> None:
        rows = [
            _rollout(
                reward=1.0,
                messages=[
                    _call("find_user_id_by_email", {"email": "a@example.com"}),
                    _result(),
                ],
                task_id="1",
            ),
            _rollout(
                reward=0.0,
                messages=[
                    _call("find_user_id_by_name_zip"),
                    _result("user-2"),
                ],
                task_id="2",
            ),
        ]

        report = build_reward_authority_audit(
            rows,
            source={"sha256": "RAW"},
            expected_user_ids={"1": "user-1", "2": "user-2"},
        )

        self.assertEqual(report["summary"]["rollout_count"], 2)
        self.assertEqual(
            report["summary"]["review_required_count"], 1
        )
        self.assertEqual(report["summary"]["final_response_unknown_count"], 2)

    def test_no_tool_opening_is_not_misclassified_as_final_response(self) -> None:
        row = _rollout(
            reward=0.0,
            messages=[
                {"role": "assistant", "content": "How can I help?"},
                {"role": "user", "content": "I need help."},
            ],
        )

        audited = audit_rollout(row, rollout_index=1, expected_user_id="user-1")

        self.assertEqual(
            audited["final_response_observation"]["verdict"], "UNKNOWN"
        )

    def test_missing_expected_user_id_is_not_claimed_as_success(self) -> None:
        row = _rollout(
            reward=0.0,
            messages=[
                _call("find_user_id_by_email", {"email": "a@example.com"}),
                _result("some-user"),
            ],
        )

        audited = audit_rollout(row, rollout_index=1)

        self.assertIsNone(audited["observable_stage_core_success"])
        self.assertEqual(audited["authority_verdict"], "NOT_EVALUATED")


if __name__ == "__main__":
    unittest.main()
