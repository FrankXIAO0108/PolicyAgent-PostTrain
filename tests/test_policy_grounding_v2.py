from __future__ import annotations

import unittest
from pathlib import Path

from src.verifiers.policy_grounding_v2 import verify_artifacts, verify_trajectory
from src.verifiers.schemas import MessageEvent, Verdict
from src.verifiers.trajectory_loader import load_task_artifacts


BASELINE = Path(
    r"D:\PolicyAgent-PostTrain\experiments"
    r"\20260722_110504_retail_baseline20_trial1_deepseek"
)


class PolicyGroundingV2UnitTests(unittest.TestCase):
    def test_clean_read_only_trajectory_still_passes(self) -> None:
        result = verify_trajectory(
            [
                MessageEvent(index=0, role="user", content="What is my order status?"),
                MessageEvent(index=1, role="assistant", content="It is pending."),
            ],
            benchmark_verdict=Verdict.PASS,
        )

        self.assertEqual(result.verdict, Verdict.PASS)
        self.assertEqual(result.metrics["verifier_version"], "2.2")
        self.assertFalse(result.metrics["uses_reference_actions"])


@unittest.skipUnless(BASELINE.exists(), "frozen baseline artifacts absent")
class PolicyGroundingV2FrozenTests(unittest.TestCase):
    def test_task_95_detects_actionable_variant_without_reference_gold(self) -> None:
        bundle = load_task_artifacts(BASELINE / "task_95")
        result = verify_artifacts(bundle)

        self.assertEqual(result.verdict, Verdict.FAIL)
        self.assertFalse(result.metrics["uses_reference_actions"])
        self.assertTrue(
            any(
                finding.code
                == "PG_GUARD_GOAL_TRANSFER_WITH_ACTIONABLE_VARIANT"
                for finding in result.findings
            )
        )

    def test_task_37_downgrades_contextual_confirmation_to_review(self) -> None:
        bundle = load_task_artifacts(BASELINE / "task_37")
        result = verify_artifacts(bundle)

        self.assertEqual(result.verdict, Verdict.REVIEW)
        self.assertFalse(
            any(
                finding.code == "PG_ACTION_ARGUMENT_NOT_CONFIRMED"
                for finding in result.findings
            )
        )

    def test_task_1_carries_forward_confirmed_exchange_details(self) -> None:
        bundle = load_task_artifacts(BASELINE / "task_1")
        result = verify_artifacts(bundle)

        self.assertEqual(result.verdict, Verdict.REVIEW)
        self.assertFalse(
            any(
                finding.code == "PG_ACTION_ARGUMENT_NOT_CONFIRMED"
                for finding in result.findings
            )
        )

    def test_task_72_detects_repeated_order_modification(self) -> None:
        bundle = load_task_artifacts(BASELINE / "task_72")
        result = verify_artifacts(bundle)

        self.assertEqual(result.verdict, Verdict.FAIL)
        self.assertTrue(
            any(
                finding.code == "PG_GUARD_POLICY_ONE_SHOT_ORDER_MUTATION"
                for finding in result.findings
            )
        )


if __name__ == "__main__":
    unittest.main()
