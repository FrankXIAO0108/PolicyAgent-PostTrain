from __future__ import annotations

import unittest

from src.evaluation.full_task_reward_shadow import build_shadow_report


def _trajectory(*, success: bool, score: float, errors: int = 0, missing: int = 0):
    return {
        "artifact": {"path": "returned_results.json", "sha256": "ABC"},
        "benchmark": {"success": success, "reward": float(success)},
        "v1_reward_proxy": {"score": score},
        "error_recovery": {"tool_error_count": errors},
        "confirmation_diagnostics": {
            "missing_confirmation_count": missing,
            "checks": [],
        },
        "claim_state_consistency": {"verdict": "REVIEW"},
    }


class FullTaskRewardShadowTests(unittest.TestCase):
    def test_flags_success_that_is_not_training_gold(self) -> None:
        process = {
            "pairs": [
                {
                    "task_id": "100",
                    "cohort": "flip",
                    "run_a": _trajectory(
                        success=True, score=0.9, errors=1, missing=1
                    ),
                    "run_b": _trajectory(success=False, score=0.1),
                }
            ],
            "gates": {"ready_to_use_v1_reward_for_grpo": False},
        }
        decisions = {
            "decisions": [
                {
                    "review_id": "100:run_a:call-1",
                    "final_label": "ACCEPTABLE",
                    "severity": "HIGH",
                    "training_disposition": "MIXED_HOLDOUT",
                }
            ]
        }

        report = build_shadow_report(process, owner_decisions=decisions)
        row = report["trajectories"][0]

        self.assertIn("BENCHMARK_SUCCESS_WITH_TOOL_ERROR", row["conflicts"])
        self.assertIn("BENCHMARK_SUCCESS_WITH_CONFIRMATION_GAP", row["conflicts"])
        self.assertIn(
            "BENCHMARK_SUCCESS_NOT_OWNER_APPROVED_AS_RAW_GOLD", row["conflicts"]
        )
        self.assertEqual(report["decision"]["status"], "DO_NOT_PROMOTE_TO_ONLINE_REWARD")
        self.assertFalse(report["scope"]["introduces_scalar_v2_reward"])
        self.assertEqual(
            report["component_coverage"]["owner_adjudication"][
                "reviewed_trajectory_count"
            ],
            1,
        )

    def test_flags_high_proxy_failure_and_flip_tie(self) -> None:
        process = {
            "pairs": [
                {
                    "task_id": "67",
                    "cohort": "flip",
                    "run_a": _trajectory(success=False, score=0.9),
                    "run_b": _trajectory(success=True, score=0.9),
                }
            ],
            "gates": {"ready_to_use_v1_reward_for_grpo": False},
        }

        report = build_shadow_report(process)

        self.assertIn(
            "HIGH_V1_PROCESS_PROXY_ON_BENCHMARK_FAILURE",
            report["trajectories"][0]["conflicts"],
        )
        self.assertEqual(report["summary"]["flip_score_tie_task_ids"], ["67"])

    def test_hidden_reference_action_signal_is_not_classed_as_production_online(self) -> None:
        process = {
            "pairs": [
                {
                    "task_id": "1",
                    "cohort": "common_success",
                    "run_a": _trajectory(success=True, score=1.0),
                    "run_b": _trajectory(success=True, score=1.0),
                }
            ],
            "gates": {"ready_to_use_v1_reward_for_grpo": False},
        }

        report = build_shadow_report(process)

        self.assertEqual(
            report["signal_authority"]["required_action_progress"]["authority"],
            "BENCHMARK_TRAINING_ONLY",
        )


if __name__ == "__main__":
    unittest.main()
