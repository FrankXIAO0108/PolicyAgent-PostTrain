from __future__ import annotations

import unittest

from src.evaluation.grpo_training_audit import build_training_audit


def _rollout(task_id: str, reward: float) -> dict:
    return {"task_id": task_id, "reward": {"reward": reward}}


class GrpoTrainingAuditTests(unittest.TestCase):
    def _build(self, raw_rollouts: list[dict]) -> dict:
        return build_training_audit(
            config={
                "data": {"task_ids": ["1", "2", "3", "4"]},
                "grpo": {
                    "max_steps": 4,
                    "num_generations": 2,
                    "max_completion_length": 128,
                    "per_device_train_batch_size": 2,
                    "gradient_accumulation_steps": 1,
                    "learning_rate": 1e-6,
                    "temperature": 0.8,
                    "beta": 0.0,
                    "loss_type": "dr_grpo",
                },
            },
            manifest={
                "status": "COMPLETED",
                "git": {"commit": "abc"},
                "bindings": {"starting_model_sha256": "MODEL"},
                "artifacts": {"raw_rollouts": {"sha256": "RAW"}},
            },
            log_history=[
                {
                    "step": 1,
                    "reward_std": 0.0,
                    "grad_norm": 0.0,
                    "entropy": 0.2,
                    "completions/mean_length": 20,
                    "completions/clipped_ratio": 0.0,
                    "step_time": 2.0,
                },
                {
                    "step": 2,
                    "reward_std": 0.7,
                    "grad_norm": 0.2,
                    "entropy": 0.18,
                    "completions/mean_length": 30,
                    "completions/clipped_ratio": 0.5,
                    "step_time": 4.0,
                },
                {
                    "step": 3,
                    "reward_std": 0.7,
                    "grad_norm": 0.3,
                    "entropy": 0.16,
                    "completions/mean_length": 40,
                    "completions/clipped_ratio": 0.0,
                    "step_time": 6.0,
                },
                {
                    "step": 4,
                    "reward_std": 0.0,
                    "grad_norm": 0.0,
                    "entropy": 0.15,
                    "completions/mean_length": 50,
                    "completions/clipped_ratio": 0.5,
                    "step_time": 8.0,
                },
            ],
            raw_rollouts=raw_rollouts,
            train_metrics={"train_runtime": 20.0, "train_loss": -0.1},
            source={"raw_rollouts_sha256": "RAW"},
        )

    def test_summarizes_group_signal_and_generation_metrics(self) -> None:
        report = self._build(
            [
                _rollout("1", 0),
                _rollout("1", 0),
                _rollout("2", 0),
                _rollout("2", 1),
                _rollout("3", 1),
                _rollout("3", 0),
                _rollout("4", 1),
                _rollout("4", 1),
            ]
        )

        self.assertEqual(
            report["training_signal"]["group_counts"],
            {
                "all_zero": 1,
                "mixed": 2,
                "all_positive": 1,
                "uniform_nonzero": 0,
            },
        )
        self.assertEqual(report["training_signal"]["effective_task_ids"], ["2", "3"])
        self.assertEqual(report["training_signal"]["zero_reward_std_steps"], 2)
        self.assertEqual(report["training_signal"]["effective_steps"], [2, 3])
        self.assertEqual(report["training_signal"]["nonzero_grad_steps"], 2)
        self.assertEqual(report["generation"]["mean_completion_length"], 35.0)
        self.assertEqual(report["generation"]["steps_with_clipping"], 2)
        self.assertEqual(report["generation"]["mean_clipped_ratio"], 0.25)
        self.assertFalse(
            report["optimization"]["rollout_vs_update_time_breakdown_available"]
        )
        self.assertTrue(report["claim_limits"]["parameter_update_observed"])
        self.assertFalse(report["claim_limits"]["behavior_improvement_assessed"])

    def test_rejects_rollout_count_not_divisible_by_group_size(self) -> None:
        with self.assertRaisesRegex(ValueError, "not divisible"):
            self._build([_rollout("1", 0), _rollout("1", 1), _rollout("2", 0)])

    def test_different_positive_rewards_are_a_mixed_group(self) -> None:
        report = self._build(
            [
                _rollout("1", 0.2),
                _rollout("1", 0.8),
                _rollout("2", 0),
                _rollout("2", 0),
                _rollout("3", 0),
                _rollout("3", 0),
                _rollout("4", 0),
                _rollout("4", 0),
            ]
        )

        self.assertEqual(report["training_signal"]["group_counts"]["mixed"], 1)
        self.assertEqual(report["training_signal"]["effective_task_ids"], ["1"])

    def test_rejects_mixed_task_ids_inside_sequential_group(self) -> None:
        with self.assertRaisesRegex(ValueError, "multiple task IDs"):
            self._build(
                [
                    _rollout("1", 0),
                    _rollout("2", 1),
                    _rollout("3", 0),
                    _rollout("3", 0),
                ]
            )


if __name__ == "__main__":
    unittest.main()
