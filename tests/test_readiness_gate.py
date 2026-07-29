from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from src.training.readiness_gate import evaluate_readiness, sha256


def write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload), encoding="utf-8")


class ReadinessGateTests(unittest.TestCase):
    def test_empty_inputs_block_every_stage(self) -> None:
        result = evaluate_readiness()
        self.assertTrue(
            all(not gate["ready"] for gate in result["gates"].values())
        )

    def test_complete_evidence_can_open_all_stage_gates(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            policy = root / "policy.json"
            write_json(
                policy,
                {"release_gate": {"official_metrics_allowed": True}},
            )
            dataset_file = root / "sft_dataset.jsonl"
            dataset_file.write_text('{"task_id":"1"}\n', encoding="utf-8")
            dataset = root / "dataset_manifest.json"
            write_json(
                dataset,
                {
                    "dataset_sha256": sha256(dataset_file),
                    "counts": {"released": 2, "train": 1, "validation": 1},
                },
            )
            checkpoint = root / "checkpoint.bin"
            checkpoint.write_bytes(b"checkpoint")
            run = root / "run.json"
            write_json(
                run,
                {
                    "stage": "SFT",
                    "status": "COMPLETED",
                    "smoke_passed": True,
                    "input_dataset_sha256": sha256(dataset_file),
                    "checkpoint": {
                        "path": str(checkpoint),
                        "sha256": sha256(checkpoint),
                    },
                },
            )
            comparison = root / "comparison.json"
            run_hash = sha256(run)
            checkpoint_hash = sha256(checkpoint)
            write_json(
                comparison,
                {
                    "stage": "BASE_VS_SFT",
                    "status": "COMPLETED",
                    "comparable_protocol": True,
                    "frozen_protocol": True,
                    "no_posthoc_tuning": True,
                    "residual_systematic_failures": ["authorization"],
                    "rl_justified": True,
                    "bindings": {
                        "sft_run_sha256": run_hash,
                        "sft_checkpoint_sha256": checkpoint_hash,
                        "task_set_sha256": "TASKS",
                        "runtime_config_sha256": "RUNTIME",
                    },
                },
            )
            preference = root / "preference.json"
            write_json(
                preference,
                {
                    "status": "READY",
                    "fully_adjudicated": True,
                    "pair_count": 2,
                    "group_leakage_detected": False,
                    "source_comparison_sha256": sha256(comparison),
                },
            )
            reward = root / "reward.json"
            write_json(
                reward,
                {
                    "held_out": True,
                    "release_gate": {"official_metrics_allowed": True},
                    "metrics": {
                        "precision": 0.95,
                        "recall": 0.96,
                        "critical_recall": 1.0,
                    },
                    "unresolved_fp_task_ids": [],
                    "unresolved_fn_task_ids": [],
                    "reward_spec_sha256": "REWARD-SPEC",
                    "source_policy_validation_sha256": sha256(policy),
                },
            )

            result = evaluate_readiness(
                policy_validation_path=policy,
                sft_dataset_manifest_path=dataset,
                sft_run_manifest_path=run,
                comparison_manifest_path=comparison,
                preference_manifest_path=preference,
                reward_validation_path=reward,
            )

            self.assertTrue(
                all(gate["ready"] for gate in result["gates"].values())
            )

    def test_dpo_and_rl_remain_blocked_without_residual_evidence(self) -> None:
        result = evaluate_readiness()
        self.assertIn(
            "No residual systematic failures justify preference work.",
            result["gates"]["dpo"]["reasons"],
        )
        self.assertIn(
            "Comparable SFT evaluation does not justify RL.",
            result["gates"]["rlhf_grpo"]["reasons"],
        )


if __name__ == "__main__":
    unittest.main()
