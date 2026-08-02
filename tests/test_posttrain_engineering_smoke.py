from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from src.training.engineering_smoke_data import build_records, write_dataset
from src.training.run_posttrain_engineering_smoke import (
    correct_arguments_reward,
    correct_tool_reward,
    directory_sha256,
    sha256,
    valid_json_reward,
    validate_inputs,
)
from src.training.verify_posttrain_engineering_smoke import verify_run


class EngineeringSmokeDataTests(unittest.TestCase):
    def test_synthetic_splits_are_disjoint_and_have_expected_shapes(self) -> None:
        records = build_records()
        self.assertEqual(len(records["sft"]), 24)
        self.assertEqual(len(records["dpo"]), 24)
        self.assertEqual(len(records["grpo"]), 8)
        self.assertEqual(len(records["holdout"]), 8)
        self.assertEqual(
            {row["scenario_id"] for row in records["sft"]},
            {row["scenario_id"] for row in records["dpo"]},
        )
        self.assertFalse(
            {row["scenario_id"] for row in records["sft"]}
            & {row["scenario_id"] for row in records["holdout"]}
        )

    def test_writer_binds_each_jsonl_with_hash(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            manifest = write_dataset(Path(temp_dir))
            self.assertTrue(manifest["leakage_checks"]["passed"])
            self.assertFalse(manifest["contains_tau2_frozen_tasks"])
            self.assertEqual(manifest["files"]["sft"]["rows"], 24)
            sft_path = Path(temp_dir) / "sft.jsonl"
            self.assertNotIn(b"\r\n", sft_path.read_bytes())
            self.assertEqual(manifest["files"]["sft"]["sha256"], sha256(sft_path))

    def test_repository_config_and_data_pass_preflight_when_dirty_is_allowed(self) -> None:
        config = Path("configs/posttrain_engineering_smoke_v1.json").resolve()
        result = validate_inputs(config, allow_dirty=True)
        self.assertEqual(result["config"]["scope"], "ISOLATED_ENGINEERING_SMOKE")


class EngineeringSmokeRewardTests(unittest.TestCase):
    def test_multidimensional_rewards_distinguish_format_tool_and_arguments(self) -> None:
        expected = ['{"tool":"get_order","arguments":{"order_id":"SYN-1"}}']
        exact = ['{"tool":"get_order","arguments":{"order_id":"SYN-1"}}']
        wrong_args = ['{"tool":"get_order","arguments":{"order_id":"SYN-2"}}']
        invalid = ["not-json"]
        self.assertEqual(valid_json_reward(exact), [1.0])
        self.assertEqual(correct_tool_reward(exact, expected), [1.0])
        self.assertEqual(correct_arguments_reward(exact, expected), [1.0])
        self.assertEqual(correct_tool_reward(wrong_args, expected), [1.0])
        self.assertEqual(correct_arguments_reward(wrong_args, expected), [0.0])
        self.assertEqual(valid_json_reward(invalid), [0.0])


class EngineeringSmokeVerificationTests(unittest.TestCase):
    def test_missing_manifest_never_allows_completion_claim(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            report = verify_run(Path(temp_dir))
            self.assertFalse(report["verified_complete"])
            self.assertFalse(report["completion_claim_allowed"])

    def test_complete_stage_evidence_allows_engineering_claim(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            config = root / "config.json"
            data_manifest = root / "data_manifest.json"
            config.write_text("{}", encoding="utf-8")
            data_manifest.write_text("{}", encoding="utf-8")
            stages = []
            for stage in ("SFT", "DPO", "GRPO"):
                adapter = root / f"{stage.lower()}_adapter"
                merged = root / f"{stage.lower()}_merged"
                checkpoint = root / f"{stage.lower()}_checkpoint"
                for directory in (adapter, merged, checkpoint):
                    directory.mkdir()
                    (directory / "weights.bin").write_bytes(stage.encode("ascii"))
                loss = root / f"{stage.lower()}_loss.jsonl"
                loss.write_text('{"loss":1.0}\n', encoding="utf-8")
                stages.append(
                    {
                        "stage": stage,
                        "status": "COMPLETED",
                        "artifact_path": str(adapter),
                        "artifact_sha256": directory_sha256(adapter),
                        "adapter": {
                            "path": str(adapter),
                            "sha256": directory_sha256(adapter),
                        },
                        "merged_model": {
                            "path": str(merged),
                            "sha256": directory_sha256(merged),
                        },
                        "checkpoint": {
                            "path": str(checkpoint),
                            "sha256": directory_sha256(checkpoint),
                        },
                        "loss_history": {
                            "path": str(loss),
                            "sha256": sha256(loss),
                            "rows": 1,
                        },
                        "train_metrics": {"train_loss": 1.0},
                    }
                )
            metrics = {
                "rows": 8,
                "valid_json_rate": 0.5,
                "tool_match_rate": 0.5,
                "arguments_match_rate": 0.5,
                "exact_action_match_rate": 0.5,
            }
            manifest = {
                "scope": "ISOLATED_ENGINEERING_SMOKE",
                "status": "COMPLETED",
                "git": {"dirty_at_start": False},
                "formal_retail_readiness_gate_opened": False,
                "business_improvement_claim_allowed": False,
                "stages": stages,
                "holdout_evaluations": {
                    name: metrics for name in ("base", "sft", "dpo", "grpo")
                },
                "bindings": {
                    "config_path": str(config),
                    "config_sha256": sha256(config),
                    "data_manifest_path": str(data_manifest),
                    "data_manifest_sha256": sha256(data_manifest),
                },
                "environment": {
                    name: "version"
                    for name in (
                        "torch",
                        "transformers",
                        "trl",
                        "datasets",
                        "peft",
                        "accelerate",
                    )
                },
            }
            (root / "run_manifest.json").write_text(
                json.dumps(manifest), encoding="utf-8"
            )
            report = verify_run(root)
            self.assertTrue(report["verified_complete"], report["reasons"])
            self.assertTrue(report["completion_claim_allowed"])


if __name__ == "__main__":
    unittest.main()
