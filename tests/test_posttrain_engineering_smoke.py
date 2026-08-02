from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from src.training.engineering_smoke_data import build_records, write_dataset
from src.training.run_posttrain_engineering_smoke import (
    correct_arguments_reward,
    correct_tool_reward,
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


if __name__ == "__main__":
    unittest.main()
