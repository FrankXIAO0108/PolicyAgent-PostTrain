from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from src.evaluation.claim_state_holdout import evaluate


PROJECT = Path(__file__).resolve().parents[1]
DATASET = PROJECT / "data" / "claim_state_holdout_v1.json"


class ClaimStateHoldoutTests(unittest.TestCase):
    def test_dataset_is_frozen_and_training_prohibited(self) -> None:
        dataset = json.loads(DATASET.read_text(encoding="utf-8"))
        self.assertFalse(dataset["policy"]["training_allowed"])
        self.assertFalse(dataset["policy"]["rule_tuning_allowed"])
        self.assertEqual(len(dataset["cases"]), 24)
        self.assertEqual(
            len({case["case_id"] for case in dataset["cases"]}), 24
        )

    def test_report_is_hash_bound_and_exposes_gate_failures(self) -> None:
        report = evaluate(DATASET)
        self.assertEqual(len(report["dataset"]["sha256"]), 64)
        self.assertEqual(report["metrics"]["case_count"], 24)
        self.assertFalse(report["ready_for_reward_penalty"])
        self.assertTrue(report["errors"])

    def test_evaluator_rejects_training_eligible_dataset(self) -> None:
        dataset = json.loads(DATASET.read_text(encoding="utf-8"))
        dataset["policy"]["training_allowed"] = True
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "invalid.json"
            path.write_text(json.dumps(dataset), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "prohibited from training"):
                evaluate(path)


if __name__ == "__main__":
    unittest.main()
