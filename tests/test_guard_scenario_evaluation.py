from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from src.guards.scenario_evaluation import (
    DEFAULT_SUITE,
    evaluate_suite,
    load_suite,
)


class GuardScenarioEvaluationTests(unittest.TestCase):
    def test_frozen_synthetic_suite_matches_expectations(self) -> None:
        result = evaluate_suite(load_suite(DEFAULT_SUITE))

        self.assertEqual(result["summary"]["case_count"], 15)
        self.assertEqual(result["summary"]["failed_count"], 0)
        self.assertEqual(result["summary"]["exact_case_accuracy"], 1.0)
        self.assertEqual(result["summary"]["decision_accuracy"], 1.0)
        self.assertEqual(result["summary"]["blocking_rule_exact_match"], 1.0)
        self.assertEqual(
            result["summary"]["blocking_detection"],
            {
                "positive_class": "expected decision is not ALLOW",
                "tp": 9,
                "fp": 0,
                "fn": 0,
                "tn": 6,
                "precision": 1.0,
                "recall": 1.0,
                "f1": 1.0,
            },
        )
        self.assertFalse(result["summary"]["uses_reference_actions"])
        self.assertEqual(result["summary"]["new_llm_calls"], 0)

    def test_suite_rejects_duplicate_case_ids(self) -> None:
        payload = {
            "official_metric": False,
            "cases": [
                {"case_id": "duplicate"},
                {"case_id": "duplicate"},
            ],
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "suite.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "unique"):
                load_suite(path)

    def test_suite_cannot_claim_official_metric(self) -> None:
        payload = {
            "official_metric": True,
            "cases": [{"case_id": "one"}],
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "suite.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "official_metric=false"):
                load_suite(path)


if __name__ == "__main__":
    unittest.main()
