from __future__ import annotations

import json
import unittest
from pathlib import Path

from src.evaluation.claim_state_holdout import evaluate
from src.training.teacher_evidence_pack import claim_state_consistency_v2


PROJECT = Path(__file__).resolve().parents[1]
DEVELOPMENT = PROJECT / "data" / "claim_state_v2_development.json"


class ClaimStateV2Tests(unittest.TestCase):
    def test_development_suite_is_not_a_holdout_or_training_data(self) -> None:
        dataset = json.loads(DEVELOPMENT.read_text(encoding="utf-8"))
        self.assertFalse(dataset["policy"]["training_allowed"])
        self.assertTrue(dataset["policy"]["rule_tuning_allowed"])
        self.assertEqual(len(dataset["cases"]), 20)

    def test_v2_matches_all_declared_development_cases(self) -> None:
        report = evaluate(DEVELOPMENT, checker_version="v2-development")
        self.assertEqual(report["metrics"]["exact_match_count"], 20)
        self.assertEqual(report["metrics"]["case_count"], 20)
        self.assertFalse(report["errors"])

    def test_v2_keeps_multi_entity_same_span_conservative(self) -> None:
        messages = [
            {
                "role": "assistant",
                "content": "Orders #W3000001 and #W3000002 have been cancelled.",
            }
        ]
        final_state = {
            "agent": {
                "orders": {
                    "#W3000001": {
                        "order_id": "#W3000001",
                        "status": "cancelled",
                    },
                    "#W3000002": {
                        "order_id": "#W3000002",
                        "status": "cancelled",
                    },
                }
            }
        }
        result = claim_state_consistency_v2(messages, final_state)
        self.assertEqual(result["verdict"], "REVIEW")
        self.assertEqual(
            result["findings"][0]["reason_code"],
            "MULTIPLE_ORDERS_IN_SINGLE_CLAIM_SPAN",
        )


if __name__ == "__main__":
    unittest.main()
