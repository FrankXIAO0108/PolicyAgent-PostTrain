from __future__ import annotations

import json
import unittest
from collections import Counter
from pathlib import Path


PROJECT = Path(__file__).resolve().parents[1]
CANDIDATE = PROJECT / "data" / "claim_state_v2_holdout_candidate.json"


class ClaimStateV2HoldoutCandidateTests(unittest.TestCase):
    def test_candidate_is_blind_pending_and_not_evaluable(self) -> None:
        dataset = json.loads(CANDIDATE.read_text(encoding="utf-8"))
        self.assertEqual(
            dataset["review_status"], "PENDING_PROJECT_OWNER_REVIEW"
        )
        self.assertFalse(dataset["policy"]["training_allowed"])
        self.assertFalse(dataset["policy"]["rule_tuning_allowed"])
        self.assertFalse(dataset["policy"]["evaluation_allowed"])
        self.assertEqual(len(dataset["cases"]), 24)
        self.assertEqual(
            len({case["case_id"] for case in dataset["cases"]}), 24
        )
        self.assertTrue(
            all(case["owner_review"] == "PENDING" for case in dataset["cases"])
        )
        self.assertTrue(
            all("expected_verdict" not in case for case in dataset["cases"])
        )

    def test_candidate_has_declared_balance(self) -> None:
        dataset = json.loads(CANDIDATE.read_text(encoding="utf-8"))
        counts = Counter(case["proposed_verdict"] for case in dataset["cases"])
        self.assertEqual(
            counts,
            {"PASS": 6, "FAIL": 6, "REVIEW": 10, "NOT_APPLICABLE": 2},
        )


if __name__ == "__main__":
    unittest.main()
