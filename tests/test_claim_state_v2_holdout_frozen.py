from __future__ import annotations

import json
import unittest
from collections import Counter
from pathlib import Path


PROJECT = Path(__file__).resolve().parents[1]
HOLDOUT = PROJECT / "data" / "claim_state_v2_holdout_v2.json"


class ClaimStateV2FrozenHoldoutTests(unittest.TestCase):
    def test_holdout_is_owner_approved_and_evaluable_but_not_tunable(self) -> None:
        dataset = json.loads(HOLDOUT.read_text(encoding="utf-8"))
        self.assertEqual(
            dataset["review_status"], "APPROVED_BY_PROJECT_OWNER"
        )
        self.assertEqual(dataset["review"]["reviewer_role"], "project_owner")
        self.assertFalse(dataset["policy"]["training_allowed"])
        self.assertFalse(dataset["policy"]["rule_tuning_allowed"])
        self.assertTrue(dataset["policy"]["evaluation_allowed"])
        self.assertEqual(len(dataset["cases"]), 24)
        self.assertEqual(
            len({case["case_id"] for case in dataset["cases"]}), 24
        )
        self.assertTrue(
            all(case["owner_review"] == "APPROVED" for case in dataset["cases"])
        )
        self.assertTrue(
            all("expected_verdict" in case for case in dataset["cases"])
        )

    def test_holdout_has_approved_label_balance(self) -> None:
        dataset = json.loads(HOLDOUT.read_text(encoding="utf-8"))
        counts = Counter(case["expected_verdict"] for case in dataset["cases"])
        self.assertEqual(
            counts,
            {"PASS": 6, "FAIL": 6, "REVIEW": 10, "NOT_APPLICABLE": 2},
        )


if __name__ == "__main__":
    unittest.main()
