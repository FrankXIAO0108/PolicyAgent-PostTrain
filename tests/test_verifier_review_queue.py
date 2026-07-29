from __future__ import annotations

import unittest

from src.verifiers.gold_validation import GoldAnnotation
from src.verifiers.review_queue import build_review_candidates


def annotation(task_id: str, status: str) -> GoldAnnotation:
    return GoldAnnotation(
        task_id=task_id,
        label=None if status == "UNREVIEWED" else "PASS",
        status=status,
        source="test",
        rationale="Targeted human review required.",
        evidence_files=(),
    )


def prediction(task_id: str, verdict: str) -> dict[str, object]:
    return {
        "task_id": task_id,
        "verdict": verdict,
        "findings": [],
        "metrics": {},
    }


class ReviewQueueTests(unittest.TestCase):
    def test_selects_only_unreviewed_and_prioritizes_fail(self) -> None:
        candidates = build_review_candidates(
            [
                annotation("19", "UNREVIEWED"),
                annotation("37", "UNREVIEWED"),
                annotation("28", "PROVISIONAL"),
            ],
            {
                "v1.2": {
                    "19": prediction("19", "REVIEW"),
                    "37": prediction("37", "FAIL"),
                    "28": prediction("28", "PASS"),
                },
                "v2.0": {
                    "19": prediction("19", "REVIEW"),
                    "37": prediction("37", "FAIL"),
                    "28": prediction("28", "PASS"),
                },
            },
        )

        self.assertEqual([row["task_id"] for row in candidates], ["37", "19"])
        self.assertEqual(candidates[0]["priority"], "P1_PREDICTED_FAIL")
        self.assertEqual(candidates[1]["priority"], "P2_PREDICTED_REVIEW")
        self.assertIsNone(candidates[0]["annotation_label"])

    def test_verifier_disagreement_has_highest_priority(self) -> None:
        candidates = build_review_candidates(
            [annotation("19", "UNREVIEWED"), annotation("37", "UNREVIEWED")],
            {
                "v1.2": {
                    "19": prediction("19", "PASS"),
                    "37": prediction("37", "FAIL"),
                },
                "v2.0": {
                    "19": prediction("19", "REVIEW"),
                    "37": prediction("37", "FAIL"),
                },
            },
        )

        self.assertEqual(candidates[0]["task_id"], "19")
        self.assertEqual(candidates[0]["priority"], "P0_VERIFIER_DISAGREEMENT")
        self.assertTrue(candidates[0]["verifier_disagreement"])

    def test_missing_prediction_fails_closed(self) -> None:
        with self.assertRaisesRegex(ValueError, "missing v2.0 prediction"):
            build_review_candidates(
                [annotation("19", "UNREVIEWED")],
                {
                    "v1.2": {"19": prediction("19", "REVIEW")},
                    "v2.0": {},
                },
            )

    def test_no_unreviewed_rows_produces_empty_queue(self) -> None:
        candidates = build_review_candidates(
            [annotation("28", "PROVISIONAL")],
            {
                "v1.2": {"28": prediction("28", "PASS")},
                "v2.0": {"28": prediction("28", "REVIEW")},
            },
        )

        self.assertEqual(candidates, [])


if __name__ == "__main__":
    unittest.main()
