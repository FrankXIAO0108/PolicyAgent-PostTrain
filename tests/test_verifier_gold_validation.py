from __future__ import annotations

import unittest

from src.verifiers.gold_validation import GoldAnnotation, evaluate_annotations


def annotation(task_id: str, label: str | None, status: str) -> GoldAnnotation:
    return GoldAnnotation(
        task_id=task_id,
        label=label,
        status=status,
        source="test",
        rationale="test",
        evidence_files=(),
    )


class GoldValidationTests(unittest.TestCase):
    def test_default_mode_excludes_provisional_and_unreviewed_rows(self) -> None:
        result = evaluate_annotations(
            [
                annotation("1", "PASS", "ADJUDICATED"),
                annotation("2", "FAIL", "PROVISIONAL"),
                annotation("3", None, "UNREVIEWED"),
            ],
            {"1": "PASS", "2": "REVIEW", "3": "FAIL"},
        )

        self.assertEqual(result["coverage"]["evaluated_rows"], 1)
        self.assertEqual(result["fail_detection"]["tn"], 1)
        self.assertFalse(result["release_gate"]["official_metrics_allowed"])

    def test_provisional_mode_reports_fp_fn_and_review_as_abstention(self) -> None:
        result = evaluate_annotations(
            [
                annotation("1", "FAIL", "PROVISIONAL"),
                annotation("2", "FAIL", "PROVISIONAL"),
                annotation("3", "PASS", "PROVISIONAL"),
                annotation("4", "PASS", "PROVISIONAL"),
                annotation("5", "REVIEW", "PROVISIONAL"),
            ],
            {
                "1": "FAIL",
                "2": "REVIEW",
                "3": "FAIL",
                "4": "PASS",
                "5": "REVIEW",
            },
            include_provisional=True,
        )

        metrics = result["fail_detection"]
        self.assertEqual(
            (metrics["tp"], metrics["fp"], metrics["fn"], metrics["tn"]),
            (1, 1, 1, 1),
        )
        self.assertEqual(metrics["false_positive_task_ids"], ["3"])
        self.assertEqual(metrics["false_negative_task_ids"], ["2"])
        self.assertAlmostEqual(metrics["precision"], 0.5)
        self.assertAlmostEqual(metrics["recall"], 0.5)
        self.assertAlmostEqual(metrics["f1"], 0.5)
        self.assertEqual(
            result["three_class"]["confusion_matrix"]["REVIEW"]["REVIEW"], 1
        )
        self.assertFalse(result["release_gate"]["official_metrics_allowed"])

    def test_release_gate_requires_complete_adjudication(self) -> None:
        result = evaluate_annotations(
            [
                annotation("1", "PASS", "ADJUDICATED"),
                annotation("2", "FAIL", "ADJUDICATED"),
            ],
            {"1": "PASS", "2": "FAIL"},
        )

        self.assertTrue(result["release_gate"]["official_metrics_allowed"])
        self.assertEqual(result["fail_detection"]["precision"], 1.0)
        self.assertEqual(result["fail_detection"]["recall"], 1.0)


if __name__ == "__main__":
    unittest.main()
