from __future__ import annotations

import unittest

from src.evaluation.comparison import (
    _classification_metrics,
    _taxonomy_metrics,
)


class ComparisonMetricTests(unittest.TestCase):
    def test_classification_metrics(self) -> None:
        result = _classification_metrics(
            {"1": False, "2": True, "3": True, "4": False},
            {"1": False, "2": False, "3": True, "4": True},
        )
        self.assertEqual(
            result["confusion_matrix"],
            {"tp": 1, "fp": 1, "fn": 1, "tn": 1},
        )
        self.assertEqual(result["accuracy"], 0.5)
        self.assertEqual(result["recall"], 0.5)

    def test_taxonomy_metrics_are_multilabel(self) -> None:
        result = _taxonomy_metrics(
            {"59": ["dataset_alignment_error", "missing_action"]},
            {"59": ["dataset_alignment_error", "missing_action"]},
        )
        self.assertEqual(result["exact_match_task_count"], 1)
        self.assertEqual(result["micro_precision"], 1.0)
        self.assertEqual(result["micro_recall"], 1.0)


if __name__ == "__main__":
    unittest.main()
