from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from src.guards.ablation import (
    DEFAULT_PROTOCOL,
    evaluate_ablation,
    load_protocol,
)
from src.guards.scenario_evaluation import DEFAULT_SUITE, load_suite


class GuardAblationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.result = evaluate_ablation(
            load_suite(DEFAULT_SUITE),
            load_protocol(DEFAULT_PROTOCOL),
        )
        self.variants = {
            row["variant_id"]: row for row in self.result["variants"]
        }

    def test_no_guard_and_full_guard_form_expected_baselines(self) -> None:
        no_guard = self.variants["no_guard"]["summary"]
        full_guard = self.variants["full_guard"]["summary"]

        self.assertEqual((no_guard["tp"], no_guard["fn"]), (0, 9))
        self.assertEqual((no_guard["fp"], no_guard["tn"]), (0, 6))
        self.assertEqual((full_guard["tp"], full_guard["fn"]), (9, 0))
        self.assertEqual((full_guard["fp"], full_guard["tn"]), (0, 6))
        self.assertEqual(full_guard["recall"], 1.0)

    def test_variant_ablation_exposes_three_semantic_dependencies(self) -> None:
        summary = self.variants["without_variant"]["summary"]

        self.assertEqual(summary["lost_risky_detections_vs_full"], 3)
        self.assertEqual(
            summary["missed_risky_case_ids"],
            [
                "cross_product_exchange",
                "same_item_exchange",
                "unavailable_replacement_variant",
            ],
        )
        self.assertEqual(summary["safe_control_regression_ids"], [])

    def test_each_single_family_ablation_has_no_safe_control_regression(self) -> None:
        for variant_id, row in self.variants.items():
            with self.subTest(variant_id=variant_id):
                self.assertEqual(
                    row["summary"]["safe_control_regression_ids"], []
                )

    def test_protocol_rejects_duplicate_variant_ids(self) -> None:
        payload = json.loads(DEFAULT_PROTOCOL.read_text(encoding="utf-8"))
        payload["variants"].append(dict(payload["variants"][0]))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "protocol.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "unique"):
                load_protocol(path)


if __name__ == "__main__":
    unittest.main()
