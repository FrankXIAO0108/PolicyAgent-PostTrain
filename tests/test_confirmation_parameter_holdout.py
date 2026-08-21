from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from src.evaluation.confirmation_parameter_holdout import evaluate_holdout
from src.rl.retail_agentic_env import confirmation_parameter_binding


PROJECT = Path(__file__).resolve().parents[1]
CONFIG = PROJECT / "configs" / "confirmation_parameter_holdout_v1.json"


class ConfirmationParameterBindingTests(unittest.TestCase):
    def test_complete_cancel_binding_passes(self) -> None:
        result = confirmation_parameter_binding(
            "cancel_pending_order",
            {"order_id": "#W100", "reason": "ordered by mistake"},
            "Cancel order #W100 because it was ordered by mistake? Yes.",
        )

        self.assertEqual(result["verdict"], "PASS")
        self.assertEqual(result["missing_fields"], [])
        self.assertFalse(result["used_as_reward"])

    def test_unpaired_item_values_require_review(self) -> None:
        result = confirmation_parameter_binding(
            "exchange_delivered_order_items",
            {
                "order_id": "#W400",
                "item_ids": ["old-1", "old-2"],
                "new_item_ids": ["new-1", "new-2"],
                "payment_method_id": "card-4",
            },
            "Exchange old-1 and old-2 for new-1 and new-2 on #W400 using card-4?",
        )

        self.assertEqual(result["verdict"], "REVIEW")
        self.assertIn("item_pairs", result["missing_fields"])

    def test_duplicate_item_quantity_must_be_explicit(self) -> None:
        arguments = {
            "order_id": "#W201",
            "item_ids": ["item-2", "item-2"],
            "payment_method_id": "gift-1",
        }
        implicit = confirmation_parameter_binding(
            "return_delivered_order_items",
            arguments,
            "Return item-2 from #W201 to gift-1?",
        )
        explicit = confirmation_parameter_binding(
            "return_delivered_order_items",
            arguments,
            "Return item-2 and item-2 from #W201 to gift-1?",
        )

        self.assertEqual(implicit["verdict"], "REVIEW")
        self.assertEqual(explicit["verdict"], "PASS")

    def test_address_is_not_evaluable_in_v1_contract(self) -> None:
        result = confirmation_parameter_binding(
            "modify_pending_order_address",
            {"order_id": "#W600", "zip": "98101"},
            "Change order #W600 to zip 98101?",
        )

        self.assertEqual(result["verdict"], "NOT_EVALUABLE")


class ConfirmationParameterHoldoutTests(unittest.TestCase):
    def test_frozen_holdout_matches_all_authored_expectations(self) -> None:
        report = evaluate_holdout(CONFIG)

        self.assertEqual(report["scope"], "SYNTHETIC_CONTRACT_HOLDOUT")
        self.assertEqual(report["summary"]["case_count"], 18)
        self.assertEqual(report["summary"]["mismatch_count"], 0)
        self.assertFalse(report["summary"]["reward_integration_allowed"])

    def test_duplicate_case_id_is_rejected(self) -> None:
        payload = json.loads(CONFIG.read_text(encoding="utf-8"))
        payload["cases"][1]["case_id"] = payload["cases"][0]["case_id"]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bad.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "duplicate case_id"):
                evaluate_holdout(path)


if __name__ == "__main__":
    unittest.main()
