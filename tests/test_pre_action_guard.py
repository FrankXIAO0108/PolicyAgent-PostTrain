from __future__ import annotations

import unittest
from pathlib import Path

from src.guards.offline_audit import audit_artifact
from src.guards.retail_pre_action import (
    GuardContext,
    GuardDecision,
    ToolProposal,
    evaluate_retail_actions,
)


PROJECT = Path(r"D:\PolicyAgent-PostTrain")
BASELINE = (
    PROJECT
    / "experiments"
    / "20260722_110504_retail_baseline20_trial1_deepseek"
)


def categories(result) -> set[str]:
    return {finding.category for finding in result.findings if finding.blocking}


class RetailPreActionGuardTests(unittest.TestCase):
    def test_same_item_exchange_is_blocked(self) -> None:
        result = evaluate_retail_actions(
            [
                ToolProposal(
                    name="exchange_delivered_order_items",
                    arguments={
                        "order_id": "#W1",
                        "item_ids": ["old"],
                        "new_item_ids": ["old"],
                        "payment_method_id": "paypal_1",
                    },
                )
            ],
            GuardContext(),
        )
        self.assertFalse(result.allowed)
        self.assertIn("policy_error", categories(result))
        self.assertEqual(result.decision, GuardDecision.REGENERATE)

    def test_item_scoped_cancel_requires_whole_order_confirmation(self) -> None:
        context = GuardContext(
            orders={
                "#W1": {
                    "order_id": "#W1",
                    "status": "pending",
                    "items": [
                        {"name": "Skateboard", "item_id": "board"},
                        {"name": "Air Purifier", "item_id": "air"},
                    ],
                }
            },
            user_texts=["Please cancel the skateboard from order #W1."],
        )
        result = evaluate_retail_actions(
            [
                ToolProposal(
                    name="cancel_pending_order",
                    arguments={"order_id": "#W1", "reason": "no longer needed"},
                )
            ],
            context,
        )
        self.assertFalse(result.allowed)
        self.assertIn("scope_error", categories(result))
        self.assertEqual(
            result.decision,
            GuardDecision.REQUIRE_CONFIRMATION,
        )

    def test_boolean_availability_can_serve_two_exchanges(self) -> None:
        context = GuardContext(
            orders={
                "#W1": {
                    "status": "delivered",
                    "items": [{"name": "Laptop", "product_id": "laptop"}],
                },
                "#W2": {
                    "status": "delivered",
                    "items": [{"name": "Laptop", "product_id": "laptop"}],
                },
            },
            products={
                "laptop": {
                    "name": "Laptop",
                    "product_id": "laptop",
                    "variants": {
                        "target": {
                            "available": True,
                            "options": {
                                "processor": "i7",
                                "ram": "8GB",
                                "storage": "1TB SSD",
                            },
                        }
                    },
                }
            },
            user_texts=[
                "I need to exchange two laptops for i7, 8GB, and 1TB SSD."
            ],
        )
        result = evaluate_retail_actions(
            [ToolProposal(name="transfer_to_human_agents", arguments={})],
            context,
        )
        self.assertFalse(result.allowed)
        self.assertIn("variant_error", categories(result))
        self.assertEqual(result.decision, GuardDecision.REGENERATE)

    def test_unknown_payment_method_is_blocked(self) -> None:
        context = GuardContext(payment_method_ids={"paypal_1"})
        result = evaluate_retail_actions(
            [
                ToolProposal(
                    name="return_delivered_order_items",
                    arguments={
                        "order_id": "#W1",
                        "item_ids": ["item"],
                        "payment_method_id": "credit_card_unknown",
                    },
                )
            ],
            context,
        )
        self.assertFalse(result.allowed)
        self.assertIn("payment_error", categories(result))
        self.assertEqual(result.decision, GuardDecision.REGENERATE)

    def test_reference_payment_mismatch_is_isolated(self) -> None:
        reference = ToolProposal(
            name="exchange_delivered_order_items",
            arguments={
                "order_id": "#W1",
                "item_ids": ["old"],
                "new_item_ids": ["new"],
                "payment_method_id": "card_gold",
            },
        )
        proposal = ToolProposal(
            name=reference.name,
            arguments={**reference.arguments, "payment_method_id": "card_actual"},
        )
        runtime = evaluate_retail_actions([proposal], GuardContext())
        diagnostic = evaluate_retail_actions(
            [proposal],
            GuardContext(reference_actions=[reference], enforce_reference=True),
        )
        self.assertTrue(runtime.allowed)
        self.assertFalse(diagnostic.allowed)
        self.assertIn("payment_error", categories(diagnostic))

    def test_invalid_order_state_is_terminal_block(self) -> None:
        context = GuardContext(
            orders={
                "#W1": {
                    "order_id": "#W1",
                    "status": "delivered",
                    "items": [],
                }
            }
        )
        result = evaluate_retail_actions(
            [
                ToolProposal(
                    name="cancel_pending_order",
                    arguments={"order_id": "#W1", "reason": "no longer needed"},
                )
            ],
            context,
        )
        self.assertEqual(result.decision, GuardDecision.BLOCK)

    def test_second_modify_on_same_order_is_blocked(self) -> None:
        context = GuardContext(
            completed_writes=[
                ToolProposal(
                    name="modify_pending_order_items",
                    arguments={
                        "order_id": "#W1",
                        "item_ids": ["old"],
                        "new_item_ids": ["new"],
                        "payment_method_id": "paypal_1",
                    },
                )
            ]
        )

        result = evaluate_retail_actions(
            [
                ToolProposal(
                    name="modify_pending_order_address",
                    arguments={
                        "order_id": "#W1",
                        "address1": "1 Main St",
                        "city": "Charlotte",
                        "state": "NC",
                        "country": "USA",
                        "zip": "28243",
                    },
                )
            ],
            context,
        )

        self.assertFalse(result.allowed)
        self.assertEqual(result.decision, GuardDecision.BLOCK)
        self.assertTrue(
            any(
                finding.rule_id == "policy.one_shot_order_mutation"
                for finding in result.findings
            )
        )


@unittest.skipUnless(BASELINE.exists(), "frozen baseline artifacts absent")
class FrozenFailureGuardTests(unittest.TestCase):
    def audit(self, task_id: str):
        return audit_artifact(
            BASELINE / f"task_{task_id}" / "returned_results.json"
        )

    def test_task_95_blocks_premature_transfer(self) -> None:
        result = self.audit("95")
        findings = result["runtime_guard"]["findings"]
        self.assertTrue(result["runtime_guard"]["would_block"])
        self.assertTrue(
            any(
                item["rule_id"] == "goal.transfer_with_actionable_variant"
                for item in findings
            )
        )

    def test_task_98_blocks_scope_and_detects_reference_payment(self) -> None:
        result = self.audit("98")
        runtime_categories = {
            item["category"]
            for item in result["runtime_guard"]["findings"]
            if item["blocking"]
        }
        reference_categories = {
            item["category"]
            for item in result["reference_diagnostic"]["findings"]
            if item["blocking"]
        }
        self.assertIn("scope_error", runtime_categories)
        self.assertIn("payment_error", reference_categories)

    def test_task_107_blocks_same_item_exchange(self) -> None:
        result = self.audit("107")
        self.assertTrue(
            any(
                item["rule_id"] == "policy.exchange_requires_different_option"
                for item in result["runtime_guard"]["findings"]
            )
        )


if __name__ == "__main__":
    unittest.main()
