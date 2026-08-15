import unittest

from src.training.correct_tau2_teacher_trajectory import (
    apply_assistant_replacements,
    serialize_and_clean,
    validate_structure,
)


def source_messages():
    return [
        {"role": "user", "content": "Please cancel both pending orders."},
        {
            "role": "assistant",
            "content": "I'll cancel both pending orders now.",
            "tool_calls": [
                {"id": "c1", "name": "cancel_pending_order", "arguments": {"order_id": "#W1", "reason": "ordered by mistake"}},
                {"id": "c2", "name": "cancel_pending_order", "arguments": {"order_id": "#W2", "reason": "ordered by mistake"}},
            ],
        },
        {"role": "tool", "id": "c1", "content": "cancelled #W1", "error": False},
        {"role": "tool", "id": "c2", "content": "cancelled #W2", "error": False},
        {"role": "assistant", "content": "Both cancelled. Paid with credit_card_1234.", "tool_calls": []},
    ]


class CorrectTrajectoryTests(unittest.TestCase):
    def test_serialize_splits_parallel_calls_and_strips_narration(self):
        corrected, change_log = serialize_and_clean(source_messages())
        roles = [m["role"] for m in corrected]
        self.assertEqual(
            roles,
            ["user", "assistant", "tool", "assistant", "tool", "assistant"],
        )
        self.assertEqual([m["tool_calls"][0]["id"] for m in corrected if m["role"] == "assistant" and m["tool_calls"]], ["c1", "c2"])
        self.assertTrue(all(not m["content"] for m in corrected if m["role"] == "assistant" and m["tool_calls"]))
        categories = {entry["category"] for entry in change_log}
        self.assertEqual(categories, {"tool_call_serialization", "assistant_text_cleanup"})

    def test_serialize_raises_when_tool_result_missing(self):
        messages = source_messages()
        del messages[3]
        with self.assertRaises(ValueError):
            serialize_and_clean(messages)

    def test_pii_masks_every_assistant_text_message(self):
        messages = [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "I can see a card (credit_card_1234)."},
            {"role": "tool", "id": "t1", "content": "card credit_card_1234 ok"},
            {"role": "assistant", "content": "Refund to credit_card_1234."},
        ]
        masked, change_log = apply_assistant_replacements(
            messages, {"credit_card_1234": "your credit card on file"}
        )
        self.assertNotIn("credit_card_1234", masked[1]["content"])
        self.assertNotIn("credit_card_1234", masked[3]["content"])
        self.assertIn("your credit card on file", masked[1]["content"])
        self.assertIn("your credit card on file", masked[3]["content"])
        self.assertIn("credit_card_1234", masked[2]["content"])
        self.assertEqual(len(change_log), 2)
        self.assertTrue(all(entry["category"] == "pii_masking" for entry in change_log))

    def test_pii_never_rewrites_tool_call_arguments(self):
        messages = [
            {
                "role": "assistant",
                "content": "paying with credit_card_1234",
                "tool_calls": [
                    {
                        "id": "c1",
                        "name": "pay",
                        "arguments": {"payment_method_id": "credit_card_1234"},
                    }
                ],
            },
            {"role": "tool", "id": "c1", "content": "ok"},
        ]
        masked, _ = apply_assistant_replacements(
            messages, {"credit_card_1234": "your credit card on file"}
        )
        self.assertEqual(
            masked[0]["tool_calls"][0]["arguments"]["payment_method_id"],
            "credit_card_1234",
        )

    def test_validate_structure_accepts_corrected(self):
        corrected, _ = serialize_and_clean(source_messages())
        validate_structure(corrected)

    def test_validate_structure_rejects_multi_call_turn(self):
        messages = [
            {"role": "user", "content": "go"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {"id": "a", "name": "t", "arguments": {}},
                    {"id": "b", "name": "t", "arguments": {}},
                ],
            },
            {"role": "tool", "id": "a", "content": "ok"},
            {"role": "tool", "id": "b", "content": "ok"},
        ]
        with self.assertRaises(ValueError):
            validate_structure(messages)

    def test_validate_structure_rejects_text_tool_mixing(self):
        messages = [
            {"role": "user", "content": "go"},
            {"role": "assistant", "content": "let me", "tool_calls": [{"id": "a", "name": "t", "arguments": {}}]},
            {"role": "tool", "id": "a", "content": "ok"},
        ]
        with self.assertRaises(ValueError):
            validate_structure(messages)


if __name__ == "__main__":
    unittest.main()
