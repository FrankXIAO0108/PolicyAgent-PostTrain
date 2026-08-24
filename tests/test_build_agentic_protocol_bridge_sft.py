import unittest

from src.training.build_agentic_protocol_bridge_sft import (
    TERMINAL_COMPLETION,
    transform_messages,
)
from src.training.run_retail_agentic_grpo import (
    wrap_retail_policy_for_agentic_protocol,
)


def _assistant_text(content):
    return {"role": "assistant", "content": content, "tool_calls": [], "loss_mask": 1}


def _user(content):
    return {"role": "user", "content": content, "tool_calls": [], "loss_mask": 0}


def _assistant_call(call_id, name, arguments):
    return {
        "role": "assistant",
        "content": "",
        "tool_calls": [{"id": call_id, "name": name, "arguments": arguments}],
        "loss_mask": 1,
    }


def _tool(content):
    return {"role": "tool", "content": content, "tool_calls": [], "loss_mask": 0}


class TransformMessagesTest(unittest.TestCase):
    def test_keeps_opening_and_bridges_later_dialogue(self):
        source = [
            _assistant_text("Hi! How can I help you today?"),
            _user("Please check order O1."),
            _assistant_text("What is your email?"),
            _user("a@example.test"),
            _assistant_call("source-1", "get_order_details", {"order_id": "O1"}),
            _tool('{"status":"pending"}'),
            _assistant_text("The order is pending. Anything else?"),
            _user("No thanks. ###STOP###"),
        ]
        converted, counts = transform_messages(source, "candidate-1")
        self.assertEqual(converted[0], _user("Please check order O1."))
        self.assertEqual(converted[1]["tool_calls"][0]["name"], "respond_to_user")
        self.assertEqual(
            converted[1]["tool_calls"][0]["arguments"],
            {"message": "What is your email?"},
        )
        self.assertEqual(converted[2], _tool("a@example.test"))
        self.assertEqual(converted[3], source[4])
        self.assertEqual(converted[4], source[5])
        self.assertEqual(converted[-1], _assistant_text(TERMINAL_COMPLETION))
        self.assertEqual(counts["existing_tool_calls_preserved"], 1)
        self.assertEqual(counts["respond_to_user_calls_added"], 2)

    def test_missing_terminal_marker_fails_closed(self):
        source = [
            _assistant_text("Hi! How can I help you today?"),
            _user("Hello"),
            _assistant_text("Anything else?"),
            _user("No thanks"),
        ]
        with self.assertRaisesRegex(ValueError, "terminal STOP/TRANSFER marker"):
            transform_messages(source, "candidate-2")

    def test_initial_greeting_mismatch_fails_closed(self):
        source = [
            _assistant_text("Welcome"),
            _user("Hello"),
            _assistant_text("Goodbye"),
            _user("###STOP###"),
        ]
        with self.assertRaisesRegex(ValueError, "initial greeting mismatch"):
            transform_messages(source, "candidate-3")

    def test_call_ids_are_deterministic(self):
        source = [
            _assistant_text("Hi! How can I help you today?"),
            _user("Hello"),
            _assistant_text("Anything else?"),
            _user("###STOP###"),
        ]
        first, _ = transform_messages(source, "candidate-4")
        second, _ = transform_messages(source, "candidate-4")
        self.assertEqual(first, second)


class PromptWrapperTest(unittest.TestCase):
    def test_matches_agentic_contract(self):
        prompt = wrap_retail_policy_for_agentic_protocol("Frozen policy.")
        self.assertIn("MUST be sent through the respond_to_user tool", prompt)
        self.assertTrue(prompt.endswith("<policy>\nFrozen policy.\n</policy>"))

    def test_empty_policy_fails_closed(self):
        with self.assertRaises(ValueError):
            wrap_retail_policy_for_agentic_protocol("  ")


if __name__ == "__main__":
    unittest.main()
