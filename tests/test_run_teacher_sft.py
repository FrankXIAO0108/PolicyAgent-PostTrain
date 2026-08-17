import unittest
import json

from src.training.run_teacher_sft import build_chat, to_chat_messages, tokenize_row


class FakeTokenizer:
    """Deterministic fake: one token per message; assistant messages trainable."""

    def apply_chat_template(
        self,
        chat,
        tools=None,
        tokenize=False,
        return_dict=False,
        return_assistant_tokens_mask=False,
        add_generation_prompt=False,
        truncation=False,
        max_length=None,
    ):
        input_ids = list(range(len(chat)))
        mask = [m["role"] == "assistant" for m in chat]
        if truncation and max_length is not None and len(input_ids) >= max_length:
            input_ids = input_ids[: max_length - 1]
            mask = mask[: max_length - 1]
        return {"input_ids": input_ids, "assistant_tokens_mask": mask}


class ToChatMessagesTest(unittest.TestCase):
    def test_plain_turns_are_preserved(self):
        messages = [
            {"role": "assistant", "content": "Hi", "tool_calls": [], "loss_mask": 1},
            {"role": "user", "content": "Hello", "tool_calls": [], "loss_mask": 0},
        ]
        chat = to_chat_messages(messages)
        self.assertEqual(
            chat,
            [
                {"role": "assistant", "content": "Hi"},
                {"role": "user", "content": "Hello"},
            ],
        )

    def test_tool_call_converted_to_openai_format(self):
        messages = [
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "call_01",
                        "name": "find_user_id_by_name_zip",
                        "arguments": {"zip": "92133"},
                        "requestor": "assistant",
                    }
                ],
                "loss_mask": 1,
            },
            {"role": "tool", "content": "ivan_hernandez_6923", "tool_calls": [], "loss_mask": 0},
        ]
        chat = to_chat_messages(messages)
        self.assertEqual(len(chat), 2)
        assistant = chat[0]
        self.assertIsNone(assistant["content"])
        call = assistant["tool_calls"][0]
        self.assertEqual(call["id"], "call_01")
        self.assertEqual(call["type"], "function")
        self.assertEqual(call["function"]["name"], "find_user_id_by_name_zip")
        self.assertEqual(json.loads(call["function"]["arguments"]), {"zip": "92133"})
        self.assertEqual(chat[1]["tool_call_id"], "call_01")

    def test_multiple_tool_calls_rejected(self):
        messages = [
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {"id": "a", "name": "x", "arguments": {}},
                    {"id": "b", "name": "y", "arguments": {}},
                ],
                "loss_mask": 1,
            }
        ]
        with self.assertRaises(ValueError):
            to_chat_messages(messages)

    def test_unsupported_role_rejected(self):
        with self.assertRaises(ValueError):
            to_chat_messages([{"role": "robot", "content": "x", "tool_calls": [], "loss_mask": 0}])


class BuildChatTest(unittest.TestCase):
    def test_prepends_system_policy(self):
        row = {
            "system_policy": "Frozen policy.",
            "messages": [
                {"role": "user", "content": "Hi", "tool_calls": [], "loss_mask": 0},
                {"role": "assistant", "content": "Hello", "tool_calls": [], "loss_mask": 1},
            ],
        }
        chat = build_chat(row)
        self.assertEqual(chat[0], {"role": "system", "content": "Frozen policy."})
        self.assertEqual([m["role"] for m in chat], ["system", "user", "assistant"])

    def test_missing_policy_rejected(self):
        row = {
            "system_policy": "  ",
            "messages": [
                {"role": "assistant", "content": "x", "tool_calls": [], "loss_mask": 1}
            ],
        }
        with self.assertRaises(ValueError):
            build_chat(row)


class TokenizeRowTest(unittest.TestCase):
    def test_labels_mask_non_assistant(self):
        tokenizer = FakeTokenizer()
        chat = [
            {"role": "system", "content": "policy"},
            {"role": "user", "content": "u1"},
            {"role": "assistant", "content": "a1"},
            {"role": "tool", "content": "t1", "tool_call_id": "c1"},
            {"role": "assistant", "content": "a2"},
        ]
        batch = tokenize_row(tokenizer, chat, tools=[], max_length=8192)
        self.assertEqual(batch["input_ids"], [0, 1, 2, 3, 4])
        self.assertEqual(batch["labels"], [-100, -100, 2, -100, 4])
        self.assertEqual(batch["attention_mask"], [1, 1, 1, 1, 1])

    def test_no_assistant_tokens_rejected(self):
        tokenizer = FakeTokenizer()
        chat = [
            {"role": "system", "content": "policy"},
            {"role": "user", "content": "u1"},
        ]
        with self.assertRaises(ValueError):
            tokenize_row(tokenizer, chat, tools=[], max_length=8192)

    def test_oversized_sequence_fails_closed(self):
        class HugeTokenizer(FakeTokenizer):
            def apply_chat_template(self, chat, **kwargs):
                return {
                    "input_ids": list(range(50)),
                    "assistant_tokens_mask": [i % 2 == 1 for i in range(50)],
                }

        chat = [
            {"role": "user", "content": "u"},
            {"role": "assistant", "content": "a"},
        ]
        with self.assertRaises(ValueError):
            tokenize_row(HugeTokenizer(), chat, tools=[], max_length=10)


if __name__ == "__main__":
    unittest.main()
