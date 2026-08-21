import unittest
import json

from src.training.run_teacher_sft import (
    build_chat,
    to_chat_messages,
    tokenize_row,
    _assistant_nll,
    _truncate_batch_tail,
    save_merged_model_enabled,
)

try:
    import torch as _torch
except ImportError:
    _torch = None


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


def _chunk_ids(text: str) -> list[int]:
    return list(range((len(text) + 1) // 2))


class QwenStyleTokenizer:
    """Fake tokenizer that reproduces the real Qwen3 failure mode.

    Renders ``<|im_start|>role\n...<|im_end|>\n`` blocks with a content-only
    guard and assistant ``<tool_call>`` blocks. The native
    ``return_assistant_tokens_mask`` key is unavailable (the template has no
    ``{% generation %}`` block), so ``tokenize_row`` must fall back to the
    render-diff mask. ``__call__`` tokenizes into 2-char chunks and returns
    ``offset_mapping`` for span-to-token mapping.
    """

    def _render(self, chat):
        parts = []
        for message in chat:
            role = message["role"]
            if role == "system":
                parts.append("<|im_start|>system\n")
                if message.get("content"):
                    parts.append(message["content"])
                parts.append("<|im_end|>\n")
            elif role == "user":
                parts.append("<|im_start|>user\n")
                parts.append(message.get("content") or "")
                parts.append("<|im_end|>\n")
            elif role == "assistant":
                parts.append("<|im_start|>assistant\n")
                if message.get("content"):
                    parts.append(message["content"])
                for call in message.get("tool_calls") or []:
                    parts.append("<tool_call>\n")
                    parts.append(call["function"]["name"])
                    parts.append(call["function"]["arguments"])
                    parts.append("\n</tool_call>\n")
                parts.append("<|im_end|>\n")
            elif role == "tool":
                parts.append("<|im_start|>tool\n")
                parts.append(message.get("content") or "")
                parts.append("<|im_end|>\n")
        return "".join(parts)

    def apply_chat_template(self, chat, tools=None, tokenize=False, **kwargs):
        rendered = self._render(chat)
        if not tokenize:
            return rendered
        return {"input_ids": _chunk_ids(rendered)}

    def __call__(
        self,
        text,
        add_special_tokens=False,
        return_offsets_mapping=True,
        truncation=False,
        padding=False,
    ):
        ids = _chunk_ids(text)
        if not return_offsets_mapping:
            return {"input_ids": ids}
        offsets = [
            (index * 2, min(index * 2 + 2, len(text))) for index in range(len(ids))
        ]
        return {"input_ids": ids, "offset_mapping": offsets}


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


class TokenizeRowFallbackTest(unittest.TestCase):
    """Render-diff mask path for templates without a generation block."""

    def _assert_mask_matches_assistant_chars(self, tokenizer, chat, batch):
        full = tokenizer.apply_chat_template(chat, tokenize=False)
        offsets = tokenizer(full)["offset_mapping"]
        labels = batch["labels"]
        assistant_chars = set()
        start = 0
        while True:
            marker = full.find("<|im_start|>assistant\n", start)
            if marker < 0:
                break
            content_start = marker + len("<|im_start|>assistant\n")
            ender = full.find("<|im_end|>", content_start)
            assistant_chars.update(range(content_start, ender))
            start = ender
        assistant_shell_chars = set()
        start = 0
        while True:
            marker = full.find("<|im_start|>assistant\n", start)
            if marker < 0:
                break
            ender = full.find("<|im_end|>", marker)
            block_end = ender + len("<|im_end|>\n")
            content_start = marker + len("<|im_start|>assistant\n")
            block_chars = set(range(marker, block_end))
            content_chars = set(range(content_start, ender))
            assistant_chars |= content_chars
            assistant_shell_chars |= block_chars - content_chars
            start = block_end
        forbidden_chars = set(range(len(full))) - assistant_chars - assistant_shell_chars
        for index, (char_start, char_end) in enumerate(offsets):
            token_chars = set(range(char_start, char_end))
            if token_chars <= assistant_chars:
                self.assertNotEqual(
                    labels[index], -100, f"token {index} under-masked"
                )
            elif token_chars <= forbidden_chars:
                self.assertEqual(
                    labels[index], -100, f"token {index} over-masked"
                )
            # tokens straddling an assistant block boundary may be either side

    def test_render_diff_marks_only_assistant_content(self):
        tokenizer = QwenStyleTokenizer()
        chat = [
            {"role": "system", "content": "policy text"},
            {"role": "user", "content": "user request one"},
            {"role": "assistant", "content": "assistant reply one"},
            {"role": "tool", "content": "tool result one", "tool_call_id": "c1"},
            {"role": "assistant", "content": "assistant reply two"},
        ]
        batch = tokenize_row(tokenizer, chat, tools=[], max_length=8192)
        self.assertEqual(len(batch["input_ids"]), len(batch["labels"]))
        self.assertEqual(batch["attention_mask"], [1] * len(batch["input_ids"]))
        self.assertIn(-100, batch["labels"])
        self._assert_mask_matches_assistant_chars(tokenizer, chat, batch)

    def test_render_diff_masks_tool_call_blocks(self):
        tokenizer = QwenStyleTokenizer()
        chat = [
            {"role": "system", "content": "policy text"},
            {"role": "user", "content": "please look up my order"},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call_01",
                        "type": "function",
                        "function": {
                            "name": "get_order_details",
                            "arguments": '{"order_id": 7}',
                        },
                    }
                ],
            },
            {"role": "tool", "content": "order found", "tool_call_id": "call_01"},
            {"role": "assistant", "content": "here it is"},
        ]
        batch = tokenize_row(tokenizer, chat, tools=[], max_length=8192)
        self._assert_mask_matches_assistant_chars(tokenizer, chat, batch)
        self.assertIn(-100, batch["labels"])

    def test_render_diff_handles_duplicate_assistant_content(self):
        tokenizer = QwenStyleTokenizer()
        chat = [
            {"role": "system", "content": "policy text"},
            {"role": "user", "content": "u1"},
            {"role": "assistant", "content": "same reply"},
            {"role": "user", "content": "u2"},
            {"role": "assistant", "content": "same reply"},
        ]
        batch = tokenize_row(tokenizer, chat, tools=[], max_length=8192)
        self._assert_mask_matches_assistant_chars(tokenizer, chat, batch)

    def test_render_diff_fails_closed_on_nonlocal_rendering(self):
        class NonlocalTokenizer(QwenStyleTokenizer):
            def apply_chat_template(self, chat, tools=None, tokenize=False, **kwargs):
                rendered = QwenStyleTokenizer._render(self, chat)
                if any(
                    m.get("role") == "assistant"
                    and not m.get("content")
                    and not m.get("tool_calls")
                    for m in chat
                ):
                    rendered = rendered.replace(
                        "<|im_start|>user\n", "<|im_start|>user\nEXTRA\n", 1
                    )
                if not tokenize:
                    return rendered
                return {"input_ids": _chunk_ids(rendered)}

        tokenizer = NonlocalTokenizer()
        chat = [
            {"role": "system", "content": "policy text"},
            {"role": "user", "content": "user request one"},
            {"role": "assistant", "content": "assistant reply one"},
        ]
        with self.assertRaises(RuntimeError):
            tokenize_row(tokenizer, chat, tools=[], max_length=8192)

class ValidationTruncationTest(unittest.TestCase):
    def test_truncate_batch_tail_keeps_tail(self):
        batch = {
            "input_ids": [0, 1, 2, 3, 4, 5],
            "attention_mask": [1, 1, 1, 1, 1, 1],
            "labels": [-100, 1, -100, 3, -100, 5],
        }
        out = _truncate_batch_tail(batch, 3)
        self.assertEqual(out["input_ids"], [3, 4, 5])
        self.assertEqual(out["attention_mask"], [1, 1, 1])
        self.assertEqual(out["labels"], [3, -100, 5])

    def test_truncate_batch_tail_noop_within_budget(self):
        batch = {
            "input_ids": [1, 2],
            "attention_mask": [1, 1],
            "labels": [-100, 2],
        }
        self.assertEqual(_truncate_batch_tail(batch, 4), batch)

    def test_truncate_batch_tail_rejects_bad_budget(self):
        batch = {
            "input_ids": [1],
            "attention_mask": [1],
            "labels": [-100],
        }
        with self.assertRaises(ValueError):
            _truncate_batch_tail(batch, 0)


class MergedArtifactPolicyTest(unittest.TestCase):
    def test_default_preserves_historical_merged_artifact(self):
        self.assertTrue(save_merged_model_enabled({}))

    def test_plateau_config_can_disable_merged_artifact(self):
        self.assertFalse(
            save_merged_model_enabled({"artifacts": {"save_merged_model": False}})
        )

    def test_rejects_non_boolean_value(self):
        with self.assertRaisesRegex(ValueError, "must be a boolean"):
            save_merged_model_enabled({"artifacts": {"save_merged_model": "no"}})


@unittest.skipUnless(_torch is not None, "torch not installed")
class AssistantNllTest(unittest.TestCase):
    def test_matches_cross_entropy_on_valid_positions(self):
        torch = _torch
        logits = torch.randn(1, 5, 6, dtype=torch.float32)
        labels = torch.tensor([[2, -100, 4, -100, 5]])
        expected = 0.0
        expected_count = 0
        for shift_index in range(4):
            target = int(labels[0, shift_index + 1])
            if target == -100:
                continue
            expected += float(
                torch.nn.functional.cross_entropy(
                    logits[0, shift_index].unsqueeze(0),
                    torch.tensor([target]),
                ).item()
            )
            expected_count += 1
        nll, count = _assistant_nll(logits, labels, torch)
        self.assertEqual(count, expected_count)
        self.assertAlmostEqual(nll, expected, places=5)

    def test_no_valid_positions_returns_zero(self):
        torch = _torch
        logits = torch.zeros(1, 3, 4)
        labels = torch.full((1, 3), -100, dtype=torch.long)
        self.assertEqual(_assistant_nll(logits, labels, torch), (0.0, 0))


if __name__ == "__main__":
    unittest.main()
