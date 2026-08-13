from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from src.training.retail_multistep_tool_sft_data import (
    build_dataset,
    build_trajectories,
    decision_rows,
    validate_dataset,
)
from src.training.run_retail_tool_sft import render_rows
from src.training.run_retail_tool_sft import SUPPORTED_SCOPES


class _Tokenizer:
    def apply_chat_template(self, messages, **kwargs):
        self.messages = messages
        return "rendered"


class RetailMultistepToolSftTests(unittest.TestCase):
    def test_multistep_scope_is_supported_without_removing_legacy_scope(self) -> None:
        self.assertEqual(
            SUPPORTED_SCOPES,
            {
                "ISOLATED_TOOL_PROTOCOL_WARMUP",
                "ISOLATED_MULTISTEP_TOOL_SFT_WARMUP",
            },
        )

    def test_trajectories_expand_to_post_tool_decisions(self) -> None:
        trajectories = build_trajectories("train", 1)
        rows = decision_rows(trajectories)
        self.assertEqual(len(trajectories), 3)
        self.assertGreaterEqual(len(rows), 15)
        self.assertTrue(any(row["prior_tool_results"] >= 3 for row in rows))
        self.assertTrue(
            any(row["expected_call"]["name"] == "modify_pending_order_items" for row in rows)
        )

    def test_train_holdout_are_disjoint_and_not_business_gold(self) -> None:
        train = build_trajectories("train", 2)
        holdout = build_trajectories("holdout", 1)
        self.assertTrue(validate_dataset(train, holdout)["passed"])

    def test_validator_rejects_unbound_tool_result(self) -> None:
        train = build_trajectories("train", 1)
        holdout = build_trajectories("holdout", 1)
        train[0]["messages"][2]["tool_call_id"] = "wrong-call"
        with self.assertRaisesRegex(RuntimeError, "tool_call_results_id_bound"):
            validate_dataset(train, holdout)

    def test_validator_rejects_write_without_explicit_confirmation(self) -> None:
        train = build_trajectories("train", 1)
        holdout = build_trajectories("holdout", 1)
        write_row = next(
            row
            for row in train[1]["messages"]
            if row.get("role") == "tool" and str(row.get("content", "")).startswith("Yes")
        )
        write_row["content"] = "No, do not proceed."
        with self.assertRaisesRegex(RuntimeError, "all_writes_have_bound_explicit_confirmation"):
            validate_dataset(train, holdout)

    def test_renderer_preserves_tool_result_context(self) -> None:
        row = next(
            row
            for row in decision_rows(build_trajectories("train", 1))
            if row["prior_tool_results"] >= 1
        )
        tokenizer = _Tokenizer()
        rendered = render_rows([row], tokenizer, tools=[], system_prompt="policy")
        self.assertEqual(rendered[0]["prompt"], "rendered")
        self.assertTrue(any(message["role"] == "tool" for message in tokenizer.messages))
        self.assertEqual(tokenizer.messages[0]["role"], "system")

    def test_all_decision_contexts_end_before_target_assistant(self) -> None:
        rows = decision_rows(build_trajectories("train", 1))
        self.assertTrue(all(row["context_messages"][-1]["role"] in {"user", "tool"} for row in rows))

    def test_builder_hash_binds_full_trajectories_and_decisions(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            manifest = build_dataset(Path(directory))
            self.assertEqual(manifest["files"]["trajectories_train"]["rows"], 24)
            self.assertGreater(manifest["files"]["sft"]["rows"], 100)
            self.assertFalse(manifest["claims"]["business_improvement_claim_allowed"])


if __name__ == "__main__":
    unittest.main()
