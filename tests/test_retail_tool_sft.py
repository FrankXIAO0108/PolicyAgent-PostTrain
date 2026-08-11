from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from src.training.retail_tool_sft_data import build_dataset, build_rows, validate_rows
from src.training.run_retail_tool_sft import parse_tool_call, tool_call_completion


class RetailToolSftDataTests(unittest.TestCase):
    def test_tool_call_completion_round_trip(self) -> None:
        expected = {
            "name": "find_user_id_by_email",
            "arguments": {"email": "holdout@example.test"},
        }
        self.assertEqual(parse_tool_call(tool_call_completion(expected)), expected)

    def test_tool_call_parser_rejects_text_and_malformed_payloads(self) -> None:
        self.assertIsNone(parse_tool_call("Please provide your email."))
        self.assertIsNone(parse_tool_call("<tool_call>{bad json}</tool_call>"))
        self.assertIsNone(
            parse_tool_call(
                '<tool_call>{"name":"respond_to_user","arguments":[]}</tool_call>'
            )
        )

    def test_curriculum_a_has_frozen_size_and_disjoint_holdout(self) -> None:
        train = build_rows("train", 16)
        holdout = build_rows("holdout", 4)
        self.assertEqual(len(train), 80)
        self.assertEqual(len(holdout), 20)
        self.assertTrue(validate_rows(train, holdout)["passed"])
        self.assertEqual(
            {row["expected_call"]["name"] for row in train},
            {
                "respond_to_user",
                "find_user_id_by_email",
                "find_user_id_by_name_zip",
            },
        )

    def test_builder_writes_hash_bound_manifest_without_business_claim(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = build_dataset(root)
            self.assertEqual(manifest["files"]["sft"]["rows"], 80)
            self.assertEqual(manifest["files"]["holdout"]["rows"], 20)
            self.assertFalse(
                manifest["claims"]["business_improvement_claim_allowed"]
            )
            stored = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(stored, manifest)


if __name__ == "__main__":
    unittest.main()
