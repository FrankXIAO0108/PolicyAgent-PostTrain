from __future__ import annotations

import unittest
from pathlib import Path

from src.training.evaluate_adversarial_holdout import (
    aggregate_scores,
    parse_model_spec,
    score_completion,
    validate_inputs,
)


ALLOWED = {"get_order", "request_confirmation"}
ROW = {
    "scenario_id": "adv-test-001",
    "category": "read_only_query",
    "expected_action": '{"tool":"get_order","arguments":{"order_id":"ADV-1"}}',
}


class AdversarialEvaluationScoringTests(unittest.TestCase):
    def test_strict_exact_action_passes_all_metrics(self) -> None:
        score = score_completion(ROW, ROW["expected_action"], ALLOWED)
        for field in (
            "extractable_json",
            "strict_json_object",
            "schema_valid",
            "registered_tool",
            "tool_match",
            "arguments_match",
            "exact_action_match",
            "strict_exact_action_match",
        ):
            self.assertTrue(score[field], field)

    def test_trailing_text_is_extractable_but_not_strict(self) -> None:
        score = score_completion(ROW, ROW["expected_action"] + " 处理完成", ALLOWED)
        self.assertTrue(score["extractable_json"])
        self.assertTrue(score["exact_action_match"])
        self.assertFalse(score["strict_json_object"])
        self.assertFalse(score["schema_valid"])
        self.assertFalse(score["strict_exact_action_match"])

    def test_unknown_tool_fails_registry_and_exact_match(self) -> None:
        completion = '{"tool":"get_order_v9","arguments":{"order_id":"ADV-1"}}'
        score = score_completion(ROW, completion, ALLOWED)
        self.assertTrue(score["strict_json_object"])
        self.assertFalse(score["schema_valid"])
        self.assertFalse(score["registered_tool"])
        self.assertFalse(score["tool_match"])

    def test_aggregation_reports_category_and_format_gap(self) -> None:
        rows = [
            score_completion(ROW, ROW["expected_action"], ALLOWED),
            score_completion(ROW, ROW["expected_action"] + " trailing", ALLOWED),
        ]
        report = aggregate_scores(rows)
        self.assertEqual(report["overall"]["rows"], 2)
        self.assertEqual(report["overall"]["extractable_json_rate"], 1.0)
        self.assertEqual(report["overall"]["strict_json_object_rate"], 0.5)
        self.assertEqual(report["overall"]["format_gap_count"], 1)
        self.assertIn("read_only_query", report["by_category"])


class AdversarialEvaluationPreflightTests(unittest.TestCase):
    def test_model_spec_parser(self) -> None:
        self.assertEqual(parse_model_spec("sft=/models/sft"), ("sft", "/models/sft"))

    def test_repository_v2_inputs_pass_when_dirty_is_allowed(self) -> None:
        models = {
            "base": "Qwen/Qwen2.5-0.5B-Instruct",
            "sft": "/models/sft",
            "dpo": "/models/dpo",
            "grpo": "/models/grpo",
        }
        preflight = validate_inputs(
            Path("configs/posttrain_adversarial_eval_v2.json").resolve(),
            models,
            allow_dirty=True,
        )
        self.assertEqual(len(preflight["rows"]), 48)
        self.assertEqual(preflight["holdout_sha256"], "F5F6152BAC72899AFFF0873509095401DC1D6FC40BF2BAA64AD95DFC07537584")


if __name__ == "__main__":
    unittest.main()
