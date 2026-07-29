from __future__ import annotations

import unittest
from pathlib import Path

from src.evaluation.db_diff import analyze_db_diff
from src.evaluation.failure_attributor import attribute_failure
from src.evaluation.nl_checker import check_recorded_nl_assertions
from src.evaluation.replay_evaluator import replay_results_artifact
from src.evaluation.taxonomy import build_three_layer_taxonomy


PROJECT = Path(r"D:\PolicyAgent-PostTrain")
TAU2 = Path(r"D:\tau2-bench")
BASELINE = (
    PROJECT
    / "experiments"
    / "20260722_110504_retail_baseline20_trial1_deepseek"
)


@unittest.skipUnless(TAU2.exists() and BASELINE.exists(), "local Tau2 artifacts absent")
class ReplayIntegrationTests(unittest.TestCase):
    def evaluate(self, task_id: str):
        replay = replay_results_artifact(
            BASELINE / f"task_{task_id}" / "returned_results.json",
            tau2_root=TAU2,
        )
        diff = analyze_db_diff(
            replay.initial_state, replay.agent_state, replay.gold_state
        )
        nl = check_recorded_nl_assertions(replay.task, replay.simulation)
        attribution = attribute_failure(replay, diff, nl)
        taxonomy = build_three_layer_taxonomy(
            official_signal={
                "db_match": replay.db_match,
                "nl_match": nl.nl_match,
            },
            detailed_root_causes=attribution["root_causes"],
            state_diff=diff.to_dict(),
        )
        return replay, diff, nl, attribution, taxonomy

    def test_task_59_detects_static_gold_conflict(self) -> None:
        replay, diff, nl, attribution, taxonomy = self.evaluate("59")
        self.assertFalse(replay.db_match)
        self.assertFalse(nl.nl_match)
        codes = {item["code"] for item in attribution["root_causes"]}
        self.assertIn("golden_mismatch", codes)
        self.assertIn("dataset_issue", codes)
        self.assertTrue(diff.flags["extra_cancel"])
        self.assertTrue(diff.flags["missing_cancel"])
        self.assertEqual(
            taxonomy["official_signal"], ["db_mismatch", "nl_failure"]
        )
        self.assertIn("dataset_alignment_error", taxonomy["root_cause"])
        self.assertTrue(taxonomy["quarantine_recommended"])

    def test_task_95_detects_inventory_variant_misread(self) -> None:
        replay, diff, _, attribution, taxonomy = self.evaluate("95")
        self.assertFalse(replay.db_match)
        self.assertTrue(diff.flags["missing_exchange"])
        codes = {item["code"] for item in attribution["root_causes"]}
        self.assertIn("variant_understanding_failure", codes)
        self.assertIn("variant_error", taxonomy["root_cause"])
        self.assertIn("incomplete_customer_request", taxonomy["business_impact"])

    def test_task_98_localizes_payment_and_scope_findings(self) -> None:
        replay, diff, nl, attribution, taxonomy = self.evaluate("98")
        self.assertFalse(replay.db_match)
        self.assertTrue(nl.nl_match)
        self.assertTrue(diff.flags["wrong_payment"])
        codes = {item["code"] for item in attribution["root_causes"]}
        self.assertIn("wrong_payment_method", codes)
        self.assertIn("scope_confirmation_failure", codes)
        self.assertIn("payment_error", taxonomy["root_cause"])
        self.assertIn("scope_error", taxonomy["root_cause"])

    def test_task_107_detects_policy_tool_gap(self) -> None:
        replay, diff, nl, attribution, taxonomy = self.evaluate("107")
        self.assertFalse(replay.db_match)
        self.assertTrue(nl.nl_match)
        self.assertTrue(diff.flags["wrong_variant"])
        codes = {item["code"] for item in attribution["root_causes"]}
        self.assertIn("policy_violation", codes)
        self.assertIn("policy_error", taxonomy["root_cause"])
        self.assertIn("policy_risk", taxonomy["business_impact"])


if __name__ == "__main__":
    unittest.main()
