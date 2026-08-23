from __future__ import annotations

import unittest
import json
from pathlib import Path

from src.evaluation.final_holdout_rebuild_cost import build_cost_analysis


class FinalHoldoutRebuildCostTests(unittest.TestCase):
    def test_current_frozen_inputs_have_expected_cost_profile(self) -> None:
        project = Path(__file__).resolve().parents[1]
        upstream = Path(r"D:\tau2-bench")
        audit_path = (
            project
            / "_local_private_runs/final_holdout_contamination_20260823/v1/audit.json"
        )
        dataset_path = (
            project
            / "_local_private_runs/merged_sft_v2_owner_corrected/release_v2/sft_dataset.jsonl"
        )
        tasks_path = upstream / "data/tau2/domains/retail/tasks.json"
        if (
            not audit_path.is_file()
            or not dataset_path.is_file()
            or not tasks_path.is_file()
        ):
            self.skipTest("frozen local audit, SFT data, or tau2 data is unavailable")

        result = build_cost_analysis(
            audit=json.loads(audit_path.read_text(encoding="utf-8-sig")),
            sft_rows=[
                json.loads(line)
                for line in dataset_path.read_text(encoding="utf-8-sig").splitlines()
                if line.strip()
            ],
            tasks=json.loads(tasks_path.read_text(encoding="utf-8-sig")),
        )

        self.assertEqual(result["summary"]["candidate_count"], 23)
        self.assertEqual(result["summary"]["current_sft_row_count"], 53)
        self.assertEqual(result["summary"]["current_train_row_count"], 40)
        self.assertEqual(result["summary"]["current_validation_row_count"], 13)
        self.assertEqual(result["summary"]["minimum_individual_exclusion_rows"], 1)
        self.assertEqual(result["summary"]["maximum_individual_exclusion_rows"], 6)
        self.assertEqual(
            result["summary"]["candidate_counts_by_stratum"],
            {
                "exchange": 6,
                "handoff": 1,
                "mixed_cancel_return": 2,
                "modify": 9,
                "query": 1,
                "return": 4,
            },
        )

    def test_counts_affected_rows_splits_and_tasks(self) -> None:
        result = build_cost_analysis(
            audit={
                "rows": [
                    {
                        "task_id": "10",
                        "used_in_development_evaluation": False,
                        "overlapping_entity_groups": ["order_id:A"],
                    },
                    {
                        "task_id": "11",
                        "used_in_development_evaluation": True,
                        "overlapping_entity_groups": [],
                    },
                ]
            },
            sft_rows=[
                {"task_id": "1", "split": "TRAIN", "group_ids": ["order_id:A"]},
                {
                    "task_id": "2",
                    "split": "VALIDATION",
                    "group_ids": ["order_id:A", "user_id:B"],
                },
                {"task_id": "3", "split": "TRAIN", "group_ids": ["order_id:C"]},
            ],
            tasks=[
                {
                    "id": 10,
                    "evaluation_criteria": {
                        "actions": [{"name": "cancel_pending_order"}]
                    },
                },
                {"id": 11, "evaluation_criteria": {"actions": []}},
            ],
        )

        self.assertEqual(result["summary"]["candidate_count"], 1)
        self.assertEqual(result["summary"]["minimum_individual_exclusion_rows"], 2)
        row = result["rows"][0]
        self.assertEqual(row["stratum"], "cancel")
        self.assertEqual(row["affected_train_row_count"], 1)
        self.assertEqual(row["affected_validation_row_count"], 1)
        self.assertEqual(row["affected_sft_task_ids"], ["1", "2"])
        self.assertEqual(row["remaining_sft_row_count"], 1)

    def test_classifies_mixed_family_without_using_model_results(self) -> None:
        result = build_cost_analysis(
            audit={
                "rows": [
                    {
                        "task_id": "20",
                        "used_in_development_evaluation": False,
                        "overlapping_entity_groups": [],
                    }
                ]
            },
            sft_rows=[{"task_id": "1", "split": "TRAIN", "group_ids": ["order_id:A"]}],
            tasks=[
                {
                    "id": 20,
                    "evaluation_criteria": {
                        "actions": [
                            {"name": "return_delivered_order_items"},
                            {"name": "modify_user_address"},
                        ]
                    },
                }
            ],
        )

        self.assertEqual(result["rows"][0]["stratum"], "mixed_modify_return")
        self.assertEqual(result["rows"][0]["affected_sft_row_count"], 0)
        self.assertFalse(result["decision"]["final_holdout_selected"])
        self.assertTrue(result["decision"]["portfolio_cost_requires_set_union"])

    def test_rejects_missing_candidate_task(self) -> None:
        with self.assertRaisesRegex(ValueError, "missing from upstream tasks"):
            build_cost_analysis(
                audit={
                    "rows": [
                        {
                            "task_id": "99",
                            "used_in_development_evaluation": False,
                            "overlapping_entity_groups": [],
                        }
                    ]
                },
                sft_rows=[
                    {
                        "task_id": "1",
                        "split": "TRAIN",
                        "group_ids": ["order_id:A"],
                    }
                ],
                tasks=[{"id": 1, "evaluation_criteria": {"actions": []}}],
            )


if __name__ == "__main__":
    unittest.main()
