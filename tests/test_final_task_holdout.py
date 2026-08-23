from __future__ import annotations

import unittest
import json
from pathlib import Path

from src.evaluation.final_task_holdout import build_holdout_manifest


def config(task_ids: list[int]) -> dict:
    return {
        "evaluation_scope": "TASK_UNSEEN_ENTITY_OVERLAP",
        "task_ids": task_ids,
        "selection_rule": {"uses_all_remaining_task_unseen_official_test_tasks": True},
        "prohibited_uses_before_unsealing": ["DEVELOPMENT_EVALUATION"],
        "unseal_gate": {"training_data_frozen": True},
        "claims": {"official_tau2_leaderboard_claim_allowed": False},
    }


class FinalTaskHoldoutTests(unittest.TestCase):
    def test_current_frozen_inputs_seal_all_23_task_unseen_rows(self) -> None:
        project = Path(__file__).resolve().parents[1]
        upstream = Path(r"D:\tau2-bench")
        audit_path = (
            project
            / "_local_private_runs/final_holdout_contamination_20260823/v1/audit.json"
        )
        tasks_path = upstream / "data/tau2/domains/retail/tasks.json"
        split_path = upstream / "data/tau2/domains/retail/split_tasks.json"
        config_path = project / "configs/retail_final_task_holdout_v1.json"
        required = [audit_path, tasks_path, split_path, config_path]
        if not all(path.is_file() for path in required):
            self.skipTest("frozen local audit or tau2 data is unavailable")

        result = build_holdout_manifest(
            config=json.loads(config_path.read_text(encoding="utf-8-sig")),
            tasks=json.loads(tasks_path.read_text(encoding="utf-8-sig")),
            split=json.loads(split_path.read_text(encoding="utf-8-sig")),
            audit=json.loads(audit_path.read_text(encoding="utf-8-sig")),
        )

        self.assertEqual(result["summary"]["task_count"], 23)
        self.assertEqual(result["summary"]["development_used_task_count"], 0)
        self.assertEqual(result["summary"]["sft_used_task_count"], 0)
        self.assertEqual(result["summary"]["entity_overlap_task_count"], 23)
        self.assertEqual(result["summary"]["model_result_count"], 0)
        self.assertEqual(
            result["task_ids_sha256"],
            "965897FEC4A5552C4D2A9534E64B5298C2E39751337BC8AB42E1C81F1E480A23",
        )

    def test_seals_all_task_unseen_rows_while_disclosing_entity_overlap(self) -> None:
        result = build_holdout_manifest(
            config=config([2, 3]),
            tasks=[{"id": 1, "goal": "dev"}, {"id": 2}, {"id": 3}],
            split={"test": [1, 2, 3]},
            audit={
                "rows": [
                    {
                        "task_id": "1",
                        "used_in_development_evaluation": True,
                        "task_used_in_sft_data": False,
                        "overlapping_entity_groups": [],
                    },
                    {
                        "task_id": "2",
                        "used_in_development_evaluation": False,
                        "task_used_in_sft_data": False,
                        "overlapping_entity_groups": ["order_id:A"],
                    },
                    {
                        "task_id": "3",
                        "used_in_development_evaluation": False,
                        "task_used_in_sft_data": False,
                        "overlapping_entity_groups": ["user_id:B"],
                    },
                ]
            },
        )

        self.assertEqual(result["status"], "SEALED_NOT_RUN")
        self.assertEqual(result["summary"]["task_count"], 2)
        self.assertEqual(result["summary"]["entity_overlap_task_count"], 2)
        self.assertEqual(result["summary"]["model_result_count"], 0)
        self.assertFalse(result["claims"]["official_tau2_leaderboard_claim_allowed"])

    def test_rejects_development_used_task(self) -> None:
        with self.assertRaisesRegex(ValueError, "do not equal all remaining"):
            build_holdout_manifest(
                config=config([1, 2]),
                tasks=[{"id": 1}, {"id": 2}],
                split={"test": [1, 2]},
                audit={
                    "rows": [
                        {
                            "task_id": "1",
                            "used_in_development_evaluation": True,
                            "task_used_in_sft_data": False,
                        },
                        {
                            "task_id": "2",
                            "used_in_development_evaluation": False,
                            "task_used_in_sft_data": False,
                        },
                    ]
                },
            )

    def test_rejects_omitting_an_eligible_task(self) -> None:
        with self.assertRaisesRegex(ValueError, "do not equal all remaining"):
            build_holdout_manifest(
                config=config([2]),
                tasks=[{"id": 2}, {"id": 3}],
                split={"test": [2, 3]},
                audit={
                    "rows": [
                        {
                            "task_id": "2",
                            "used_in_development_evaluation": False,
                            "task_used_in_sft_data": False,
                        },
                        {
                            "task_id": "3",
                            "used_in_development_evaluation": False,
                            "task_used_in_sft_data": False,
                        },
                    ]
                },
            )

    def test_rejects_task_outside_official_test(self) -> None:
        with self.assertRaisesRegex(ValueError, "outside official test"):
            build_holdout_manifest(
                config=config([2]),
                tasks=[{"id": 2}],
                split={"test": [1]},
                audit={
                    "rows": [
                        {
                            "task_id": "2",
                            "used_in_development_evaluation": False,
                            "task_used_in_sft_data": False,
                        }
                    ]
                },
            )


if __name__ == "__main__":
    unittest.main()
