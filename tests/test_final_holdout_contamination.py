from __future__ import annotations

import json
import unittest
from pathlib import Path

from src.evaluation.final_holdout_contamination import build_audit


class FinalHoldoutContaminationTests(unittest.TestCase):
    def test_current_frozen_inputs_have_no_strict_candidate(self) -> None:
        project = Path(__file__).resolve().parents[1]
        upstream = Path(r"D:\tau2-bench")
        tasks_path = upstream / "data/tau2/domains/retail/tasks.json"
        split_path = upstream / "data/tau2/domains/retail/split_tasks.json"
        if not tasks_path.is_file() or not split_path.is_file():
            self.skipTest("pinned local tau2 data is unavailable")

        result = build_audit(
            tasks=json.loads(tasks_path.read_text(encoding="utf-8-sig")),
            split=json.loads(split_path.read_text(encoding="utf-8-sig")),
            training_groups=json.loads(
                (
                    project
                    / "data/retail_teacher_eval_v2/teacher_sft_entity_groups.json"
                ).read_text(encoding="utf-8-sig")
            ),
            development_config=json.loads(
                (project / "configs/retail_teacher_eval_v1.json").read_text(
                    encoding="utf-8-sig"
                )
            ),
        )

        self.assertEqual(result["summary"]["official_test_task_count"], 40)
        self.assertEqual(result["summary"]["development_used_test_task_count"], 17)
        self.assertEqual(result["summary"]["development_unseen_test_task_count"], 23)
        self.assertEqual(
            result["summary"]["development_unseen_with_sft_entity_overlap_count"],
            23,
        )
        self.assertEqual(result["summary"]["strict_holdout_candidate_count"], 0)

    def test_blocks_development_use_training_task_and_entity_overlap(self) -> None:
        result = build_audit(
            tasks=[
                {"id": 1, "goal": "order A"},
                {"id": 2, "goal": "clean task"},
                {"id": 3, "goal": "another order A"},
            ],
            split={"test": [1, 2, 3]},
            training_groups={
                "teacher_task_ids": [1],
                "entity_groups": ["order_id:A"],
            },
            development_config={"tasks": [{"task_id": "2", "source": "test_clean"}]},
        )

        self.assertEqual(result["summary"]["official_test_task_count"], 3)
        self.assertEqual(result["summary"]["strict_holdout_candidate_count"], 0)
        self.assertTrue(result["decision"]["rebuild_training_split_required"])
        by_id = {row["task_id"]: row for row in result["rows"]}
        self.assertIn("TASK_USED_IN_SFT_DATA", by_id["1"]["blockers"])
        self.assertIn("USED_IN_DEVELOPMENT_EVALUATION", by_id["2"]["blockers"])
        self.assertIn("ENTITY_OVERLAP_WITH_SFT_DATA", by_id["3"]["blockers"])

    def test_keeps_clean_unseen_task_as_candidate_but_not_frozen(self) -> None:
        result = build_audit(
            tasks=[{"id": 1, "goal": "order A"}, {"id": 4, "goal": "order B"}],
            split={"test": [1, 4]},
            training_groups={
                "teacher_task_ids": [1],
                "entity_groups": ["order_id:A"],
            },
            development_config={"tasks": []},
        )

        candidate = next(row for row in result["rows"] if row["task_id"] == "4")
        self.assertTrue(candidate["strict_holdout_candidate"])
        self.assertEqual(result["summary"]["strict_holdout_candidate_count"], 1)
        self.assertFalse(result["decision"]["final_holdout_frozen"])
        self.assertFalse(result["decision"]["rebuild_training_split_required"])

    def test_rejects_development_rows_outside_official_test(self) -> None:
        with self.assertRaisesRegex(ValueError, "outside the official test split"):
            build_audit(
                tasks=[{"id": 1, "goal": "clean"}],
                split={"test": [1]},
                training_groups={
                    "teacher_task_ids": [],
                    "entity_groups": ["order_id:A"],
                },
                development_config={
                    "tasks": [{"task_id": "2", "source": "test_clean"}]
                },
            )


if __name__ == "__main__":
    unittest.main()
