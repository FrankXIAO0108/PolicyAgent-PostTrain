from __future__ import annotations

import copy
import json
import unittest
from pathlib import Path

from src.training.prepare_teacher_expansion_150 import SCHEMA_VERSION, build_plan


def protocol() -> dict:
    return {
        "schema_version": SCHEMA_VERSION,
        "retention_evidence": {"observed_retained_rate": 0.5},
        "wave_a": {
            "smoke_task_ids": [1],
            "candidates_per_task": 2,
            "temperature_ladder": [0.2, 0.6],
            "seed": 20260824,
        },
        "wave_b": {"maximum_additional_candidates_per_task": 1},
    }


def template() -> dict:
    return {
        "generation": {
            "agent": {"temperature_ladder": [0.6]},
            "candidates_per_task": 1,
            "seed": 1,
        },
        "claims": {"official_test_reserved": True},
    }


class TeacherExpansionPlanTests(unittest.TestCase):
    def test_current_frozen_inputs_produce_expected_wave_a_scope(self) -> None:
        project = Path(__file__).resolve().parents[1]
        upstream = Path(r"D:\tau2-bench")
        protocol_path = project / "configs/retail_teacher_expansion_150_v1.json"
        required = [
            protocol_path,
            upstream / "data/tau2/domains/retail/tasks.json",
            upstream / "data/tau2/domains/retail/split_tasks.json",
            project
            / "_local_private_runs/merged_sft_v2_owner_corrected/release_v2/sft_dataset.jsonl",
        ]
        if not all(path.is_file() for path in required):
            self.skipTest("frozen local SFT data or tau2 data is unavailable")

        frozen = json.loads(protocol_path.read_text(encoding="utf-8-sig"))
        current_sft_path = project / frozen["current_sft"]["path"]
        result = build_plan(
            protocol=frozen,
            tasks=json.loads(required[1].read_text(encoding="utf-8-sig")),
            split=json.loads(required[2].read_text(encoding="utf-8-sig")),
            current_sft=[
                json.loads(line)
                for line in current_sft_path.read_text(
                    encoding="utf-8-sig"
                ).splitlines()
                if line.strip()
            ],
            development=json.loads(
                (project / frozen["development_evaluation"]["path"]).read_text(
                    encoding="utf-8-sig"
                )
            ),
            final_holdout=json.loads(
                (project / frozen["final_holdout"]["path"]).read_text(
                    encoding="utf-8-sig"
                )
            ),
            template=json.loads(
                (project / frozen["generation_template"]["path"]).read_text(
                    encoding="utf-8-sig"
                )
            ),
            task_split_path="_local_private_runs/example/task_split.json",
        )

        self.assertEqual(
            result["summary"],
            {
                "upstream_train_task_count": 74,
                "excluded_development_task_count": 13,
                "eligible_task_count": 61,
                "eligible_already_present_in_current_sft": 37,
                "eligible_new_to_current_sft": 24,
                "current_sft_rows": 53,
                "wave_a_candidate_count": 122,
                "wave_a_expected_retained_rows": 65,
                "wave_a_projected_total_rows": 118,
                "wave_b_max_candidate_count": 61,
                "wave_b_expected_retained_rows": 32,
                "wave_a_plus_b_projected_total_rows": 150,
            },
        )
        self.assertEqual(result["gates"]["development_overlap_count"], 0)
        self.assertEqual(result["gates"]["final_holdout_overlap_count"], 0)

    def test_builds_task_unseen_wave_a_and_estimates_yield(self) -> None:
        result = build_plan(
            protocol=protocol(),
            tasks=[
                {"id": 1, "goal": "cancel an order"},
                {"id": 2, "goal": "return an item"},
                {"id": 3, "goal": "exchange an item"},
                {"id": 4, "goal": "final holdout"},
            ],
            split={"train": [1, 2, 3], "test": [4]},
            current_sft=[{"task_id": 1}, {"task_id": 1}],
            development={"tasks": [{"task_id": 2}]},
            final_holdout={"task_ids": [4]},
            template=template(),
            task_split_path="_local_private_runs/example/task_split.json",
        )

        summary = result["summary"]
        self.assertEqual(summary["eligible_task_count"], 2)
        self.assertEqual(summary["eligible_already_present_in_current_sft"], 1)
        self.assertEqual(summary["eligible_new_to_current_sft"], 1)
        self.assertEqual(summary["wave_a_candidate_count"], 4)
        self.assertEqual(summary["wave_a_expected_retained_rows"], 2)
        self.assertEqual(summary["wave_a_projected_total_rows"], 4)
        self.assertEqual(result["task_split"]["splits"]["generation_pool"], ["1", "3"])
        self.assertEqual(
            result["wave_a_generation_config"]["generation"]["agent"][
                "temperature_ladder"
            ],
            [0.2, 0.6],
        )
        self.assertEqual(
            [row["task_id"] for row in result["wave_a_smoke_config"]["tasks"]],
            ["1"],
        )
        self.assertFalse(result["gates"]["external_api_called"])

    def test_rejects_current_sft_overlap_with_final_holdout(self) -> None:
        with self.assertRaisesRegex(ValueError, "Current SFT data overlaps"):
            build_plan(
                protocol=protocol(),
                tasks=[{"id": 1}, {"id": 4}],
                split={"train": [1], "test": [4]},
                current_sft=[{"task_id": 4}],
                development={"tasks": []},
                final_holdout={"task_ids": [4]},
                template=template(),
                task_split_path="task_split.json",
            )

    def test_rejects_development_overlap_with_final_holdout(self) -> None:
        with self.assertRaisesRegex(ValueError, "Development evaluation overlaps"):
            build_plan(
                protocol=protocol(),
                tasks=[{"id": 1}, {"id": 4}],
                split={"train": [1], "test": [4]},
                current_sft=[],
                development={"tasks": [{"task_id": 4}]},
                final_holdout={"task_ids": [4]},
                template=template(),
                task_split_path="task_split.json",
            )

    def test_rejects_schema_drift(self) -> None:
        changed = copy.deepcopy(protocol())
        changed["schema_version"] = "unexpected"
        with self.assertRaisesRegex(ValueError, "schema version mismatch"):
            build_plan(
                protocol=changed,
                tasks=[{"id": 1}],
                split={"train": [1]},
                current_sft=[],
                development={"tasks": []},
                final_holdout={"task_ids": []},
                template=template(),
                task_split_path="task_split.json",
            )

    def test_rejects_smoke_task_outside_generation_pool(self) -> None:
        changed = copy.deepcopy(protocol())
        changed["wave_a"]["smoke_task_ids"] = [99]
        with self.assertRaisesRegex(ValueError, "smoke tasks must be"):
            build_plan(
                protocol=changed,
                tasks=[{"id": 1}],
                split={"train": [1]},
                current_sft=[],
                development={"tasks": []},
                final_holdout={"task_ids": []},
                template=template(),
                task_split_path="task_split.json",
            )


if __name__ == "__main__":
    unittest.main()
