import json
import tempfile
import unittest
from pathlib import Path

from src.training.teacher_candidate_decision_builder import (
    build_teacher_candidate_decisions,
)
from src.training.sft_release import sha256


def jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )


class TeacherCandidateDecisionBuilderTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)
        self.source = self.root / "source.json"
        self.source.write_text(
            json.dumps({"user_id": "u1", "order_id": "#W1", "product_id": "p1"}),
            encoding="utf-8",
        )
        self.correction = self.root / "corrected.json"
        self.correction.write_text(
            json.dumps(
                {
                    "task_id": "0",
                    "author_id": "tau2_teacher_correction_pipeline_v1",
                    "messages": [
                        {"role": "user", "content": "hi"},
                        {"role": "assistant", "content": "done"},
                    ],
                }
            ),
            encoding="utf-8",
        )
        self.validation = self.root / "validation.json"

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def write_validation(self, ready: bool = True, correction_hash: str | None = None) -> Path:
        payload = {
            "ready": ready,
            "task_id": "0",
            "correction_sha256": correction_hash or sha256(self.correction),
        }
        self.validation.write_text(json.dumps(payload), encoding="utf-8")
        return self.validation

    def reviews(
        self,
        label: str = "CORRECTION_REQUIRED",
        conflict_label: str | None = None,
    ) -> tuple[Path, Path]:
        path_a = self.root / "reviews_a.jsonl"
        path_b = self.root / "reviews_b.jsonl"
        rows_a = [
            {
                "candidate_id": "c1",
                "task_id": "0",
                "quality_label": label,
                "reviewer_id": "user_reviewer_a",
            },
            {
                "candidate_id": "c2",
                "task_id": "1",
                "quality_label": "HOLDOUT",
                "reviewer_id": "user_reviewer_a",
            },
        ]
        label_b = conflict_label or label
        rows_b = [
            {
                "candidate_id": "c1",
                "task_id": "0",
                "quality_label": label_b,
                "reviewer_id": "assistant_reviewer_b",
            },
            {
                "candidate_id": "c2",
                "task_id": "1",
                "quality_label": "HOLDOUT",
                "reviewer_id": "assistant_reviewer_b",
            },
        ]
        jsonl(path_a, rows_a)
        jsonl(path_b, rows_b)
        return path_a, path_b

    def corrections_registry(self) -> Path:
        path = self.root / "corrections.jsonl"
        jsonl(
            path,
            [
                {
                    "candidate_id": "c1",
                    "source_path": str(self.source),
                    "source_sha256": sha256(self.source),
                    "correction_path": str(self.correction),
                    "correction_sha256": sha256(self.correction),
                    "correction_validation_path": str(self.validation),
                    "correction_validation_sha256": sha256(self.validation),
                }
            ],
        )
        return path

    def split_plan(self) -> Path:
        path = self.root / "splits.jsonl"
        jsonl(
            path,
            [
                {
                    "candidate_id": "c1",
                    "split": "TRAIN",
                    "source_split": "TRAIN",
                }
            ],
        )
        return path

    def test_happy_path_maps_correction_and_holdout(self):
        self.write_validation()
        result = build_teacher_candidate_decisions(
            *self.reviews(),
            corrections_path=self.corrections_registry(),
            split_plan_path=self.split_plan(),
        )
        self.assertTrue(result["ready"], result["reasons"])
        by_id = {row["candidate_id"]: row for row in result["decisions"]}
        self.assertEqual(by_id["c1"]["disposition"], "CORRECTED_POSITIVE")
        self.assertEqual(by_id["c1"]["split"], "TRAIN")
        self.assertEqual(by_id["c1"]["status"], "SECOND_REVIEWED")
        self.assertEqual(by_id["c1"]["group_ids"], ["order_id:#W1", "product_id:p1", "user_id:u1"])
        self.assertEqual(by_id["c2"]["disposition"], "HOLDOUT")
        self.assertIsNone(by_id["c2"]["split"])
        self.assertEqual(result["counts"]["corrected_positive"], 1)
        self.assertEqual(result["counts"]["holdout"], 1)

    def test_reviewer_label_conflict_fails_closed(self):
        self.write_validation()
        result = build_teacher_candidate_decisions(
            *self.reviews(conflict_label="HOLDOUT"),
            corrections_path=self.corrections_registry(),
            split_plan_path=self.split_plan(),
        )
        self.assertFalse(result["ready"])
        self.assertTrue(any("conflict" in reason for reason in result["reasons"]))
        self.assertEqual(result["decisions"], [])

    def test_missing_correction_fails_closed(self):
        self.write_validation()
        path_a, path_b = self.reviews()
        result = build_teacher_candidate_decisions(
            path_a,
            path_b,
            split_plan_path=self.split_plan(),
        )
        self.assertFalse(result["ready"])
        self.assertTrue(any("correction registry" in reason for reason in result["reasons"]))

    def test_unready_validation_fails_closed(self):
        self.write_validation(ready=False)
        result = build_teacher_candidate_decisions(
            *self.reviews(),
            corrections_path=self.corrections_registry(),
            split_plan_path=self.split_plan(),
        )
        self.assertFalse(result["ready"])
        self.assertTrue(any("not ready" in reason for reason in result["reasons"]))

    def test_author_cannot_approve_own_artifact(self):
        self.correction.write_text(
            json.dumps(
                {
                    "task_id": "0",
                    "author_id": "user_reviewer_a",
                    "messages": [{"role": "assistant", "content": "done"}],
                }
            ),
            encoding="utf-8",
        )
        self.write_validation()
        result = build_teacher_candidate_decisions(
            *self.reviews(),
            corrections_path=self.corrections_registry(),
            split_plan_path=self.split_plan(),
        )
        self.assertFalse(result["ready"])
        self.assertTrue(any("author" in reason for reason in result["reasons"]))

    def test_group_leakage_across_splits_fails_closed(self):
        self.write_validation()
        path_a = self.root / "reviews_a_two_released.jsonl"
        jsonl(
            path_a,
            [
                {
                    "candidate_id": "c1",
                    "task_id": "0",
                    "quality_label": "CORRECTION_REQUIRED",
                    "reviewer_id": "user_reviewer_a",
                },
                {
                    "candidate_id": "c3",
                    "task_id": "2",
                    "quality_label": "CORRECTION_REQUIRED",
                    "reviewer_id": "user_reviewer_a",
                },
            ],
        )
        path_b = self.root / "reviews_b_two_released.jsonl"
        jsonl(
            path_b,
            [
                {
                    "candidate_id": "c1",
                    "task_id": "0",
                    "quality_label": "CORRECTION_REQUIRED",
                    "reviewer_id": "assistant_reviewer_b",
                },
                {
                    "candidate_id": "c3",
                    "task_id": "2",
                    "quality_label": "CORRECTION_REQUIRED",
                    "reviewer_id": "assistant_reviewer_b",
                },
            ],
        )
        second_correction = self.root / "corrected_c3.json"
        second_correction.write_text(
            json.dumps(
                {
                    "task_id": "2",
                    "author_id": "tau2_teacher_correction_pipeline_v1",
                    "messages": [{"role": "assistant", "content": "done"}],
                }
            ),
            encoding="utf-8",
        )
        second_source = self.root / "source_c3.json"
        second_source.write_text(
            json.dumps({"user_id": "u1", "order_id": "#W2"}),
            encoding="utf-8",
        )
        second_validation = self.root / "validation_c3.json"
        second_validation.write_text(
            json.dumps(
                {
                    "ready": True,
                    "task_id": "2",
                    "correction_sha256": sha256(second_correction),
                }
            ),
            encoding="utf-8",
        )
        registry = self.root / "corrections_leak.jsonl"
        jsonl(
            registry,
            [
                {
                    "candidate_id": "c1",
                    "source_path": str(self.source),
                    "source_sha256": sha256(self.source),
                    "correction_path": str(self.correction),
                    "correction_sha256": sha256(self.correction),
                    "correction_validation_path": str(self.validation),
                    "correction_validation_sha256": sha256(self.validation),
                },
                {
                    "candidate_id": "c3",
                    "source_path": str(second_source),
                    "source_sha256": sha256(second_source),
                    "correction_path": str(second_correction),
                    "correction_sha256": sha256(second_correction),
                    "correction_validation_path": str(second_validation),
                    "correction_validation_sha256": sha256(second_validation),
                },
            ],
        )
        splits = self.root / "splits_leak2.jsonl"
        jsonl(
            splits,
            [
                {
                    "candidate_id": "c1",
                    "split": "TRAIN",
                    "source_split": "TRAIN",
                },
                {
                    "candidate_id": "c3",
                    "split": "VALIDATION",
                    "source_split": "TRAIN",
                },
            ],
        )
        result = build_teacher_candidate_decisions(
            path_a,
            path_b,
            corrections_path=registry,
            split_plan_path=splits,
        )
        self.assertFalse(result["ready"])
        self.assertTrue(any("Group leakage" in reason for reason in result["reasons"]))

    def test_holdout_with_correction_data_fails_closed(self):
        self.write_validation()
        path_a, path_b = self.reviews(label="HOLDOUT")
        result = build_teacher_candidate_decisions(
            path_a,
            path_b,
            corrections_path=self.corrections_registry(),
        )
        self.assertFalse(result["ready"])
        self.assertTrue(any("HOLDOUT must not have" in reason for reason in result["reasons"]))


if __name__ == "__main__":
    unittest.main()
