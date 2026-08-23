import json
import tempfile
import unittest
from pathlib import Path

from src.training.assemble_owner_review_candidate_inputs import assemble_inputs
from src.training.sft_release import sha256


def jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


class AssembleOwnerReviewCandidateInputsTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.source = self.root / "source.json"
        self.source.write_text(json.dumps({"user_id": "u1", "order_id": "o1"}), encoding="utf-8")
        self.correction = self.root / "corrected_c1.json"
        self.correction.write_text(
            json.dumps(
                {
                    "candidate_id": "c1",
                    "task_id": "1",
                    "source": {"path": str(self.source), "sha256": sha256(self.source)},
                    "messages": [{"role": "assistant", "content": "done"}],
                }
            ),
            encoding="utf-8",
        )
        self.validation_root = self.root / "validations"
        self.validation_root.mkdir()
        validation = self.validation_root / "correction_validation.json"
        validation.write_text(
            json.dumps({"ready": True, "approval_count": 2, "correction_sha256": sha256(self.correction)}),
            encoding="utf-8",
        )
        self.owner = self.root / "owner.jsonl"
        jsonl(
            self.owner,
            [
                {"candidate_id": "c1", "task_id": "1", "quality_label": "CORRECTION_REQUIRED", "reviewer_id": "user_reviewer_a"},
                {"candidate_id": "c2", "task_id": "2", "quality_label": "HOLDOUT", "reviewer_id": "user_reviewer_a"},
            ],
        )
        self.approval = self.root / "approval.jsonl"
        jsonl(
            self.approval,
            [{"correction_sha256": sha256(self.correction), "reviewer_id": "assistant_reviewer_b", "verdict": "APPROVE", "rationale": "checked", "evidence_files": [str(self.correction)]}],
        )
        self.holdouts = self.root / "holdouts.jsonl"
        jsonl(self.holdouts, [{"candidate_id": "c2", "task_id": "2", "quality_label": "HOLDOUT", "reviewer_id": "assistant_reviewer_b", "rationale": "failed"}])
        self.existing_dataset = self.root / "dataset.jsonl"
        jsonl(self.existing_dataset, [{"task_id": "1", "split": "VALIDATION", "group_ids": ["user_id:u1"]}])
        self.existing_splits = self.root / "splits.jsonl"
        jsonl(self.existing_splits, [{"task_id": "1", "split": "VALIDATION"}])

    def tearDown(self) -> None:
        self.temp.cleanup()

    def build(self):
        return assemble_inputs(
            owner_review_paths=[self.owner],
            assistant_approval_paths=[self.approval],
            assistant_holdouts_path=self.holdouts,
            validation_root=self.validation_root,
            existing_dataset_path=self.existing_dataset,
            existing_split_plan_path=self.existing_splits,
            split_seed=1,
        )

    def test_assembles_hash_bound_inputs_and_preserves_existing_split(self):
        result = self.build()
        self.assertEqual(result["counts"], {"reviewed": 2, "corrected": 1, "holdout": 1, "train": 0, "validation": 1, "unique_tasks": 2})
        self.assertEqual(result["split_plan"][0]["split"], "VALIDATION")
        self.assertEqual(result["reviews_b"][0]["quality_label"], "CORRECTION_REQUIRED")
        self.assertEqual(result["reviews_b"][1]["quality_label"], "HOLDOUT")

    def test_missing_assistant_holdout_fails_closed(self):
        jsonl(self.holdouts, [])
        with self.assertRaisesRegex(ValueError, "HOLDOUT coverage mismatch"):
            self.build()

    def test_unready_validation_fails_closed(self):
        validation = self.validation_root / "correction_validation.json"
        validation.write_text(json.dumps({"ready": False, "approval_count": 2, "correction_sha256": sha256(self.correction)}), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "no final validation"):
            self.build()


if __name__ == "__main__":
    unittest.main()
