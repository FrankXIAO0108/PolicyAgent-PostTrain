import json
import tempfile
import unittest
from pathlib import Path

from src.training.sft_release import sha256
from src.training.teacher_sft_release import assess_candidate_release


def jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )


class TeacherSftReleaseTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)
        self.source = self.root / "source.json"
        self.source.write_text(
            json.dumps(
                {
                    "simulations": [
                        {
                            "id": "c1",
                            "task_id": "0",
                            "messages": [
                                {"role": "user", "content": "hi"},
                                {"role": "assistant", "content": "done"},
                            ],
                        }
                    ],
                    "info": {"environment_info": {"policy": ""}},
                    "user_id": "u1",
                    "order_id": "#W1",
                }
            ),
            encoding="utf-8",
        )
        self.correction = self.root / "corrected.json"
        self.correction.write_text(
            json.dumps(
                {
                    "task_id": "0",
                    "author_id": "tau2_teacher_correction_pipeline_v1",
                    "system_policy": "policy text",
                    "messages": [
                        {"role": "user", "content": "cancel both"},
                        {
                            "role": "assistant",
                            "content": "",
                            "tool_calls": [
                                {
                                    "id": "c1",
                                    "name": "cancel_pending_order",
                                    "arguments": {"order_id": "#W1"},
                                }
                            ],
                        },
                        {"role": "tool", "id": "c1", "content": "cancelled #W1"},
                        {"role": "assistant", "content": "Both cancelled."},
                    ],
                }
            ),
            encoding="utf-8",
        )
        self.validation = self.root / "validation.json"
        self.write_validation()

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def write_validation(self, ready: bool = True) -> None:
        self.validation.write_text(
            json.dumps(
                {
                    "ready": ready,
                    "task_id": "0",
                    "correction_sha256": sha256(self.correction),
                }
            ),
            encoding="utf-8",
        )

    def decisions(self, extra_rows: list[dict[str, object]] | None = None) -> Path:
        path = self.root / "decisions.jsonl"
        rows: list[dict[str, object]] = [
            {
                "candidate_id": "c1",
                "task_id": "0",
                "status": "SECOND_REVIEWED",
                "quality_label": "CORRECTION_REQUIRED",
                "disposition": "CORRECTED_POSITIVE",
                "split": "TRAIN",
                "source_split": "TRAIN",
                "source_path": str(self.source),
                "source_sha256": sha256(self.source),
                "correction_path": str(self.correction),
                "correction_sha256": sha256(self.correction),
                "correction_validation_path": str(self.validation),
                "correction_validation_sha256": sha256(self.validation),
                "group_ids": ["user_id:u1", "order_id:#W1"],
                "rationale": "test",
            },
            {
                "candidate_id": "c2",
                "task_id": "57",
                "status": "SECOND_REVIEWED",
                "quality_label": "HOLDOUT",
                "disposition": "HOLDOUT",
                "split": None,
                "source_split": "TRAIN",
                "source_path": None,
                "source_sha256": None,
                "correction_path": None,
                "correction_sha256": None,
                "correction_validation_path": None,
                "correction_validation_sha256": None,
                "group_ids": [],
                "rationale": "holdout",
            },
        ]
        if extra_rows:
            rows.extend(extra_rows)
        jsonl(path, rows)
        return path

    def test_release_emits_normalized_records_and_skips_holdout(self):
        result = assess_candidate_release(self.decisions())
        self.assertTrue(result["ready"], result["reasons"])
        self.assertEqual(result["counts"]["released"], 1)
        self.assertEqual(result["counts"]["holdout"], 1)
        record = result["records"][0]
        self.assertEqual(record["candidate_id"], "c1")
        self.assertEqual(record["split"], "TRAIN")
        self.assertEqual(record["system_policy"], "policy text")
        self.assertTrue(all(row["loss_mask"] == 1 for row in record["messages"] if row["role"] == "assistant"))
        self.assertEqual(
            {row["role"] for row in record["messages"]},
            {"user", "assistant", "tool"},
        )

    def test_unready_validation_fails_closed(self):
        self.write_validation(ready=False)
        result = assess_candidate_release(self.decisions())
        self.assertFalse(result["ready"])
        self.assertTrue(any("not ready" in reason for reason in result["reasons"]))

    def test_text_tool_mixing_fails_closed(self):
        self.correction.write_text(
            json.dumps(
                {
                    "task_id": "0",
                    "author_id": "pipeline",
                    "system_policy": "policy text",
                    "messages": [
                        {"role": "user", "content": "go"},
                        {
                            "role": "assistant",
                            "content": "let me",
                            "tool_calls": [
                                {
                                    "id": "c1",
                                    "name": "cancel_pending_order",
                                    "arguments": {"order_id": "#W1"},
                                }
                            ],
                        },
                        {"role": "tool", "id": "c1", "content": "ok"},
                    ],
                }
            ),
            encoding="utf-8",
        )
        self.write_validation()
        result = assess_candidate_release(self.decisions())
        self.assertFalse(result["ready"])
        self.assertTrue(any("mixes text" in reason for reason in result["reasons"]))

    def test_group_leakage_across_splits_fails_closed(self):
        second_source = self.root / "source_c3.json"
        second_source.write_text(
            json.dumps({"user_id": "u1", "order_id": "#W2"}),
            encoding="utf-8",
        )
        second_correction = self.root / "corrected_c3.json"
        second_correction.write_text(
            json.dumps(
                {
                    "task_id": "2",
                    "author_id": "pipeline",
                    "system_policy": "policy text",
                    "messages": [
                        {"role": "user", "content": "go"},
                        {"role": "assistant", "content": "done."},
                    ],
                }
            ),
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
        extra = [
            {
                "candidate_id": "c3",
                "task_id": "2",
                "status": "SECOND_REVIEWED",
                "quality_label": "CORRECTION_REQUIRED",
                "disposition": "CORRECTED_POSITIVE",
                "split": "VALIDATION",
                "source_split": "TRAIN",
                "source_path": str(second_source),
                "source_sha256": sha256(second_source),
                "correction_path": str(second_correction),
                "correction_sha256": sha256(second_correction),
                "correction_validation_path": str(second_validation),
                "correction_validation_sha256": sha256(second_validation),
                "group_ids": ["user_id:u1", "order_id:#W2"],
                "rationale": "test",
            }
        ]
        result = assess_candidate_release(self.decisions(extra))
        self.assertFalse(result["ready"])
        self.assertTrue(any("leakage" in reason for reason in result["reasons"]))


if __name__ == "__main__":
    unittest.main()
