from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from src.training.sft_release import (
    QualityDecision,
    assess_release,
    sha256,
    write_release,
)
from src.verifiers.gold_validation import GoldAnnotation


def annotation(task_id: str, label: str, status: str) -> GoldAnnotation:
    return GoldAnnotation(task_id, label, status, "test", "test", ())


def raw_source(path: Path) -> None:
    path.write_text(
        json.dumps(
            {
                "info": {"environment_info": {"policy": "policy"}},
                "simulations": [
                    {
                        "messages": [
                            {"role": "user", "content": "help"},
                            {"role": "assistant", "content": "done"},
                        ]
                    }
                ],
            }
        ),
        encoding="utf-8",
    )


def decision(
    task_id: str,
    source: Path,
    *,
    disposition: str = "RAW_POSITIVE",
    split: str | None = "TRAIN",
    groups: tuple[str, ...] = ("user:1",),
) -> QualityDecision:
    return QualityDecision(
        task_id=task_id,
        status="ADJUDICATED",
        disposition=disposition,
        split=split,
        source_split="TRAIN",
        source_path=source,
        source_sha256=sha256(source),
        correction_path=None,
        correction_sha256=None,
        correction_validation_path=None,
        correction_validation_sha256=None,
        group_ids=groups,
        rationale="audited",
    )


class SftReleaseTests(unittest.TestCase):
    def test_owner_reviewed_development_requires_explicit_mode(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            source = Path(temp_dir) / "raw.json"
            raw_source(source)
            row = {
                "task_id": "1",
                "status": "HUMAN_ADJUDICATED",
                "disposition": "RAW_POSITIVE",
                "split": "TRAIN",
                "source_split": "TRAIN",
                "source_path": str(source),
                "source_sha256": sha256(source),
                "group_ids": ["user:1"],
                "rationale": "owner reviewed development row",
            }
            with self.assertRaisesRegex(ValueError, "not permitted"):
                QualityDecision.from_dict(row)

            owner_decision = QualityDecision.from_dict(
                row, allow_owner_reviewed_development=True
            )
            owner_annotation = annotation(
                "1", "PASS", "HUMAN_ADJUDICATED"
            )
            blocked = assess_release(
                [owner_annotation], {"1": owner_decision}
            )
            self.assertFalse(blocked["ready"])

            allowed = assess_release(
                [owner_annotation],
                {"1": owner_decision},
                allow_owner_reviewed_development=True,
            )
            self.assertTrue(allowed["ready"])
            self.assertEqual(
                allowed["review_mode"], "OWNER_REVIEWED_DEVELOPMENT"
            )

    def test_corrected_positive_requires_ready_bound_validation(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source = root / "raw.json"
            raw_source(source)
            correction = root / "correction.json"
            correction.write_text(
                json.dumps(
                    {
                        "system_policy": "policy",
                        "messages": [
                            {"role": "user", "content": "help"},
                            {"role": "assistant", "content": "corrected"},
                        ],
                    }
                ),
                encoding="utf-8",
            )
            validation = root / "validation.json"
            validation.write_text(
                json.dumps(
                    {
                        "ready": True,
                        "task_id": "1",
                        "correction_sha256": sha256(correction),
                    }
                ),
                encoding="utf-8",
            )
            corrected = QualityDecision(
                task_id="1",
                status="ADJUDICATED",
                disposition="CORRECTED_POSITIVE",
                split="TRAIN",
                source_split="TRAIN",
                source_path=source,
                source_sha256=sha256(source),
                correction_path=correction,
                correction_sha256=sha256(correction),
                correction_validation_path=validation,
                correction_validation_sha256=sha256(validation),
                group_ids=("user:1",),
                rationale="independently approved",
            )

            result = assess_release(
                [annotation("1", "REVIEW", "ADJUDICATED")],
                {"1": corrected},
            )

            self.assertTrue(result["ready"])
            self.assertTrue(result["records"][0]["corrected"])

    def test_official_test_source_is_rejected_for_release(self) -> None:
        with self.assertRaisesRegex(ValueError, "official TEST"):
            QualityDecision.from_dict(
                {
                    "task_id": "1",
                    "status": "ADJUDICATED",
                    "disposition": "RAW_POSITIVE",
                    "split": "TRAIN",
                    "source_split": "TEST",
                    "source_path": "raw.json",
                    "source_sha256": "ABC",
                    "group_ids": ["user:1"],
                    "rationale": "test",
                }
            )

    def test_provisional_annotations_fail_closed_without_dataset(self) -> None:
        result = assess_release([annotation("1", "PASS", "PROVISIONAL")], None)
        self.assertFalse(result["ready"])
        self.assertEqual(result["records"], [])
        with tempfile.TemporaryDirectory() as temp_dir:
            output = Path(temp_dir) / "release"
            write_release(result, output)
            self.assertFalse((output / "sft_dataset.jsonl").exists())

    def test_adjudicated_pass_emits_assistant_only_loss_mask(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            source = Path(temp_dir) / "raw.json"
            raw_source(source)
            result = assess_release(
                [annotation("1", "PASS", "ADJUDICATED")],
                {"1": decision("1", source)},
            )
            self.assertTrue(result["ready"])
            self.assertEqual(
                [message["loss_mask"] for message in result["records"][0]["messages"]],
                [0, 1],
            )
            self.assertTrue(
                result["records"][0]["candidate_id"].startswith(
                    "policy-agent-task-1-"
                )
            )
            output = Path(temp_dir) / "release"
            write_release(result, output)
            dataset = output / "sft_dataset.jsonl"
            manifest = json.loads(
                (output / "dataset_manifest.json").read_text(encoding="utf-8")
            )
            self.assertTrue(dataset.is_file())
            self.assertEqual(manifest["dataset_sha256"], sha256(dataset))

    def test_raw_positive_requires_policy_pass(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            source = Path(temp_dir) / "raw.json"
            raw_source(source)
            with self.assertRaisesRegex(ValueError, "requires adjudicated policy PASS"):
                assess_release(
                    [annotation("1", "REVIEW", "ADJUDICATED")],
                    {"1": decision("1", source)},
                )

    def test_group_leakage_across_splits_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            source_1 = Path(temp_dir) / "one.json"
            source_2 = Path(temp_dir) / "two.json"
            raw_source(source_1)
            raw_source(source_2)
            result = assess_release(
                [
                    annotation("1", "PASS", "ADJUDICATED"),
                    annotation("2", "PASS", "ADJUDICATED"),
                ],
                {
                    "1": decision("1", source_1, groups=("user:shared",)),
                    "2": decision(
                        "2",
                        source_2,
                        split="VALIDATION",
                        groups=("user:shared",),
                    ),
                },
            )
            self.assertFalse(result["ready"])
            self.assertEqual(result["records"], [])
            self.assertIn("Group leakage", result["reasons"][0])


if __name__ == "__main__":
    unittest.main()
