from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from src.training.sft_decision_builder import build_sft_decisions
from src.training.sft_release import sha256


def jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )


class SftDecisionBuilderTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)
        self.source = self.root / "source.json"
        self.source.write_text(
            json.dumps({"user_id": "u1", "order_id": "#W1"}),
            encoding="utf-8",
        )

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def policy(self, status: str = "ADJUDICATED") -> Path:
        path = self.root / f"policy_{status}.jsonl"
        jsonl(
            path,
            [
                {
                    "task_id": "1",
                    "label": "PASS",
                    "status": status,
                    "source": "test",
                    "rationale": "test",
                    "evidence_files": ["policy.json"],
                }
            ],
        )
        return path

    def quality(self, label: str) -> Path:
        path = self.root / f"quality_{label}.jsonl"
        jsonl(
            path,
            [
                {
                    "task_id": "1",
                    "status": "ADJUDICATED",
                    "quality_label": label,
                    "policy_label": "PASS",
                    "source_path": str(self.source),
                    "source_sha256": sha256(self.source),
                    "rationale": "quality decision",
                }
            ],
        )
        return path

    def test_provisional_policy_fails_closed(self) -> None:
        result = build_sft_decisions(self.policy("PROVISIONAL"))
        self.assertFalse(result["ready"])
        self.assertEqual(result["decisions"], [])

    def test_raw_gold_maps_to_raw_positive_with_groups(self) -> None:
        split = self.root / "split.jsonl"
        jsonl(
            split,
            [{"task_id": "1", "split": "TRAIN", "source_split": "TRAIN"}],
        )
        result = build_sft_decisions(
            self.policy(),
            adjudicated_quality_path=self.quality("RAW_GOLD"),
            split_plan_path=split,
        )
        self.assertTrue(result["ready"])
        row = result["decisions"][0]
        self.assertEqual(row["disposition"], "RAW_POSITIVE")
        self.assertIn("user_id:u1", row["group_ids"])

    def test_segment_required_maps_to_holdout_without_split(self) -> None:
        result = build_sft_decisions(
            self.policy(),
            adjudicated_quality_path=self.quality("SEGMENT_REQUIRED"),
        )
        self.assertTrue(result["ready"])
        self.assertEqual(result["decisions"][0]["disposition"], "HOLDOUT")
        self.assertIsNone(result["decisions"][0]["split"])

    def test_correction_required_without_registry_is_blocked(self) -> None:
        split = self.root / "split.jsonl"
        jsonl(
            split,
            [{"task_id": "1", "split": "TRAIN", "source_split": "TRAIN"}],
        )
        result = build_sft_decisions(
            self.policy(),
            adjudicated_quality_path=self.quality("CORRECTION_REQUIRED"),
            split_plan_path=split,
        )
        self.assertFalse(result["ready"])
        self.assertIn("Correction registry", result["reasons"][0])


if __name__ == "__main__":
    unittest.main()
