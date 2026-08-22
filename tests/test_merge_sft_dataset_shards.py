from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from src.training.merge_sft_dataset_shards import merge_shards, write_merge
from src.training.sft_release import sha256


def make_shard(root: Path, name: str, rows: list[dict[str, object]]) -> Path:
    shard = root / name
    shard.mkdir()
    dataset = shard / "sft_dataset.jsonl"
    dataset.write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )
    (shard / "dataset_manifest.json").write_text(
        json.dumps(
            {
                "schema_version": "test",
                "dataset_sha256": sha256(dataset),
                "counts": {"released": len(rows)},
            }
        ),
        encoding="utf-8",
    )
    return shard


def row(candidate: str, task: str, split: str, group: str) -> dict[str, object]:
    return {
        "candidate_id": candidate,
        "task_id": task,
        "split": split,
        "disposition": "CORRECTED_POSITIVE",
        "source_sha256": (candidate * 64)[:64].upper(),
        "group_ids": [group],
        "system_policy": "policy",
        "messages": [{"role": "assistant", "content": "done"}],
    }


class MergeSftDatasetShardsTests(unittest.TestCase):
    def test_merge_rechecks_global_leakage_and_writes_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            first = make_shard(root, "one", [row("a", "1", "TRAIN", "u:1")])
            second = make_shard(
                root, "two", [row("b", "2", "VALIDATION", "u:2")]
            )
            result = merge_shards([first, second])
            self.assertTrue(result["ready"])
            self.assertEqual(result["counts"]["released"], 2)
            output = root / "merged"
            write_merge(result, output)
            manifest = json.loads(
                (output / "dataset_manifest.json").read_text(encoding="utf-8")
            )
            self.assertTrue(manifest["leakage_checks"]["passed"])
            self.assertEqual(
                manifest["dataset_sha256"], sha256(output / "sft_dataset.jsonl")
            )
            self.assertEqual(
                manifest["split_plan_sha256"], sha256(output / "split_plan.jsonl")
            )
            self.assertEqual(manifest["split_plan_rows"], 2)

    def test_cross_shard_entity_leakage_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            first = make_shard(
                root, "one", [row("a", "1", "TRAIN", "user:shared")]
            )
            second = make_shard(
                root,
                "two",
                [row("b", "2", "VALIDATION", "user:shared")],
            )
            result = merge_shards([first, second])
            self.assertFalse(result["ready"])
            self.assertEqual(result["records"], [])
            self.assertIn("user:shared", result["reasons"][0])

    def test_duplicate_selected_trajectory_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            duplicate = row("a", "1", "TRAIN", "u:1")
            first = make_shard(root, "one", [duplicate])
            second_row = {**duplicate, "candidate_id": "b", "task_id": "2"}
            second = make_shard(root, "two", [second_row])
            with self.assertRaisesRegex(ValueError, "Duplicate selected trajectory"):
                merge_shards([first, second])

    def test_same_task_cannot_cross_splits(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            first = make_shard(root, "one", [row("a", "1", "TRAIN", "u:1")])
            second = make_shard(
                root, "two", [row("b", "1", "VALIDATION", "u:2")]
            )
            result = merge_shards([first, second])
            self.assertFalse(result["ready"])
            self.assertIn("Task IDs cross merged splits", result["reasons"][0])


if __name__ == "__main__":
    unittest.main()
