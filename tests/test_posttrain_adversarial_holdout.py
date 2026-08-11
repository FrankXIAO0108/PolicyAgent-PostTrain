from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from src.training.adversarial_holdout import CATEGORY_TEMPLATES, build_records, write_dataset


class AdversarialHoldoutTests(unittest.TestCase):
    def test_dataset_is_balanced_and_evaluation_only(self) -> None:
        rows = build_records()
        self.assertEqual(len(rows), 48)
        self.assertEqual(len({row["scenario_id"] for row in rows}), 48)
        for category in CATEGORY_TEMPLATES:
            self.assertEqual(sum(row["category"] == category for row in rows), 6)
        self.assertTrue(all(row["training_use_prohibited"] for row in rows))
        self.assertTrue(all(row["split"] == "evaluation_only" for row in rows))

    def test_manifest_binds_hash_when_reference_is_disjoint(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            reference = root / "reference"
            reference.mkdir()
            reference.joinpath("sft.jsonl").write_text(
                '{"scenario_id":"train-000","prompt":"unrelated"}\n',
                encoding="utf-8",
            )
            manifest = write_dataset(root / "output", reference)
            self.assertTrue(manifest["leakage_checks"]["passed"])
            self.assertTrue(manifest["training_use_prohibited"])
            self.assertEqual(manifest["rows"], 48)
            self.assertEqual(set(manifest["categories"].values()), {6})
            self.assertEqual(len(manifest["file"]["sha256"]), 64)

    def test_manifest_detects_reference_leakage(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            reference = root / "reference"
            reference.mkdir()
            leaked = build_records()[0]
            reference.joinpath("sft.jsonl").write_text(
                json.dumps(
                    {
                        "scenario_id": leaked["scenario_id"],
                        "prompt": leaked["prompt"],
                    },
                    ensure_ascii=False,
                )
                + "\n",
                encoding="utf-8",
            )
            manifest = write_dataset(root / "output", reference)
            self.assertFalse(manifest["leakage_checks"]["passed"])
            self.assertEqual(
                manifest["leakage_checks"]["scenario_id_overlap"],
                [leaked["scenario_id"]],
            )
            self.assertEqual(
                manifest["leakage_checks"]["exact_prompt_overlap"],
                [leaked["prompt"]],
            )


if __name__ == "__main__":
    unittest.main()
