from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path

from src.verifiers.blind_review_packet import build_packet, write_packet


class BlindReviewPacketTests(unittest.TestCase):
    def test_packet_excludes_seed_labels_and_predictions(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            annotations = root / "annotations.jsonl"
            annotations.write_text(
                json.dumps(
                    {
                        "task_id": "1",
                        "label": "FAIL",
                        "status": "PROVISIONAL",
                        "source": "seed",
                        "rationale": "must stay hidden",
                        "evidence_files": ["biased/audit.md"],
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            task_dir = root / "experiment" / "task_1"
            task_dir.mkdir(parents=True)
            (task_dir / "returned_results.json").write_text(
                "{}\n", encoding="utf-8"
            )
            (task_dir / "summary.json").write_text("{}\n", encoding="utf-8")
            policy = root / "policy.md"
            policy.write_text("policy\n", encoding="utf-8")

            rows, manifest = build_packet(
                annotations,
                root / "experiment",
                policy,
            )
            output = root / "packet"
            write_packet(rows, manifest, output)

            rendered = (output / "review_template.jsonl").read_text(
                encoding="utf-8"
            )
            self.assertNotIn('"FAIL"', rendered)
            self.assertNotIn("must stay hidden", rendered)
            self.assertNotIn("biased/audit.md", rendered)
            self.assertEqual(rows[0]["label"], "")
            self.assertFalse(manifest["annotation_source"]["labels_exposed_in_packet"])

    def test_portable_packet_bundles_evidence_with_relative_paths(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            annotations = root / "annotations.jsonl"
            annotations.write_text(
                json.dumps(
                    {
                        "task_id": "1",
                        "label": "REVIEW",
                        "status": "PROVISIONAL",
                        "source": "seed",
                        "rationale": "hidden",
                        "evidence_files": [],
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            task_dir = root / "experiment" / "task_1"
            task_dir.mkdir(parents=True)
            (task_dir / "returned_results.json").write_text(
                '{"trajectory":true}\n', encoding="utf-8"
            )
            (task_dir / "summary.json").write_text(
                '{"summary":true}\n', encoding="utf-8"
            )
            policy = root / "policy.md"
            policy.write_text("policy\n", encoding="utf-8")
            rows, manifest = build_packet(
                annotations, root / "experiment", policy
            )
            output = root / "portable"

            write_packet(rows, manifest, output, bundle_evidence=True)

            packet_row = json.loads(
                (output / "review_template.jsonl").read_text(encoding="utf-8")
            )
            self.assertEqual(
                packet_row["blind_evidence"]["trajectory"]["path"],
                "evidence/task_1/returned_results.json",
            )
            self.assertTrue(
                (output / "evidence" / "task_1" / "summary.json").is_file()
            )
            self.assertTrue((output / "evidence" / "policy.md").is_file())
            self.assertTrue((output / "REVIEWER_INSTRUCTIONS.md").is_file())
            with (output / "review_sheet.csv").open(
                encoding="utf-8-sig", newline=""
            ) as handle:
                csv_rows = list(csv.DictReader(handle))
            self.assertEqual(len(csv_rows), 1)
            self.assertEqual(csv_rows[0]["task_id"], "1")
            self.assertEqual(csv_rows[0]["label"], "")
            self.assertEqual(
                json.loads(csv_rows[0]["evidence_files"]),
                packet_row["evidence_files"],
            )
            instructions = (output / "REVIEWER_INSTRUCTIONS.md").read_text(
                encoding="utf-8"
            )
            self.assertIn("独立政策盲审", instructions)
            self.assertIn("Independent policy review", instructions)
            self.assertNotIn('"REVIEW"', json.dumps(packet_row))


if __name__ == "__main__":
    unittest.main()
