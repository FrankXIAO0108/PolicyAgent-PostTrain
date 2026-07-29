from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path

from src.verifiers.blind_review_packet import build_packet, write_packet
from src.verifiers.review_submission import preflight_submission, write_outputs


class ReviewSubmissionTests(unittest.TestCase):
    def _packet(self, root: Path, task_ids: tuple[str, ...] = ("1", "2")) -> Path:
        annotations = root / "annotations.jsonl"
        with annotations.open("w", encoding="utf-8") as handle:
            for task_id in task_ids:
                handle.write(
                    json.dumps(
                        {
                            "task_id": task_id,
                            "label": "PASS",
                            "status": "PROVISIONAL",
                            "source": "seed",
                            "rationale": "hidden",
                            "evidence_files": [],
                        }
                    )
                    + "\n"
                )
                task_dir = root / "experiment" / f"task_{task_id}"
                task_dir.mkdir(parents=True)
                (task_dir / "returned_results.json").write_text(
                    f'{{"task_id":"{task_id}"}}\n', encoding="utf-8"
                )
                (task_dir / "summary.json").write_text(
                    f'{{"task_id":"{task_id}"}}\n', encoding="utf-8"
                )
        policy = root / "policy.md"
        policy.write_text("policy\n", encoding="utf-8")
        rows, manifest = build_packet(annotations, root / "experiment", policy)
        packet = root / "packet"
        write_packet(rows, manifest, packet, bundle_evidence=True)
        return packet

    def _complete_sheet(
        self,
        packet: Path,
        *,
        reviewer_ids: tuple[str, ...] = ("reviewer-a",),
    ) -> Path:
        source = packet / "review_sheet.csv"
        with source.open(encoding="utf-8-sig", newline="") as handle:
            rows = list(csv.DictReader(handle))
        output = packet.parent / "completed.csv"
        with output.open("w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            for index, row in enumerate(rows):
                row["label"] = "PASS"
                row["reviewer_id"] = reviewer_ids[index % len(reviewer_ids)]
                row["reviewed_at"] = "2026-07-29T14:00:00+08:00"
                row["rationale"] = "Policy evidence supports this label."
                writer.writerow(row)
        return output

    def test_valid_csv_is_normalized_for_adjudication(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            packet = self._packet(root)
            submission = self._complete_sheet(packet)

            result = preflight_submission(packet, submission)
            output = root / "preflight"
            write_outputs(result, output)

            self.assertEqual(result["status"], "READY")
            self.assertEqual(result["coverage"]["valid_rows"], 2)
            normalized = (output / "normalized_review.jsonl").read_text(
                encoding="utf-8"
            )
            self.assertEqual(len(normalized.splitlines()), 2)

    def test_changed_evidence_paths_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            packet = self._packet(root)
            submission = self._complete_sheet(packet)
            with submission.open(encoding="utf-8-sig", newline="") as handle:
                rows = list(csv.DictReader(handle))
            rows[0]["evidence_files"] = json.dumps(["different.json"])
            with submission.open("w", encoding="utf-8-sig", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)

            result = preflight_submission(packet, submission)

            self.assertEqual(result["status"], "REJECTED")
            self.assertTrue(
                any("evidence_files changed" in error for error in result["errors"])
            )

    def test_multiple_reviewer_identities_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            packet = self._packet(root)
            submission = self._complete_sheet(
                packet, reviewer_ids=("reviewer-a", "reviewer-b")
            )

            result = preflight_submission(packet, submission)

            self.assertEqual(result["status"], "REJECTED")
            self.assertTrue(
                any("Exactly one reviewer_id" in error for error in result["errors"])
            )

    def test_modified_bundled_evidence_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            packet = self._packet(root)
            submission = self._complete_sheet(packet)
            (packet / "evidence" / "task_1" / "summary.json").write_text(
                '{"tampered":true}\n', encoding="utf-8"
            )

            result = preflight_submission(packet, submission)

            self.assertEqual(result["status"], "REJECTED")
            self.assertTrue(
                any("Evidence hash mismatch" in error for error in result["errors"])
            )

    def test_incomplete_rows_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            packet = self._packet(root)
            submission = packet.parent / "incomplete.csv"
            source = packet / "review_sheet.csv"
            submission.write_bytes(source.read_bytes())

            result = preflight_submission(packet, submission)
            output = root / "preflight"
            write_outputs(result, output)

            self.assertEqual(result["status"], "REJECTED")
            self.assertFalse((output / "normalized_review.jsonl").exists())


if __name__ == "__main__":
    unittest.main()
