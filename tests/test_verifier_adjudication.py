from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from src.verifiers.adjudication import adjudicate, write_outputs


def write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )


def review(task_id: str, label: str, reviewer_id: str) -> dict[str, object]:
    return {
        "task_id": task_id,
        "label": label,
        "reviewer_id": reviewer_id,
        "reviewed_at": "2026-07-27T12:00:00+08:00",
        "rationale": f"{reviewer_id} rationale",
        "evidence_files": [f"evidence/task_{task_id}.json"],
    }


class AdjudicationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)
        self.annotations = self.root / "annotations.jsonl"
        write_jsonl(
            self.annotations,
            [
                {
                    "task_id": "1",
                    "label": "REVIEW",
                    "status": "PROVISIONAL",
                    "source": "test",
                    "rationale": "seed",
                    "evidence_files": ["seed/1.json"],
                },
                {
                    "task_id": "2",
                    "label": "FAIL",
                    "status": "PROVISIONAL",
                    "source": "test",
                    "rationale": "seed",
                    "evidence_files": ["seed/2.json"],
                },
            ],
        )

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def test_two_independent_agreements_release_complete_gold(self) -> None:
        reviewer_a = self.root / "a.jsonl"
        reviewer_b = self.root / "b.jsonl"
        write_jsonl(
            reviewer_a,
            [review("1", "REVIEW", "alice"), review("2", "FAIL", "alice")],
        )
        write_jsonl(
            reviewer_b,
            [review("1", "REVIEW", "bob"), review("2", "FAIL", "bob")],
        )

        result = adjudicate(self.annotations, reviewer_a, reviewer_b)

        self.assertTrue(result["release_gate"]["adjudicated_annotations_ready"])
        self.assertEqual(result["coverage"]["adjudicated_rows"], 2)
        self.assertTrue(
            all(
                row["status"] == "ADJUDICATED"
                for row in result["adjudicated_annotations"]
            )
        )

    def test_unresolved_conflict_fails_closed_without_gold_file(self) -> None:
        reviewer_a = self.root / "a.jsonl"
        reviewer_b = self.root / "b.jsonl"
        output = self.root / "output"
        write_jsonl(
            reviewer_a,
            [review("1", "REVIEW", "alice"), review("2", "FAIL", "alice")],
        )
        write_jsonl(
            reviewer_b,
            [review("1", "PASS", "bob"), review("2", "FAIL", "bob")],
        )

        result = adjudicate(self.annotations, reviewer_a, reviewer_b)
        write_outputs(result, output)

        self.assertFalse(result["release_gate"]["adjudicated_annotations_ready"])
        self.assertEqual(result["release_gate"]["unresolved_task_ids"], ["1"])
        self.assertFalse((output / "adjudicated_annotations.jsonl").exists())
        self.assertTrue((output / "conflicts.jsonl").exists())

    def test_independent_resolver_closes_conflicts(self) -> None:
        reviewer_a = self.root / "a.jsonl"
        reviewer_b = self.root / "b.jsonl"
        resolver = self.root / "resolver.jsonl"
        write_jsonl(
            reviewer_a,
            [review("1", "REVIEW", "alice"), review("2", "FAIL", "alice")],
        )
        write_jsonl(
            reviewer_b,
            [review("1", "PASS", "bob"), review("2", "FAIL", "bob")],
        )
        write_jsonl(resolver, [review("1", "REVIEW", "carol")])

        result = adjudicate(
            self.annotations,
            reviewer_a,
            reviewer_b,
            resolver_path=resolver,
        )

        self.assertTrue(result["release_gate"]["adjudicated_annotations_ready"])
        task_1 = next(
            row
            for row in result["adjudicated_annotations"]
            if row["task_id"] == "1"
        )
        self.assertEqual(task_1["label"], "REVIEW")
        self.assertEqual(
            task_1["adjudication"]["resolution"],
            "third_reviewer_resolution",
        )

    def test_same_reviewer_identity_is_rejected(self) -> None:
        reviewer_a = self.root / "a.jsonl"
        reviewer_b = self.root / "b.jsonl"
        rows = [review("1", "REVIEW", "alice"), review("2", "FAIL", "alice")]
        write_jsonl(reviewer_a, rows)
        write_jsonl(reviewer_b, rows)

        with self.assertRaisesRegex(ValueError, "independent identities"):
            adjudicate(self.annotations, reviewer_a, reviewer_b)

    def test_existing_output_is_not_overwritten(self) -> None:
        output = self.root / "output"
        output.mkdir()
        (output / "frozen.json").write_text("{}\n", encoding="utf-8")
        result = {
            "adjudicated_annotations": [],
            "conflicts": [],
            "release_gate": {"adjudicated_annotations_ready": False},
        }

        with self.assertRaisesRegex(FileExistsError, "Refusing to overwrite"):
            write_outputs(result, output)


if __name__ == "__main__":
    unittest.main()
