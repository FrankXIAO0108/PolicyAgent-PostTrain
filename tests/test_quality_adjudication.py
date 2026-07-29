from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from src.training.quality_adjudication import (
    evaluate_quality_adjudication,
    write_outputs,
)


def jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )


def review(task_id: str, label: str, reviewer: str) -> dict[str, object]:
    return {
        "task_id": task_id,
        "quality_label": label,
        "reviewer_id": reviewer,
        "reviewed_at": "2026-07-28T12:00:00+08:00",
        "rationale": "evidence checked",
        "evidence_files": ["evidence.json"],
    }


class QualityAdjudicationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)
        self.experiment = self.root / "experiment"
        task_dir = self.experiment / "task_1"
        task_dir.mkdir(parents=True)
        (task_dir / "returned_results.json").write_text("{}\n", encoding="utf-8")
        (task_dir / "summary.json").write_text("{}\n", encoding="utf-8")

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def annotations(self, status: str, label: str = "PASS") -> Path:
        path = self.root / f"annotations_{status}.jsonl"
        jsonl(
            path,
            [
                {
                    "task_id": "1",
                    "label": label,
                    "status": status,
                    "source": "test",
                    "rationale": "test",
                    "evidence_files": ["policy.json"],
                }
            ],
        )
        return path

    def test_provisional_policy_blocks_quality_template(self) -> None:
        result = evaluate_quality_adjudication(
            self.annotations("PROVISIONAL"), self.experiment
        )
        output = self.root / "output"
        write_outputs(result, output)
        self.assertFalse(result["ready"])
        self.assertFalse((output / "quality_review_template.jsonl").exists())

    def test_adjudicated_policy_emits_unlabeled_template(self) -> None:
        result = evaluate_quality_adjudication(
            self.annotations("ADJUDICATED"), self.experiment
        )
        self.assertFalse(result["ready"])
        self.assertEqual(result["template_rows"][0]["quality_label"], "")

    def test_two_agreements_emit_adjudicated_quality(self) -> None:
        a = self.root / "a.jsonl"
        b = self.root / "b.jsonl"
        jsonl(a, [review("1", "RAW_GOLD", "alice")])
        jsonl(b, [review("1", "RAW_GOLD", "bob")])
        result = evaluate_quality_adjudication(
            self.annotations("ADJUDICATED"),
            self.experiment,
            reviewer_a_path=a,
            reviewer_b_path=b,
        )
        self.assertTrue(result["ready"])
        self.assertEqual(
            result["adjudicated_quality"][0]["quality_label"], "RAW_GOLD"
        )

    def test_raw_gold_requires_policy_pass(self) -> None:
        a = self.root / "a.jsonl"
        b = self.root / "b.jsonl"
        jsonl(a, [review("1", "RAW_GOLD", "alice")])
        jsonl(b, [review("1", "RAW_GOLD", "bob")])
        with self.assertRaisesRegex(ValueError, "policy PASS"):
            evaluate_quality_adjudication(
                self.annotations("ADJUDICATED", "REVIEW"),
                self.experiment,
                reviewer_a_path=a,
                reviewer_b_path=b,
            )

    def test_unresolved_conflict_fails_closed(self) -> None:
        a = self.root / "a.jsonl"
        b = self.root / "b.jsonl"
        jsonl(a, [review("1", "RAW_GOLD", "alice")])
        jsonl(b, [review("1", "HOLDOUT", "bob")])
        result = evaluate_quality_adjudication(
            self.annotations("ADJUDICATED"),
            self.experiment,
            reviewer_a_path=a,
            reviewer_b_path=b,
        )
        self.assertFalse(result["ready"])
        self.assertEqual(result["adjudicated_quality"], [])


if __name__ == "__main__":
    unittest.main()
