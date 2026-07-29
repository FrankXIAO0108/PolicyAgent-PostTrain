from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from src.training.correction_validation import sha256, validate_correction


def write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload), encoding="utf-8")


class CorrectionValidationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)
        self.policy = self.root / "policy.md"
        self.policy.write_text("frozen policy\n", encoding="utf-8")
        self.source = self.root / "source.json"
        self.source_messages = [
            {"role": "user", "content": "Help"},
            {"role": "assistant", "content": "Unsupported claim"},
        ]
        write_json(
            self.source,
            {"simulations": [{"messages": self.source_messages}]},
        )

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def correction(self) -> Path:
        path = self.root / "correction.json"
        write_json(
            path,
            {
                "task_id": "1",
                "author_id": "author",
                "authored_at": "2026-07-28T10:00:00+08:00",
                "generation_mode": "ASSISTANT_TEXT_EDIT",
                "source": {
                    "path": str(self.source),
                    "sha256": sha256(self.source),
                },
                "policy": {
                    "path": str(self.policy),
                    "sha256": sha256(self.policy),
                },
                "system_policy": "frozen policy\n",
                "change_log": [
                    {
                        "category": "FINAL_CLAIM",
                        "reason": "Remove unsupported claim",
                    }
                ],
                "messages": [
                    {"role": "user", "content": "Help"},
                    {"role": "assistant", "content": "Supported answer"},
                ],
            },
        )
        return path

    def approvals(self, correction: Path) -> Path:
        path = self.root / "approvals.jsonl"
        rows = [
            {
                "task_id": "1",
                "correction_sha256": sha256(correction),
                "reviewer_id": reviewer,
                "verdict": "APPROVE",
                "reviewed_at": "2026-07-28T11:00:00+08:00",
                "rationale": "Evidence checked",
                "evidence_files": [str(self.source), str(self.policy)],
            }
            for reviewer in ("reviewer-a", "reviewer-b")
        ]
        path.write_text(
            "".join(json.dumps(row) + "\n" for row in rows),
            encoding="utf-8",
        )
        return path

    def test_no_correction_fails_closed(self) -> None:
        result = validate_correction(None, None)
        self.assertFalse(result["ready"])

    def test_two_independent_approvals_release_correction(self) -> None:
        correction = self.correction()
        result = validate_correction(correction, self.approvals(correction))
        self.assertTrue(result["ready"])
        self.assertEqual(result["correction_sha256"], sha256(correction))

    def test_missing_approvals_fails_closed(self) -> None:
        result = validate_correction(self.correction(), None)
        self.assertFalse(result["ready"])
        self.assertIn("No independent", result["reasons"][0])

    def test_author_cannot_self_approve(self) -> None:
        correction = self.correction()
        approvals = self.approvals(correction)
        text = approvals.read_text(encoding="utf-8").replace(
            "reviewer-a", "author"
        )
        approvals.write_text(text, encoding="utf-8")
        result = validate_correction(correction, approvals)
        self.assertFalse(result["ready"])
        self.assertTrue(any("author" in reason.lower() for reason in result["reasons"]))

    def test_text_edit_cannot_change_user_observation(self) -> None:
        correction = self.correction()
        payload = json.loads(correction.read_text(encoding="utf-8"))
        payload["messages"][0]["content"] = "Changed user request"
        write_json(correction, payload)
        with self.assertRaisesRegex(ValueError, "preserve all user/tool"):
            validate_correction(correction, None)


if __name__ == "__main__":
    unittest.main()
