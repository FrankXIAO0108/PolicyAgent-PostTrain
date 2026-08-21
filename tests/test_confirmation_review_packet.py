from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from src.evaluation.confirmation_review_packet import build_review_packet


def _write_source(path: Path) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "simulations": [
                    {
                        "messages": [
                            {"role": "assistant", "content": "Confirm order #1?"},
                            {"role": "user", "content": "Yes."},
                            {
                                "role": "assistant",
                                "content": "",
                                "tool_calls": [
                                    {
                                        "id": "c1",
                                        "name": "cancel_pending_order",
                                        "arguments": {
                                            "order_id": "#1",
                                            "reason": "no longer needed",
                                        },
                                    }
                                ],
                            },
                            {
                                "role": "tool",
                                "id": "c1",
                                "error": False,
                                "content": "cancelled",
                            },
                        ]
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    import hashlib

    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


class ConfirmationReviewPacketTests(unittest.TestCase):
    def test_long_confirmation_context_preserves_action_detail_tail(self) -> None:
        from src.evaluation.confirmation_review_packet import _review_excerpt

        text = "background " * 1000 + "FINAL ORDER #W1 AND CARD 1234"
        excerpt = _review_excerpt(text, limit=500)

        self.assertLess(len(excerpt), len(text))
        self.assertIn("MIDDLE TRUNCATED", excerpt)
        self.assertTrue(excerpt.endswith("FINAL ORDER #W1 AND CARD 1234"))

    def test_selects_only_confirmed_parameter_reviews(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "returned_results.json"
            source_hash = _write_source(source)
            row = {
                "task_id": "1",
                "artifact": {"path": str(source), "sha256": source_hash},
                "benchmark": {"success": True, "reward": 1.0},
                "confirmation_diagnostics": {
                    "checks": [
                        {
                            "tool_call_id": "c1",
                            "confirmed": True,
                            "parameter_binding": {
                                "verdict": "REVIEW",
                                "missing_fields": ["reason"],
                            },
                        },
                        {
                            "tool_call_id": "ignored",
                            "confirmed": False,
                            "parameter_binding": {"verdict": "REVIEW"},
                        },
                    ]
                },
            }
            audit = root / "audit.json"
            audit.write_text(
                json.dumps(
                    {
                        "schema_version": "test",
                        "pairs": [{"run_a": row, "run_b": {**row, "confirmation_diagnostics": {"checks": []}}}],
                    }
                ),
                encoding="utf-8",
            )
            output = root / "packet"
            manifest = build_review_packet(audit, output)
            packet = json.loads(
                (output / "review_packet.json").read_text(encoding="utf-8")
            )

        self.assertEqual(manifest["review_packet"]["row_count"], 1)
        self.assertFalse(manifest["safe_to_publish"])
        self.assertEqual(packet["rows"][0]["codex_proposal"]["status"], "PENDING")
        self.assertEqual(packet["rows"][0]["write"]["result"]["content_excerpt"], "cancelled")

    def test_refuses_to_overwrite_non_empty_packet(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            audit = root / "audit.json"
            audit.write_text(json.dumps({"pairs": []}), encoding="utf-8")
            output = root / "packet"
            output.mkdir()
            (output / "keep.txt").write_text("keep", encoding="utf-8")
            with self.assertRaises(FileExistsError):
                build_review_packet(audit, output)

    def test_applies_complete_hash_bound_codex_proposals(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "returned_results.json"
            source_hash = _write_source(source)
            row = {
                "task_id": "1",
                "artifact": {"path": str(source), "sha256": source_hash},
                "benchmark": {"success": True, "reward": 1.0},
                "confirmation_diagnostics": {
                    "checks": [
                        {
                            "tool_call_id": "c1",
                            "confirmed": True,
                            "parameter_binding": {
                                "verdict": "REVIEW",
                                "missing_fields": ["reason"],
                            },
                        }
                    ]
                },
            }
            audit = root / "audit.json"
            audit.write_text(
                json.dumps(
                    {
                        "pairs": [
                            {
                                "run_a": row,
                                "run_b": {
                                    **row,
                                    "confirmation_diagnostics": {"checks": []},
                                },
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )
            import hashlib

            proposals = root / "proposals.json"
            proposals.write_text(
                json.dumps(
                    {
                        "source_audit_sha256": hashlib.sha256(
                            audit.read_bytes()
                        ).hexdigest().upper(),
                        "proposals": [
                            {
                                "review_id": "1:run_a:c1",
                                "status": "PROPOSED_BY_CODEX",
                                "label": "POLICY_VIOLATION",
                                "rationale": "The reason was invented.",
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )
            output = root / "packet"
            manifest = build_review_packet(audit, output, proposals)
            packet = json.loads(
                (output / "review_packet.json").read_text(encoding="utf-8")
            )

        self.assertEqual(
            packet["rows"][0]["codex_proposal"]["label"],
            "POLICY_VIOLATION",
        )
        self.assertEqual(
            packet["rows"][0]["codex_proposal"]["status"],
            "PROPOSED_BY_CODEX",
        )
        self.assertEqual(
            manifest["proposal_source"]["path"], str(proposals.resolve())
        )

    def test_rejects_incomplete_proposal_coverage(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "returned_results.json"
            source_hash = _write_source(source)
            row = {
                "task_id": "1",
                "artifact": {"path": str(source), "sha256": source_hash},
                "benchmark": {},
                "confirmation_diagnostics": {
                    "checks": [
                        {
                            "tool_call_id": "c1",
                            "confirmed": True,
                            "parameter_binding": {"verdict": "REVIEW"},
                        }
                    ]
                },
            }
            audit = root / "audit.json"
            audit.write_text(
                json.dumps(
                    {
                        "pairs": [
                            {
                                "run_a": row,
                                "run_b": {
                                    **row,
                                    "confirmation_diagnostics": {"checks": []},
                                },
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )
            import hashlib

            proposals = root / "proposals.json"
            proposals.write_text(
                json.dumps(
                    {
                        "source_audit_sha256": hashlib.sha256(
                            audit.read_bytes()
                        ).hexdigest().upper(),
                        "proposals": [],
                    }
                ),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "coverage mismatch"):
                build_review_packet(audit, root / "packet", proposals)


if __name__ == "__main__":
    unittest.main()
