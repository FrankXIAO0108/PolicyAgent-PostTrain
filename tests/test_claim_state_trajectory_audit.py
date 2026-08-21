from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from src.evaluation.claim_state_trajectory_audit import audit


def _artifact(path: Path, answer: str) -> Path:
    path.write_text(
        json.dumps(
            {
                "simulations": [
                    {
                        "messages": [
                            {
                                "role": "tool",
                                "content": json.dumps(
                                    {
                                        "order_id": "#W4000001",
                                        "status": "cancelled",
                                    }
                                ),
                                "error": False,
                            },
                            {"role": "assistant", "content": answer},
                        ]
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    return path


class ClaimStateTrajectoryAuditTests(unittest.TestCase):
    def test_audit_reports_false_failures_on_successes(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            success = _artifact(
                root / "success.json", "Order #W4000001 has been cancelled."
            )
            failure = _artifact(
                root / "failure.json", "Order #W4000001 is now cancelled."
            )
            process_audit = root / "process.json"
            process_audit.write_text(
                json.dumps(
                    {
                        "pairs": [
                            {
                                "task_id": "1",
                                "cohort": "flip",
                                "run_a": {
                                    "artifact": {"path": str(success), "sha256": "a" * 64},
                                    "benchmark": {"success": True},
                                },
                                "run_b": {
                                    "artifact": {"path": str(failure), "sha256": "b" * 64},
                                    "benchmark": {"success": False},
                                },
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )
            report = audit(process_audit)
            self.assertEqual(report["summary"]["trajectory_count"], 2)
            self.assertEqual(
                report["summary"]["claim_state_failure_on_success_count"], 0
            )
            self.assertFalse(report["gates"]["ready_for_reward_penalty"])


if __name__ == "__main__":
    unittest.main()
