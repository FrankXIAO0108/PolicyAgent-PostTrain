import json
import tempfile
import unittest
from pathlib import Path

from src.training.prepare_teacher_review_queue import build_review_queue, write_outputs


def simulation(candidate_id: str, task_id: str, tool_name: str) -> dict:
    return {
        "id": candidate_id,
        "task_id": task_id,
        "messages": [
            {
                "role": "assistant",
                "tool_calls": [{"name": tool_name, "arguments": {"order_id": "O1"}}],
            }
        ],
    }


def audit(candidate_id: str, task_id: str, trial: int, label: str, reward: float, db_match: bool) -> dict:
    return {
        "candidate_id": candidate_id,
        "task_id": task_id,
        "trial": trial,
        "automatic_label": label,
        "hard_rejection_reasons": [] if label != "REJECTED" else ["tool_error_present"],
        "review_reasons": ["review"],
        "metrics": {
            "tau2_reward": reward,
            "db_match": db_match,
            "tool_error_count": int(label == "REJECTED"),
            "unexpected_write_count": 0,
        },
        "evidence_pack": {"path": f"candidate_{candidate_id}.json", "sha256": candidate_id},
    }


class ReviewQueueTest(unittest.TestCase):
    def test_deduplicates_review_candidates_and_keeps_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            run_dir = Path(temporary) / "run"
            task_dir = run_dir / "public_candidates" / "task_1"
            task_dir.mkdir(parents=True)
            simulations = [
                simulation("a", "1", "get_order"),
                simulation("b", "1", "get_order"),
                simulation("c", "1", "cancel_order"),
            ]
            (task_dir / "candidate_trajectories.jsonl").write_text(
                "".join(json.dumps(row) + "\n" for row in simulations), encoding="utf-8"
            )
            audits = [
                audit("a", "1", 0, "REVIEW_REQUIRED", 1.0, True),
                audit("b", "1", 1, "REVIEW_REQUIRED", 0.0, True),
                audit("c", "1", 2, "REJECTED", 0.0, False),
            ]
            (run_dir / "candidate_audit.jsonl").write_text(
                "".join(json.dumps(row) + "\n" for row in audits), encoding="utf-8"
            )

            payload = build_review_queue(run_dir, benchmark_anomaly_tasks={"1"})

            self.assertEqual(payload["counts"]["source_candidates"], 3)
            self.assertEqual(payload["counts"]["review_queue"], 2)
            self.assertEqual(payload["counts"]["duplicate_holdup"], 1)
            self.assertEqual(payload["duplicate_holdup"][0]["candidate_id"], "b")
            self.assertEqual(payload["duplicate_holdup"][0]["representative_candidate_id"], "a")
            self.assertTrue(all(row["priority"] == "P0" for row in payload["queue"]))

    def test_refuses_to_overwrite_output(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "output"
            output.mkdir()
            (output / "existing").write_text("x", encoding="utf-8")
            with self.assertRaises(FileExistsError):
                write_outputs({"source": {}, "counts": {}}, output)


if __name__ == "__main__":
    unittest.main()
