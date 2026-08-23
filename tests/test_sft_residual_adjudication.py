from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from src.evaluation.sft_residual_adjudication import build_adjudication


def _hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def _write_fixture(root: Path) -> tuple[Path, Path]:
    result = root / "returned_results.json"
    result.write_text(
        json.dumps(
            {
                "simulations": [
                    {
                        "termination_reason": "user_stop",
                        "messages": [
                            {
                                "role": "assistant",
                                "content": "Would you like to switch it to Visa instead?",
                            },
                            {
                                "role": "user",
                                "content": "Yes, please switch it to Visa.",
                            },
                            {
                                "role": "assistant",
                                "tool_calls": [
                                    {
                                        "id": "w1",
                                        "name": "modify_user_address",
                                        "arguments": {"user_id": "u1"},
                                    }
                                ],
                            },
                            {
                                "role": "tool",
                                "id": "w1",
                                "content": "done",
                                "error": False,
                            },
                        ],
                    },
                    {
                        "termination_reason": "max_steps",
                        "messages": [
                            (
                                {
                                    "role": "assistant",
                                    "tool_calls": [
                                        {
                                            "id": f"c{i}",
                                            "name": "get_item_details",
                                            "arguments": {"item_id": "1"},
                                        }
                                    ],
                                }
                                if i % 2 == 0
                                else {
                                    "role": "tool",
                                    "id": f"c{i - 1}",
                                    "content": "same",
                                    "error": False,
                                }
                            )
                            for i in range(6)
                        ],
                    },
                ]
            }
        ),
        encoding="utf-8",
    )
    packet = root / "review_packet.json"
    rows = []
    for trial, role, reward, dialogue in (
        (0, "PREFERRED_POSITIVE_CANDIDATE", 1, False),
        (1, "NEGATIVE_TOOL_LOOP", 0, False),
    ):
        rows.append(
            {
                "review_id": f"candidate:7:{trial}",
                "task_id": "7",
                "trial_index": trial,
                "candidate_role": role,
                "post_run_benchmark_evidence": {"outcome": {"reward": reward}},
                "trajectory_digest": {
                    "evaluation_card": {
                        "dialogue_use": {"dialogue_repeat_candidate": dialogue}
                    }
                },
                "source": {
                    "returned_results": str(result),
                    "returned_results_sha256": _hash(result),
                },
            }
        )
    packet.write_text(json.dumps({"rows": rows}), encoding="utf-8")
    reviews = root / "reviews"
    reviews.mkdir()
    review = {
        "source_review_packet": {
            "path": str(packet.resolve()),
            "sha256": _hash(packet),
        },
        "decisions": [
            {
                "review_id": "candidate:7:0",
                "decision": "ACCEPT_AS_PREFERRED_POSITIVE",
                "label": "PREFERRED_SUCCESS",
                "rationale": "clean",
            },
            {
                "review_id": "candidate:7:1",
                "decision": "ACCEPT_AS_VALID_NEGATIVE",
                "label": "NO_PROGRESS_TOOL_LOOP",
                "rationale": "loop",
            },
        ],
    }
    (reviews / "task_7_owner_review.json").write_text(
        json.dumps(review), encoding="utf-8"
    )
    return packet, reviews


class SftResidualAdjudicationTests(unittest.TestCase):
    def test_aggregates_full_coverage_and_validates_loop_signal(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            packet, reviews = _write_fixture(root)
            manifest = build_adjudication(
                review_packet=packet,
                owner_reviews_dir=reviews,
                output_dir=root / "output",
            )
            result = json.loads(
                (root / "output" / "adjudication.json").read_text(encoding="utf-8")
            )

        self.assertEqual(manifest["summary"]["row_count"], 2)
        self.assertEqual(manifest["summary"]["preference_pair_candidate_count"], 1)
        self.assertTrue(result["reward_signal_validation"]["development_checks_passed"])
        self.assertFalse(result["reward_signal_validation"]["eligible_for_grpo_reward"])
        self.assertFalse(
            result["preference_pair_candidates"][0]["training_release_allowed"]
        )

    def test_rejects_incomplete_owner_coverage(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            packet, reviews = _write_fixture(root)
            review_path = reviews / "task_7_owner_review.json"
            review = json.loads(review_path.read_text(encoding="utf-8"))
            review["decisions"].pop()
            review_path.write_text(json.dumps(review), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "coverage mismatch"):
                build_adjudication(
                    review_packet=packet,
                    owner_reviews_dir=reviews,
                    output_dir=root / "output",
                )


if __name__ == "__main__":
    unittest.main()
