from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from src.evaluation.sft_residual_review import (
    build_preference_review_packet,
    verify_residual_tasks,
)


def _hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def _fixture(root: Path) -> tuple[Path, Path]:
    task_dir = root / "run" / "private_evaluation" / "task_7"
    task_dir.mkdir(parents=True)
    snapshot = task_dir / "task_snapshot.json"
    snapshot.write_text(
        json.dumps(
            {
                "id": "7",
                "user_scenario": {"instructions": {"reason_for_call": "change it"}},
                "evaluation_criteria": {"actions": []},
            }
        ),
        encoding="utf-8",
    )
    result = task_dir / "returned_results.json"
    result.write_text(
        json.dumps(
            {
                "simulations": [
                    {
                        "task_id": "7",
                        "termination_reason": "agent_stop",
                        "messages": [
                            {"role": "user", "content": "change it"},
                            {
                                "role": "assistant",
                                "tool_calls": [
                                    {
                                        "id": "c1",
                                        "name": "modify_user_address",
                                        "arguments": {"user_id": "u1"},
                                    }
                                ],
                            },
                            {
                                "role": "tool",
                                "id": "c1",
                                "content": "price 12.34",
                                "error": False,
                            },
                            {"role": "assistant", "content": "Done; price 12.34"},
                        ],
                        "reward_info": {"reward": 1, "db_check": {"db_match": True}},
                    },
                    {
                        "task_id": "7",
                        "termination_reason": "max_steps",
                        "messages": [{"role": "assistant", "content": "stuck"}],
                        "reward_info": {"reward": 0, "db_check": {"db_match": False}},
                    },
                ]
            }
        ),
        encoding="utf-8",
    )
    cards = root / "cards.json"
    cards.write_text(
        json.dumps(
            {
                "candidate": [
                    {
                        "task_id": "7",
                        "trial_index": index,
                        "artifact": {
                            "returned_results": str(result),
                            "sha256": _hash(result),
                        },
                        "infrastructure": {"valid": True, "termination_reason": reason},
                        "outcome": {"reward": reward, "success": bool(reward)},
                        "tool_use": {"total_calls": 1 if index == 0 else 0},
                        "dialogue_use": {"max_steps_reached": index == 1},
                        "policy_diagnostic": {"verdict": "PASS"},
                    }
                    for index, (reason, reward) in enumerate(
                        (("agent_stop", 1), ("max_steps", 0))
                    )
                ]
            }
        ),
        encoding="utf-8",
    )
    return cards, snapshot


class SftResidualReviewTests(unittest.TestCase):
    def test_task_bound_verifier_checks_each_trial(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cards, snapshot = _fixture(root)
            spec = root / "spec.json"
            spec.write_text(
                json.dumps(
                    {
                        "run_name": "candidate",
                        "source_evaluation_cards_sha256": _hash(cards),
                        "tasks": [
                            {
                                "task_id": "7",
                                "task_snapshot_sha256": _hash(snapshot),
                                "required_writes": [
                                    {
                                        "name": "modify_user_address",
                                        "arguments": {"user_id": "u1"},
                                    }
                                ],
                                "required_final_claims": [
                                    {
                                        "name": "price",
                                        "value": "12.34",
                                        "require_tool_evidence": True,
                                    }
                                ],
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )
            manifest = verify_residual_tasks(
                evaluation_cards=cards, spec_path=spec, output_dir=root / "verified"
            )
            result = json.loads(
                (root / "verified" / "verifier_results.json").read_text(
                    encoding="utf-8"
                )
            )

        self.assertEqual(manifest["pass_count"], 1)
        self.assertEqual(manifest["fail_count"], 1)
        self.assertEqual([row["verdict"] for row in result["rows"]], ["PASS", "FAIL"])
        self.assertFalse(result["scope"]["eligible_as_general_reward"])

    def test_review_packet_selects_exact_trials_and_keeps_release_closed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cards, _ = _fixture(root)
            selections = root / "selections.json"
            selections.write_text(
                json.dumps(
                    {
                        "run_name": "candidate",
                        "source_evaluation_cards_sha256": _hash(cards),
                        "selections": [
                            {
                                "task_id": "7",
                                "trial_index": 1,
                                "candidate_role": "NEGATIVE_TOOL_LOOP",
                                "rationale": "max steps",
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )
            manifest = build_preference_review_packet(
                evaluation_cards=cards,
                selection_path=selections,
                output_dir=root / "packet",
            )
            packet = json.loads(
                (root / "packet" / "review_packet.json").read_text(encoding="utf-8")
            )

        self.assertEqual(manifest["review_packet"]["row_count"], 1)
        self.assertEqual(packet["rows"][0]["trial_index"], 1)
        self.assertEqual(packet["rows"][0]["human_review"]["status"], "PENDING")
        self.assertFalse(packet["rows"][0]["training_release_allowed"])

    def test_review_packet_rejects_duplicate_trial_selection(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cards, _ = _fixture(root)
            selection = {
                "task_id": "7",
                "trial_index": 1,
                "candidate_role": "NEGATIVE_TOOL_LOOP",
                "rationale": "max steps",
            }
            selections = root / "selections.json"
            selections.write_text(
                json.dumps(
                    {
                        "run_name": "candidate",
                        "source_evaluation_cards_sha256": _hash(cards),
                        "selections": [selection, selection],
                    }
                ),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "Duplicate selected"):
                build_preference_review_packet(
                    evaluation_cards=cards,
                    selection_path=selections,
                    output_dir=root / "packet",
                )


if __name__ == "__main__":
    unittest.main()
