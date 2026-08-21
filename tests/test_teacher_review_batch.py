from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from src.evaluation.teacher_review_batch import build_batch


def _write_run(root: Path, task_id: str, *, reward: float, calls: int) -> None:
    task_dir = root / "private_evaluation" / f"task_{task_id}"
    task_dir.mkdir(parents=True)
    messages: list[dict] = [{"role": "user", "content": "cancel it"}]
    for index in range(calls):
        call_id = f"call-{index}"
        messages.extend(
            [
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": call_id,
                            "name": "get_order_details",
                            "arguments": {"order_id": "#W1"},
                        }
                    ],
                },
                {"role": "tool", "id": call_id, "content": "{}", "error": False},
            ]
        )
    messages.append({"role": "assistant", "content": "done"})
    (task_dir / "returned_results.json").write_text(
        json.dumps(
            {
                "simulations": [
                    {
                        "id": f"sim-{task_id}",
                        "task_id": task_id,
                        "termination_reason": "agent_stop",
                        "messages": messages,
                        "reward_info": {
                            "reward": reward,
                            "db_check": {"db_match": reward == 1},
                            "action_checks": [],
                            "nl_assertions": [],
                            "communicate_checks": [],
                            "info": {"action": {"note": "No actions to evaluate"}},
                        },
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    (task_dir / "task_snapshot.json").write_text(
        json.dumps(
            {
                "id": task_id,
                "user_scenario": {
                    "instructions": {"reason_for_call": "cancel order"}
                },
                "evaluation_criteria": {
                    "actions": [],
                    "communicate_info": [],
                    "nl_assertions": [],
                    "reward_basis": ["DB"],
                },
            }
        ),
        encoding="utf-8",
    )


class TeacherReviewBatchTests(unittest.TestCase):
    def test_builds_compact_paired_pack_and_blank_human_template(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            base = root / "base"
            sft = root / "sft"
            output = root / "review"
            _write_run(base, "1", reward=0, calls=1)
            _write_run(sft, "1", reward=1, calls=3)

            manifest = build_batch(
                base_dir=base,
                candidate_dir=sft,
                task_ids=["1"],
                output_dir=output,
            )
            pack = json.loads(
                (output / "evidence_packs" / "task_1.json").read_text(
                    encoding="utf-8"
                )
            )
            template = json.loads(
                (output / "human_review_template.jsonl")
                .read_text(encoding="utf-8")
                .strip()
            )

        self.assertEqual(pack["base"]["evaluation_card"]["tool_use"]["total_calls"], 1)
        self.assertEqual(pack["sft"]["consecutive_tool_groups"][0]["count"], 3)
        self.assertEqual(template["human_decision"], "")
        self.assertFalse(manifest["label_identity"]["codex_proposals_are_human_gold"])

    def test_refuses_to_overwrite_review_batch(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            base = root / "base"
            sft = root / "sft"
            output = root / "review"
            _write_run(base, "1", reward=0, calls=1)
            _write_run(sft, "1", reward=1, calls=1)
            build_batch(
                base_dir=base,
                candidate_dir=sft,
                task_ids=["1"],
                output_dir=output,
            )

            with self.assertRaises(FileExistsError):
                build_batch(
                    base_dir=base,
                    candidate_dir=sft,
                    task_ids=["1"],
                    output_dir=output,
                )


if __name__ == "__main__":
    unittest.main()
