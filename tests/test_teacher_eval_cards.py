from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from src.evaluation.teacher_eval_cards import (
    analyze_task,
    compare_cards,
    load_run_cards,
)


def _write_result(
    root: Path,
    task_id: str,
    *,
    reward: float | None,
    calls: list[tuple[str, dict]],
    action_matches: list[bool] | None = None,
    premature: bool = False,
) -> Path:
    task_dir = root / "private_evaluation" / f"task_{task_id}"
    task_dir.mkdir(parents=True)
    messages: list[dict] = [{"role": "user", "content": "help"}]
    for index, (name, arguments) in enumerate(calls):
        call_id = f"call-{index}"
        messages.extend(
            [
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {"id": call_id, "name": name, "arguments": arguments}
                    ],
                },
                {
                    "id": call_id,
                    "role": "tool",
                    "content": "{}",
                    "error": False,
                },
            ]
        )
    messages.append({"role": "assistant", "content": "done"})
    reward_info = None
    if reward is not None:
        reward_info = {
            "reward": reward,
            "db_check": {"db_match": reward == 1},
            "action_checks": (
                None
                if premature
                else [
                    {"action_match": matched}
                    for matched in (action_matches or [])
                ]
            ),
            "nl_assertions": [],
            "communicate_checks": [],
            "info": (
                {"note": "Simulation terminated prematurely"}
                if premature
                else {"action": {"note": "No actions to evaluate"}}
            ),
        }
    payload = {
        "simulations": [
            {
                "task_id": task_id,
                "termination_reason": "agent_stop",
                "reward_info": reward_info,
                "messages": messages,
            }
        ]
    }
    path = task_dir / "returned_results.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


class TeacherEvalCardTests(unittest.TestCase):
    def test_analyze_task_separates_recall_from_call_efficiency(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            path = _write_result(
                root,
                "1",
                reward=1,
                calls=[
                    ("get_order_details", {"order_id": "1"}),
                    ("get_order_details", {"order_id": "1"}),
                    ("cancel_pending_order", {"order_id": "1"}),
                ],
                action_matches=[True, False],
            )
            card = analyze_task(path, run_name="base")

        self.assertEqual(card["tool_use"]["total_calls"], 3)
        self.assertEqual(card["tool_use"]["read_calls"], 2)
        self.assertEqual(card["tool_use"]["write_calls"], 1)
        self.assertEqual(card["tool_use"]["repeated_exact_calls"], 1)
        self.assertEqual(card["tool_use"]["consecutive_exact_repeats"], 1)
        self.assertEqual(card["tool_use"]["max_consecutive_same_tool_name"], 2)
        self.assertEqual(card["tool_use"]["reference_action_recall"], 0.5)
        self.assertAlmostEqual(card["tool_use"]["reference_action_density"], 1 / 3)

    def test_reward_info_none_is_infrastructure_invalid(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = _write_result(Path(tmp), "2", reward=None, calls=[])
            card = analyze_task(path, run_name="sft")

        self.assertFalse(card["infrastructure"]["valid"])
        self.assertIsNone(card["outcome"]["reward"])

    def test_premature_model_failure_has_unavailable_action_evaluation(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = _write_result(
                Path(tmp), "3", reward=0, calls=[], premature=True
            )
            card = analyze_task(path, run_name="sft")

        self.assertTrue(card["infrastructure"]["valid"])
        self.assertEqual(
            card["tool_use"]["reference_action_evaluation_status"],
            "unavailable",
        )

    def test_replacement_overrides_original_task_artifact(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            original = root / "original"
            replacement = root / "replacement"
            _write_result(original, "7", reward=None, calls=[])
            _write_result(
                replacement,
                "7",
                reward=1,
                calls=[("get_order_details", {"order_id": "7"})],
            )
            cards = load_run_cards(
                original,
                run_name="sft",
                replacements={"7": replacement},
            )

        self.assertTrue(cards[0]["artifact"]["is_replacement"])
        self.assertTrue(cards[0]["infrastructure"]["valid"])
        self.assertEqual(cards[0]["tool_use"]["total_calls"], 1)

    def test_compare_stratifies_outcome_transitions(self) -> None:
        def card(task_id: str, success: bool, calls: int) -> dict:
            return {
                "task_id": task_id,
                "infrastructure": {"valid": True},
                "outcome": {"success": success},
                "tool_use": {
                    "total_calls": calls,
                    "read_calls": calls,
                    "write_calls": 0,
                    "repeated_exact_calls": 0,
                    "consecutive_exact_repeats": 0,
                    "dominant_tool_call_count": calls,
                    "max_consecutive_same_tool_name": calls,
                    "tool_error_results": 0,
                    "reference_action_evaluation_status": "no_reference_actions",
                    "matched_reference_actions": 0,
                    "reference_action_count": 0,
                },
                "policy_diagnostic": {"verdict": "PASS"},
                "artifact": {"is_replacement": False},
            }

        result = compare_cards(
            [card("1", True, 4), card("2", False, 2)],
            [card("1", True, 3), card("2", True, 5)],
        )

        self.assertEqual(result["strata"]["both_success"]["task_ids"], ["1"])
        self.assertEqual(result["strata"]["both_success"]["mean_tool_call_delta"], -1)
        self.assertEqual(result["strata"]["improved"]["task_ids"], ["2"])


if __name__ == "__main__":
    unittest.main()
