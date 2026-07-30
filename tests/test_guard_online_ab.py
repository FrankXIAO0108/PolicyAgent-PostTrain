from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from src.guards.online_ab import (
    DEFAULT_PROTOCOL,
    compare_arms,
    load_protocol,
    preflight,
)


def write_result(
    root: Path,
    task_id: str,
    reward: float,
    *,
    db_reward: float | None = None,
    nl_reward: float | None = None,
) -> None:
    task_dir = root / f"task_{task_id}"
    task_dir.mkdir(parents=True)
    payload = {
        "simulations": [
            {
                "task_id": task_id,
                "reward_info": {
                    "reward": reward,
                    "reward_breakdown": {
                        "DB": reward if db_reward is None else db_reward,
                        "NL_ASSERTION": reward if nl_reward is None else nl_reward,
                    },
                },
                "duration": 3.0,
                "agent_cost": 0.001,
                "user_cost": 0.0005,
                "termination_reason": "user_stop",
            }
        ]
    }
    (task_dir / "returned_results.json").write_text(
        json.dumps(payload),
        encoding="utf-8",
    )


def write_trace(root: Path, task_id: str, *, blocked: bool) -> None:
    task_dir = root / f"task_{task_id}"
    event = {
        "schema_version": "guard-live-trace-v1.0.0",
        "event": "proposal_evaluated",
        "retry_index": 0,
        "allowed": not blocked,
        "decision": "REGENERATE" if blocked else "ALLOW",
        "tool_proposals": [],
        "blocking_findings": [{"rule_id": "synthetic"}] if blocked else [],
    }
    (task_dir / "guard_trace.jsonl").write_text(
        json.dumps(event) + "\n",
        encoding="utf-8",
    )


class GuardOnlineABTests(unittest.TestCase):
    def test_preflight_blocks_without_clean_tree_key_and_approval(self) -> None:
        result = preflight(
            load_protocol(DEFAULT_PROTOCOL),
            paid_approval=False,
            api_key_configured=False,
            git_dirty=True,
        )
        self.assertEqual(result["status"], "BLOCKED")
        self.assertEqual(
            set(result["blocking_failure_ids"]),
            {
                "clean_git_tree",
                "api_key_configured",
                "explicit_paid_approval",
            },
        )
        self.assertFalse(result["paid_calls_executed"])
        self.assertFalse(result["secrets"]["api_key_value_recorded"])

    def test_preflight_ready_only_when_external_gates_pass(self) -> None:
        result = preflight(
            load_protocol(DEFAULT_PROTOCOL),
            paid_approval=True,
            api_key_configured=True,
            git_dirty=False,
        )
        self.assertEqual(result["status"], "READY")
        self.assertEqual(result["blocking_failure_ids"], [])

    def test_paired_comparison_counts_recovery_and_regression(self) -> None:
        protocol = {
            **load_protocol(DEFAULT_PROTOCOL),
            "task_ids": ["95", "98", "107"],
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            base = root / "base"
            guarded = root / "guarded"
            write_result(base, "95", 0.0)
            write_result(guarded, "95", 1.0)
            write_trace(guarded, "95", blocked=True)
            write_result(base, "98", 0.0)
            write_result(guarded, "98", 0.0)
            write_trace(guarded, "98", blocked=True)
            write_result(base, "107", 1.0)
            write_result(guarded, "107", 0.0)
            write_trace(guarded, "107", blocked=False)

            result = compare_arms(
                protocol,
                base_dir=base,
                guarded_dir=guarded,
            )

        self.assertEqual(result["summary"]["base_success_count"], 1)
        self.assertEqual(result["summary"]["guarded_success_count"], 1)
        self.assertEqual(result["summary"]["paired_business_success_delta"], 0)
        self.assertEqual(result["summary"]["guard_recovery_count"], 1)
        self.assertEqual(result["summary"]["guard_regression_count"], 1)
        self.assertEqual(result["summary"]["guard_intervention_count"], 2)
        self.assertTrue(result["summary"]["guard_trace_complete"])
        self.assertTrue(
            result["summary"]["v7_replay_required_before_final_claim"]
        )

    def test_protocol_rejects_quarantined_task(self) -> None:
        payload = json.loads(DEFAULT_PROTOCOL.read_text(encoding="utf-8"))
        payload["task_ids"].append("59")
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "protocol.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "quarantined"):
                load_protocol(path)


if __name__ == "__main__":
    unittest.main()
