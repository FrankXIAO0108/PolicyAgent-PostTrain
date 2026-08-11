from __future__ import annotations

import json
import hashlib
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from src.rl.retail_agentic_env import (
    DEFAULT_REWARD_CONFIG,
    RetailAgenticEnvironment,
    confirmation_diagnostics,
    one_to_one_action_progress,
)
from src.training.run_retail_agentic_grpo import validate_upstream_checkout
from src.analysis.analyze_agentic_rollout_diagnostic import analyze


PROJECT = Path(__file__).resolve().parents[1]


class _ExpectedAction:
    def __init__(self, action_id: str, name: str, arguments: dict) -> None:
        self.action_id = action_id
        self.name = name
        self.arguments = arguments
        self.requestor = "assistant"

    def compare_with_tool_call(self, call) -> bool:
        return self.name == call.name and self.arguments == call.arguments


def _call(call_id: str, name: str, arguments: dict):
    return SimpleNamespace(
        id=call_id,
        name=name,
        arguments=arguments,
        requestor="assistant",
    )


def _assistant(*, content: str = "", calls: list | None = None):
    return SimpleNamespace(role="assistant", content=content, tool_calls=calls or [])


def _user(content: str):
    return SimpleNamespace(role="user", content=content, tool_calls=[])


def _task_with_actions(actions: list[_ExpectedAction]):
    return SimpleNamespace(
        evaluation_criteria=SimpleNamespace(actions=actions)
    )


class _ScriptedUser:
    def generate_next_message(self, message, state):
        from tau2.data_model.message import UserMessage

        state.append(message.content)
        reply = UserMessage(role="user", content="Yes, I confirm that exact action.")
        state.append(reply.content)
        return reply, state


def _scripted_user_factory(environment, task, messages, seed):
    del environment, task, seed
    return _ScriptedUser(), [message.content for message in messages]


class RetailAgenticEnvironmentTests(unittest.TestCase):
    def make_env(self, reward: float = 1.0) -> RetailAgenticEnvironment:
        return RetailAgenticEnvironment(
            user_factory=_scripted_user_factory,
            evaluator=lambda task, messages: {
                "reward": reward,
                "task_id": str(task.id),
                "message_count": len(messages),
            },
        )

    def test_reset_requires_a_frozen_opening_utterance(self) -> None:
        env = self.make_env()
        with self.assertRaises(ValueError):
            env.reset(task_id="1", initial_user_message="")

    def test_reset_and_real_retail_read_tool(self) -> None:
        env = self.make_env()
        result = env.reset(
            task_id="1",
            initial_user_message="I need help with an order.",
            user_seed=20260810,
        )
        self.assertIsNone(result)
        products = json.loads(env.list_all_product_types())
        self.assertEqual(len(products), 50)
        self.assertEqual(env.get_reward(), 1.0)
        self.assertFalse(env._last_reward_info["policy_findings_are_reward_authority"])

    def test_dynamic_customer_turn_is_part_of_internal_trajectory(self) -> None:
        env = self.make_env(reward=0.25)
        env.reset(
            task_id="1",
            initial_user_message="Please help me update my order.",
        )
        reply = env.respond_to_user(
            "I can do that after you confirm the exact change. Do you confirm?"
        )
        self.assertIn("confirm", reply.lower())
        self.assertEqual(env.get_reward(), 0.25)
        self.assertGreaterEqual(env._last_reward_info["message_count"], 4)

    def test_reward_persists_one_raw_rollout_without_hidden_scenario(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "rollouts.jsonl"
            with patch.dict(os.environ, {"POLICYAGENT_ROLLOUT_LOG": str(path)}):
                env = self.make_env(reward=0.75)
                env.reset(
                    task_id="1",
                    initial_user_message="I need help with an order.",
                    user_seed=17,
                )
                self.assertEqual(env.get_reward(), 0.75)
                self.assertEqual(env.get_reward(), 0.75)
            rows = path.read_text(encoding="utf-8").splitlines()
            self.assertEqual(len(rows), 1)
            record = json.loads(rows[0])
            self.assertEqual(record["task_id"], "1")
            self.assertEqual(record["user_seed"], 17)
            self.assertFalse(record["hidden_user_scenario_persisted"])

    def test_public_methods_are_only_reserved_hooks_and_agent_tools(self) -> None:
        public = {
            name
            for name, value in RetailAgenticEnvironment.__dict__.items()
            if not name.startswith("_") and callable(value)
        }
        self.assertEqual(
            public,
            {
                "reset",
                "get_reward",
                "respond_to_user",
                "calculate",
                "cancel_pending_order",
                "exchange_delivered_order_items",
                "find_user_id_by_name_zip",
                "find_user_id_by_email",
                "get_order_details",
                "get_product_details",
                "get_item_details",
                "get_user_details",
                "list_all_product_types",
                "modify_pending_order_address",
                "modify_pending_order_items",
                "modify_pending_order_payment",
                "modify_user_address",
                "return_delivered_order_items",
                "transfer_to_human_agents",
            },
        )


class ProcessRewardSignalTests(unittest.TestCase):
    def test_one_call_cannot_satisfy_two_duplicate_expected_actions(self) -> None:
        arguments = {"order_id": "#1"}
        task = _task_with_actions(
            [
                _ExpectedAction("a1", "get_order_details", arguments),
                _ExpectedAction("a2", "get_order_details", arguments),
            ]
        )
        messages = [
            _assistant(calls=[_call("c1", "get_order_details", arguments)])
        ]
        progress = one_to_one_action_progress(task, messages)
        self.assertEqual(progress["matched_count"], 1)
        self.assertEqual(progress["recall"], 0.5)

    def test_two_calls_can_satisfy_two_duplicate_expected_actions(self) -> None:
        arguments = {"order_id": "#1"}
        task = _task_with_actions(
            [
                _ExpectedAction("a1", "get_order_details", arguments),
                _ExpectedAction("a2", "get_order_details", arguments),
            ]
        )
        messages = [
            _assistant(
                calls=[
                    _call("c1", "get_order_details", arguments),
                    _call("c2", "get_order_details", arguments),
                ]
            )
        ]
        progress = one_to_one_action_progress(task, messages)
        self.assertEqual(progress["matched_count"], 2)
        self.assertEqual(progress["recall"], 1.0)
        self.assertEqual(progress["duplicate_excess_count"], 0)

    def test_excess_repeated_call_and_unexpected_write_are_counted(self) -> None:
        expected_args = {"order_id": "#1"}
        unexpected_args = {"order_id": "#1", "payment_method_id": "pm_2"}
        task = _task_with_actions(
            [_ExpectedAction("a1", "get_order_details", expected_args)]
        )
        messages = [
            _assistant(
                calls=[
                    _call("c1", "get_order_details", expected_args),
                    _call("c2", "get_order_details", expected_args),
                    _call(
                        "c3", "modify_pending_order_payment", unexpected_args
                    ),
                ]
            )
        ]
        progress = one_to_one_action_progress(task, messages)
        self.assertEqual(progress["duplicate_excess_count"], 1)
        self.assertEqual(progress["unexpected_write_count"], 1)

    def test_confirmation_diagnostic_requires_question_then_affirmation(self) -> None:
        write = _call(
            "c1",
            "modify_pending_order_payment",
            {"order_id": "#1", "payment_method_id": "pm_2"},
        )
        confirmed = confirmation_diagnostics(
            [
                _assistant(content="Do you confirm this exact payment change?"),
                _user("Yes, I confirm."),
                _assistant(calls=[write]),
            ]
        )
        unconfirmed = confirmation_diagnostics([_assistant(calls=[write])])
        self.assertEqual(confirmed["confirmed_write_count"], 1)
        self.assertEqual(confirmed["missing_confirmation_count"], 0)
        self.assertEqual(unconfirmed["missing_confirmation_count"], 1)
        self.assertFalse(confirmed["used_as_reward"])


class RetailAgenticSplitTests(unittest.TestCase):
    def test_frozen_split_is_disjoint_and_reserves_official_test(self) -> None:
        manifest = json.loads(
            (PROJECT / "data" / "retail_agentic_rl_v1" / "task_split.json").read_text(
                encoding="utf-8"
            )
        )
        splits = manifest["splits"]
        train = set(splits["rl_train"])
        validation = set(splits["rl_validation"])
        development = set(splits["development_audit"])
        self.assertEqual(len(train), 44)
        self.assertEqual(len(validation), 10)
        self.assertEqual(len(development), 20)
        self.assertFalse(train & validation)
        self.assertFalse(train & development)
        self.assertFalse(validation & development)
        self.assertEqual(len(train | validation | development), 74)
        self.assertEqual(
            manifest["leakage_checks"]["official_test_overlap_count"], 0
        )

    def test_training_configs_bind_the_implemented_reward_spec(self) -> None:
        for name in (
            "retail_agentic_grpo_v1.json",
            "retail_agentic_grpo_sanity_v1.json",
            "retail_agentic_qwen3_4b_rollout_diagnostic_v1.json",
        ):
            config = json.loads(
                (PROJECT / "configs" / name).read_text(encoding="utf-8")
            )
            self.assertEqual(config["reward"], DEFAULT_REWARD_CONFIG)
            self.assertIn(
                "data/tau2/user_simulator/simulation_guidelines.md",
                config["upstream"]["required_files"],
            )
        sanity = json.loads(
            (
                PROJECT / "configs" / "retail_agentic_grpo_sanity_v1.json"
            ).read_text(encoding="utf-8")
        )
        self.assertEqual(sanity["data"]["max_tasks"], 1)
        self.assertEqual(sanity["grpo"]["max_steps"], 1)
        diagnostic = json.loads(
            (
                PROJECT
                / "configs"
                / "retail_agentic_qwen3_4b_rollout_diagnostic_v1.json"
            ).read_text(encoding="utf-8")
        )
        self.assertEqual(diagnostic["execution_mode"], "ROLLOUT_DIAGNOSTIC")
        self.assertEqual(diagnostic["data"]["max_tasks"], 8)
        self.assertEqual(diagnostic["grpo"]["learning_rate"], 0.0)
        self.assertEqual(diagnostic["grpo"]["beta"], 0.0)
        self.assertEqual(
            diagnostic["diagnostic"]["expected_rollouts"],
            diagnostic["grpo"]["max_steps"]
            * diagnostic["grpo"]["num_generations"],
        )

    def test_rollout_diagnostic_requires_behavior_and_reward_variance(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "rollouts.jsonl"
            rows = []
            for index in range(8):
                rows.append(
                    {
                        "task_id": str(index // 4),
                        "tool_calls": index % 2,
                        "customer_turns": index % 2,
                        "reward": {
                            "reward": float(index % 2),
                            "tool_error_count": 0,
                            "action_progress": {"recall": float(index % 2)},
                        },
                    }
                )
            path.write_text(
                "".join(json.dumps(row) + "\n" for row in rows),
                encoding="utf-8",
            )
            report = analyze(path, expected_rollouts=8)
            self.assertEqual(report["observed_rollouts"], 8)
            self.assertEqual(report["unique_tasks"], 2)
            self.assertTrue(report["gates"]["ready_to_consider_optimization"])

    def test_transferred_upstream_requires_commit_and_package_hash(self) -> None:
        commit = "58e5e1ace69302e6982d27014569c03e0ffccdd2"
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "tau2-bench"
            (root / "src").mkdir(parents=True)
            (root / "data" / "tau2" / "domains" / "retail").mkdir(parents=True)
            guidelines = (
                root
                / "data"
                / "tau2"
                / "user_simulator"
                / "simulation_guidelines.md"
            )
            guidelines.parent.mkdir(parents=True)
            guidelines.write_text("frozen guidelines\n", encoding="utf-8")
            guidelines_digest = (
                hashlib.sha256(guidelines.read_bytes()).hexdigest().upper()
            )
            archive = Path(directory) / "tau2.tar.gz"
            archive.write_bytes(b"frozen tau2 retail package")
            digest = hashlib.sha256(archive.read_bytes()).hexdigest().upper()
            (root / "PINNED_UPSTREAM_COMMIT.txt").write_text(
                commit + "\n", encoding="utf-8"
            )
            (root / "TRANSFER_MANIFEST.json").write_text(
                json.dumps(
                    {
                        "commit": commit,
                        "source_package_path": str(archive),
                    }
                ),
                encoding="utf-8",
            )
            with patch.dict(
                os.environ, {"POLICYAGENT_TAU2_ROOT": str(root)}, clear=False
            ):
                result = validate_upstream_checkout(
                    commit,
                    digest,
                    {
                        "data/tau2/user_simulator/simulation_guidelines.md": (
                            guidelines_digest
                        )
                    },
                )
            self.assertEqual(
                result["verification_method"],
                "commit_marker_and_source_package_sha256",
            )
            self.assertEqual(result["source_package_sha256"], digest)
            self.assertEqual(
                result["required_file_sha256"][
                    "data/tau2/user_simulator/simulation_guidelines.md"
                ],
                guidelines_digest,
            )

    def test_transferred_upstream_rejects_missing_required_file(self) -> None:
        commit = "58e5e1ace69302e6982d27014569c03e0ffccdd2"
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "tau2-bench"
            (root / "src").mkdir(parents=True)
            (root / "data" / "tau2" / "domains" / "retail").mkdir(parents=True)
            archive = Path(directory) / "tau2.tar.gz"
            archive.write_bytes(b"frozen tau2 retail package")
            digest = hashlib.sha256(archive.read_bytes()).hexdigest().upper()
            (root / "PINNED_UPSTREAM_COMMIT.txt").write_text(
                commit + "\n", encoding="utf-8"
            )
            (root / "TRANSFER_MANIFEST.json").write_text(
                json.dumps(
                    {"commit": commit, "source_package_path": str(archive)}
                ),
                encoding="utf-8",
            )
            with patch.dict(
                os.environ, {"POLICYAGENT_TAU2_ROOT": str(root)}, clear=False
            ):
                with self.assertRaisesRegex(FileNotFoundError, "Required tau2 file"):
                    validate_upstream_checkout(
                        commit,
                        digest,
                        {
                            "data/tau2/user_simulator/simulation_guidelines.md": (
                                "0" * 64
                            )
                        },
                    )


if __name__ == "__main__":
    unittest.main()
