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
    IDENTITY_AUTHENTICATION_ACTIONS,
    IDENTITY_AUTHENTICATION_STAGE,
    REQUIRE_TRANSPORT_COMPLETE_ENV,
    RetailAgenticEnvironment,
    TERMINAL_ONLY_REWARD_CONFIG,
    TIERED_TERMINAL_PROCESS_MODE,
    confirmation_diagnostics,
    gate_environment_state_reward,
    identity_authentication_stage_reward,
    load_reward_config,
    one_to_one_action_progress,
    terminal_environment_reward,
    tiered_terminal_process_reward,
    transport_invalid_reasons,
)
from src.training.run_retail_agentic_grpo import (
    selected_task_ids,
    validate_config_and_split,
    validate_optimization_contract,
    validate_sft_manifest_binding,
    validate_terminal_task_eligibility,
    validate_upstream_checkout,
    wrap_retail_policy_for_agentic_protocol,
)
from src.rl.task_split import protect_validation_from_sft_tasks
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


def _tool(content: str):
    return SimpleNamespace(role="tool", content=content, tool_calls=[])


def _tool_result(call_id: str, content: str, *, error: bool = False):
    return SimpleNamespace(
        role="tool",
        id=call_id,
        content=content,
        error=error,
        tool_calls=[],
    )


def _task_with_actions(actions: list[_ExpectedAction]):
    return SimpleNamespace(evaluation_criteria=SimpleNamespace(actions=actions))


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


def _completion_telemetry(**overrides):
    payload = {
        "stop_reason": "MODEL_EOS_BEFORE_USER_STOP",
        "stop_reason_source": "guarded_grpo_trainer_v1",
        "stop_flags": [],
        "model_ended": True,
        "model_eos_observed": True,
        "completion_token_budget_exhausted": False,
        "context_limit_reached": False,
        "tool_iteration_limit_reached": False,
        "unresolved_tool_call": False,
        "framework_loop_abnormal_end": False,
        "prompt_tokens": 100,
        "completion_tokens": 40,
        "model_tokens_retained": 30,
        "observation_tokens_retained": 10,
        "model_completion_truncated": False,
        "model_completion_truncation_source": "guarded_grpo_trainer_v1",
    }
    payload.update(overrides)
    return payload


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
            evidence_path = Path(directory) / "rollout_evidence.jsonl"
            with patch.dict(
                os.environ,
                {
                    "POLICYAGENT_ROLLOUT_LOG": str(path),
                    "POLICYAGENT_ROLLOUT_EVIDENCE_LOG": str(evidence_path),
                },
            ):
                env = self.make_env(reward=0.75)
                env.reset(
                    task_id="1",
                    initial_user_message="I need help with an order.",
                    user_seed=17,
                )
                env.list_all_product_types()
                self.assertEqual(env.get_reward(), 0.75)
                self.assertEqual(env.get_reward(), 0.75)
            rows = path.read_text(encoding="utf-8").splitlines()
            self.assertEqual(len(rows), 1)
            record = json.loads(rows[0])
            self.assertEqual(record["task_id"], "1")
            self.assertEqual(record["user_seed"], 17)
            self.assertFalse(record["hidden_user_scenario_persisted"])
            evidence_rows = evidence_path.read_text(encoding="utf-8").splitlines()
            self.assertEqual(len(evidence_rows), 1)
            evidence = json.loads(evidence_rows[0])
            self.assertEqual(record["evidence_sha256"], evidence["evidence_sha256"])
            self.assertIn("agent", evidence["initial_state"])
            self.assertIn("agent", evidence["final_state"])
            self.assertIsInstance(evidence["state_diff"], list)
            self.assertEqual(len(evidence["state_hashes"]["initial_sha256"]), 64)
            self.assertEqual(len(evidence["state_hashes"]["final_sha256"]), 64)
            self.assertEqual(evidence["terminal_evaluator"]["reward"], 0.75)
            self.assertEqual(
                evidence["tool_trace"][0]["name"], "list_all_product_types"
            )
            self.assertIsNone(evidence["completion"]["model_completion_truncated"])
            self.assertFalse(evidence["hidden_user_scenario_persisted"])

    def test_raw_rollout_requires_evidence_sidecar_path(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "rollouts.jsonl"
            with patch.dict(
                os.environ,
                {
                    "POLICYAGENT_ROLLOUT_LOG": str(path),
                    "POLICYAGENT_ROLLOUT_EVIDENCE_LOG": "",
                },
            ):
                env = self.make_env()
                env.reset(
                    task_id="1",
                    initial_user_message="I need help with an order.",
                )
                with self.assertRaisesRegex(
                    RuntimeError, "POLICYAGENT_ROLLOUT_EVIDENCE_LOG"
                ):
                    env.get_reward()

    def test_guarded_completion_telemetry_is_hash_bound_in_raw_and_evidence(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "rollouts.jsonl"
            evidence_path = Path(directory) / "rollout_evidence.jsonl"
            with patch.dict(
                os.environ,
                {
                    "POLICYAGENT_ROLLOUT_LOG": str(path),
                    "POLICYAGENT_ROLLOUT_EVIDENCE_LOG": str(evidence_path),
                },
            ):
                env = self.make_env(reward=0.5)
                env.reset(
                    task_id="1",
                    initial_user_message="I need help with an order.",
                    user_seed=19,
                )
                telemetry = _completion_telemetry()
                env._set_trainer_completion_telemetry(telemetry)
                self.assertEqual(env.get_reward(), 0.5)
            record = json.loads(path.read_text(encoding="utf-8"))
            evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
            self.assertEqual(record["completion"], evidence["completion"])
            self.assertEqual(
                record["completion"]["stop_reason"], telemetry["stop_reason"]
            )
            self.assertFalse(record["completion"]["model_completion_truncated"])
            without_hash = dict(evidence)
            persisted_hash = without_hash.pop("evidence_sha256")
            canonical = json.dumps(
                without_hash,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
                default=str,
            ).encode("utf-8")
            self.assertEqual(
                hashlib.sha256(canonical).hexdigest().upper(), persisted_hash
            )

    def test_completion_telemetry_resets_and_cannot_be_bound_after_reward(self) -> None:
        env = self.make_env()
        env.reset(task_id="1", initial_user_message="First task")
        env._set_trainer_completion_telemetry(_completion_telemetry())
        env.reset(task_id="1", initial_user_message="Second task")
        self.assertIsNone(env._trainer_completion_telemetry)
        self.assertEqual(env._runtime_stop_signals, [])
        env.get_reward()
        with self.assertRaisesRegex(RuntimeError, "before reward"):
            env._set_trainer_completion_telemetry(_completion_telemetry())

    def test_transport_gate_rejects_censored_rollout_before_reward(self) -> None:
        with patch.dict(os.environ, {REQUIRE_TRANSPORT_COMPLETE_ENV: "1"}):
            env = self.make_env(reward=1.0)
            env.reset(task_id="1", initial_user_message="Need help")
            with self.assertRaisesRegex(RuntimeError, "missing_completion_telemetry"):
                env.get_reward()

            env._set_trainer_completion_telemetry(
                _completion_telemetry(
                    stop_reason="COMPLETION_BUDGET_EXHAUSTED",
                    stop_flags=["COMPLETION_BUDGET_EXHAUSTED"],
                    completion_token_budget_exhausted=True,
                    model_completion_truncated=True,
                )
            )
            with self.assertRaisesRegex(
                RuntimeError, "completion_token_budget_exhausted"
            ):
                env.get_reward()
            self.assertIsNone(env._last_reward_info)

    def test_transport_gate_keeps_model_eos_behavior_failure_reward_eligible(self) -> None:
        telemetry = _completion_telemetry()
        self.assertEqual(transport_invalid_reasons(telemetry), [])
        with patch.dict(os.environ, {REQUIRE_TRANSPORT_COMPLETE_ENV: "1"}):
            env = self.make_env(reward=0.0)
            env.reset(task_id="1", initial_user_message="Need help")
            env._set_trainer_completion_telemetry(telemetry)
            self.assertEqual(env.get_reward(), 0.0)

    def test_transport_gate_environment_value_is_strict(self) -> None:
        with patch.dict(os.environ, {REQUIRE_TRANSPORT_COMPLETE_ENV: "true"}):
            with self.assertRaisesRegex(RuntimeError, "exactly '0' or '1'"):
                self.make_env()

    def test_completion_telemetry_rejects_semantic_inconsistency(self) -> None:
        env = self.make_env()
        env.reset(task_id="1", initial_user_message="Need help")
        with self.assertRaisesRegex(ValueError, "conserve length"):
            env._set_trainer_completion_telemetry(
                _completion_telemetry(model_tokens_retained=31)
            )

        with self.assertRaisesRegex(ValueError, "contradicts stop_flags"):
            env._set_trainer_completion_telemetry(
                _completion_telemetry(model_completion_truncated=True)
            )

        with self.assertRaisesRegex(ValueError, "primary stop reason"):
            env._set_trainer_completion_telemetry(
                _completion_telemetry(
                    stop_reason="TOOL_CALL_LIMIT",
                    stop_flags=["TOOL_CALL_LIMIT"],
                )
            )

    def test_runtime_limits_are_signals_only_after_an_over_limit_attempt(self) -> None:
        env = self.make_env()
        env.reset(task_id="1", initial_user_message="Need help")
        env._max_customer_turns = 0
        self.assertEqual(env._runtime_stop_signals, [])
        with self.assertRaisesRegex(RuntimeError, "Maximum customer turns"):
            env.respond_to_user("Can you confirm?")
        self.assertEqual(env._runtime_stop_signals, ["CUSTOMER_TURN_LIMIT"])

        env.reset(task_id="1", initial_user_message="Need help")
        env._max_tool_calls = 0
        self.assertEqual(env._runtime_stop_signals, [])
        with self.assertRaisesRegex(RuntimeError, "Maximum Retail tool calls"):
            env.list_all_product_types()
        self.assertEqual(env._runtime_stop_signals, ["TOOL_CALL_LIMIT"])

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
    def test_terminal_environment_reward_requires_state_and_normal_user_stop(
        self,
    ) -> None:
        self.assertEqual(terminal_environment_reward(1.0, user_stopped=True), 1.0)
        self.assertEqual(terminal_environment_reward(1.0, user_stopped=False), 0.0)
        self.assertEqual(terminal_environment_reward(0.0, user_stopped=True), 0.0)

    def test_terminal_config_contains_no_process_reward_or_penalty(self) -> None:
        self.assertEqual(
            TERMINAL_ONLY_REWARD_CONFIG["process_reward_mode"],
            "terminal_environment_state",
        )
        self.assertEqual(
            TERMINAL_ONLY_REWARD_CONFIG["environment_state_action_progress_gate"],
            "none",
        )
        self.assertEqual(TERMINAL_ONLY_REWARD_CONFIG["required_action_weight"], 0.0)
        self.assertEqual(TERMINAL_ONLY_REWARD_CONFIG["communication_weight"], 0.0)
        penalty_keys = [key for key in TERMINAL_ONLY_REWARD_CONFIG if "penalty" in key]
        self.assertTrue(penalty_keys)
        self.assertTrue(
            all(TERMINAL_ONLY_REWARD_CONFIG[key] == 0.0 for key in penalty_keys)
        )

    def test_terminal_config_round_trips_through_environment_loader(self) -> None:
        with patch.dict(
            os.environ,
            {"POLICYAGENT_REWARD_CONFIG_JSON": json.dumps(TERMINAL_ONLY_REWARD_CONFIG)},
        ):
            self.assertEqual(load_reward_config(), TERMINAL_ONLY_REWARD_CONFIG)

    def test_tiered_reward_reuses_frozen_v2_scorer_online(self) -> None:
        spec = json.loads(
            (
                PROJECT
                / "configs"
                / "evaluation"
                / "staged_reward_shadow_tasks43_72_v2.json"
            ).read_text(encoding="utf-8")
        )
        messages = [
            _assistant(
                calls=[_call("c1", "find_user_id_by_email", {"email": "x@y.z"})]
            ),
            _tool_result("c1", "lucas_santos_6600"),
            _assistant(
                calls=[
                    _call("c2", "get_user_details", {"user_id": "lucas_santos_6600"})
                ]
            ),
            _tool_result("c2", json.dumps({"orders": ["#W1588712", "#W7895761"]})),
            _assistant(
                calls=[_call("c3", "get_order_details", {"order_id": "#W1588712"})]
            ),
            _tool_result("c3", "The order is delayed."),
            _assistant(
                calls=[_call("c4", "get_order_details", {"order_id": "#W7895761"})]
            ),
            _tool_result("c4", "Tablet order found."),
            _assistant(
                calls=[
                    _call(
                        "c5",
                        "modify_user_address",
                        {
                            "user_id": "lucas_santos_6600",
                            "address1": "943 Maple Drive",
                            "address2": "Suite 356",
                            "city": "Chicago",
                            "state": "IL",
                            "country": "USA",
                            "zip": "60621",
                        },
                    )
                ]
            ),
            _tool_result("c5", "Address updated."),
            _assistant(content="The order is delayed."),
        ]
        score = tiered_terminal_process_reward(
            task_id="43",
            messages=messages,
            action_progress={
                "matches": [
                    {
                        "action_id": "43_4",
                        "matched": True,
                        "matched_call_index": 4,
                        "name": "modify_user_address",
                    }
                ],
                "unexpected_write_count": 0,
            },
            environment_payload={"reward": 1.0},
            communication_payload={
                "communicate_checks": [{"info": "delayed", "met": True}]
            },
            environment_state_reward=1.0,
            user_stopped=True,
            completion={
                "customer_turn_limit_reached": False,
                "tool_call_limit_reached": False,
            },
            staged_reward_spec=spec,
        )
        self.assertEqual(score["staged_reward"], 1.0)
        self.assertEqual(score["terminal_environment_reward"], 1.0)
        self.assertTrue(score["complete_success"])

    def test_tiered_reward_loader_requires_a_frozen_task_spec(self) -> None:
        configured = {
            **DEFAULT_REWARD_CONFIG,
            "process_reward_mode": TIERED_TERMINAL_PROCESS_MODE,
            "environment_state_action_progress_gate": "none",
        }
        with patch.dict(
            os.environ,
            {"POLICYAGENT_REWARD_CONFIG_JSON": json.dumps(configured)},
        ):
            with self.assertRaisesRegex(RuntimeError, "requires staged_reward_spec"):
                load_reward_config()

    def test_confirmation_v4_loader_requires_explicit_reward_authority(self) -> None:
        spec = json.loads(
            (
                PROJECT
                / "configs"
                / "evaluation"
                / "staged_reward_shadow_task43_n4_confirmation_v4_candidate.json"
            ).read_text(encoding="utf-8")
        )
        configured = {
            **DEFAULT_REWARD_CONFIG,
            "process_reward_mode": TIERED_TERMINAL_PROCESS_MODE,
            "environment_state_action_progress_gate": "none",
            "confirmation_signal_used_as_reward": True,
            "staged_reward_spec": spec,
        }
        with patch.dict(
            os.environ,
            {"POLICYAGENT_REWARD_CONFIG_JSON": json.dumps(configured)},
        ):
            loaded = load_reward_config()
        self.assertTrue(loaded["confirmation_signal_used_as_reward"])

        configured["confirmation_signal_used_as_reward"] = False
        with patch.dict(
            os.environ,
            {"POLICYAGENT_REWARD_CONFIG_JSON": json.dumps(configured)},
        ):
            with self.assertRaisesRegex(
                RuntimeError,
                "must match the staged reward composition mode",
            ):
                load_reward_config()

    def test_confirmation_v4_online_scorer_uses_bound_prompt_diagnostic(self) -> None:
        spec = json.loads(
            (
                PROJECT
                / "configs"
                / "evaluation"
                / "staged_reward_shadow_task43_n4_confirmation_v4_candidate.json"
            ).read_text(encoding="utf-8")
        )
        address = {
            "user_id": "lucas_santos_6600",
            "address1": "943 Maple Drive",
            "address2": "Suite 356",
            "city": "Chicago",
            "state": "IL",
            "country": "USA",
            "zip": "60621",
        }
        messages = [
            _assistant(
                calls=[_call("c1", "find_user_id_by_email", {"email": "x@y.z"})]
            ),
            _tool_result("c1", "lucas_santos_6600"),
            _assistant(
                calls=[
                    _call("c2", "get_user_details", {"user_id": "lucas_santos_6600"})
                ]
            ),
            _tool_result("c2", json.dumps({"orders": ["#W1588712", "#W7895761"]})),
            _assistant(
                calls=[_call("c3", "get_order_details", {"order_id": "#W1588712"})]
            ),
            _tool_result("c3", "The order is delayed."),
            _assistant(
                calls=[_call("c4", "get_order_details", {"order_id": "#W7895761"})]
            ),
            _tool_result("c4", "Tablet order found."),
            _assistant(
                content=(
                    "The order is delayed. Do you confirm changing the default "
                    "address to 943 Maple Drive, Suite 356, Chicago, IL 60621, USA?"
                )
            ),
            _user("Yes, please proceed with that exact address."),
            _assistant(calls=[_call("c5", "modify_user_address", address)]),
            _tool_result("c5", "Address updated."),
        ]
        score = tiered_terminal_process_reward(
            task_id="43",
            messages=messages,
            action_progress={
                "matches": [
                    {
                        "action_id": "43_4",
                        "matched": True,
                        "matched_call_index": 4,
                        "name": "modify_user_address",
                    }
                ],
                "unexpected_write_count": 0,
            },
            environment_payload={"reward": 1.0},
            communication_payload={
                "communicate_checks": [{"info": "delayed", "met": True}]
            },
            environment_state_reward=1.0,
            user_stopped=True,
            completion={
                "customer_turn_limit_reached": False,
                "tool_call_limit_reached": False,
            },
            staged_reward_spec=spec,
        )
        self.assertEqual(score["components"]["confirmation_binding"]["value"], 1.0)
        self.assertEqual(score["staged_reward"], 1.0)

    def test_hierarchical_v5_online_scorer_matches_shared_offline_scorer(self) -> None:
        from src.evaluation.staged_reward_shadow import score_rollout

        spec = json.loads(
            (
                PROJECT
                / "configs"
                / "evaluation"
                / "staged_reward_shadow_task43_adaptive_quality_n4_v5_audit.json"
            ).read_text(encoding="utf-8")
        )
        address = {
            "user_id": "lucas_santos_6600",
            "address1": "943 Maple Drive",
            "address2": "Suite 356",
            "city": "Chicago",
            "state": "IL",
            "country": "USA",
            "zip": "60621",
        }
        messages = [
            _assistant(
                calls=[_call("c1", "find_user_id_by_email", {"email": "x@y.z"})]
            ),
            _tool_result("c1", "lucas_santos_6600"),
            _assistant(
                calls=[
                    _call("c2", "get_user_details", {"user_id": "lucas_santos_6600"})
                ]
            ),
            _tool_result("c2", json.dumps({"orders": ["#W1588712", "#W7895761"]})),
            _assistant(
                calls=[_call("c3", "get_order_details", {"order_id": "#W1588712"})]
            ),
            _tool_result("c3", "The order is delayed."),
            _assistant(
                calls=[_call("c4", "get_order_details", {"order_id": "#W7895761"})]
            ),
            _tool_result("c4", "Tablet order found."),
            _assistant(
                content=(
                    "The order is delayed. Do you confirm changing the default "
                    "address to 943 Maple Drive, Suite 356, Chicago, IL 60621, USA?"
                )
            ),
            _user("Yes, please proceed with that exact address."),
            _assistant(calls=[_call("c5", "modify_user_address", address)]),
            _tool_result("c5", "Address updated."),
            _assistant(content="The address update is complete."),
        ]
        action_progress = {
            "matches": [
                {
                    "action_id": "43_4",
                    "matched": True,
                    "matched_call_index": 4,
                    "name": "modify_user_address",
                }
            ],
            "unexpected_write_count": 0,
        }
        environment_payload = {"reward": 1.0}
        communication_payload = {
            "communicate_checks": [{"info": "delayed", "met": True}]
        }
        completion = {
            "customer_turn_limit_reached": False,
            "tool_call_limit_reached": False,
        }
        configured = {
            **DEFAULT_REWARD_CONFIG,
            "process_reward_mode": TIERED_TERMINAL_PROCESS_MODE,
            "environment_state_action_progress_gate": "none",
            "confirmation_signal_used_as_reward": True,
            "staged_reward_spec": spec,
        }
        with patch.dict(
            os.environ,
            {"POLICYAGENT_REWARD_CONFIG_JSON": json.dumps(configured)},
        ):
            loaded = load_reward_config()
        self.assertEqual(
            loaded["staged_reward_spec"]["reward"]["composition_mode"],
            "hierarchical_state_authorization_v5",
        )

        # Capture the exact bound raw/evidence inputs given to the shared scorer,
        # then call that scorer directly to prove online/offline parity.
        with patch(
            "src.evaluation.staged_reward_shadow.score_rollout",
            wraps=score_rollout,
        ) as shared:
            online = tiered_terminal_process_reward(
                task_id="43",
                messages=messages,
                action_progress=action_progress,
                environment_payload=environment_payload,
                communication_payload=communication_payload,
                environment_state_reward=1.0,
                user_stopped=True,
                completion=completion,
                staged_reward_spec=spec,
            )
        shared.assert_called_once()
        raw_arg, evidence_arg, spec_arg = shared.call_args.args
        confirmation_arg = shared.call_args.kwargs["confirmation_diagnostic"]
        offline = score_rollout(
            raw_arg,
            evidence_arg,
            spec_arg,
            confirmation_diagnostic=confirmation_arg,
        )
        self.assertEqual(online["staged_reward"], offline["staged_reward"])
        self.assertEqual(online["additive_components"], offline["additive_components"])
        self.assertEqual(online["staged_reward"], 1.0)

    def test_hierarchical_v6_review_matches_shared_offline_scorer(self) -> None:
        from src.evaluation.staged_reward_shadow import score_rollout

        spec = json.loads(
            (
                PROJECT
                / "configs"
                / "evaluation"
                / "staged_reward_shadow_task44_sft_prescreen_hierarchical_v6_candidate.json"
            ).read_text(encoding="utf-8")
        )
        write_arguments = {
            "order_id": "#W9300146",
            "item_ids": ["9190635437"],
            "new_item_ids": ["5320792178"],
            "payment_method_id": "gift_card_7245904",
        }
        messages = [
            _assistant(
                calls=[
                    _call(
                        "c1",
                        "find_user_id_by_name_zip",
                        {"first_name": "Aarav", "last_name": "Anderson", "zip": "19031"},
                    )
                ]
            ),
            _tool_result("c1", "aarav_anderson_8794"),
            _assistant(
                calls=[
                    _call(
                        "c2",
                        "get_user_details",
                        {"user_id": "aarav_anderson_8794"},
                    )
                ]
            ),
            _tool_result(
                "c2",
                json.dumps(
                    {
                        "orders": ["#W9300146"],
                        "payment_methods": {
                            "gift_card_7245904": {"source": "gift_card"}
                        },
                    }
                ),
            ),
            _assistant(
                calls=[
                    _call(
                        "c3", "get_order_details", {"order_id": "#W9300146"}
                    )
                ]
            ),
            _tool_result("c3", "Pending order with desk lamp 9190635437."),
            _assistant(
                calls=[
                    _call(
                        "c4",
                        "get_product_details",
                        {"product_id": "6817146515"},
                    )
                ]
            ),
            _tool_result("c4", "Available replacement item 5320792178."),
            _assistant(
                content=(
                    "Do you confirm replacing item 9190635437 with 5320792178 "
                    "for order #W9300146?"
                )
            ),
            _user("Yes, please proceed."),
            _assistant(
                calls=[
                    _call(
                        "c5", "modify_pending_order_items", write_arguments
                    )
                ]
            ),
            _tool_result("c5", '{"refund":17.98999999999998}'),
            _assistant(content="The $17.99 refund is complete."),
        ]
        action_progress = {
            "matches": [
                {
                    "action_id": "44_4",
                    "matched": True,
                    "matched_call_index": 4,
                    "name": "modify_pending_order_items",
                }
            ],
            "unexpected_write_count": 0,
        }
        completion = {
            "customer_turn_limit_reached": False,
            "tool_call_limit_reached": False,
        }

        with patch(
            "src.evaluation.staged_reward_shadow.score_rollout",
            wraps=score_rollout,
        ) as shared:
            online = tiered_terminal_process_reward(
                task_id="44",
                messages=messages,
                action_progress=action_progress,
                environment_payload={"reward": 1.0},
                communication_payload={
                    "communicate_checks": [{"info": "17.99", "met": True}]
                },
                environment_state_reward=1.0,
                user_stopped=True,
                completion=completion,
                staged_reward_spec=spec,
            )
        shared.assert_called_once()
        raw_arg, evidence_arg, spec_arg = shared.call_args.args
        confirmation_arg = shared.call_args.kwargs["confirmation_diagnostic"]
        offline = score_rollout(
            raw_arg,
            evidence_arg,
            spec_arg,
            confirmation_diagnostic=confirmation_arg,
        )

        self.assertEqual(
            {
                key: value
                for key, value in online.items()
                if key != "terminal_environment_reward"
            },
            offline,
        )
        self.assertEqual(online["staged_reward"], 0.75)
        self.assertEqual(
            online["components"]["confirmation_binding"]["verdict"], "REVIEW"
        )
        self.assertEqual(
            online["components"]["grounded_communication"]["value"], 1.0
        )

    def test_hierarchical_v6_loader_accepts_explicit_review_semantics(self) -> None:
        spec = json.loads(
            (
                PROJECT
                / "configs"
                / "evaluation"
                / "staged_reward_shadow_task44_sft_prescreen_hierarchical_v6_candidate.json"
            ).read_text(encoding="utf-8")
        )
        configured = {
            **DEFAULT_REWARD_CONFIG,
            "process_reward_mode": TIERED_TERMINAL_PROCESS_MODE,
            "environment_state_action_progress_gate": "none",
            "confirmation_signal_used_as_reward": True,
            "staged_reward_spec": spec,
        }
        with patch.dict(
            os.environ,
            {"POLICYAGENT_REWARD_CONFIG_JSON": json.dumps(configured)},
        ):
            loaded = load_reward_config()
        self.assertEqual(
            loaded["staged_reward_spec"]["reward"]["composition_mode"],
            "hierarchical_state_authorization_review_v6",
        )

    def test_terminal_task_gate_requires_environment_reward_basis(self) -> None:
        eligible = SimpleNamespace(
            id="1",
            evaluation_criteria=SimpleNamespace(
                reward_basis=["db"],
                actions=[SimpleNamespace(name="cancel_pending_order")],
            ),
        )
        query_only = SimpleNamespace(
            id="2",
            evaluation_criteria=SimpleNamespace(
                reward_basis=["db"],
                actions=[SimpleNamespace(name="get_order_details")],
            ),
        )
        result = validate_terminal_task_eligibility(["1"], [eligible, query_only])
        self.assertEqual(result["status"], "TERMINAL_TASKS_ELIGIBLE")
        with self.assertRaisesRegex(ValueError, r"ineligible task IDs: \['2'\]"):
            validate_terminal_task_eligibility(["2"], [eligible, query_only])

    def test_environment_state_reward_is_gated_by_action_progress(self) -> None:
        self.assertEqual(gate_environment_state_reward(1.0, 0.0), (0.0, 0.0))
        self.assertEqual(gate_environment_state_reward(1.0, 0.4), (0.4, 0.4))
        self.assertEqual(gate_environment_state_reward(1.0, None), (1.0, 1.0))

    def test_one_call_cannot_satisfy_two_duplicate_expected_actions(self) -> None:
        arguments = {"order_id": "#1"}
        task = _task_with_actions(
            [
                _ExpectedAction("a1", "get_order_details", arguments),
                _ExpectedAction("a2", "get_order_details", arguments),
            ]
        )
        messages = [_assistant(calls=[_call("c1", "get_order_details", arguments)])]
        progress = one_to_one_action_progress(task, messages)
        self.assertEqual(progress["matched_count"], 1)
        self.assertEqual(progress["recall"], 0.5)

    def test_action_progress_can_be_scoped_to_identity_authentication(self) -> None:
        task = _task_with_actions(
            [
                _ExpectedAction(
                    "auth",
                    "find_user_id_by_email",
                    {"email": "user@example.com"},
                ),
                _ExpectedAction(
                    "order",
                    "get_order_details",
                    {"order_id": "#1"},
                ),
            ]
        )
        progress = one_to_one_action_progress(
            task,
            [
                _assistant(
                    calls=[
                        _call(
                            "c1",
                            "find_user_id_by_email",
                            {"email": "user@example.com"},
                        )
                    ]
                )
            ],
            expected_action_names=IDENTITY_AUTHENTICATION_ACTIONS,
        )
        self.assertEqual(progress["expected_count"], 1)
        self.assertEqual(progress["matched_count"], 1)
        self.assertEqual(progress["recall"], 1.0)

    def test_identity_stage_rewards_correct_hidden_authentication_action(self) -> None:
        task = _task_with_actions(
            [
                _ExpectedAction(
                    "auth",
                    "find_user_id_by_email",
                    {"email": "user@example.com"},
                ),
                _ExpectedAction(
                    "order",
                    "get_order_details",
                    {"order_id": "#1"},
                ),
            ]
        )
        correct = identity_authentication_stage_reward(
            task,
            [
                _assistant(
                    calls=[
                        _call(
                            "c1",
                            "find_user_id_by_email",
                            {"email": "user@example.com"},
                        )
                    ]
                )
            ],
            DEFAULT_REWARD_CONFIG,
        )
        wrong = identity_authentication_stage_reward(
            task,
            [
                _assistant(
                    calls=[
                        _call(
                            "c2",
                            "find_user_id_by_email",
                            {"email": "wrong@example.com"},
                        )
                    ]
                )
            ],
            DEFAULT_REWARD_CONFIG,
        )
        self.assertEqual(correct["reward"], 1.0)
        self.assertTrue(correct["stage_complete"])
        self.assertEqual(correct["unfinished_interaction_penalty"], 0.0)
        self.assertEqual(wrong["reward"], 0.0)
        self.assertFalse(wrong["stage_complete"])
        self.assertEqual(
            wrong["unfinished_interaction_penalty"],
            DEFAULT_REWARD_CONFIG["unfinished_interaction_penalty"],
        )

    def test_identity_stage_requires_exactly_one_hidden_auth_action(self) -> None:
        task = _task_with_actions(
            [
                _ExpectedAction(
                    "order",
                    "get_order_details",
                    {"order_id": "#1"},
                )
            ]
        )
        with self.assertRaisesRegex(RuntimeError, "exactly one"):
            identity_authentication_stage_reward(task, [], DEFAULT_REWARD_CONFIG)

    def test_identity_stage_rejects_correct_auth_followed_by_extra_tool(self) -> None:
        task = _task_with_actions(
            [
                _ExpectedAction(
                    "auth",
                    "find_user_id_by_email",
                    {"email": "user@example.com"},
                )
            ]
        )
        result = identity_authentication_stage_reward(
            task,
            [
                _assistant(
                    calls=[
                        _call(
                            "c1",
                            "find_user_id_by_email",
                            {"email": "user@example.com"},
                        ),
                        _call(
                            "c2",
                            "get_user_details",
                            {"user_id": "user_1"},
                        ),
                    ]
                )
            ],
            DEFAULT_REWARD_CONFIG,
        )
        self.assertEqual(result["action_progress"]["recall"], 1.0)
        self.assertFalse(result["stage_complete"])
        self.assertEqual(result["reward"], 0.0)

    def test_identity_stage_prompt_is_explicit_but_does_not_expose_gold(self) -> None:
        policy = "Authenticate users before accessing account data."
        full_task = wrap_retail_policy_for_agentic_protocol(policy)
        staged = wrap_retail_policy_for_agentic_protocol(
            policy, IDENTITY_AUTHENTICATION_STAGE
        )
        self.assertNotIn("<stage_contract>", full_task)
        self.assertIn("<stage_contract>", staged)
        self.assertIn("stop immediately", staged)
        self.assertNotIn("user@example.com", staged)

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

    def test_item_id_order_does_not_invalidate_equivalent_return(self) -> None:
        task = _task_with_actions(
            [
                _ExpectedAction(
                    "a1",
                    "return_delivered_order_items",
                    {
                        "order_id": "#1",
                        "item_ids": ["A", "B", "C"],
                        "payment_method_id": "pm_1",
                    },
                )
            ]
        )
        messages = [
            _assistant(
                calls=[
                    _call(
                        "c1",
                        "return_delivered_order_items",
                        {
                            "order_id": "#1",
                            "item_ids": ["C", "B", "A"],
                            "payment_method_id": "pm_1",
                        },
                    )
                ]
            )
        ]

        progress = one_to_one_action_progress(task, messages)

        self.assertEqual(progress["recall"], 1.0)
        self.assertEqual(progress["unexpected_write_count"], 0)

    def test_reordered_item_pairs_match_but_changed_mapping_does_not(self) -> None:
        expected = {
            "order_id": "#1",
            "item_ids": ["A", "B"],
            "new_item_ids": ["X", "Y"],
            "payment_method_id": "pm_1",
        }
        task = _task_with_actions(
            [_ExpectedAction("a1", "modify_pending_order_items", expected)]
        )
        equivalent = _call(
            "c1",
            "modify_pending_order_items",
            {
                **expected,
                "item_ids": ["B", "A"],
                "new_item_ids": ["Y", "X"],
            },
        )
        wrong_mapping = _call(
            "c2",
            "modify_pending_order_items",
            {
                **expected,
                "item_ids": ["B", "A"],
                "new_item_ids": ["X", "Y"],
            },
        )

        equivalent_progress = one_to_one_action_progress(
            task, [_assistant(calls=[equivalent])]
        )
        wrong_progress = one_to_one_action_progress(
            task, [_assistant(calls=[wrong_mapping])]
        )

        self.assertEqual(equivalent_progress["recall"], 1.0)
        self.assertEqual(wrong_progress["recall"], 0.0)

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
                    _call("c3", "modify_pending_order_payment", unexpected_args),
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

    def test_modify_user_address_requires_confirmation_diagnostic(self) -> None:
        write = _call(
            "c1",
            "modify_user_address",
            {
                "user_id": "user_1",
                "new_address": {
                    "address1": "101 Highway",
                    "city": "New York",
                    "state": "NY",
                    "zip": "10001",
                    "country": "USA",
                },
            },
        )
        confirmed = confirmation_diagnostics(
            [
                _assistant(content="Do you confirm this default address change?"),
                _user("Yes, I confirm."),
                _assistant(calls=[write]),
            ]
        )
        unconfirmed = confirmation_diagnostics([_assistant(calls=[write])])

        self.assertEqual(confirmed["write_count"], 1)
        self.assertEqual(confirmed["confirmed_write_count"], 1)
        self.assertEqual(unconfirmed["missing_confirmation_count"], 1)

    def test_modify_user_address_confirmation_binds_all_flat_address_fields(self) -> None:
        write = _call(
            "c1",
            "modify_user_address",
            {
                "user_id": "lucas_santos_6600",
                "address1": "943 Maple Drive",
                "address2": "Suite 356",
                "city": "Chicago",
                "state": "IL",
                "country": "USA",
                "zip": "60621",
            },
        )
        result = confirmation_diagnostics(
            [
                _assistant(
                    content=(
                        "Do you confirm changing the default address to "
                        "943 Maple Drive, Suite 356, Chicago, IL 60621, USA?"
                    )
                ),
                _user("Yes, please make that my default address."),
                _assistant(calls=[write]),
            ]
        )

        binding = result["checks"][0]["parameter_binding"]
        self.assertEqual(binding["verdict"], "PASS")
        self.assertEqual(binding["missing_fields"], [])
        self.assertFalse(binding["used_as_reward"])

    def test_is_that_correct_binds_explicit_address_confirmation(self) -> None:
        write = _call(
            "c1",
            "modify_user_address",
            {
                "user_id": "lucas_santos_6600",
                "address1": "943 Maple Drive",
                "address2": "Suite 356",
                "city": "Chicago",
                "state": "IL",
                "country": "USA",
                "zip": "60621",
            },
        )
        result = confirmation_diagnostics(
            [
                _assistant(
                    content=(
                        "I will set the default address to 943 Maple Drive, "
                        "Suite 356, Chicago, IL 60621, USA. Is that correct?"
                    )
                ),
                _user("Yes, that is correct. Please update it."),
                _assistant(calls=[write]),
            ]
        )

        self.assertEqual(result["confirmed_write_count"], 1)
        self.assertEqual(
            result["checks"][0]["parameter_binding"]["verdict"], "PASS"
        )

    def test_modify_user_address_confirmation_reviews_missing_address_field(self) -> None:
        write = _call(
            "c1",
            "modify_user_address",
            {
                "user_id": "lucas_santos_6600",
                "address1": "943 Maple Drive",
                "address2": "Suite 356",
                "city": "Chicago",
                "state": "IL",
                "country": "USA",
                "zip": "60621",
            },
        )
        result = confirmation_diagnostics(
            [
                _assistant(
                    content=(
                        "Do you confirm changing the default address to "
                        "943 Maple Drive, Chicago, IL 60621, USA?"
                    )
                ),
                _user("Yes, please proceed."),
                _assistant(calls=[write]),
            ]
        )

        binding = result["checks"][0]["parameter_binding"]
        self.assertEqual(binding["verdict"], "REVIEW")
        self.assertEqual(binding["missing_fields"], ["address2"])

    def test_user_reply_cannot_supply_missing_confirmation_summary_fields(self) -> None:
        write = _call(
            "c1",
            "modify_user_address",
            {
                "user_id": "lucas_santos_6600",
                "address1": "943 Maple Drive",
                "address2": "Suite 356",
                "city": "Chicago",
                "state": "IL",
                "country": "USA",
                "zip": "60621",
            },
        )
        result = confirmation_diagnostics(
            [
                _assistant(content="Do you confirm this default address change?"),
                _user(
                    "Yes, change it to 943 Maple Drive, Suite 356, Chicago, "
                    "IL 60621, USA."
                ),
                _assistant(calls=[write]),
            ]
        )

        self.assertEqual(result["confirmed_write_count"], 1)
        binding = result["checks"][0]["parameter_binding"]
        self.assertEqual(binding["verdict"], "REVIEW")
        self.assertEqual(
            binding["missing_fields"],
            ["address1", "address2", "city", "state", "country", "zip"],
        )

    def test_delivery_confirmation_mention_is_not_write_confirmation(self) -> None:
        write = _call(
            "c1",
            "modify_user_address",
            {
                "user_id": "lucas_santos_6600",
                "address1": "943 Maple Drive",
                "address2": "Suite 356",
                "city": "Chicago",
                "state": "IL",
                "country": "USA",
                "zip": "60621",
            },
        )
        result = confirmation_diagnostics(
            [
                _assistant(
                    content=(
                        "I do not have delivery confirmation or a photo. "
                        "What default address would you like to use?"
                    )
                ),
                _user(
                    "Yes, use 943 Maple Drive, Suite 356, Chicago, IL 60621."
                ),
                _assistant(calls=[write]),
            ]
        )

        self.assertEqual(result["confirmed_write_count"], 0)
        self.assertEqual(result["missing_confirmation_count"], 1)

    def test_current_state_confirmation_is_not_change_confirmation(self) -> None:
        write = _call(
            "c1",
            "modify_user_address",
            {
                "user_id": "lucas_santos_6600",
                "address1": "943 Maple Drive",
                "address2": "Suite 356",
                "city": "Chicago",
                "state": "IL",
                "country": "USA",
                "zip": "60621",
            },
        )
        result = confirmation_diagnostics(
            [
                _assistant(
                    content=(
                        "Let me confirm your current address is in Denver. "
                        "What new address would you like to use?"
                    )
                ),
                _user(
                    "Use 943 Maple Drive, Suite 356, Chicago, IL 60621. Go ahead."
                ),
                _assistant(calls=[write]),
            ]
        )

        self.assertEqual(result["confirmed_write_count"], 0)
        self.assertEqual(result["missing_confirmation_count"], 1)

    def test_one_explicit_batch_confirmation_covers_writes_in_same_turn(self) -> None:
        first = _call(
            "c1",
            "cancel_pending_order",
            {"order_id": "#1", "reason": "ordered by mistake"},
        )
        second = _call(
            "c2",
            "cancel_pending_order",
            {"order_id": "#2", "reason": "ordered by mistake"},
        )
        result = confirmation_diagnostics(
            [
                _assistant(content="Do you confirm cancelling orders #1 and #2?"),
                _user("Yes, cancel orders #1 and #2."),
                _assistant(calls=[first, second]),
            ]
        )
        self.assertEqual(result["write_count"], 2)
        self.assertEqual(result["confirmed_write_count"], 2)
        self.assertEqual(result["missing_confirmation_count"], 0)

    def test_batch_confirmation_does_not_authorize_unmentioned_order(self) -> None:
        first = _call(
            "c1",
            "cancel_pending_order",
            {"order_id": "#1", "reason": "ordered by mistake"},
        )
        second = _call(
            "c2",
            "cancel_pending_order",
            {"order_id": "#2", "reason": "ordered by mistake"},
        )
        result = confirmation_diagnostics(
            [
                _assistant(content="Do you confirm cancelling order #1?"),
                _user("Yes, cancel order #1."),
                _assistant(calls=[first, second]),
            ]
        )
        self.assertEqual(result["confirmed_write_count"], 1)
        self.assertEqual(result["missing_confirmation_count"], 1)

    def test_confirmation_covers_sequential_writes_for_mentioned_order(self) -> None:
        address = _call(
            "c1",
            "modify_pending_order_address",
            {"order_id": "#1", "new_address": {"zip": "10001"}},
        )
        items = _call(
            "c2",
            "modify_pending_order_items",
            {"order_id": "#1", "item_ids": ["a"], "new_item_ids": ["b"]},
        )

        result = confirmation_diagnostics(
            [
                _assistant(content="Do you confirm both changes for order #1?"),
                _user("Yes, proceed with both changes for order #1."),
                _assistant(calls=[address]),
                _assistant(calls=[items]),
            ]
        )

        self.assertEqual(result["confirmed_write_count"], 2)
        self.assertEqual(result["missing_confirmation_count"], 0)

    def test_new_user_message_invalidates_unused_batch_confirmation(self) -> None:
        first = _call(
            "c1",
            "modify_pending_order_address",
            {"order_id": "#1", "new_address": {"zip": "10001"}},
        )
        second = _call(
            "c2",
            "modify_pending_order_items",
            {"order_id": "#1", "item_ids": ["a"], "new_item_ids": ["b"]},
        )

        result = confirmation_diagnostics(
            [
                _assistant(content="Do you confirm both changes for order #1?"),
                _user("Yes, proceed with both changes for order #1."),
                _assistant(calls=[first]),
                _user("Actually, I want something different now."),
                _assistant(calls=[second]),
            ]
        )

        self.assertEqual(result["confirmed_write_count"], 1)
        self.assertEqual(result["missing_confirmation_count"], 1)

    def test_payment_alias_is_resolved_from_tool_observed_order_state(self) -> None:
        action = _call(
            "c1",
            "return_delivered_order_items",
            {
                "order_id": "#1",
                "item_ids": ["item-1"],
                "payment_method_id": "credit_card_123",
            },
        )
        order = json.dumps(
            {
                "order_id": "#1",
                "payment_history": [
                    {
                        "transaction_type": "payment",
                        "payment_method_id": "credit_card_123",
                    }
                ],
            }
        )

        result = confirmation_diagnostics(
            [
                _tool(order),
                _assistant(
                    content=(
                        "Confirm returning item-1 from order #1 to the original "
                        "payment method?"
                    )
                ),
                _user("Yes, proceed."),
                _assistant(calls=[action]),
            ]
        )

        binding = result["checks"][0]["parameter_binding"]
        self.assertEqual(binding["verdict"], "PASS")
        self.assertEqual(
            binding["field_checks"][2]["values"][0]["matched_aliases"],
            ["original payment method"],
        )

    def test_unique_card_brand_is_resolved_from_tool_observed_user_state(self) -> None:
        action = _call(
            "c1",
            "modify_pending_order_payment",
            {"order_id": "#1", "payment_method_id": "credit_card_visa"},
        )
        user = json.dumps(
            {
                "payment_methods": {
                    "credit_card_visa": {
                        "source": "credit_card",
                        "brand": "visa",
                        "last_four": "8902",
                    },
                    "credit_card_mastercard": {
                        "source": "credit_card",
                        "brand": "mastercard",
                        "last_four": "4336",
                    },
                }
            }
        )

        result = confirmation_diagnostics(
            [
                _tool(user),
                _assistant(content="Confirm changing order #1 to your Visa?"),
                _user("Yes, use my Visa."),
                _assistant(calls=[action]),
            ]
        )

        binding = result["checks"][0]["parameter_binding"]
        self.assertEqual(binding["verdict"], "PASS")
        self.assertIn(
            "visa", binding["field_checks"][1]["values"][0]["matched_aliases"]
        )

    def test_ambiguous_card_brand_is_not_accepted_as_payment_binding(self) -> None:
        action = _call(
            "c1",
            "modify_pending_order_payment",
            {"order_id": "#1", "payment_method_id": "credit_card_visa_a"},
        )
        user = json.dumps(
            {
                "payment_methods": {
                    "credit_card_visa_a": {
                        "source": "credit_card",
                        "brand": "visa",
                        "last_four": "1111",
                    },
                    "credit_card_visa_b": {
                        "source": "credit_card",
                        "brand": "visa",
                        "last_four": "2222",
                    },
                }
            }
        )

        result = confirmation_diagnostics(
            [
                _tool(user),
                _assistant(content="Confirm changing order #1 to your Visa?"),
                _user("Yes, use my Visa."),
                _assistant(calls=[action]),
            ]
        )

        binding = result["checks"][0]["parameter_binding"]
        self.assertEqual(binding["verdict"], "REVIEW")
        self.assertIn("payment_method_id", binding["missing_fields"])


class RetailAgenticSplitTests(unittest.TestCase):
    def test_sft_seen_validation_tasks_are_reassigned_to_rl_train(self) -> None:
        tasks = [
            SimpleNamespace(
                id=str(task_id),
                evaluation_criteria=SimpleNamespace(actions=[]),
            )
            for task_id in range(5)
        ]
        split = {
            "rl_train": ["0", "1"],
            "rl_validation": ["2", "3"],
            "development_audit": ["4"],
            "strata": {"candidate_counts": {"query": 4}, "validation_counts": {}},
        }
        reassigned = protect_validation_from_sft_tasks(
            split,
            tasks=tasks,
            sft_task_ids={"2"},
        )
        self.assertEqual(reassigned, ["2"])
        self.assertEqual(split["rl_train"], ["0", "1", "2"])
        self.assertEqual(split["rl_validation"], ["3"])
        self.assertEqual(split["strata"]["validation_counts"], {"query": 1})

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
        self.assertEqual(manifest["leakage_checks"]["official_test_overlap_count"], 0)

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
            (PROJECT / "configs" / "retail_agentic_grpo_sanity_v1.json").read_text(
                encoding="utf-8"
            )
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
        self.assertTrue(diagnostic["quantization"]["enabled"])
        self.assertEqual(diagnostic["quantization"]["mode"], "4bit_nf4")
        self.assertEqual(
            diagnostic["diagnostic"]["expected_rollouts"],
            diagnostic["grpo"]["max_steps"] * diagnostic["grpo"]["num_generations"],
        )
        self.assertEqual(
            diagnostic["diagnostic"]["expected_tasks"],
            diagnostic["data"]["max_tasks"],
        )

    def test_v2_split_and_sft_diagnostic_are_bound_and_clean(self) -> None:
        split = json.loads(
            (PROJECT / "data" / "retail_agentic_rl_v2" / "task_split.json").read_text(
                encoding="utf-8"
            )
        )
        train = set(split["splits"]["rl_train"])
        validation = set(split["splits"]["rl_validation"])
        development = set(split["splits"]["development_audit"])
        self.assertEqual((len(train), len(validation), len(development)), (47, 7, 20))
        self.assertFalse(train & validation)
        self.assertFalse(train & development)
        self.assertFalse(validation & development)
        self.assertEqual(len(train | validation | development), 74)
        self.assertEqual(
            split["sft_task_isolation"]["reassigned_from_rl_validation_to_rl_train"],
            ["2", "3", "4"],
        )
        self.assertEqual(
            split["leakage_checks"]["rl_validation_sft_task_overlap_count"], 0
        )

        config = json.loads(
            (
                PROJECT
                / "configs"
                / "retail_agentic_qwen3_4b_sft_v3_rollout_diagnostic_v2.json"
            ).read_text(encoding="utf-8")
        )
        self.assertEqual(config["reward"], DEFAULT_REWARD_CONFIG)
        self.assertEqual(config["model"]["source_stage"], "SFT")
        self.assertEqual(
            config["model"]["training_data_manifest_sha256"],
            split["source"]["sft_data_manifest_sha256"],
        )
        self.assertEqual(config["data"]["max_tasks"], 4)
        self.assertEqual(config["data"]["task_ids"], ["0", "7", "10", "15"])
        self.assertEqual(selected_task_ids(config, split), ["0", "7", "10", "15"])
        self.assertEqual(config["grpo"]["num_generations"], 2)
        self.assertEqual(config["grpo"]["max_steps"], 8)
        self.assertEqual(config["diagnostic"]["expected_rollouts"], 16)
        self.assertEqual(config["diagnostic"]["expected_rollouts_per_task"], 4)

        openings_path = PROJECT / config["data"]["openings"]
        openings_manifest = json.loads(
            (PROJECT / config["data"]["openings_manifest"]).read_text(encoding="utf-8")
        )
        opening_rows = [
            json.loads(line)
            for line in openings_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        self.assertEqual(
            [row["task_id"] for row in opening_rows], config["data"]["task_ids"]
        )
        self.assertEqual(
            hashlib.sha256(openings_path.read_bytes()).hexdigest().upper(),
            openings_manifest["output_sha256"],
        )
        self.assertEqual(
            openings_manifest["task_split_sha256"],
            hashlib.sha256((PROJECT / config["data"]["task_split"]).read_bytes())
            .hexdigest()
            .upper(),
        )
        self.assertNotIn(
            b"\r\n",
            (PROJECT / config["data"]["task_split"]).read_bytes(),
            "Frozen RL v2 data must remain LF-stable across Windows and Linux",
        )

    def test_identity_authentication_diagnostic_is_stage_bound(self) -> None:
        config = json.loads(
            (
                PROJECT
                / "configs"
                / "retail_agentic_qwen3_4b_identity_auth_rollout_diagnostic_v1.json"
            ).read_text(encoding="utf-8")
        )
        self.assertEqual(config["execution_mode"], "ROLLOUT_DIAGNOSTIC")
        self.assertEqual(config["rollout"]["stage"], IDENTITY_AUTHENTICATION_STAGE)
        self.assertEqual(config["reward"], DEFAULT_REWARD_CONFIG)
        self.assertEqual(config["grpo"]["learning_rate"], 0.0)
        self.assertEqual(config["grpo"]["beta"], 0.0)
        self.assertEqual(config["grpo"]["max_completion_length"], 384)
        self.assertEqual(config["data"]["task_ids"], ["0", "7", "10", "15"])

    def test_terminal_prescreen_uses_derived_sft_manifest_chain(self) -> None:
        config = json.loads(
            (
                PROJECT
                / "configs"
                / "retail_agentic_qwen3_4b_terminal_only_prescreen_tasks43_72_v1.json"
            ).read_text(encoding="utf-8")
        )
        split_path = PROJECT / config["data"]["task_split"]
        split = json.loads(split_path.read_text(encoding="utf-8"))
        binding = validate_sft_manifest_binding(config["model"], split)
        self.assertEqual(binding["binding_type"], "DERIVED")
        self.assertEqual(
            binding["training_data_manifest_sha256"],
            config["model"]["training_data_manifest_sha256"],
        )
        self.assertEqual(
            binding["source_manifest_sha256"],
            split["source"]["sft_data_manifest_sha256"],
        )
        self.assertEqual(
            selected_task_ids(config, split),
            ["43", "72"],
        )
        self.assertEqual(config["reward"], TERMINAL_ONLY_REWARD_CONFIG)
        self.assertEqual(config["grpo"]["learning_rate"], 0.0)
        self.assertEqual(config["grpo"]["beta"], 0.0)
        self.assertEqual(config["diagnostic"]["expected_rollouts"], 16)
        self.assertTrue(config["diagnostic"]["task_selection_uses_reward"])

        openings_path = PROJECT / config["data"]["openings"]
        openings_manifest = json.loads(
            (PROJECT / config["data"]["openings_manifest"]).read_text(encoding="utf-8")
        )
        rows = [
            json.loads(line)
            for line in openings_path.read_text(encoding="utf-8").splitlines()
        ]
        self.assertEqual([row["task_id"] for row in rows], ["43", "72"])
        self.assertEqual(
            hashlib.sha256(openings_path.read_bytes()).hexdigest().upper(),
            openings_manifest["output_sha256"],
        )
        self.assertEqual(
            hashlib.sha256(split_path.read_bytes()).hexdigest().upper(),
            openings_manifest["task_split_sha256"],
        )
        self.assertTrue(openings_manifest["selection_used_reward_values"])
        self.assertTrue(
            all(row["hidden_user_scenario_persisted"] is False for row in rows)
        )

        wrong_source = json.loads(json.dumps(config["model"]))
        wrong_source["training_data_source_manifest_sha256"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "differs from the RL split"):
            validate_sft_manifest_binding(wrong_source, split)

    def test_staged_prescreen_changes_only_the_reward_configuration(self) -> None:
        terminal_path = (
            PROJECT
            / "configs"
            / "retail_agentic_qwen3_4b_terminal_only_prescreen_tasks43_72_v1.json"
        )
        staged_path = (
            PROJECT
            / "configs"
            / "retail_agentic_qwen3_4b_staged_reward_prescreen_tasks43_72_v2.json"
        )
        terminal = json.loads(terminal_path.read_text(encoding="utf-8"))
        staged = json.loads(staged_path.read_text(encoding="utf-8"))
        terminal_without_reward = {k: v for k, v in terminal.items() if k != "reward"}
        staged_without_reward = {k: v for k, v in staged.items() if k != "reward"}
        self.assertEqual(staged_without_reward, terminal_without_reward)
        validated = validate_config_and_split(staged_path)
        self.assertEqual(
            validated["config"]["reward"]["process_reward_mode"],
            TIERED_TERMINAL_PROCESS_MODE,
        )

    def test_task43_confirmation_v4_changes_only_reward_from_terminal_n4(self) -> None:
        terminal = json.loads(
            (
                PROJECT
                / "configs"
                / "retail_agentic_qwen3_4b_terminal_grpo_closure_task43_n4_v1.json"
            ).read_text(encoding="utf-8")
        )
        staged = json.loads(
            (
                PROJECT
                / "configs"
                / "retail_agentic_qwen3_4b_staged_confirmation_grpo_closure_task43_n4_v1.json"
            ).read_text(encoding="utf-8")
        )
        self.assertEqual(
            {key: value for key, value in terminal.items() if key != "reward"},
            {key: value for key, value in staged.items() if key != "reward"},
        )
        self.assertEqual(staged["grpo"]["num_generations"], 4)
        self.assertEqual(staged["grpo"]["max_steps"], 2)
        self.assertEqual(staged["grpo"]["beta"], 0.02)
        self.assertTrue(staged["reward"]["confirmation_signal_used_as_reward"])
        self.assertEqual(
            staged["reward"]["staged_reward_spec"]["reward"]["composition_mode"],
            "additive_terminal_process_confirmation_v4",
        )

    def test_write_coverage_diagnostic_changes_only_sampling_count(self) -> None:
        staged_path = (
            PROJECT
            / "configs"
            / "retail_agentic_qwen3_4b_staged_reward_prescreen_tasks43_72_v2.json"
        )
        coverage_path = (
            PROJECT
            / "configs"
            / "retail_agentic_qwen3_4b_staged_reward_write_coverage_tasks43_72_v1.json"
        )
        staged = json.loads(staged_path.read_text(encoding="utf-8"))
        coverage = json.loads(coverage_path.read_text(encoding="utf-8"))
        staged["grpo"]["max_steps"] = 16
        staged["diagnostic"]["expected_rollouts"] = 32
        staged["diagnostic"]["expected_rollouts_per_task"] = 16
        self.assertEqual(coverage, staged)
        validated = validate_config_and_split(coverage_path)
        self.assertEqual(validated["config"]["grpo"]["learning_rate"], 0.0)
        self.assertEqual(validated["config"]["grpo"]["num_generations"], 2)
        self.assertEqual(validated["config"]["grpo"]["temperature"], 0.8)
        self.assertEqual(validated["config"]["diagnostic"]["expected_rollouts"], 32)

    def test_expanded_identity_diagnostic_reuses_frozen_openings(self) -> None:
        config = json.loads(
            (
                PROJECT
                / "configs"
                / "retail_agentic_qwen3_4b_identity_auth_rollout_diagnostic_v2.json"
            ).read_text(encoding="utf-8")
        )
        manifest = json.loads(
            (PROJECT / config["data"]["openings_manifest"]).read_text(encoding="utf-8")
        )
        openings_path = PROJECT / config["data"]["openings"]
        rows = [
            json.loads(line)
            for line in openings_path.read_text(encoding="utf-8").splitlines()
        ]
        self.assertEqual(config["rollout"]["stage"], IDENTITY_AUTHENTICATION_STAGE)
        self.assertEqual(config["data"]["max_tasks"], 8)
        self.assertEqual(config["diagnostic"]["expected_rollouts"], 32)
        self.assertEqual(config["grpo"]["max_steps"], 16)
        self.assertEqual(config["grpo"]["num_generations"], 2)
        self.assertEqual([row["task_id"] for row in rows], config["data"]["task_ids"])
        self.assertEqual(manifest["rows"], len(rows))
        self.assertEqual(
            hashlib.sha256(openings_path.read_bytes()).hexdigest().upper(),
            manifest["output_sha256"],
        )
        self.assertEqual(
            hashlib.sha256((PROJECT / config["data"]["task_split"]).read_bytes())
            .hexdigest()
            .upper(),
            manifest["task_split_sha256"],
        )
        self.assertIn("no_external_api_call", manifest["derivation"])

    def test_clean_identity_diagnostic_excludes_only_system_failure_task(self) -> None:
        config = json.loads(
            (
                PROJECT
                / "configs"
                / "retail_agentic_qwen3_4b_identity_auth_rollout_diagnostic_clean_v3.json"
            ).read_text(encoding="utf-8")
        )
        manifest = json.loads(
            (PROJECT / config["data"]["openings_manifest"]).read_text(encoding="utf-8")
        )
        openings_path = PROJECT / config["data"]["openings"]
        rows = [
            json.loads(line)
            for line in openings_path.read_text(encoding="utf-8").splitlines()
        ]
        source_openings_path = PROJECT / manifest["source_openings_path"]
        source_rows = [
            json.loads(line)
            for line in source_openings_path.read_text(encoding="utf-8").splitlines()
        ]
        self.assertEqual(config["execution_mode"], "ROLLOUT_DIAGNOSTIC")
        self.assertEqual(config["rollout"]["stage"], IDENTITY_AUTHENTICATION_STAGE)
        self.assertEqual(
            config["data"]["task_ids"],
            ["7", "10", "11", "13", "15", "20", "22"],
        )
        self.assertEqual(config["data"]["max_tasks"], 7)
        self.assertEqual(config["grpo"]["max_steps"], 14)
        self.assertEqual(config["grpo"]["learning_rate"], 0.0)
        self.assertEqual(config["diagnostic"]["expected_rollouts"], 28)
        self.assertEqual(config["diagnostic"]["expected_tasks"], 7)
        self.assertEqual([row["task_id"] for row in rows], config["data"]["task_ids"])
        self.assertEqual(source_rows[1:], rows)
        self.assertEqual(manifest["excluded_task_ids"], ["0"])
        self.assertFalse(manifest["selection_used_reward_values"])
        self.assertIn("no_external_api_call", manifest["derivation"])
        self.assertEqual(
            hashlib.sha256(openings_path.read_bytes()).hexdigest().upper(),
            manifest["output_sha256"],
        )
        self.assertEqual(
            hashlib.sha256(source_openings_path.read_bytes()).hexdigest().upper(),
            manifest["source_openings_sha256"],
        )
        self.assertNotIn(b"\r\n", openings_path.read_bytes())
        self.assertEqual(
            hashlib.sha256((PROJECT / config["data"]["task_split"]).read_bytes())
            .hexdigest()
            .upper(),
            manifest["task_split_sha256"],
        )

        seed2_config = json.loads(
            (
                PROJECT
                / "configs"
                / "retail_agentic_qwen3_4b_identity_auth_rollout_diagnostic_clean_seed20260825_v1.json"
            ).read_text(encoding="utf-8")
        )
        self.assertEqual(seed2_config["seed"], 20260825)
        seed1_normalized = dict(config)
        seed2_normalized = dict(seed2_config)
        seed1_normalized["seed"] = None
        seed2_normalized["seed"] = None
        self.assertEqual(seed1_normalized, seed2_normalized)

    def test_identity_authentication_grpo_is_bound_to_eligible_evidence(self) -> None:
        config = json.loads(
            (
                PROJECT
                / "configs"
                / "retail_agentic_qwen3_4b_identity_auth_grpo_v1.json"
            ).read_text(encoding="utf-8")
        )
        evidence = config["eligibility_evidence"]
        self.assertEqual(config["execution_mode"], "OPTIMIZE")
        self.assertEqual(config["rollout"]["stage"], IDENTITY_AUTHENTICATION_STAGE)
        self.assertEqual(config["reward"], DEFAULT_REWARD_CONFIG)
        self.assertEqual(
            config["data"]["task_ids"],
            ["7", "10", "11", "13", "15", "20", "22"],
        )
        self.assertEqual(config["grpo"]["max_steps"], 28)
        self.assertEqual(config["grpo"]["num_generations"], 2)
        self.assertGreater(config["grpo"]["learning_rate"], 0.0)
        self.assertEqual(config["grpo"]["beta"], 0.0)
        self.assertTrue(config["quantization"]["enabled"])
        self.assertTrue(evidence["ready_to_consider_optimization"])
        self.assertGreaterEqual(
            evidence["selected_variance_task_count"],
            evidence["minimum_variance_tasks"],
        )
        self.assertFalse(evidence["selection_used_reward_values"])

        smoke = json.loads(
            (
                PROJECT
                / "configs"
                / "retail_agentic_qwen3_4b_identity_auth_grpo_smoke_v1.json"
            ).read_text(encoding="utf-8")
        )
        self.assertEqual(smoke["execution_mode"], "OPTIMIZE")
        self.assertEqual(smoke["model"], config["model"])
        self.assertEqual(smoke["data"], config["data"])
        self.assertEqual(smoke["rollout"], config["rollout"])
        self.assertEqual(smoke["reward"], config["reward"])
        self.assertEqual(smoke["grpo"]["max_steps"], 1)
        self.assertEqual(smoke["grpo"]["num_generations"], 2)

        post_grpo = json.loads(
            (
                PROJECT
                / "configs"
                / "retail_agentic_qwen3_4b_identity_auth_grpo_s28_rollout_diagnostic_v1.json"
            ).read_text(encoding="utf-8")
        )
        self.assertEqual(post_grpo["execution_mode"], "ROLLOUT_DIAGNOSTIC")
        self.assertEqual(post_grpo["data"], config["data"])
        self.assertEqual(post_grpo["rollout"], config["rollout"])
        self.assertEqual(post_grpo["reward"], config["reward"])
        self.assertEqual(post_grpo["grpo"]["max_steps"], 14)
        self.assertEqual(post_grpo["grpo"]["num_generations"], 2)
        self.assertEqual(post_grpo["grpo"]["learning_rate"], 0.0)
        self.assertEqual(post_grpo["grpo"]["beta"], 0.0)
        self.assertFalse(post_grpo["diagnostic"]["weight_update_expected"])
        self.assertEqual(
            post_grpo["model"]["expected_sha256"],
            "AF9CDEE0DAEE8DD9701AB0CF7DF7FDB4A576335879E55DE1632BD48EA075C72E",
        )

        post_grpo_seed2 = json.loads(
            (
                PROJECT
                / "configs"
                / "retail_agentic_qwen3_4b_identity_auth_grpo_s28_rollout_diagnostic_seed20260825_v1.json"
            ).read_text(encoding="utf-8")
        )
        self.assertEqual(post_grpo_seed2["seed"], 20260825)
        first_seed_normalized = dict(post_grpo)
        second_seed_normalized = dict(post_grpo_seed2)
        first_seed_normalized["seed"] = None
        second_seed_normalized["seed"] = None
        self.assertEqual(first_seed_normalized, second_seed_normalized)

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
            report = analyze(path, expected_rollouts=8, expected_tasks=2)
            self.assertEqual(report["observed_rollouts"], 8)
            self.assertEqual(report["unique_tasks"], 2)
            self.assertTrue(report["gates"]["ready_to_consider_optimization"])
            self.assertEqual(report["group_variance"]["joint_variance_task_count"], 2)

    def test_rollout_diagnostic_rejects_global_variance_from_one_task(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "rollouts.jsonl"
            rows = []
            for task_id in range(8):
                for sample in range(4):
                    varying = task_id == 7 and sample % 2 == 1
                    rows.append(
                        {
                            "task_id": str(task_id),
                            "tool_calls": 2,
                            "customer_turns": 1,
                            "reward": {
                                "reward": 0.1 if varying else 0.0,
                                "tool_error_count": 0,
                                "action_progress": {"recall": 0.5 if varying else 0.25},
                            },
                        }
                    )
            path.write_text(
                "".join(json.dumps(row) + "\n" for row in rows),
                encoding="utf-8",
            )
            report = analyze(path, expected_rollouts=32, expected_tasks=8)
            self.assertTrue(report["gates"]["reward_has_variance"])
            self.assertTrue(report["gates"]["action_progress_has_variance"])
            self.assertEqual(report["group_variance"]["joint_variance_task_count"], 1)
            self.assertFalse(
                report["gates"]["sufficient_task_groups_have_joint_variance"]
            )
            self.assertFalse(report["gates"]["ready_to_consider_optimization"])

    def test_rollout_diagnostic_allows_one_task_sanity_to_produce_signal(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "rollouts.jsonl"
            rows = [
                {
                    "task_id": "0",
                    "tool_calls": 1,
                    "customer_turns": 1,
                    "reward": {
                        "reward": float(index),
                        "tool_error_count": 0,
                        "action_progress": {"recall": float(index)},
                    },
                }
                for index in range(2)
            ]
            path.write_text(
                "".join(json.dumps(row) + "\n" for row in rows),
                encoding="utf-8",
            )
            report = analyze(path, expected_rollouts=2, expected_tasks=1)
            self.assertEqual(report["group_variance"]["minimum_signal_task_count"], 1)
            self.assertTrue(report["gates"]["ready_to_consider_optimization"])

    def test_rollout_diagnostic_rejects_variance_caused_by_regression(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            baseline_path = root / "baseline.jsonl"
            candidate_path = root / "candidate.jsonl"
            baseline = []
            candidate = []
            for task_id in range(2):
                for sample in range(4):
                    baseline.append(
                        {
                            "task_id": str(task_id),
                            "tool_calls": 2,
                            "customer_turns": 1,
                            "reward": {
                                "reward": 0.2 if sample % 2 else 0.1,
                                "tool_error_count": 0,
                                "unfinished_interaction_penalty": 0.1,
                                "action_progress": {
                                    "recall": 0.5 if sample % 2 else 0.4,
                                    "duplicate_excess_count": 0,
                                },
                            },
                        }
                    )
                    candidate.append(
                        {
                            "task_id": str(task_id),
                            "tool_calls": 5,
                            "customer_turns": 1,
                            "reward": {
                                "reward": 0.1 if sample % 2 else 0.0,
                                "tool_error_count": 1,
                                "unfinished_interaction_penalty": 0.1,
                                "action_progress": {
                                    "recall": 0.2 if sample % 2 else 0.1,
                                    "duplicate_excess_count": 2,
                                },
                            },
                        }
                    )
            for path, rows in ((baseline_path, baseline), (candidate_path, candidate)):
                path.write_text(
                    "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
                )
            report = analyze(candidate_path, 8, 2, baseline_path)
            self.assertTrue(report["gates"]["baseline_protocol_comparable"])
            self.assertTrue(
                report["gates"]["sufficient_task_groups_have_joint_variance"]
            )
            self.assertFalse(report["gates"]["mean_action_recall_not_regressed"])
            self.assertFalse(report["gates"]["tool_error_count_not_increased"])
            self.assertFalse(report["gates"]["ready_to_consider_optimization"])

    def test_rollout_diagnostic_requires_a_normal_termination(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "rollouts.jsonl"
            rows = [
                {
                    "task_id": "0",
                    "tool_calls": 1,
                    "customer_turns": 1,
                    "reward": {
                        "reward": float(index),
                        "tool_error_count": 0,
                        "unfinished_interaction_penalty": 0.1,
                        "action_progress": {"recall": float(index)},
                    },
                }
                for index in range(2)
            ]
            path.write_text(
                "".join(json.dumps(row) + "\n" for row in rows),
                encoding="utf-8",
            )
            report = analyze(path, expected_rollouts=2, expected_tasks=1)
            self.assertEqual(report["behavior"]["normal_termination_rollout_count"], 0)
            self.assertFalse(report["gates"]["normal_termination_observed"])
            self.assertFalse(report["gates"]["ready_to_consider_optimization"])

    def test_staged_rollout_uses_stage_completion_instead_of_user_stop(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "rollouts.jsonl"
            rows = [
                {
                    "task_id": "0",
                    "rollout_stage": IDENTITY_AUTHENTICATION_STAGE,
                    "tool_calls": 1,
                    "customer_turns": 1,
                    "reward": {
                        "reward": float(index),
                        "rollout_stage": IDENTITY_AUTHENTICATION_STAGE,
                        "stage_complete": bool(index),
                        "tool_error_count": 0,
                        "unfinished_interaction_penalty": 0.0 if index else 0.1,
                        "action_progress": {"recall": float(index)},
                    },
                }
                for index in range(2)
            ]
            path.write_text(
                "".join(json.dumps(row) + "\n" for row in rows),
                encoding="utf-8",
            )
            report = analyze(path, expected_rollouts=2, expected_tasks=1)
            self.assertEqual(report["behavior"]["normal_termination_rollout_count"], 0)
            self.assertEqual(report["behavior"]["stage_complete_rollout_count"], 1)
            self.assertEqual(report["behavior"]["completion_target_rollout_count"], 1)
            self.assertTrue(report["gates"]["normal_termination_observed"])
            self.assertTrue(report["gates"]["stage_completion_observed"])
            self.assertTrue(report["gates"]["ready_to_consider_optimization"])

    def test_staged_rollout_does_not_require_action_recall_variance(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "rollouts.jsonl"
            rows = []
            for task_id in range(2):
                for sample in range(4):
                    stage_complete = sample == 3
                    rows.append(
                        {
                            "task_id": str(task_id),
                            "rollout_stage": IDENTITY_AUTHENTICATION_STAGE,
                            "tool_calls": 1 if stage_complete else 2,
                            "customer_turns": 1,
                            "reward": {
                                "reward": 1.0 if stage_complete else 0.0,
                                "rollout_stage": IDENTITY_AUTHENTICATION_STAGE,
                                "stage_complete": stage_complete,
                                "tool_error_count": 0,
                                "unfinished_interaction_penalty": (
                                    0.0 if stage_complete else 0.1
                                ),
                                "action_progress": {"recall": 1.0},
                            },
                        }
                    )
            path.write_text(
                "".join(json.dumps(row) + "\n" for row in rows),
                encoding="utf-8",
            )
            report = analyze(path, expected_rollouts=8, expected_tasks=2)
            self.assertEqual(
                report["group_variance"]["variance_gate_mode"],
                "REWARD_AND_STAGE_COMPLETION",
            )
            self.assertEqual(report["group_variance"]["joint_variance_task_count"], 0)
            self.assertEqual(
                report["group_variance"]["stage_target_variance_task_count"], 2
            )
            self.assertTrue(
                report["gates"]["sufficient_task_groups_have_stage_target_variance"]
            )
            self.assertNotIn("action_progress_has_variance", report["gates"])
            self.assertTrue(report["gates"]["ready_to_consider_optimization"])

    def test_rollout_diagnostic_rejects_system_failures(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "rollouts.jsonl"
            failure_path = root / "system_failures.jsonl"
            rows = [
                {
                    "task_id": "0",
                    "tool_calls": 1,
                    "customer_turns": 1,
                    "reward": {
                        "reward": float(index),
                        "tool_error_count": 0,
                        "action_progress": {"recall": float(index)},
                    },
                }
                for index in range(2)
            ]
            path.write_text(
                "".join(json.dumps(row) + "\n" for row in rows),
                encoding="utf-8",
            )
            failure_path.write_text(
                json.dumps({"category": "INVALID_RESPONSE"}) + "\n",
                encoding="utf-8",
            )
            report = analyze(
                path,
                expected_rollouts=2,
                expected_tasks=1,
                system_failure_path=failure_path,
            )
            self.assertEqual(report["system_failures"]["count"], 1)
            self.assertEqual(
                report["system_failures"]["categories"],
                {"INVALID_RESPONSE": 1},
            )
            self.assertFalse(report["gates"]["no_system_failures"])
            self.assertFalse(report["gates"]["ready_to_consider_optimization"])

    def test_transferred_upstream_requires_commit_and_package_hash(self) -> None:
        commit = "58e5e1ace69302e6982d27014569c03e0ffccdd2"
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "tau2-bench"
            (root / "src").mkdir(parents=True)
            (root / "data" / "tau2" / "domains" / "retail").mkdir(parents=True)
            guidelines = (
                root / "data" / "tau2" / "user_simulator" / "simulation_guidelines.md"
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
                json.dumps({"commit": commit, "source_package_path": str(archive)}),
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

    def test_task44_explicit_context_prescreen_is_short_and_hash_bound(self) -> None:
        config_path = (
            PROJECT
            / "configs"
            / "retail_agentic_qwen3_4b_terminal_prescreen_task44_explicit_context_n4_v1.json"
        )
        config = json.loads(config_path.read_text(encoding="utf-8"))
        opening_path = PROJECT / config["data"]["openings"]
        manifest = json.loads(
            (PROJECT / config["data"]["openings_manifest"]).read_text(
                encoding="utf-8"
            )
        )
        opening = json.loads(opening_path.read_text(encoding="utf-8").strip())

        self.assertEqual(config["execution_mode"], "ROLLOUT_DIAGNOSTIC")
        self.assertEqual(config["data"]["task_ids"], ["44"])
        self.assertEqual(config["grpo"]["num_generations"], 4)
        self.assertEqual(config["grpo"]["max_completion_length"], 4096)
        self.assertEqual(config["reward"], TERMINAL_ONLY_REWARD_CONFIG)
        self.assertEqual(opening["user_seed"], 814494)
        self.assertFalse(opening["hidden_user_scenario_persisted"])
        self.assertEqual(
            hashlib.sha256(opening_path.read_bytes()).hexdigest().upper(),
            manifest["output_sha256"],
        )
        self.assertEqual(
            hashlib.sha256(
                (PROJECT / config["data"]["task_split"]).read_bytes()
            ).hexdigest().upper(),
            manifest["task_split_sha256"],
        )
        text = opening["initial_user_message"]
        self.assertIn("Aarav Anderson", text)
        self.assertIn("19031", text)
        self.assertIn("#W9300146", text)
        for hidden_answer in ("9190635437", "5320792178", "135.24", "17.99"):
            self.assertNotIn(hidden_answer, text)

    def test_task44_explicit_context_grpo_reuses_frozen_prescreen_contract(self) -> None:
        prescreen_path = (
            PROJECT
            / "configs"
            / "retail_agentic_qwen3_4b_terminal_prescreen_task44_explicit_context_n4_v1.json"
        )
        training_path = (
            PROJECT
            / "configs"
            / "retail_agentic_qwen3_4b_terminal_grpo_closure_task44_explicit_context_n4_v1.json"
        )
        prescreen = json.loads(prescreen_path.read_text(encoding="utf-8"))
        training = json.loads(training_path.read_text(encoding="utf-8"))

        self.assertEqual(training["execution_mode"], "OPTIMIZE")
        self.assertEqual(
            training["engineering_acceptance"]["contract_version"],
            "grpo-engineering-closure-v1",
        )
        self.assertEqual(training["engineering_acceptance"]["expected_optimizer_steps"], 2)
        self.assertEqual(training["engineering_acceptance"]["expected_rollouts"], 8)
        self.assertEqual(training["engineering_acceptance"]["expected_groups"], 2)
        self.assertEqual(training["data"], prescreen["data"])
        self.assertEqual(training["model"], prescreen["model"])
        self.assertEqual(training["reward"], prescreen["reward"])
        self.assertEqual(training["rollout"], prescreen["rollout"])
        for field in (
            "num_generations",
            "max_completion_length",
            "temperature",
            "per_device_train_batch_size",
            "gradient_accumulation_steps",
            "steps_per_generation",
        ):
            self.assertEqual(training["grpo"][field], prescreen["grpo"][field])
        self.assertEqual(training["grpo"]["beta"], 0.02)
        self.assertEqual(training["grpo"]["learning_rate"], 0.000005)
        self.assertTrue(training["claims"]["explicit_context_short_variant"])
        self.assertFalse(training["claims"]["comparison_with_original_opening_allowed"])
        contract = validate_optimization_contract(training, selected_task_count=1)
        self.assertEqual(
            contract,
            {
                "status": "VALIDATED",
                "generation_batch_size": 4,
                "steps_per_generation": 4,
                "expected_optimizer_steps": 2,
                "expected_groups": 2,
                "expected_rollouts": 8,
                "configured_task_pool_size": 1,
                "kl_reference_required": True,
            },
        )

    def test_task44_reward_ab_configs_change_only_reward(self) -> None:
        terminal_path = (
            PROJECT
            / "configs"
            / "retail_agentic_qwen3_4b_task44_reward_ab_terminal_n4_v1.json"
        )
        staged_path = (
            PROJECT
            / "configs"
            / "retail_agentic_qwen3_4b_task44_reward_ab_staged_v6_n4_v1.json"
        )
        terminal = json.loads(terminal_path.read_text(encoding="utf-8"))
        staged = json.loads(staged_path.read_text(encoding="utf-8"))

        terminal_common = {key: value for key, value in terminal.items() if key != "reward"}
        staged_common = {key: value for key, value in staged.items() if key != "reward"}
        self.assertEqual(staged_common, terminal_common)
        self.assertEqual(terminal["reward"], TERMINAL_ONLY_REWARD_CONFIG)
        self.assertEqual(terminal["grpo"]["max_completion_length"], 5120)
        self.assertEqual(terminal["grpo"]["num_generations"], 4)
        self.assertEqual(terminal["grpo"]["beta"], 0.02)
        self.assertTrue(
            terminal["engineering_acceptance"]["transport_complete_groups_required"]
        )

        staged_reward = staged["reward"]
        self.assertEqual(
            staged_reward["process_reward_mode"], TIERED_TERMINAL_PROCESS_MODE
        )
        self.assertTrue(staged_reward["confirmation_signal_used_as_reward"])
        staged_spec = staged_reward["staged_reward_spec"]
        self.assertEqual(
            staged_spec["reward"]["composition_mode"],
            "hierarchical_state_authorization_review_v6",
        )
        self.assertEqual(set(staged_spec["tasks"]), {"44"})
        self.assertNotIn("source_run", staged_spec)
        self.assertNotIn("shadow_gate", staged_spec)

        for config_path in (terminal_path, staged_path):
            validated = validate_config_and_split(config_path)
            self.assertEqual(validated["config"]["data"]["task_ids"], ["44"])

    def test_task44_reward_ab_presample_configs_change_only_reward(self) -> None:
        terminal_path = (
            PROJECT
            / "configs"
            / "retail_agentic_qwen3_4b_task44_reward_ab_terminal_presample_n4_v1.json"
        )
        staged_path = (
            PROJECT
            / "configs"
            / "retail_agentic_qwen3_4b_task44_reward_ab_staged_v6_presample_n4_v1.json"
        )
        terminal = json.loads(terminal_path.read_text(encoding="utf-8"))
        staged = json.loads(staged_path.read_text(encoding="utf-8"))

        self.assertEqual(
            {key: value for key, value in terminal.items() if key != "reward"},
            {key: value for key, value in staged.items() if key != "reward"},
        )
        self.assertEqual(terminal["execution_mode"], "ROLLOUT_DIAGNOSTIC")
        self.assertEqual(terminal["model_loading"]["mode"], "qwen3_bf16_inference_v1")
        self.assertEqual(terminal["grpo"]["learning_rate"], 0)
        self.assertEqual(terminal["grpo"]["beta"], 0)
        self.assertEqual(terminal["grpo"]["max_completion_length"], 5120)
        self.assertEqual(terminal["sampling"]["contract_version"], "fixed-n4-single-group-v1")
        self.assertTrue(
            terminal["engineering_acceptance"]["transport_complete_groups_required"]
        )
        for config_path in (terminal_path, staged_path):
            validated = validate_config_and_split(config_path)
            self.assertEqual(validated["config"]["data"]["task_ids"], ["44"])

    def test_task44_post_grpo_eval_changes_only_bound_model_and_claims(self) -> None:
        prescreen = json.loads(
            (
                PROJECT
                / "configs"
                / "retail_agentic_qwen3_4b_terminal_prescreen_task44_explicit_context_n4_v1.json"
            ).read_text(encoding="utf-8")
        )
        post_eval = json.loads(
            (
                PROJECT
                / "configs"
                / "retail_agentic_qwen3_4b_terminal_post_grpo_eval_task44_explicit_context_n4_v1.json"
            ).read_text(encoding="utf-8")
        )

        self.assertEqual(post_eval["execution_mode"], "ROLLOUT_DIAGNOSTIC")
        self.assertEqual(post_eval["data"], prescreen["data"])
        self.assertEqual(post_eval["sampling"], prescreen["sampling"])
        self.assertEqual(post_eval["seed"], prescreen["seed"])
        self.assertEqual(post_eval["reward"], prescreen["reward"])
        self.assertEqual(post_eval["rollout"], prescreen["rollout"])
        self.assertEqual(post_eval["grpo"], prescreen["grpo"])
        self.assertEqual(post_eval["model_loading"], prescreen["model_loading"])
        self.assertEqual(
            post_eval["model"]["expected_sha256"],
            "8E876183425B06A742D8CE6E5B5B8B74F95528777B62614ACDE7F2C719E4D6C6",
        )
        self.assertEqual(
            post_eval["model"]["grpo_run_manifest_sha256"],
            "55AEB96EC61D3DCA11833E320BB3531B0E4C4F8C971434F6D300C28A93BA6C87",
        )
        self.assertTrue(post_eval["claims"]["training_exposed_task_evaluation"])
        self.assertFalse(post_eval["claims"]["generalization_claim_allowed"])
        self.assertFalse(post_eval["claims"]["business_improvement_claim_allowed"])

    def test_task44_reward_ab_post_eval_changes_only_model_binding(self) -> None:
        config_dir = PROJECT / "configs"
        paths = (
            config_dir
            / "retail_agentic_qwen3_4b_task44_reward_ab_terminal_presample_n4_v1.json",
            config_dir
            / "retail_agentic_qwen3_4b_task44_reward_ab_terminal_post_eval_n4_v1.json",
            config_dir
            / "retail_agentic_qwen3_4b_task44_reward_ab_staged_v6_post_eval_n4_v1.json",
        )
        configs = [json.loads(path.read_text(encoding="utf-8")) for path in paths]

        protocols = [
            {key: value for key, value in config.items() if key != "model"}
            for config in configs
        ]
        self.assertEqual(protocols[0], protocols[1])
        self.assertEqual(protocols[0], protocols[2])

        self.assertEqual(
            [config["model"]["expected_sha256"] for config in configs],
            [
                "0A2E06C9BCA6082F3FE6723EC54A46D4CE1D37A8C6DD6B9BC116BBA4BAB16576",
                "AD7328404716C5554D7DC5F6E9311DE0D4ABFCFF1E49BC958945346EDEB1A5D1",
                "DD166DF2E7B1CDC3FECBE515DCE93DE3C274E9D929D69C29EF72689547168148",
            ],
        )
        for path in paths:
            validated = validate_config_and_split(path)
            config = validated["config"]
            self.assertEqual(config["execution_mode"], "ROLLOUT_DIAGNOSTIC")
            self.assertEqual(config["data"]["task_ids"], ["44"])
            self.assertEqual(config["grpo"]["num_generations"], 4)
            self.assertEqual(config["grpo"]["learning_rate"], 0)
            self.assertEqual(config["reward"]["process_reward_mode"], "terminal_environment_state")

    def test_task44_reward_ab_10step_configs_change_only_reward(self) -> None:
        config_dir = PROJECT / "configs"
        terminal_path = (
            config_dir
            / "retail_agentic_qwen3_4b_task44_reward_ab_terminal_n4_10step_v1.json"
        )
        staged_path = (
            config_dir
            / "retail_agentic_qwen3_4b_task44_reward_ab_staged_v6_n4_10step_v1.json"
        )
        terminal = json.loads(terminal_path.read_text(encoding="utf-8"))
        staged = json.loads(staged_path.read_text(encoding="utf-8"))

        self.assertEqual(
            {key: value for key, value in terminal.items() if key != "reward"},
            {key: value for key, value in staged.items() if key != "reward"},
        )
        self.assertEqual(terminal["execution_mode"], "OPTIMIZE")
        self.assertEqual(terminal["grpo"]["max_steps"], 10)
        self.assertEqual(terminal["grpo"]["num_generations"], 4)
        self.assertEqual(terminal["grpo"]["save_steps"], 5)
        self.assertEqual(terminal["grpo"]["beta"], 0.02)
        self.assertEqual(terminal["model"]["expected_sha256"], (
            "0A2E06C9BCA6082F3FE6723EC54A46D4CE1D37A8C6DD6B9BC116BBA4BAB16576"
        ))
        self.assertEqual(
            terminal["engineering_acceptance"]["expected_optimizer_steps"], 10
        )
        self.assertEqual(
            terminal["engineering_acceptance"]["expected_rollouts"], 40
        )
        self.assertEqual(
            terminal["engineering_acceptance"]["expected_groups"], 10
        )
        self.assertEqual(terminal["reward"], TERMINAL_ONLY_REWARD_CONFIG)
        self.assertEqual(
            staged["reward"]["process_reward_mode"], TIERED_TERMINAL_PROCESS_MODE
        )
        for path in (terminal_path, staged_path):
            validated = validate_config_and_split(path)
            contract = validate_optimization_contract(
                validated["config"], selected_task_count=1
            )
            self.assertEqual(contract["expected_optimizer_steps"], 10)
            self.assertEqual(contract["expected_rollouts"], 40)
            self.assertEqual(contract["expected_groups"], 10)


if __name__ == "__main__":
    unittest.main()
