from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from src.evaluation.grpo_training_audit import (
    _successful_required_write_execution,
    audit_run,
    build_training_audit,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def _write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _rewrite_hashed_jsonl_artifact(
    run_dir: Path, filename: str, rows: list[dict]
) -> None:
    artifact_path = run_dir / filename
    artifact_path.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n",
        encoding="utf-8",
    )
    manifest_path = run_dir / "run_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["artifacts"][filename]["sha256"] = _sha256(artifact_path)
    _write_json(manifest_path, manifest)
    _write_json(
        run_dir / "run_state.json",
        {
            "status": "COMPLETED",
            "run_manifest_sha256": _sha256(manifest_path),
        },
    )


def _write_audit_fixture(root: Path) -> tuple[Path, Path]:
    run_dir = root / "run"
    run_dir.mkdir()
    config_path = root / "config.json"
    config = {
        "data": {"task_ids": ["1"]},
        "grpo": {
            "max_steps": 1,
            "num_generations": 2,
            "max_completion_length": 128,
            "per_device_train_batch_size": 1,
            "gradient_accumulation_steps": 2,
            "learning_rate": 1e-6,
            "temperature": 0.8,
            "beta": 0.02,
            "loss_type": "dr_grpo",
        },
    }
    log_history = [
        {
            "step": 1,
            "loss": 0.1,
            "reward": 0.5,
            "reward_std": 0.5,
            "grad_norm": 0.2,
            "kl": 0.01,
        }
    ]
    train_metrics = {"train_runtime": 1.0, "train_loss": 0.1}
    rollouts = [
        _rollout(
            "1",
            reward,
            [("get_order_details", {"order_id": "O1"})],
            terminal_success=bool(reward),
            complete_success=bool(reward),
            write_progress=reward,
        )
        for reward in (0.0, 1.0)
    ]
    for row in rollouts:
        row.update({"prompt": "p", "final_answer": "done"})
    evidence = {
        "status": "PASSED",
        "trainable_parameter_change_detected": True,
        "final_trainable_parameters": {"all_finite": True},
    }
    _write_json(config_path, config)
    _write_json(run_dir / "log_history.json", log_history)
    _write_json(run_dir / "train_metrics.json", train_metrics)
    (run_dir / "raw_rollouts.jsonl").write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in rollouts) + "\n",
        encoding="utf-8",
    )
    _write_json(run_dir / "optimization_evidence.json", evidence)
    manifest = {
        "status": "COMPLETED",
        "optimization_enabled": True,
        "git": {"commit": "abc"},
        "bindings": {
            "config_sha256": _sha256(config_path),
            "starting_model_sha256": "MODEL",
        },
        "artifacts": {
            name: {"sha256": _sha256(run_dir / filename)}
            for name, filename in {
                "raw_rollouts": "raw_rollouts.jsonl",
                "log_history": "log_history.json",
                "train_metrics": "train_metrics.json",
                "optimization_evidence": "optimization_evidence.json",
            }.items()
        },
    }
    _write_json(run_dir / "run_manifest.json", manifest)
    return run_dir, config_path


def _write_sampling_audit_fixture(
    root: Path,
    *,
    sampling_mode: str = "TRUE_GREEDY",
    group_size: int = 1,
    groups_per_task: int = 1,
    terminal_matrix: dict[tuple[str, int], list[bool]] | None = None,
    reward_matrix: dict[tuple[str, int], list[float]] | None = None,
    transport_invalid_raw_indices: set[int] | None = None,
) -> tuple[Path, Path]:
    run_dir = root / "sampling_run"
    run_dir.mkdir()
    config_path = root / "sampling_config.json"
    do_sample = sampling_mode == "STOCHASTIC_GROUP_SAMPLING"
    terminal_matrix = terminal_matrix or {
        ("43", repeat): [True] * group_size for repeat in range(groups_per_task)
    } | {
        ("72", repeat): [False] * group_size for repeat in range(groups_per_task)
    }
    task_ids = sorted({task_id for task_id, _ in terminal_matrix})
    reward_matrix = reward_matrix or {
        key: [float(value) for value in values]
        for key, values in terminal_matrix.items()
    }
    transport_invalid_raw_indices = transport_invalid_raw_indices or set()
    sampling = {"mode": sampling_mode, "do_sample": do_sample}
    if do_sample:
        sampling.update({"temperature": 0.8, "top_p": 1.0, "top_k": 0})
    configured_temperature = 0.8 if do_sample else 1.0
    config = {
        "execution_mode": "ROLLOUT_DIAGNOSTIC",
        "model": {"expected_sha256": "MODEL"},
        "data": {"task_ids": task_ids},
        "grpo": {
            "num_generations": group_size,
            "temperature": configured_temperature,
        },
        "sampling": sampling,
        "diagnostic": {
            "expected_tasks": len(task_ids),
            "expected_rollouts": len(task_ids) * groups_per_task * group_size,
            "expected_rollouts_per_task": groups_per_task * group_size,
            "groups_per_task": groups_per_task,
            "group_size": group_size,
            "trainer_max_steps_unused": True,
        },
    }
    effective_config = {
        **config,
        "sampling": {
            **sampling,
            "actual_num_generations": group_size,
            "trl_constructor_num_generations": 2,
            "groups_per_task": groups_per_task,
            "trainer_max_steps_unused": True,
            "configured_temperature": configured_temperature,
        },
    }
    raw_rows = []
    evidence_rows = []
    groups = []
    task_seeds = {
        task_id: 350291 + index for index, task_id in enumerate(task_ids)
    }
    for repeat in range(groups_per_task):
        for task_id in task_ids:
            completions = []
            diagnostics = []
            for candidate_index in range(group_size):
                raw_index = len(raw_rows)
                success = terminal_matrix[(task_id, repeat)][candidate_index]
                reward_value = reward_matrix[(task_id, repeat)][candidate_index]
                transport_complete = raw_index not in transport_invalid_raw_indices
                stop_reason = (
                    "USER_STOP_AND_MODEL_EOS"
                    if success
                    else "MODEL_EOS_BEFORE_USER_STOP"
                )
                completion = {
                    "stop_reason_source": "guarded_grpo_trainer_v1",
                    "stop_reason": stop_reason,
                    "model_ended": True,
                    "completion_tokens": 10 + raw_index,
                }
                evidence = {
                    "task_id": task_id,
                    "user_seed": task_seeds[task_id],
                    "initial_state": {},
                    "final_state": {},
                    "state_diff": [],
                    "state_hashes": {},
                    "tool_trace": [],
                    "terminal_evaluator": {"reward": int(success)},
                    "completion": completion,
                }
                evidence_hash = hashlib.sha256(
                    json.dumps(
                        evidence,
                        ensure_ascii=False,
                        sort_keys=True,
                        separators=(",", ":"),
                        default=str,
                    ).encode("utf-8")
                ).hexdigest().upper()
                evidence["evidence_sha256"] = evidence_hash
                reward = {
                    "reward": reward_value,
                    "terminal_environment_reward": float(success),
                    "complete_success": success,
                    "components": {
                        "required_write_progress": {"value": float(success)}
                    },
                }
                answer = f"answer-{task_id}-{repeat}-{candidate_index}"
                raw_rows.append(
                    {
                        "task_id": task_id,
                        "user_seed": task_seeds[task_id],
                        "messages": [
                            {"role": "user", "content": f"request-{task_id}"},
                            {"role": "assistant", "content": answer},
                        ],
                        "tool_calls": 0,
                        "reward": reward,
                        "completion": completion,
                        "evidence_sha256": evidence_hash,
                    }
                )
                evidence_rows.append(evidence)
                completions.append([{"role": "assistant", "content": answer}])
                diagnostics.append(
                    {
                        "training_eligible": False,
                        "trajectory_transport_complete": transport_complete,
                        "raw_row_index": raw_index,
                        "candidate_index": candidate_index,
                        "stop_reason": (
                            stop_reason if transport_complete else "CONTEXT_LIMIT"
                        ),
                        "model_ended": True,
                        "completion_tokens": 10 + raw_index,
                        "model_tokens_retained": 6 + raw_index,
                        "observation_tokens_retained": 4,
                        "context_limit_reached": not transport_complete,
                        "model_completion_truncated": False,
                    }
                )
            groups.append(
                {
                    "group_id": f"{task_id}:{repeat}",
                    "task_id": task_id,
                    "user_seed": task_seeds[task_id],
                    "status": "COMPLETED",
                    "prompt": [
                        {"role": "user", "content": f"request-{task_id}"}
                    ],
                    "completions": completions,
                    "rewards_observational_only": reward_matrix[(task_id, repeat)],
                    "diagnostics": diagnostics,
                    "training_eligible": False,
                }
            )

    _write_json(config_path, config)
    _write_json(run_dir / "effective_config.json", effective_config)
    _write_json(run_dir / "command.json", {"parameter_update_requested": False})
    _write_json(
        run_dir / "user_simulator_preflight.json",
        {
            "status": "PASSED",
            "model": "deepseek/test-user",
            "external_api_called": True,
        },
    )
    for name, rows in {
        "raw_rollouts.jsonl": raw_rows,
        "rollout_evidence.jsonl": evidence_rows,
        "generation_events.jsonl": [{"event": "generation_stop_observed"}],
        "sampling_groups.jsonl": groups,
    }.items():
        (run_dir / name).write_text(
            "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n",
            encoding="utf-8",
        )
    manifest = {
        "schema_version": "retail-pure-sampling-v1",
        "status": "COMPLETED",
        "execution_mode": "PURE_SAMPLING",
        "optimization_enabled": False,
        "backward_called": False,
        "optimizer_created": False,
        "loss_computed": False,
        "training_eligible": False,
        "config_sha256": _sha256(config_path),
        "starting_model_sha256": "MODEL",
        "groups": len(groups),
        "rollouts": len(raw_rows),
        "git": {"commit": "abc"},
        "runtime": {
            "sampling_adapter": {
                "direct_merged_checkpoint_inference": True,
                "decode_contract": {
                    **effective_config["sampling"],
                    "effective_do_sample": do_sample,
                    "effective_num_generations": group_size,
                    **(
                        {
                            "effective_temperature": 0.8,
                            "effective_top_p": 1.0,
                            "effective_top_k": 0,
                            "temperature_is_decode_authority": True,
                        }
                        if do_sample
                        else {"temperature_is_decode_authority": False}
                    ),
                }
            },
            "user_simulator": {
                "model": "deepseek/test-user",
                "llm_args": {"temperature": 0.0},
                "llm_args_sha256": "A" * 64,
                "seed_source": "per_opening_user_seed",
                "preflight_status": "PASSED",
                "external_api_called": True,
            },
            "model_loading": {
                "mode": "qwen3_bf16_inference_v1",
                "quantized": False,
                "peft_adapter_applied": False,
            },
        },
        "decode_contract": effective_config["sampling"],
        "artifacts": {
            name: {"sha256": _sha256(run_dir / name)}
            for name in (
                "raw_rollouts.jsonl",
                "rollout_evidence.jsonl",
                "generation_events.jsonl",
                "sampling_groups.jsonl",
                "effective_config.json",
                "command.json",
                "user_simulator_preflight.json",
            )
        },
    }
    _write_json(run_dir / "run_manifest.json", manifest)
    _write_json(
        run_dir / "run_state.json",
        {
            "status": "COMPLETED",
            "run_manifest_sha256": _sha256(run_dir / "run_manifest.json"),
        },
    )
    return run_dir, config_path


def _inject_required_write_evidence(
    run_dir: Path,
    *,
    missing_result_indices: set[int] | None = None,
    error_result_indices: set[int] | None = None,
) -> None:
    missing_result_indices = missing_result_indices or set()
    error_result_indices = error_result_indices or set()
    raw_path = run_dir / "raw_rollouts.jsonl"
    rows = [
        json.loads(line)
        for line in raw_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    for index, row in enumerate(rows):
        terminal_success = bool(row["reward"]["terminal_environment_reward"])
        matches = [
            {
                "action_id": f"95_{write_index}",
                "name": "exchange_delivered_order_items",
                "matched": terminal_success,
                "matched_call_index": write_index if terminal_success else None,
            }
            for write_index in range(2)
        ]
        row["reward"]["action_progress"] = {
            "matches": matches,
            "unexpected_write_count": 0,
        }
        opening = row["messages"][0]["content"]
        answer = row["messages"][-1]["content"]
        messages: list[dict] = [{"role": "user", "content": opening}]
        if terminal_success:
            for write_index in range(2):
                call_id = f"write-{index}-{write_index}"
                messages.append(
                    {
                        "role": "assistant",
                        "tool_calls": [
                            {
                                "id": call_id,
                                "name": "exchange_delivered_order_items",
                                "arguments": {"order_id": f"#W{write_index}"},
                                "requestor": "assistant",
                            }
                        ],
                    }
                )
                if not (
                    index in missing_result_indices and write_index == 1
                ):
                    messages.append(
                        {
                            "id": call_id,
                            "role": "tool",
                            "content": "exchange requested",
                            "requestor": "assistant",
                            "error": (
                                index in error_result_indices and write_index == 1
                            ),
                        }
                    )
        messages.append({"role": "assistant", "content": answer})
        row["messages"] = messages
        row["tool_calls"] = 2 if terminal_success else 0
    _rewrite_hashed_jsonl_artifact(run_dir, "raw_rollouts.jsonl", rows)
    groups_path = run_dir / "sampling_groups.jsonl"
    groups = [
        json.loads(line)
        for line in groups_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    for group in groups:
        for diagnostic in group["diagnostics"]:
            candidate_index = diagnostic["candidate_index"]
            raw_index = diagnostic["raw_row_index"]
            group["completions"][candidate_index] = rows[raw_index]["messages"][1:]
    _rewrite_hashed_jsonl_artifact(run_dir, "sampling_groups.jsonl", groups)


def _rollout(
    task_id: str,
    reward: float,
    tools: list[tuple[str, dict]] | None = None,
    *,
    stage: str | None = None,
    terminal_success: bool | None = None,
    complete_success: bool | None = None,
    write_progress: float | None = None,
) -> dict:
    messages = []
    for index, (name, arguments) in enumerate(tools or [], start=1):
        messages.extend(
            [
                {
                    "role": "assistant",
                    "tool_calls": [
                        {
                            "id": f"call-{index}",
                            "name": name,
                            "arguments": arguments,
                        }
                    ],
                },
                {"role": "tool", "content": "ok", "error": False},
            ]
        )
    reward_payload = {"reward": reward}
    if terminal_success is not None:
        reward_payload["terminal_environment_reward"] = float(terminal_success)
    if complete_success is not None:
        reward_payload["complete_success"] = complete_success
    if write_progress is not None:
        reward_payload["components"] = {
            "required_write_progress": {"value": write_progress}
        }
    row = {"task_id": task_id, "reward": reward_payload, "messages": messages}
    if stage is not None:
        row["rollout_stage"] = stage
    return row


class GrpoTrainingAuditTests(unittest.TestCase):
    def _build(
        self,
        raw_rollouts: list[dict],
        *,
        optimization_enabled: bool = True,
        learning_rate: float = 1e-6,
        optimization_evidence: dict | None = None,
        beta: float = 0.0,
        kl_value: float | None = None,
        temperature: float = 0.8,
        runtime_decode_contract: dict | None = None,
    ) -> dict:
        if optimization_enabled and optimization_evidence is None:
            optimization_evidence = {
                "status": "PASSED",
                "trainable_parameter_change_detected": True,
                "final_trainable_parameters": {"all_finite": True},
            }
        log_history = [
            {
                "step": step,
                "reward_std": reward_std,
                "grad_norm": grad_norm,
                "entropy": entropy,
                "completions/mean_length": mean_length,
                "completions/clipped_ratio": clipped_ratio,
                "step_time": step_time,
                **({"kl": kl_value} if kl_value is not None else {}),
            }
            for step, reward_std, grad_norm, entropy, mean_length, clipped_ratio, step_time in [
                (1, 0.0, 0.0, 0.2, 20, 0.0, 2.0),
                (2, 0.7, 0.2, 0.18, 30, 0.5, 4.0),
                (3, 0.7, 0.3, 0.16, 40, 0.0, 6.0),
                (4, 0.0, 0.0, 0.15, 50, 0.5, 8.0),
            ]
        ]
        return build_training_audit(
            config={
                "data": {"task_ids": ["1", "2", "3", "4"]},
                "grpo": {
                    "max_steps": 4,
                    "num_generations": 2,
                    "max_completion_length": 128,
                    "per_device_train_batch_size": 2,
                    "gradient_accumulation_steps": 1,
                    "learning_rate": learning_rate,
                    "temperature": temperature,
                    "beta": beta,
                    "loss_type": "dr_grpo",
                },
            },
            manifest={
                "status": "COMPLETED",
                "optimization_enabled": optimization_enabled,
                "git": {"commit": "abc"},
                "bindings": {"starting_model_sha256": "MODEL"},
                "artifacts": {"raw_rollouts": {"sha256": "RAW"}},
                **(
                    {
                        "runtime": {
                            "sampling_adapter": {
                                "decode_contract": runtime_decode_contract
                            }
                        }
                    }
                    if runtime_decode_contract is not None
                    else {}
                ),
            },
            log_history=log_history,
            raw_rollouts=raw_rollouts,
            train_metrics={"train_runtime": 20.0, "train_loss": -0.1},
            source={"raw_rollouts_sha256": "RAW"},
            optimization_evidence=optimization_evidence,
        )

    def test_summarizes_group_signal_and_generation_metrics(self) -> None:
        report = self._build(
            [
                _rollout("1", 0),
                _rollout("1", 0),
                _rollout("2", 0),
                _rollout("2", 1),
                _rollout("3", 1),
                _rollout("3", 0),
                _rollout("4", 1),
                _rollout("4", 1),
            ]
        )

        self.assertEqual(
            report["training_signal"]["group_counts"],
            {
                "all_zero": 1,
                "mixed": 2,
                "all_positive": 1,
                "uniform_nonzero": 0,
            },
        )
        self.assertEqual(report["training_signal"]["effective_task_ids"], ["2", "3"])
        self.assertEqual(report["training_signal"]["zero_reward_std_steps"], 2)
        self.assertEqual(report["training_signal"]["effective_steps"], [2, 3])
        self.assertEqual(report["training_signal"]["nonzero_grad_steps"], 2)
        self.assertEqual(report["generation"]["mean_completion_length"], 35.0)
        self.assertEqual(report["generation"]["steps_with_clipping"], 2)
        self.assertEqual(report["generation"]["mean_clipped_ratio"], 0.25)
        self.assertFalse(
            report["optimization"]["rollout_vs_update_time_breakdown_available"]
        )
        self.assertTrue(report["claim_limits"]["parameter_update_observed"])
        self.assertFalse(report["claim_limits"]["behavior_improvement_assessed"])
        self.assertEqual(
            report["training_signal"]["per_task"]["2"]["nonzero_std_group_count"],
            1,
        )

    def test_forecasts_mixed_group_probability_from_terminal_success_rate(self) -> None:
        report = self._build(
            [
                _rollout("1", 0, terminal_success=False),
                _rollout("1", 1, terminal_success=True),
                _rollout("2", 0, terminal_success=False),
                _rollout("2", 0, terminal_success=False),
                _rollout("3", 1, terminal_success=True),
                _rollout("3", 1, terminal_success=True),
                _rollout("4", 0, terminal_success=False),
                _rollout("4", 1, terminal_success=True),
            ]
        )
        forecast = report["training_signal"]["per_task"]["1"][
            "group_size_signal_forecast"
        ]
        self.assertEqual(forecast["observed_terminal_success_rate"], 0.5)
        self.assertEqual(
            forecast["estimated_mixed_group_probability"],
            {"2": 0.5, "4": 0.875, "8": 0.9921875},
        )
        self.assertEqual(
            forecast["minimum_candidate_n_for_80_percent_mixed_probability"],
            4,
        )

    def test_no_update_diagnostic_does_not_claim_a_parameter_update(self) -> None:
        report = self._build(
            [
                _rollout("1", 0),
                _rollout("1", 1),
                _rollout("2", 0),
                _rollout("2", 1),
                _rollout("3", 0),
                _rollout("3", 1),
                _rollout("4", 0),
                _rollout("4", 1),
            ],
            optimization_enabled=False,
            learning_rate=0.0,
        )
        self.assertFalse(report["optimization"]["optimization_enabled"])
        self.assertFalse(report["claim_limits"]["parameter_update_observed"])
        self.assertNotIn(
            "NO_KL_CONSTRAINT",
            {warning["code"] for warning in report["warnings"]},
        )

    def test_gradients_without_weight_evidence_do_not_prove_update(self) -> None:
        report = self._build(
            [
                _rollout("1", 0),
                _rollout("1", 1),
                _rollout("2", 0),
                _rollout("2", 1),
                _rollout("3", 0),
                _rollout("3", 1),
                _rollout("4", 0),
                _rollout("4", 1),
            ],
            optimization_evidence={},
        )
        self.assertGreater(report["training_signal"]["nonzero_grad_steps"], 0)
        self.assertFalse(report["claim_limits"]["parameter_update_observed"])
        self.assertFalse(
            report["claim_limits"]["parameter_update_inferred_from_gradients"]
        )

    def test_failed_optimization_evidence_does_not_prove_update(self) -> None:
        rows = [
            _rollout(str(task), reward) for task in range(1, 5) for reward in (0, 1)
        ]
        report = self._build(
            rows,
            optimization_evidence={
                "status": "FAILED",
                "trainable_parameter_change_detected": True,
                "final_trainable_parameters": {"all_finite": True},
            },
        )
        self.assertFalse(report["claim_limits"]["parameter_update_observed"])

    def test_kl_curve_requires_finite_logged_values(self) -> None:
        rows = [
            _rollout("1", 0),
            _rollout("1", 1),
            _rollout("2", 0),
            _rollout("2", 1),
            _rollout("3", 0),
            _rollout("3", 1),
            _rollout("4", 0),
            _rollout("4", 1),
        ]
        missing = self._build(rows, beta=0.02)
        self.assertFalse(missing["optimization"]["kl_curve_available"])
        self.assertIn(
            "KL_CONFIGURED_BUT_CURVE_MISSING",
            {warning["code"] for warning in missing["warnings"]},
        )
        present = self._build(rows, beta=0.02, kl_value=0.01)
        self.assertTrue(present["optimization"]["kl_curve_available"])
        self.assertEqual(present["optimization"]["kl_values"], [0.01] * 4)

    def test_rejects_rollout_count_not_divisible_by_group_size(self) -> None:
        with self.assertRaisesRegex(ValueError, "not divisible"):
            self._build([_rollout("1", 0), _rollout("1", 1), _rollout("2", 0)])

    def test_different_positive_rewards_are_a_mixed_group(self) -> None:
        report = self._build(
            [
                _rollout("1", 0.2),
                _rollout("1", 0.8),
                _rollout("2", 0),
                _rollout("2", 0),
                _rollout("3", 0),
                _rollout("3", 0),
                _rollout("4", 0),
                _rollout("4", 0),
            ]
        )

        self.assertEqual(report["training_signal"]["group_counts"]["mixed"], 1)
        self.assertEqual(report["training_signal"]["effective_task_ids"], ["1"])

    def test_rejects_mixed_task_ids_inside_sequential_group(self) -> None:
        with self.assertRaisesRegex(ValueError, "multiple task IDs"):
            self._build(
                [
                    _rollout("1", 0),
                    _rollout("2", 1),
                    _rollout("3", 0),
                    _rollout("3", 0),
                ]
            )

    def test_summarizes_action_patterns_and_stage_boundary_violations(self) -> None:
        auth = ("find_user_id_by_email", {"email": "a@example.com"})
        details = ("get_user_details", {"user_id": "u1"})
        report = self._build(
            [
                _rollout(
                    "1",
                    0,
                    [auth, details],
                    stage="IDENTITY_AUTHENTICATION",
                ),
                _rollout("1", 1, [auth], stage="IDENTITY_AUTHENTICATION"),
                _rollout("2", 0, []),
                _rollout("2", 1, []),
                _rollout("3", 0, []),
                _rollout("3", 0, []),
                _rollout("4", 0, []),
                _rollout("4", 0, []),
            ]
        )

        behavior = report["behavior"]
        self.assertEqual(behavior["unique_action_pattern_count"], 3)
        self.assertEqual(behavior["no_tool_rollout_count"], 6)
        self.assertEqual(behavior["identity_stage_boundary_violation_count"], 1)
        self.assertEqual(
            behavior["action_pattern_counts"][
                "find_user_id_by_email -> get_user_details"
            ],
            1,
        )
        self.assertGreater(behavior["action_pattern_entropy_nats"], 0.0)
        coverage = report["trajectory_artifact_coverage"]
        self.assertFalse(coverage["tool_calls_in_raw_rollout"])
        self.assertEqual(coverage["rollout_rows_with_tool_calls"], 2)

    def test_tool_use_is_conditioned_on_terminal_outcome(self) -> None:
        auth = ("find_user_id_by_email", {"email": "a@example.com"})
        details = ("get_user_details", {"user_id": "u1"})
        order = ("get_order_details", {"order_id": "#1"})
        rows = [
            _rollout("1", 1, [auth], terminal_success=True),
            _rollout("1", 1, [auth, details, order], terminal_success=True),
            _rollout("2", 0, [], terminal_success=False),
            _rollout("2", 0, [auth, details], terminal_success=False),
            _rollout("3", 0, [], terminal_success=False),
            _rollout("3", 0, [], terminal_success=False),
            _rollout("4", 0, [], terminal_success=False),
            _rollout("4", 0, [], terminal_success=False),
        ]
        report = self._build(rows)

        behavior = report["behavior"]
        self.assertFalse(behavior["mean_tool_calls_per_rollout_is_quality_metric"])
        conditioned = behavior["tool_use_by_terminal_outcome"]
        self.assertEqual(conditioned["success"]["rollouts"], 2)
        self.assertEqual(conditioned["success"]["mean_tool_calls"], 2.0)
        self.assertEqual(conditioned["failure"]["rollouts"], 6)
        self.assertEqual(conditioned["failure"]["mean_tool_calls"], 1 / 3)
        self.assertEqual(conditioned["unknown"]["rollouts"], 0)
        self.assertEqual(
            report["training_signal"]["per_task"]["1"][
                "tool_use_by_terminal_outcome"
            ]["success"]["mean_tool_calls"],
            2.0,
        )

    def test_reports_sampled_pass_at_k_and_terminal_reward_alignment(self) -> None:
        report = self._build(
            [
                _rollout(
                    "1",
                    1.0,
                    terminal_success=True,
                    complete_success=True,
                    write_progress=1.0,
                ),
                _rollout(
                    "1",
                    0.2,
                    terminal_success=False,
                    complete_success=False,
                    write_progress=0.0,
                ),
                _rollout(
                    "2",
                    0.3,
                    terminal_success=False,
                    complete_success=False,
                    write_progress=0.0,
                ),
                _rollout(
                    "2",
                    0.4,
                    terminal_success=False,
                    complete_success=False,
                    write_progress=0.0,
                ),
                _rollout(
                    "3",
                    0.2,
                    terminal_success=False,
                    complete_success=False,
                    write_progress=0.0,
                ),
                _rollout(
                    "3",
                    0.2,
                    terminal_success=False,
                    complete_success=False,
                    write_progress=0.0,
                ),
                _rollout(
                    "4",
                    0.2,
                    terminal_success=False,
                    complete_success=False,
                    write_progress=0.0,
                ),
                _rollout(
                    "4",
                    0.2,
                    terminal_success=False,
                    complete_success=False,
                    write_progress=0.0,
                ),
            ],
            runtime_decode_contract={
                "effective_do_sample": True,
                "effective_num_generations": 2,
            },
        )

        capability = report["capability"]
        self.assertEqual(capability["sampling_mode"], "stochastic")
        self.assertFalse(capability["greedy_pass_at_1_assessed"])
        self.assertEqual(capability["terminal_success_rollouts"], 1)
        self.assertEqual(capability["complete_success_rollouts"], 1)
        self.assertEqual(capability["correct_write_rollouts"], 1)
        self.assertEqual(capability["positive_reward_terminal_failure_rollouts"], 7)
        ordering = capability["same_group_terminal_reward_ordering"]
        self.assertTrue(ordering["assessed"])
        self.assertEqual(ordering["strict_ordering_rate"], 1.0)
        task = report["training_signal"]["per_task"]["1"]
        self.assertEqual(task["sampled_terminal_pass_at_k"]["1"], 0.5)
        self.assertEqual(task["sampled_terminal_pass_at_k"]["2"], 1.0)
        self.assertIn(
            "PARTIAL_REWARD_WITHOUT_TERMINAL_SUCCESS",
            {warning["code"] for warning in report["warnings"]},
        )

    def test_temperature_zero_without_runtime_contract_is_not_greedy(self) -> None:
        rows = [
            _rollout(str(task), reward) for task in range(1, 5) for reward in (0.0, 1.0)
        ]
        report = self._build(rows, temperature=0.0)

        capability = report["capability"]
        self.assertEqual(capability["sampling_mode"], "decode_mode_not_execution_bound")
        self.assertFalse(capability["greedy_pass_at_1_assessed"])
        self.assertIsNone(capability["effective_do_sample"])
        self.assertIsNone(capability["effective_num_generations"])

    def test_greedy_requires_runtime_do_sample_false_and_one_completion(
        self,
    ) -> None:
        rows = [
            _rollout(str(task), reward) for task in range(1, 5) for reward in (0.0, 1.0)
        ]
        greedy = self._build(
            rows,
            runtime_decode_contract={
                "effective_do_sample": False,
                "effective_num_generations": 1,
            },
        )["capability"]
        multiple = self._build(
            rows,
            runtime_decode_contract={
                "effective_do_sample": False,
                "effective_num_generations": 2,
            },
        )["capability"]
        sampled = self._build(
            rows,
            temperature=0.0,
            runtime_decode_contract={
                "effective_do_sample": True,
                "effective_num_generations": 1,
            },
        )["capability"]

        self.assertEqual(greedy["sampling_mode"], "greedy")
        self.assertTrue(greedy["greedy_pass_at_1_assessed"])
        self.assertEqual(multiple["sampling_mode"], "decode_mode_not_execution_bound")
        self.assertFalse(multiple["greedy_pass_at_1_assessed"])
        self.assertEqual(sampled["sampling_mode"], "stochastic")
        self.assertFalse(sampled["greedy_pass_at_1_assessed"])

    def test_partial_write_progress_is_not_a_correct_write(self) -> None:
        report = self._build(
            [
                _rollout(str(task), reward, write_progress=progress)
                for task, reward, progress in [
                    (1, 0.5, 0.5),
                    (1, 0.0, 0.0),
                    (2, 0.0, 0.0),
                    (2, 0.0, 0.0),
                    (3, 0.0, 0.0),
                    (3, 0.0, 0.0),
                    (4, 0.0, 0.0),
                    (4, 0.0, 0.0),
                ]
            ]
        )
        self.assertEqual(report["capability"]["correct_write_rollouts"], 0)

    def test_audit_run_rejects_tampered_log_history(self) -> None:
        with TemporaryDirectory() as temp_dir:
            run_dir, config_path = _write_audit_fixture(Path(temp_dir))
            _write_json(run_dir / "log_history.json", [{"reward_std": 0.0}])
            with self.assertRaisesRegex(ValueError, "Log history SHA-256"):
                audit_run(run_dir, config_path)

    def test_audit_run_rejects_tampered_train_metrics(self) -> None:
        with TemporaryDirectory() as temp_dir:
            run_dir, config_path = _write_audit_fixture(Path(temp_dir))
            _write_json(run_dir / "train_metrics.json", {"train_loss": 9.0})
            with self.assertRaisesRegex(ValueError, "Train metrics SHA-256"):
                audit_run(run_dir, config_path)

    def test_audit_run_rejects_unbound_optimization_evidence(self) -> None:
        with TemporaryDirectory() as temp_dir:
            run_dir, config_path = _write_audit_fixture(Path(temp_dir))
            manifest_path = run_dir / "run_manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["artifacts"].pop("optimization_evidence")
            _write_json(manifest_path, manifest)
            with self.assertRaisesRegex(ValueError, "not bound"):
                audit_run(run_dir, config_path)

    def test_audit_run_accepts_true_greedy_pure_sampling_without_training_logs(
        self,
    ) -> None:
        with TemporaryDirectory() as temp_dir:
            run_dir, config_path = _write_sampling_audit_fixture(Path(temp_dir))

            report = audit_run(run_dir, config_path)

            self.assertEqual(report["status"], "PASSED")
            self.assertEqual(
                report["scope"],
                "PURE_SAMPLING_CAPABILITY_AND_TRAJECTORY_AUDIT_ONLY",
            )
            self.assertTrue(report["capability"]["greedy_pass_at_1_assessed"])
            self.assertEqual(report["capability"]["greedy_terminal_pass_at_1"], 0.5)
            self.assertFalse(report["training"]["applicable"])
            self.assertEqual(report["training"]["optimizer_steps"], 0)
            self.assertTrue(all(report["release_criteria"].values()))
            self.assertFalse((run_dir / "log_history.json").exists())
            self.assertFalse((run_dir / "train_metrics.json").exists())

    def test_audit_run_accepts_stochastic_groups_and_reports_pass_signal(
        self,
    ) -> None:
        terminals = {
            ("43", 0): [True, False],
            ("72", 0): [True, True],
            ("43", 1): [False, False],
            ("72", 1): [False, False],
        }
        rewards = {
            ("43", 0): [0.9, 0.2],
            ("72", 0): [0.8, 0.8],
            ("43", 1): [0.2, 0.4],
            ("72", 1): [0.1, 0.1],
        }
        with TemporaryDirectory() as temp_dir:
            run_dir, config_path = _write_sampling_audit_fixture(
                Path(temp_dir),
                sampling_mode="STOCHASTIC_GROUP_SAMPLING",
                group_size=2,
                groups_per_task=2,
                terminal_matrix=terminals,
                reward_matrix=rewards,
            )

            report = audit_run(run_dir, config_path)

        self.assertEqual(report["status"], "PASSED")
        self.assertTrue(all(report["release_criteria"].values()))
        self.assertEqual(report["capability"]["sampling_mode"], "stochastic")
        self.assertFalse(report["capability"]["greedy_pass_at_1_assessed"])
        self.assertEqual(report["trajectory"]["rollouts"], 8)
        task_43 = report["capability"]["per_task"]["43"]
        self.assertEqual(task_43["valid_terminal_rollouts"], 4)
        self.assertEqual(task_43["terminal_success_rollouts"], 1)
        self.assertEqual(task_43["sampled_terminal_success_rate"], 0.25)
        self.assertEqual(task_43["sampled_terminal_pass_at_k"]["1"], 0.25)
        self.assertEqual(task_43["sampled_terminal_pass_at_k"]["2"], 0.5)
        self.assertEqual(task_43["sampled_terminal_pass_at_k"]["4"], 1.0)
        task_72 = report["capability"]["per_task"]["72"]
        self.assertEqual(task_72["valid_terminal_rollouts"], 4)
        self.assertEqual(task_72["terminal_success_rollouts"], 2)
        self.assertEqual(task_72["sampled_terminal_success_rate"], 0.5)
        self.assertTrue(task_72["observed_any_terminal_success"])
        self.assertEqual(
            report["capability"]["actual_group_any_terminal_success"],
            {"valid_groups": 4, "groups_with_any_success": 2, "rate": 0.5},
        )
        self.assertEqual(
            report["group_signal"]["terminal_outcome_group_counts"],
            {"all_correct": 1, "all_wrong": 2, "mixed": 1},
        )
        self.assertEqual(
            report["group_signal"]["staged_reward"]["zero_std_groups"], 2
        )
        self.assertEqual(
            report["group_signal"]["terminal_binary"]["zero_std_groups"], 3
        )
        self.assertEqual(report["trajectory"]["transport_invalid_rollouts"], 0)
        self.assertEqual(report["trajectory"]["censored_rollouts"], 0)
        self.assertEqual(
            report["trajectory"]["model_eos_before_user_stop_valid_failures"], 5
        )
        self.assertEqual(report["trajectory"]["completion_tokens"]["mean"], 13.5)
        self.assertEqual(
            report["trajectory"]["observation_tokens_retained"]["mean"], 4.0
        )
        self.assertTrue(
            report["capability"]["same_group_terminal_reward_ordering"][
                "assessed"
            ]
        )

    def test_transport_invalid_rollout_is_excluded_and_closes_release(self) -> None:
        terminals = {
            ("43", 0): [True, False],
            ("72", 0): [False, False],
            ("43", 1): [False, False],
            ("72", 1): [False, False],
        }
        with TemporaryDirectory() as temp_dir:
            run_dir, config_path = _write_sampling_audit_fixture(
                Path(temp_dir),
                sampling_mode="STOCHASTIC_GROUP_SAMPLING",
                group_size=2,
                groups_per_task=2,
                terminal_matrix=terminals,
                transport_invalid_raw_indices={1},
            )

            report = audit_run(run_dir, config_path)

        self.assertEqual(report["status"], "REJECTED")
        self.assertFalse(
            report["release_criteria"]["all_trajectory_transport_complete"]
        )
        self.assertEqual(report["trajectory"]["transport_invalid_rollouts"], 1)
        self.assertEqual(report["trajectory"]["censored_rollouts"], 1)
        task_43 = report["capability"]["per_task"]["43"]
        self.assertEqual(task_43["rollouts"], 4)
        self.assertEqual(task_43["valid_terminal_rollouts"], 3)
        self.assertEqual(task_43["terminal_success_rollouts"], 1)
        self.assertEqual(task_43["sampled_terminal_success_rate"], 1 / 3)
        self.assertEqual(
            report["trajectory"]["model_eos_before_user_stop_valid_failures"], 6
        )

    def test_duplicate_diagnostic_raw_index_closes_release(self) -> None:
        with TemporaryDirectory() as temp_dir:
            run_dir, config_path = _write_sampling_audit_fixture(
                Path(temp_dir),
                sampling_mode="STOCHASTIC_GROUP_SAMPLING",
                group_size=2,
                groups_per_task=2,
            )
            groups_path = run_dir / "sampling_groups.jsonl"
            groups = [
                json.loads(line)
                for line in groups_path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            groups[1]["diagnostics"][0]["raw_row_index"] = 0
            groups_path.write_text(
                "\n".join(json.dumps(row, ensure_ascii=False) for row in groups)
                + "\n",
                encoding="utf-8",
            )
            manifest_path = run_dir / "run_manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["artifacts"]["sampling_groups.jsonl"]["sha256"] = _sha256(
                groups_path
            )
            _write_json(manifest_path, manifest)
            _write_json(
                run_dir / "run_state.json",
                {
                    "status": "COMPLETED",
                    "run_manifest_sha256": _sha256(manifest_path),
                },
            )

            report = audit_run(run_dir, config_path)

        self.assertEqual(report["status"], "REJECTED")
        self.assertFalse(
            report["release_criteria"][
                "diagnostic_raw_candidate_indices_bound"
            ]
        )

    def test_pure_sampling_decode_drift_is_not_reported_as_greedy(self) -> None:
        with TemporaryDirectory() as temp_dir:
            run_dir, config_path = _write_sampling_audit_fixture(Path(temp_dir))
            manifest_path = run_dir / "run_manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            contract = manifest["runtime"]["sampling_adapter"]["decode_contract"]
            contract["effective_do_sample"] = True
            _write_json(manifest_path, manifest)
            _write_json(
                run_dir / "run_state.json",
                {
                    "status": "COMPLETED",
                    "run_manifest_sha256": _sha256(manifest_path),
                },
            )

            report = audit_run(run_dir, config_path)

            self.assertEqual(report["status"], "REJECTED")
            self.assertEqual(report["capability"]["sampling_mode"], "stochastic")
            self.assertFalse(report["capability"]["greedy_pass_at_1_assessed"])
            self.assertIsNone(report["capability"]["greedy_terminal_pass_at_1"])

    def test_stochastic_runtime_decode_knob_drift_closes_release(self) -> None:
        for field, value in (
            ("effective_temperature", 0.7),
            ("effective_top_p", 0.9),
            ("effective_top_k", 10),
        ):
            with self.subTest(field=field), TemporaryDirectory() as temp_dir:
                run_dir, config_path = _write_sampling_audit_fixture(
                    Path(temp_dir),
                    sampling_mode="STOCHASTIC_GROUP_SAMPLING",
                    group_size=2,
                    groups_per_task=2,
                )
                manifest_path = run_dir / "run_manifest.json"
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                manifest["runtime"]["sampling_adapter"]["decode_contract"][
                    field
                ] = value
                _write_json(manifest_path, manifest)
                _write_json(
                    run_dir / "run_state.json",
                    {
                        "status": "COMPLETED",
                        "run_manifest_sha256": _sha256(manifest_path),
                    },
                )

                report = audit_run(run_dir, config_path)

            self.assertEqual(report["status"], "REJECTED")
            self.assertFalse(
                report["release_criteria"]["runtime_sampling_mode_bound"]
            )

    def test_effective_non_sampling_config_drift_fails_closed(self) -> None:
        with TemporaryDirectory() as temp_dir:
            run_dir, config_path = _write_sampling_audit_fixture(
                Path(temp_dir),
                sampling_mode="STOCHASTIC_GROUP_SAMPLING",
                group_size=2,
                groups_per_task=2,
            )
            effective_path = run_dir / "effective_config.json"
            effective = json.loads(effective_path.read_text(encoding="utf-8"))
            effective["grpo"]["temperature"] = 0.7
            _write_json(effective_path, effective)
            manifest_path = run_dir / "run_manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["artifacts"]["effective_config.json"]["sha256"] = _sha256(
                effective_path
            )
            _write_json(manifest_path, manifest)
            _write_json(
                run_dir / "run_state.json",
                {
                    "status": "COMPLETED",
                    "run_manifest_sha256": _sha256(manifest_path),
                },
            )

            with self.assertRaisesRegex(ValueError, "non-sampling"):
                audit_run(run_dir, config_path)

    def test_effective_sampling_adapter_field_drift_fails_closed(self) -> None:
        cases = (
            ("actual_num_generations", 3, "actual_num_generations"),
            ("actual_num_generations", True, "actual_num_generations"),
            ("trl_constructor_num_generations", 3, "constructor"),
            ("groups_per_task", 3, "groups_per_task"),
            ("groups_per_task", True, "groups_per_task"),
            ("trainer_max_steps_unused", False, "max_steps"),
        )
        for field, value, message in cases:
            with self.subTest(field=field), TemporaryDirectory() as temp_dir:
                run_dir, config_path = _write_sampling_audit_fixture(
                    Path(temp_dir),
                    sampling_mode="STOCHASTIC_GROUP_SAMPLING",
                    group_size=2,
                    groups_per_task=2,
                )
                effective_path = run_dir / "effective_config.json"
                effective = json.loads(effective_path.read_text(encoding="utf-8"))
                effective["sampling"][field] = value
                _write_json(effective_path, effective)
                manifest_path = run_dir / "run_manifest.json"
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                manifest["artifacts"]["effective_config.json"]["sha256"] = (
                    _sha256(effective_path)
                )
                _write_json(manifest_path, manifest)
                _write_json(
                    run_dir / "run_state.json",
                    {
                        "status": "COMPLETED",
                        "run_manifest_sha256": _sha256(manifest_path),
                    },
                )

                with self.assertRaisesRegex(ValueError, message):
                    audit_run(run_dir, config_path)

    def test_frozen_diagnostic_rollout_shape_drift_fails_closed(self) -> None:
        cases = (
            ("expected_tasks", 3),
            ("expected_tasks", True),
            ("expected_rollouts", 9),
            ("expected_rollouts_per_task", 5),
            ("groups_per_task", 3),
            ("group_size", 3),
        )
        for field, value in cases:
            with self.subTest(field=field), TemporaryDirectory() as temp_dir:
                run_dir, config_path = _write_sampling_audit_fixture(
                    Path(temp_dir),
                    sampling_mode="STOCHASTIC_GROUP_SAMPLING",
                    group_size=2,
                    groups_per_task=2,
                )
                frozen = json.loads(config_path.read_text(encoding="utf-8"))
                frozen["diagnostic"][field] = value
                _write_json(config_path, frozen)
                effective_path = run_dir / "effective_config.json"
                effective = json.loads(effective_path.read_text(encoding="utf-8"))
                effective["diagnostic"][field] = value
                _write_json(effective_path, effective)
                manifest_path = run_dir / "run_manifest.json"
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                manifest["config_sha256"] = _sha256(config_path)
                manifest["artifacts"]["effective_config.json"]["sha256"] = (
                    _sha256(effective_path)
                )
                _write_json(manifest_path, manifest)
                _write_json(
                    run_dir / "run_state.json",
                    {
                        "status": "COMPLETED",
                        "run_manifest_sha256": _sha256(manifest_path),
                    },
                )

                with self.assertRaisesRegex(ValueError, "Frozen diagnostic"):
                    audit_run(run_dir, config_path)

    def test_completion_suffix_is_allowed_only_by_enumerated_terminal_rule(
        self,
    ) -> None:
        cases = (
            (
                "post_user_stop_text",
                0,
                [{"role": "assistant", "content": "Thanks. Goodbye."}],
                None,
                "PASSED",
                "POST_USER_STOP_ASSISTANT_TEXT",
            ),
            (
                "unconsumed_model_eos_text",
                1,
                [
                    {
                        "role": "assistant",
                        "content": '<tool_call>{"name":"respond_to_user"}',
                    }
                ],
                None,
                "PASSED",
                "UNCONSUMED_TERMINAL_MODEL_TEXT",
            ),
            (
                "multiple_messages",
                0,
                [
                    {"role": "assistant", "content": "one"},
                    {"role": "assistant", "content": "two"},
                ],
                None,
                "REJECTED",
                None,
            ),
            (
                "parsed_tool_call",
                0,
                [
                    {
                        "role": "assistant",
                        "tool_calls": [
                            {
                                "name": "get_order_details",
                                "arguments": {"order_id": "#X"},
                            }
                        ],
                    }
                ],
                None,
                "REJECTED",
                None,
            ),
            (
                "wrong_stop_reason",
                0,
                [{"role": "assistant", "content": "tail"}],
                "CUSTOM_STOP",
                "REJECTED",
                None,
            ),
        )
        for name, group_index, suffix, stop_override, expected, rule in cases:
            with self.subTest(name=name), TemporaryDirectory() as temp_dir:
                run_dir, config_path = _write_sampling_audit_fixture(Path(temp_dir))
                groups_path = run_dir / "sampling_groups.jsonl"
                groups = [
                    json.loads(line)
                    for line in groups_path.read_text(encoding="utf-8").splitlines()
                    if line.strip()
                ]
                groups[group_index]["completions"][0].extend(suffix)
                if stop_override is not None:
                    groups[group_index]["diagnostics"][0]["stop_reason"] = stop_override
                _rewrite_hashed_jsonl_artifact(
                    run_dir, "sampling_groups.jsonl", groups
                )

                report = audit_run(run_dir, config_path)

            self.assertEqual(
                report["status"],
                expected,
                {
                    "case": name,
                    "failed_release_criteria": [
                        key
                        for key, value in report["release_criteria"].items()
                        if not value
                    ],
                    "trajectory": report["trajectory"],
                },
            )
            if expected == "PASSED":
                self.assertTrue(
                    report["release_criteria"][
                        "group_prompt_and_completion_bound"
                    ]
                )
                self.assertEqual(
                    report["trajectory"][
                        "allowed_unconsumed_terminal_model_message_rules"
                    ],
                    {rule: 1},
                )
            else:
                self.assertFalse(
                    report["release_criteria"][
                        "group_prompt_and_completion_bound"
                    ]
                )
                self.assertEqual(
                    report["trajectory"]["rejected_completion_suffixes"], 1
                )

    def test_censored_transport_complete_rollout_is_excluded_and_rejected(self) -> None:
        with TemporaryDirectory() as temp_dir:
            run_dir, config_path = _write_sampling_audit_fixture(
                Path(temp_dir),
                sampling_mode="STOCHASTIC_GROUP_SAMPLING",
                group_size=2,
                groups_per_task=2,
            )
            groups_path = run_dir / "sampling_groups.jsonl"
            groups = [
                json.loads(line)
                for line in groups_path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            groups[0]["diagnostics"][0]["model_completion_truncated"] = True
            groups_path.write_text(
                "\n".join(json.dumps(row, ensure_ascii=False) for row in groups)
                + "\n",
                encoding="utf-8",
            )
            manifest_path = run_dir / "run_manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["artifacts"]["sampling_groups.jsonl"]["sha256"] = _sha256(
                groups_path
            )
            _write_json(manifest_path, manifest)
            _write_json(
                run_dir / "run_state.json",
                {
                    "status": "COMPLETED",
                    "run_manifest_sha256": _sha256(manifest_path),
                },
            )

            report = audit_run(run_dir, config_path)

        self.assertEqual(report["status"], "REJECTED")
        self.assertFalse(report["release_criteria"]["no_censored_rollouts"])
        self.assertEqual(report["trajectory"]["censored_rollouts"], 1)
        self.assertEqual(report["capability"]["valid_terminal_rollouts"], 7)

    def test_duplicate_group_id_closes_release(self) -> None:
        with TemporaryDirectory() as temp_dir:
            run_dir, config_path = _write_sampling_audit_fixture(
                Path(temp_dir),
                sampling_mode="STOCHASTIC_GROUP_SAMPLING",
                group_size=2,
                groups_per_task=2,
            )
            groups_path = run_dir / "sampling_groups.jsonl"
            groups = [
                json.loads(line)
                for line in groups_path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            groups[1]["group_id"] = groups[0]["group_id"]
            groups_path.write_text(
                "\n".join(json.dumps(row, ensure_ascii=False) for row in groups)
                + "\n",
                encoding="utf-8",
            )
            manifest_path = run_dir / "run_manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["artifacts"]["sampling_groups.jsonl"]["sha256"] = _sha256(
                groups_path
            )
            _write_json(manifest_path, manifest)
            _write_json(
                run_dir / "run_state.json",
                {
                    "status": "COMPLETED",
                    "run_manifest_sha256": _sha256(manifest_path),
                },
            )

            report = audit_run(run_dir, config_path)

        self.assertEqual(report["status"], "REJECTED")
        self.assertFalse(
            report["release_criteria"]["group_ids_unique_and_complete"]
        )

    def test_group_reward_or_completion_drift_closes_release(self) -> None:
        for mutation, criterion in (
            ("reward", "observational_rewards_bound"),
            ("completion", "group_prompt_and_completion_bound"),
        ):
            with self.subTest(mutation=mutation), TemporaryDirectory() as temp_dir:
                run_dir, config_path = _write_sampling_audit_fixture(
                    Path(temp_dir),
                    sampling_mode="STOCHASTIC_GROUP_SAMPLING",
                    group_size=2,
                    groups_per_task=2,
                )
                groups_path = run_dir / "sampling_groups.jsonl"
                groups = [
                    json.loads(line)
                    for line in groups_path.read_text(encoding="utf-8").splitlines()
                    if line.strip()
                ]
                if mutation == "reward":
                    groups[0]["rewards_observational_only"][0] = 0.123
                else:
                    groups[0]["completions"][0][-1]["content"] = "drifted"
                groups_path.write_text(
                    "\n".join(json.dumps(row, ensure_ascii=False) for row in groups)
                    + "\n",
                    encoding="utf-8",
                )
                manifest_path = run_dir / "run_manifest.json"
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                manifest["artifacts"]["sampling_groups.jsonl"]["sha256"] = _sha256(
                    groups_path
                )
                _write_json(manifest_path, manifest)
                _write_json(
                    run_dir / "run_state.json",
                    {
                        "status": "COMPLETED",
                        "run_manifest_sha256": _sha256(manifest_path),
                    },
                )

                report = audit_run(run_dir, config_path)

            self.assertEqual(report["status"], "REJECTED")
            self.assertFalse(report["release_criteria"][criterion])

    def test_pure_sampling_rejects_any_training_flag(self) -> None:
        with TemporaryDirectory() as temp_dir:
            run_dir, config_path = _write_sampling_audit_fixture(Path(temp_dir))
            manifest_path = run_dir / "run_manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["optimizer_created"] = True
            _write_json(manifest_path, manifest)
            _write_json(
                run_dir / "run_state.json",
                {
                    "status": "COMPLETED",
                    "run_manifest_sha256": _sha256(manifest_path),
                },
            )

            report = audit_run(run_dir, config_path)

            self.assertEqual(report["status"], "REJECTED")
            self.assertFalse(report["release_criteria"]["no_training_or_update_path"])

    def test_pure_sampling_rejects_tampered_hash_bound_artifact(self) -> None:
        with TemporaryDirectory() as temp_dir:
            run_dir, config_path = _write_sampling_audit_fixture(Path(temp_dir))
            with (run_dir / "sampling_groups.jsonl").open("a", encoding="utf-8") as handle:
                handle.write("{}\n")

            with self.assertRaisesRegex(ValueError, "artifact SHA-256 mismatch"):
                audit_run(run_dir, config_path)

    def test_pure_sampling_rejects_opening_prompt_seed_drift(self) -> None:
        with TemporaryDirectory() as temp_dir:
            run_dir, config_path = _write_sampling_audit_fixture(Path(temp_dir))
            groups_path = run_dir / "sampling_groups.jsonl"
            groups = [
                json.loads(line)
                for line in groups_path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            groups[0]["prompt"][0]["content"] = "drifted request"
            groups_path.write_text(
                "\n".join(json.dumps(row, ensure_ascii=False) for row in groups) + "\n",
                encoding="utf-8",
            )
            manifest_path = run_dir / "run_manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["artifacts"]["sampling_groups.jsonl"]["sha256"] = _sha256(
                groups_path
            )
            _write_json(manifest_path, manifest)
            _write_json(
                run_dir / "run_state.json",
                {
                    "status": "COMPLETED",
                    "run_manifest_sha256": _sha256(manifest_path),
                },
            )

            report = audit_run(run_dir, config_path)

            self.assertEqual(report["status"], "REJECTED")
            self.assertFalse(
                report["release_criteria"]["opening_prompt_and_seed_bound"]
            )

    def test_marks_terminal_capability_unavailable_for_unstructured_reward(
        self,
    ) -> None:
        report = self._build(
            [
                _rollout("1", 0),
                _rollout("1", 1),
                _rollout("2", 0),
                _rollout("2", 1),
                _rollout("3", 0),
                _rollout("3", 1),
                _rollout("4", 0),
                _rollout("4", 1),
            ]
        )
        self.assertFalse(report["capability"]["terminal_success_label_available"])
        self.assertFalse(report["capability"]["correct_write_label_available"])
        self.assertFalse(
            report["capability"]["same_group_terminal_reward_ordering"]["assessed"]
        )


class WriteTaskGrpoPrescreenTests(unittest.TestCase):
    @staticmethod
    def _run(
        root: Path,
        terminal_matrix: dict[tuple[str, int], list[bool]],
        *,
        missing_result_indices: set[int] | None = None,
        error_result_indices: set[int] | None = None,
    ) -> dict:
        run_dir, config_path = _write_sampling_audit_fixture(
            root,
            sampling_mode="STOCHASTIC_GROUP_SAMPLING",
            group_size=2,
            groups_per_task=2,
            terminal_matrix=terminal_matrix,
        )
        _inject_required_write_evidence(
            run_dir,
            missing_result_indices=missing_result_indices,
            error_result_indices=error_result_indices,
        )
        return audit_run(run_dir, config_path)

    def test_mixed_group_with_grounded_success_passes_prescreen(self) -> None:
        with TemporaryDirectory() as temp_dir:
            report = self._run(
                Path(temp_dir),
                {("95", 0): [True, False], ("95", 1): [True, True]},
            )

        prescreen = report["write_task_grpo_prescreen"]
        self.assertEqual(report["status"], "PASSED")
        self.assertEqual(prescreen["status"], "PASSED")
        self.assertTrue(all(prescreen["criteria"].values()))
        self.assertEqual(
            report["capability"]["grounded_terminal_success_rollouts"], 3
        )

    def test_two_terminal_constant_groups_block_prescreen(self) -> None:
        with TemporaryDirectory() as temp_dir:
            report = self._run(
                Path(temp_dir),
                {("95", 0): [True, True], ("95", 1): [False, False]},
            )

        prescreen = report["write_task_grpo_prescreen"]
        self.assertEqual(report["status"], "PASSED")
        self.assertEqual(prescreen["status"], "BLOCKED")
        self.assertFalse(prescreen["criteria"]["mixed_terminal_group_observed"])

    def test_missing_write_result_blocks_prescreen_as_unverifiable(self) -> None:
        with TemporaryDirectory() as temp_dir:
            report = self._run(
                Path(temp_dir),
                {("95", 0): [True, False], ("95", 1): [True, True]},
                missing_result_indices={0},
            )

        prescreen = report["write_task_grpo_prescreen"]
        self.assertEqual(prescreen["status"], "BLOCKED")
        self.assertFalse(
            prescreen["criteria"]["required_write_execution_labels_available"]
        )

    def test_tool_errors_on_all_terminal_successes_block_prescreen(self) -> None:
        with TemporaryDirectory() as temp_dir:
            report = self._run(
                Path(temp_dir),
                {("95", 0): [True, False], ("95", 1): [True, True]},
                error_result_indices={0, 2, 3},
            )

        prescreen = report["write_task_grpo_prescreen"]
        self.assertEqual(prescreen["status"], "BLOCKED")
        self.assertTrue(
            prescreen["criteria"]["required_write_execution_labels_available"]
        )
        self.assertFalse(
            prescreen["criteria"]["grounded_terminal_success_observed"]
        )


class SuccessfulRequiredWriteExecutionTests(unittest.TestCase):
    @staticmethod
    def _row() -> dict:
        return {
            "reward": {
                "reward": 1.0,
                "reward_mode": "terminal_environment_state",
                "action_progress": {
                    "matches": [
                        {
                            "action_id": "95_0",
                            "name": "exchange_delivered_order_items",
                            "matched": True,
                            "matched_call_index": 0,
                        },
                        {
                            "action_id": "95_1",
                            "name": "exchange_delivered_order_items",
                            "matched": True,
                            "matched_call_index": 1,
                        },
                    ],
                    "unexpected_write_count": 0,
                },
            },
            "messages": [
                {
                    "role": "assistant",
                    "tool_calls": [
                        {
                            "id": "write-1",
                            "name": "exchange_delivered_order_items",
                            "arguments": {"order_id": "#W1"},
                            "requestor": "assistant",
                        }
                    ],
                },
                {
                    "id": "write-1",
                    "role": "tool",
                    "content": "first exchange requested",
                    "requestor": "assistant",
                    "error": False,
                },
                {
                    "role": "assistant",
                    "tool_calls": [
                        {
                            "id": "write-2",
                            "name": "exchange_delivered_order_items",
                            "arguments": {"order_id": "#W2"},
                            "requestor": "assistant",
                        }
                    ],
                },
                {
                    "id": "write-2",
                    "role": "tool",
                    "content": "second exchange requested",
                    "requestor": "assistant",
                    "error": False,
                },
            ],
        }

    def test_two_required_writes_need_two_successful_bound_results(self) -> None:
        self.assertTrue(_successful_required_write_execution(self._row()))

    def test_tool_reported_error_is_a_failed_required_write(self) -> None:
        row = deepcopy(self._row())
        row["messages"][-1]["error"] = True
        self.assertFalse(_successful_required_write_execution(row))

    def test_missing_tool_result_is_unverifiable_not_a_model_failure(self) -> None:
        row = deepcopy(self._row())
        row["messages"].pop()
        self.assertIsNone(_successful_required_write_execution(row))

    def test_unmatched_or_unexpected_write_is_not_successful(self) -> None:
        unmatched = deepcopy(self._row())
        unmatched["reward"]["action_progress"]["matches"][1]["matched"] = False
        self.assertFalse(_successful_required_write_execution(unmatched))
        unexpected = deepcopy(self._row())
        unexpected["reward"]["action_progress"]["unexpected_write_count"] = 1
        self.assertFalse(_successful_required_write_execution(unexpected))

    def test_malformed_match_closes_evidence(self) -> None:
        row = deepcopy(self._row())
        row["reward"]["action_progress"]["matches"].append(None)
        self.assertIsNone(_successful_required_write_execution(row))

    def test_call_id_reused_by_user_call_is_ambiguous(self) -> None:
        row = deepcopy(self._row())
        row["messages"].insert(
            0,
            {
                "role": "user",
                "tool_calls": [
                    {
                        "id": "write-1",
                        "name": "get_order_details",
                        "arguments": {"order_id": "#W1"},
                        "requestor": "user",
                    }
                ],
            },
        )
        self.assertIsNone(_successful_required_write_execution(row))


if __name__ == "__main__":
    unittest.main()
