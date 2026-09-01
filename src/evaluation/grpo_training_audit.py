from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any

from src.guards.retail_pre_action import WRITE_TOOLS


SCHEMA_VERSION = "retail-agentic-grpo-training-audit-v4"
SAMPLING_SCHEMA_VERSION = "retail-agentic-pure-sampling-audit-v1"
IDENTITY_AUTHENTICATION_ACTIONS = {
    "find_user_id_by_email",
    "find_user_id_by_name_zip",
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def _read_json(path: Path) -> dict[str, Any] | list[dict[str, Any]]:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not rows:
        raise ValueError(f"No JSONL rows found in {path}")
    return rows


def _mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def _range(values: list[float]) -> dict[str, float | None]:
    return {
        "min": min(values) if values else None,
        "max": max(values) if values else None,
    }


def _reward_value(row: dict[str, Any]) -> float:
    reward = row.get("reward")
    if not isinstance(reward, dict) or "reward" not in reward:
        raise ValueError("Rollout row is missing reward.reward")
    return float(reward["reward"])


def _reward_payload(row: dict[str, Any]) -> dict[str, Any]:
    reward = row.get("reward")
    if not isinstance(reward, dict):
        raise ValueError("Rollout row is missing a structured reward payload")
    return reward


def _terminal_success(row: dict[str, Any]) -> bool | None:
    reward = _reward_payload(row)
    if "terminal_environment_reward" in reward:
        value = reward["terminal_environment_reward"]
    elif reward.get("reward_mode") == "terminal_environment_state":
        value = reward.get("reward")
    else:
        component = (reward.get("components") or {}).get(
            "terminal_environment_state"
        )
        value = component.get("value") if isinstance(component, dict) else None
    if value is None:
        return None
    return math.isclose(float(value), 1.0, abs_tol=1e-12)


def _complete_success(row: dict[str, Any]) -> bool | None:
    reward = _reward_payload(row)
    if "complete_success" in reward:
        return bool(reward["complete_success"])
    return _terminal_success(row)


def _correct_write(row: dict[str, Any]) -> bool | None:
    component = (_reward_payload(row).get("components") or {}).get(
        "required_write_progress"
    )
    if not isinstance(component, dict) or component.get("value") is None:
        return None
    return math.isclose(float(component["value"]), 1.0, abs_tol=1e-12)


def _successful_required_write_execution(row: dict[str, Any]) -> bool | None:
    """Verify required writes succeeded and no unexpected write was issued.

    This is an audit label, not a reward and not a policy-compliance verdict.
    ``action_progress`` binds calls to hidden required actions; serialized messages
    independently prove that each selected write reached a non-error tool result.
    """

    reward = _reward_payload(row)
    progress = reward.get("action_progress")
    if not isinstance(progress, dict) or not isinstance(progress.get("matches"), list):
        return None
    matches = progress["matches"]
    if any(not isinstance(match, dict) for match in matches):
        return None
    if any(
        not isinstance(match.get("name"), str) or not match["name"]
        for match in matches
    ):
        return None
    required = [match for match in matches if match["name"] in WRITE_TOOLS]
    if not required:
        return None
    action_ids = [match.get("action_id") for match in required]
    if (
        any(not isinstance(action_id, str) or not action_id for action_id in action_ids)
        or len(action_ids) != len(set(action_ids))
    ):
        return None
    unexpected_write_count = progress.get("unexpected_write_count")
    if type(unexpected_write_count) is not int or unexpected_write_count < 0:
        return None
    if unexpected_write_count:
        return False

    messages = row.get("messages")
    if not isinstance(messages, list):
        return None
    calls: list[tuple[int, dict[str, Any]]] = []
    all_call_ids: list[Any] = []
    results: dict[str, list[tuple[int, dict[str, Any]]]] = {}
    for position, message in enumerate(messages):
        if not isinstance(message, dict):
            return None
        if message.get("role") in {"assistant", "user"}:
            tool_calls = message.get("tool_calls") or []
            if not isinstance(tool_calls, list):
                return None
            for call in tool_calls:
                if not isinstance(call, dict):
                    return None
                all_call_ids.append(call.get("id"))
                if (
                    message.get("role") == "assistant"
                    and call.get("requestor", "assistant") == "assistant"
                ):
                    calls.append((position, call))
        elif message.get("role") == "tool":
            call_id = message.get("id")
            if isinstance(call_id, str) and call_id:
                results.setdefault(call_id, []).append((position, message))

    selected_indices: list[int] = []
    for match in required:
        if match.get("matched") is not True:
            return False
        index = match.get("matched_call_index")
        if type(index) is not int or not 0 <= index < len(calls):
            return None
        selected_indices.append(index)
        call_position, call = calls[index]
        if call.get("name") != match.get("name"):
            return None
        call_id = call.get("id")
        if (
            not isinstance(call_id, str)
            or not call_id
            or all_call_ids.count(call_id) != 1
            or len(results.get(call_id, [])) != 1
        ):
            return None
        result_position, result = results[call_id][0]
        if (
            result_position <= call_position
            or result.get("requestor", "assistant") != "assistant"
            or type(result.get("error")) is not bool
            or (
                result.get("name") is not None
                and result.get("name") != call.get("name")
            )
            or not isinstance(result.get("content"), str)
            or not result["content"].strip()
        ):
            return None
        if result["error"]:
            return False
    if len(selected_indices) != len(set(selected_indices)):
        return None
    return True


def _pass_at_k(total: int, correct: int, k: int) -> float | None:
    if total <= 0 or k <= 0 or k > total:
        return None
    if correct <= 0:
        return 0.0
    if total - correct < k:
        return 1.0
    return 1.0 - math.comb(total - correct, k) / math.comb(total, k)


def _sampled_pass_curve(total: int, correct: int) -> dict[str, float]:
    ks = sorted({value for value in [1, 2, 4, 8, total] if value <= total})
    return {str(k): float(_pass_at_k(total, correct, k) or 0.0) for k in ks}


def _group_size_signal_forecast(total: int, correct: int) -> dict[str, Any] | None:
    """Estimate binary-reward mixed-group probability under an IID stationary model."""
    if total <= 0 or correct < 0 or correct > total:
        return None
    success_rate = correct / total
    candidate_sizes = (2, 4, 8)
    probabilities = {
        str(size): 1.0 - success_rate**size - (1.0 - success_rate) ** size
        for size in candidate_sizes
    }
    minimum_at_least_80_percent = next(
        (
            size
            for size in candidate_sizes
            if probabilities[str(size)] >= 0.8
        ),
        None,
    )
    return {
        "observed_rollouts": total,
        "observed_terminal_success_rate": success_rate,
        "iid_stationary_binary_reward_assumption": True,
        "estimated_mixed_group_probability": probabilities,
        "minimum_candidate_n_for_80_percent_mixed_probability": (
            minimum_at_least_80_percent
        ),
        "decision_limit": (
            "Diagnostic only: the estimate is unreliable when policy updates occur "
            "between groups, rollouts are correlated, or Reward is non-binary."
        ),
    }


def _same_group_terminal_ordering(
    groups: list[list[dict[str, Any]]],
) -> dict[str, Any]:
    correct = 0
    tied = 0
    inverted = 0
    comparable_groups = 0
    for group in groups:
        successes = [row for row in group if _terminal_success(row) is True]
        failures = [row for row in group if _terminal_success(row) is False]
        if not successes or not failures:
            continue
        comparable_groups += 1
        for success in successes:
            for failure in failures:
                delta = _reward_value(success) - _reward_value(failure)
                if delta > 1e-12:
                    correct += 1
                elif delta < -1e-12:
                    inverted += 1
                else:
                    tied += 1
    pairs = correct + tied + inverted
    return {
        "assessed": pairs > 0,
        "comparable_groups": comparable_groups,
        "comparable_pairs": pairs,
        "correctly_ordered_pairs": correct,
        "tied_pairs": tied,
        "inverted_pairs": inverted,
        "strict_ordering_rate": correct / pairs if pairs else None,
    }


def _tool_calls(row: dict[str, Any]) -> list[dict[str, Any]]:
    calls: list[dict[str, Any]] = []
    for message in row.get("messages") or []:
        for call in message.get("tool_calls") or []:
            calls.append(
                {
                    "name": str(call.get("name") or ""),
                    "arguments": dict(call.get("arguments") or {}),
                }
            )
    return calls


def _action_pattern(row: dict[str, Any]) -> str:
    names = [call["name"] for call in _tool_calls(row)]
    return " -> ".join(names) if names else "<none>"


def _exact_action_signature(row: dict[str, Any]) -> str:
    return json.dumps(_tool_calls(row), ensure_ascii=False, sort_keys=True)


def _duplicate_ratio(values: list[str]) -> float | None:
    return (len(values) - len(set(values))) / len(values) if values else None


def _categorical_entropy(counts: Counter[str]) -> float | None:
    total = sum(counts.values())
    if not total:
        return None
    return -sum(
        (count / total) * math.log(count / total)
        for count in counts.values()
        if count
    )


def _tool_use_by_terminal_outcome(
    rows: list[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    partitions = {
        "success": [row for row in rows if _terminal_success(row) is True],
        "failure": [row for row in rows if _terminal_success(row) is False],
        "unknown": [row for row in rows if _terminal_success(row) is None],
    }
    output: dict[str, dict[str, Any]] = {}
    for name, partition in partitions.items():
        counts = [float(len(_tool_calls(row))) for row in partition]
        patterns = Counter(_action_pattern(row) for row in partition)
        output[name] = {
            "rollouts": len(partition),
            "mean_tool_calls": _mean(counts),
            "tool_call_count_range": _range(counts),
            "action_pattern_counts": dict(patterns.most_common()),
        }
    return output


def _failure_behavior_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    failures = [row for row in rows if _terminal_success(row) is False]
    missing_required_action_rollouts = 0
    missing_required_action_total = 0
    action_progress_evaluable_rollouts = 0
    unexpected_write_rollouts = 0
    unexpected_write_total = 0
    tool_error_rollouts = 0
    tool_error_total = 0
    repeated_call_rollouts = 0
    repeated_call_total = 0
    premature_model_eos_rollouts = 0

    for row in failures:
        reward = _reward_payload(row)
        progress = reward.get("action_progress")
        if isinstance(progress, dict) and isinstance(progress.get("matches"), list):
            matches = progress["matches"]
            if all(isinstance(match, dict) for match in matches):
                action_progress_evaluable_rollouts += 1
                missing = sum(match.get("matched") is not True for match in matches)
                missing_required_action_total += missing
                missing_required_action_rollouts += missing > 0
            unexpected = progress.get("unexpected_write_count")
            if type(unexpected) is int and unexpected >= 0:
                unexpected_write_total += unexpected
                unexpected_write_rollouts += unexpected > 0
            repeated = progress.get("duplicate_excess_count")
            if type(repeated) is int and repeated >= 0:
                repeated_call_total += repeated
                repeated_call_rollouts += repeated > 0

        tool_errors = reward.get("tool_error_count")
        if type(tool_errors) is int and tool_errors >= 0:
            tool_error_total += tool_errors
            tool_error_rollouts += tool_errors > 0

        completion = row.get("completion") or {}
        if completion.get("stop_reason") == "MODEL_EOS_BEFORE_USER_STOP":
            premature_model_eos_rollouts += 1

    return {
        "terminal_failure_rollouts": len(failures),
        "action_progress_evaluable_rollouts": action_progress_evaluable_rollouts,
        "missing_required_action_rollouts": missing_required_action_rollouts,
        "missing_required_action_total": missing_required_action_total,
        "unexpected_write_rollouts": unexpected_write_rollouts,
        "unexpected_write_total": unexpected_write_total,
        "tool_error_rollouts": tool_error_rollouts,
        "tool_error_total": tool_error_total,
        "repeated_call_rollouts": repeated_call_rollouts,
        "repeated_call_total": repeated_call_total,
        "premature_model_eos_rollouts": premature_model_eos_rollouts,
    }


def _population_std(values: list[float]) -> float | None:
    if not values:
        return None
    mean = sum(values) / len(values)
    return math.sqrt(sum((value - mean) ** 2 for value in values) / len(values))


def _identity_stage_boundary_violation(row: dict[str, Any]) -> bool:
    if str(row.get("rollout_stage") or "") != "IDENTITY_AUTHENTICATION":
        return False
    calls = _tool_calls(row)
    first_auth = next(
        (
            index
            for index, call in enumerate(calls)
            if call["name"] in IDENTITY_AUTHENTICATION_ACTIONS
        ),
        None,
    )
    if first_auth is None:
        return False
    return any(
        call["name"] not in IDENTITY_AUTHENTICATION_ACTIONS
        for call in calls[first_auth + 1 :]
    )


def _group_rollouts(
    rows: list[dict[str, Any]], num_generations: int
) -> list[list[dict[str, Any]]]:
    if num_generations < 2:
        raise ValueError("num_generations must be at least 2 for a GRPO audit")
    if len(rows) % num_generations:
        raise ValueError(
            f"Rollout rows ({len(rows)}) are not divisible by num_generations "
            f"({num_generations})"
        )
    groups: list[list[dict[str, Any]]] = []
    for start in range(0, len(rows), num_generations):
        group = rows[start : start + num_generations]
        task_ids = {str(row.get("task_id")) for row in group}
        if len(task_ids) != 1:
            raise ValueError(
                "Sequential rollout group contains multiple task IDs: "
                f"rows {start}-{start + num_generations - 1}: {sorted(task_ids)}"
            )
        groups.append(group)
    return groups


def _canonical_sha256(value: dict[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            default=str,
        ).encode("utf-8")
    ).hexdigest().upper()


def _final_answer(row: dict[str, Any]) -> str | None:
    for message in reversed(row.get("messages") or []):
        if message.get("role") == "assistant" and message.get("content") is not None:
            return str(message["content"])
    return None


def _normalized_tool_call(call: dict[str, Any]) -> dict[str, Any]:
    function = call.get("function")
    if isinstance(function, dict):
        return {
            "name": str(function.get("name") or ""),
            "arguments": dict(function.get("arguments") or {}),
        }
    return {
        "name": str(call.get("name") or ""),
        "arguments": dict(call.get("arguments") or {}),
    }


def _project_completion_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Normalize TRL tool messages to the environment trajectory representation."""

    projected: list[dict[str, Any]] = []
    pending_tool_names: list[str] = []
    for message in messages:
        role = message.get("role")
        calls = [
            _normalized_tool_call(call)
            for call in (message.get("tool_calls") or [])
            if isinstance(call, dict)
        ]
        if role == "assistant" and calls:
            if len(calls) == 1 and calls[0]["name"] == "respond_to_user":
                projected.append(
                    {
                        "role": "assistant",
                        "content": str(calls[0]["arguments"].get("message") or ""),
                    }
                )
            else:
                projected.append({"role": "assistant", "tool_calls": calls})
                pending_tool_names = [call["name"] for call in calls]
        elif role == "tool" and message.get("name") == "respond_to_user":
            projected.append(
                {"role": "user", "content": str(message.get("content") or "")}
            )
        elif role == "tool":
            tool_name = str(message.get("name") or "")
            if not tool_name and pending_tool_names:
                tool_name = pending_tool_names.pop(0)
            projected.append(
                {
                    "role": "tool",
                    "name": tool_name,
                    "content": str(message.get("content") or ""),
                }
            )
        elif role == "user":
            projected.append(
                {"role": "user", "content": str(message.get("content") or "")}
            )
        elif role == "assistant":
            projected.append(
                {"role": "assistant", "content": str(message.get("content") or "")}
            )
    return projected


def _project_raw_messages_after_opening(
    row: dict[str, Any], opening: Any
) -> list[dict[str, Any]] | None:
    messages = row.get("messages") or []
    opening_index = next(
        (
            index
            for index, message in enumerate(messages)
            if message.get("role") == "user" and message.get("content") == opening
        ),
        None,
    )
    if opening_index is None:
        return None
    return _project_completion_messages(messages[opening_index + 1 :])


def _numeric_summary(values: list[float]) -> dict[str, float | int | None]:
    return {
        "count": len(values),
        "total": sum(values) if values else 0.0,
        "mean": _mean(values),
        **_range(values),
    }


def _diagnostic_is_censored(row: dict[str, Any]) -> bool:
    return any(
        bool(row.get(name))
        for name in (
            "completion_token_budget_exhausted",
            "context_limit_reached",
            "customer_turn_limit_reached",
            "tool_call_limit_reached",
            "tool_iteration_limit_reached",
            "unresolved_tool_call",
            "framework_loop_abnormal_end",
            "model_completion_truncated",
        )
    )


def _validate_effective_config_matches_frozen(
    frozen: dict[str, Any], effective: dict[str, Any]
) -> None:
    """Reject runtime config drift outside the explicit sampling adapter fields."""

    frozen_non_sampling = {
        key: value for key, value in frozen.items() if key != "sampling"
    }
    effective_non_sampling = {
        key: value for key, value in effective.items() if key != "sampling"
    }
    if effective_non_sampling != frozen_non_sampling:
        raise ValueError("Effective config drifted from frozen non-sampling fields")

    frozen_sampling = frozen.get("sampling")
    effective_sampling = effective.get("sampling")
    if not isinstance(frozen_sampling, dict) or not isinstance(
        effective_sampling, dict
    ):
        raise ValueError("Pure-sampling audit requires an explicit frozen contract")
    if any(
        effective_sampling.get(key) != value
        for key, value in frozen_sampling.items()
    ):
        raise ValueError("Effective sampling request drifted from the frozen config")
    adapter_fields = {
        "actual_num_generations",
        "trl_constructor_num_generations",
        "groups_per_task",
        "trainer_max_steps_unused",
        "configured_temperature",
    }
    unexpected = set(effective_sampling) - set(frozen_sampling) - adapter_fields
    missing = adapter_fields - set(effective_sampling)
    if unexpected or missing:
        raise ValueError(
            "Effective sampling adapter fields are incomplete or unexpected: "
            f"missing={sorted(missing)}, unexpected={sorted(unexpected)}"
        )
    if effective_sampling["configured_temperature"] != (
        frozen.get("grpo") or {}
    ).get("temperature"):
        raise ValueError("Effective decode temperature drifted from frozen GRPO config")

    task_ids = (frozen.get("data") or {}).get("task_ids") or []
    diagnostic = frozen.get("diagnostic") or {}
    frozen_group_size = (frozen.get("grpo") or {}).get("num_generations")
    if type(frozen_group_size) is not int or frozen_group_size < 1:
        raise ValueError("Frozen GRPO num_generations must be a positive integer")

    expected_rollouts_per_task = diagnostic.get("expected_rollouts_per_task")
    if (
        type(expected_rollouts_per_task) is not int
        or expected_rollouts_per_task < 1
        or expected_rollouts_per_task % frozen_group_size != 0
    ):
        raise ValueError("Frozen diagnostic rollout shape is not divisible by group size")
    derived_groups_per_task = expected_rollouts_per_task // frozen_group_size
    configured_groups_per_task = diagnostic.get("groups_per_task")
    if configured_groups_per_task is not None and (
        type(configured_groups_per_task) is not int
        or configured_groups_per_task != derived_groups_per_task
    ):
        raise ValueError("Frozen diagnostic groups_per_task is inconsistent")
    configured_diagnostic_group_size = diagnostic.get("group_size")
    if configured_diagnostic_group_size is not None and (
        type(configured_diagnostic_group_size) is not int
        or configured_diagnostic_group_size != frozen_group_size
    ):
        raise ValueError("Frozen diagnostic group_size is inconsistent")
    if type(diagnostic.get("expected_tasks")) is not int or diagnostic.get(
        "expected_tasks"
    ) != len(task_ids):
        raise ValueError("Frozen diagnostic expected_tasks is inconsistent")
    expected_rollouts = len(task_ids) * expected_rollouts_per_task
    if type(diagnostic.get("expected_rollouts")) is not int or diagnostic.get(
        "expected_rollouts"
    ) != expected_rollouts:
        raise ValueError("Frozen diagnostic expected_rollouts is inconsistent")
    frozen_max_steps_unused = diagnostic.get("trainer_max_steps_unused")
    if frozen_max_steps_unused is not None and frozen_max_steps_unused is not True:
        raise ValueError("Frozen diagnostic must mark trainer max_steps unused")

    if (
        type(effective_sampling["actual_num_generations"]) is not int
        or effective_sampling["actual_num_generations"] != frozen_group_size
    ):
        raise ValueError("Effective actual_num_generations drifted from frozen config")
    expected_constructor_generations = max(2, frozen_group_size)
    if (
        type(effective_sampling["trl_constructor_num_generations"]) is not int
        or effective_sampling["trl_constructor_num_generations"]
        != expected_constructor_generations
    ):
        raise ValueError("TRL constructor num_generations violated sampling protocol")
    if (
        type(effective_sampling["groups_per_task"]) is not int
        or effective_sampling["groups_per_task"] != derived_groups_per_task
    ):
        raise ValueError("Effective groups_per_task drifted from frozen diagnostic")
    if effective_sampling["trainer_max_steps_unused"] is not True:
        raise ValueError("Pure sampling must not consume trainer max_steps")


def _bind_completion_to_environment_log(
    *,
    candidate: list[dict[str, Any]],
    projected_completion: list[dict[str, Any]],
    projected_raw: list[dict[str, Any]],
    diagnostic: dict[str, Any],
    raw_completion: dict[str, Any],
) -> tuple[bool, int, str | None]:
    """Bind a sampled completion while enumerating the only permitted suffixes."""

    if len(candidate) != len(projected_completion):
        return False, 0, None
    if projected_completion[: len(projected_raw)] != projected_raw:
        return False, 0, None

    suffix = projected_completion[len(projected_raw) :]
    if not suffix:
        return True, 0, None
    if len(suffix) != 1:
        return False, len(suffix), None

    original_tail = candidate[-1]
    projected_tail = suffix[0]
    plain_assistant_tail = (
        original_tail.get("role") == "assistant"
        and not (original_tail.get("tool_calls") or [])
        and projected_tail.get("role") == "assistant"
        and not (projected_tail.get("tool_calls") or [])
        and isinstance(projected_tail.get("content"), str)
        and bool(projected_tail["content"].strip())
    )
    stop_reason = diagnostic.get("stop_reason")
    stop_contract_bound = (
        raw_completion.get("stop_reason") == stop_reason
        and diagnostic.get("model_ended") is True
        and raw_completion.get("model_ended") is True
    )
    if not plain_assistant_tail or not stop_contract_bound:
        return False, 1, None

    if stop_reason == "USER_STOP_AND_MODEL_EOS" and "<tool_call>" not in str(
        projected_tail["content"]
    ):
        return True, 1, "POST_USER_STOP_ASSISTANT_TEXT"
    if stop_reason == "MODEL_EOS_BEFORE_USER_STOP":
        return True, 1, "UNCONSUMED_TERMINAL_MODEL_TEXT"
    return False, 1, None


def build_sampling_audit(
    *,
    config: dict[str, Any],
    manifest: dict[str, Any],
    raw_rollouts: list[dict[str, Any]],
    evidence_rows: list[dict[str, Any]],
    sampling_groups: list[dict[str, Any]],
    source: dict[str, Any],
) -> dict[str, Any]:
    """Audit a no-update rollout baseline without fabricating training metrics."""

    runtime = manifest.get("runtime") or {}
    sampling_adapter = runtime.get("sampling_adapter") or {}
    decode_contract = sampling_adapter.get("decode_contract") or {}
    effective_do_sample = decode_contract.get("effective_do_sample")
    effective_num_generations = decode_contract.get("effective_num_generations")
    true_greedy = effective_do_sample is False and effective_num_generations == 1
    if effective_do_sample is True:
        sampling_mode = "stochastic"
    elif true_greedy:
        sampling_mode = "greedy"
    else:
        sampling_mode = "decode_mode_not_execution_bound"

    expected_task_ids = [
        str(value) for value in (config.get("data") or {}).get("task_ids") or []
    ]
    expected_task_set = set(expected_task_ids)
    raw_task_ids = [str(row.get("task_id")) for row in raw_rollouts]
    evidence_task_ids = [str(row.get("task_id")) for row in evidence_rows]
    group_task_ids = [str(row.get("task_id")) for row in sampling_groups]
    group_size = (
        effective_num_generations
        if type(effective_num_generations) is int
        and effective_num_generations > 0
        else 0
    )
    groups_per_task = (config.get("sampling") or {}).get("groups_per_task", 1)
    groups_per_task = groups_per_task if type(groups_per_task) is int else 0
    expected_groups = len(expected_task_ids) * groups_per_task
    expected_rollouts = expected_groups * group_size

    diagnostic_indices: list[int] = []
    bound_groups: list[list[dict[str, Any]]] = []
    bound_group_diagnostics: list[list[dict[str, Any]]] = []
    diagnostic_index_binding_valid = group_size > 0
    group_structure_bound = group_size > 0
    opening_bindings = group_size > 0
    identity_bound = len(raw_rollouts) == len(evidence_rows)
    evidence_hashes_bound = identity_bound
    complete_trajectory_artifacts = identity_bound
    observational_rewards_bound = group_size > 0
    completion_bindings = group_size > 0
    trailing_group_completion_messages = 0
    allowed_terminal_suffixes: Counter[str] = Counter()
    rejected_completion_suffixes = 0
    all_transport_complete = True

    for group in sampling_groups:
        task_id = str(group.get("task_id"))
        diagnostics = group.get("diagnostics")
        completions = group.get("completions")
        observational_rewards = group.get("rewards_observational_only")
        structurally_valid = (
            group.get("status") == "COMPLETED"
            and group.get("training_eligible") is False
            and isinstance(group.get("prompt"), list)
            and isinstance(completions, list)
            and len(completions) == group_size
            and isinstance(diagnostics, list)
            and len(diagnostics) == group_size
            and isinstance(observational_rewards, list)
            and len(observational_rewards) == group_size
        )
        group_structure_bound = group_structure_bound and structurally_valid
        if not structurally_valid:
            diagnostic_index_binding_valid = False
            opening_bindings = False
            continue

        candidate_indices = [item.get("candidate_index") for item in diagnostics]
        raw_indices = [item.get("raw_row_index") for item in diagnostics]
        indices_valid = (
            all(type(value) is int for value in candidate_indices)
            and set(candidate_indices) == set(range(group_size))
            and len(set(candidate_indices)) == group_size
            and all(type(value) is int for value in raw_indices)
            and len(set(raw_indices)) == group_size
            and all(0 <= value < len(raw_rollouts) for value in raw_indices)
        )
        diagnostic_index_binding_valid = diagnostic_index_binding_valid and indices_valid
        if not indices_valid:
            opening_bindings = False
            continue

        ordered = sorted(diagnostics, key=lambda item: item["candidate_index"])
        raw_indices = [item["raw_row_index"] for item in ordered]
        diagnostic_indices.extend(raw_indices)
        group_rows = [raw_rollouts[index] for index in raw_indices]
        group_evidence = [evidence_rows[index] for index in raw_indices]
        bound_groups.append(group_rows)
        bound_group_diagnostics.append(ordered)
        try:
            observational_rewards_bound = observational_rewards_bound and all(
                math.isclose(
                    float(observational_rewards[candidate_index]),
                    _reward_value(row),
                    abs_tol=1e-12,
                )
                for candidate_index, row in enumerate(group_rows)
            )
        except (TypeError, ValueError):
            observational_rewards_bound = False

        prompt_initial_user = next(
            (
                message.get("content")
                for message in group.get("prompt") or []
                if message.get("role") == "user"
            ),
            None,
        )
        opening_bindings = opening_bindings and (
            prompt_initial_user is not None
            and all(
                row.get("user_seed") == group.get("user_seed")
                and str(row.get("task_id")) == task_id
                and next(
                    (
                        message.get("content")
                        for message in row.get("messages") or []
                        if message.get("role") == "user"
                    ),
                    None,
                )
                == prompt_initial_user
                for row in group_rows
            )
        )
        for candidate_index, row in enumerate(group_rows):
            candidate = completions[candidate_index]
            projected_raw = _project_raw_messages_after_opening(
                row, prompt_initial_user
            )
            projected_completion = (
                _project_completion_messages(candidate)
                if isinstance(candidate, list)
                else None
            )
            if projected_raw is not None and projected_completion is not None:
                candidate_bound, suffix_count, suffix_rule = (
                    _bind_completion_to_environment_log(
                        candidate=candidate,
                        projected_completion=projected_completion,
                        projected_raw=projected_raw,
                        diagnostic=ordered[candidate_index],
                        raw_completion=row.get("completion") or {},
                    )
                )
            else:
                candidate_bound, suffix_count, suffix_rule = False, 0, None
            completion_bindings = completion_bindings and candidate_bound
            trailing_group_completion_messages += suffix_count
            if candidate_bound and suffix_rule is not None:
                allowed_terminal_suffixes[suffix_rule] += 1
            elif suffix_count:
                rejected_completion_suffixes += 1

        for raw, evidence, diagnostic in zip(
            group_rows, group_evidence, ordered, strict=True
        ):
            identity_bound = identity_bound and (
                str(raw.get("task_id")) == task_id
                and str(evidence.get("task_id")) == task_id
                and raw.get("user_seed") == group.get("user_seed")
                and evidence.get("user_seed") == group.get("user_seed")
            )
            canonical = dict(evidence)
            claimed_hash = canonical.pop("evidence_sha256", None)
            evidence_hashes_bound = evidence_hashes_bound and (
                claimed_hash is not None
                and claimed_hash == _canonical_sha256(canonical)
                and raw.get("evidence_sha256") == claimed_hash
                and raw.get("completion") == evidence.get("completion")
            )
            complete_trajectory_artifacts = complete_trajectory_artifacts and (
                isinstance(raw.get("messages"), list)
                and type(raw.get("tool_calls")) is int
                and raw.get("tool_calls") >= 0
                and isinstance(raw.get("reward"), dict)
                and isinstance(evidence.get("tool_trace"), list)
                and isinstance(evidence.get("terminal_evaluator"), dict)
                and isinstance(evidence.get("initial_state"), dict)
                and isinstance(evidence.get("final_state"), dict)
                and isinstance(evidence.get("state_diff"), list)
                and isinstance(evidence.get("state_hashes"), dict)
                and _final_answer(raw) is not None
                and (raw.get("completion") or {}).get("stop_reason_source")
                == "guarded_grpo_trainer_v1"
            )
            all_transport_complete = all_transport_complete and (
                diagnostic.get("training_eligible") is False
                and diagnostic.get("trajectory_transport_complete") is True
            )

    diagnostic_index_binding_valid = diagnostic_index_binding_valid and (
        len(diagnostic_indices) == len(raw_rollouts)
        and len(set(diagnostic_indices)) == len(raw_rollouts)
        and set(diagnostic_indices) == set(range(len(raw_rollouts)))
    )
    if not diagnostic_index_binding_valid:
        identity_bound = False
        evidence_hashes_bound = False
        complete_trajectory_artifacts = False

    diagnostic_by_raw_index = {
        item["raw_row_index"]: item
        for group in bound_group_diagnostics
        for item in group
    }
    censored_indices = {
        index
        for index, diagnostic in diagnostic_by_raw_index.items()
        if _diagnostic_is_censored(diagnostic)
    }
    transport_complete_indices = {
        index
        for index, diagnostic in diagnostic_by_raw_index.items()
        if diagnostic.get("trajectory_transport_complete") is True
    }
    valid_raw_indices = transport_complete_indices - censored_indices
    transport_invalid_indices = (
        set(range(len(raw_rollouts))) - transport_complete_indices
    )

    terminal_labels = [_terminal_success(row) for row in raw_rollouts]
    complete_labels = [_complete_success(row) for row in raw_rollouts]
    write_labels = [_correct_write(row) for row in raw_rollouts]
    successful_write_labels = [
        _successful_required_write_execution(row) for row in raw_rollouts
    ]
    action_patterns = [_action_pattern(row) for row in raw_rollouts]
    tool_call_counts = [len(_tool_calls(row)) for row in raw_rollouts]
    per_task: dict[str, dict[str, Any]] = {}
    for task_id in expected_task_ids:
        task_indices = [
            index for index, value in enumerate(raw_task_ids) if value == task_id
        ]
        valid_terminal_indices = [
            index
            for index in task_indices
            if index in valid_raw_indices and terminal_labels[index] is not None
        ]
        total = len(valid_terminal_indices)
        correct = sum(terminal_labels[index] is True for index in valid_terminal_indices)
        first_terminal = (
            terminal_labels[valid_terminal_indices[0]]
            if true_greedy and total == 1
            else None
        )
        first_complete = (
            complete_labels[valid_terminal_indices[0]]
            if true_greedy and total == 1
            else None
        )
        per_task[task_id] = {
            "rollouts": len(task_indices),
            "valid_terminal_rollouts": total,
            "transport_invalid_rollouts": sum(
                index in transport_invalid_indices for index in task_indices
            ),
            "terminal_success_rollouts": correct,
            "sampled_terminal_success_rate": correct / total if total else None,
            "sampled_terminal_pass_at_k": (
                _sampled_pass_curve(total, correct)
                if sampling_mode == "stochastic" and total
                else None
            ),
            "observed_any_terminal_success": correct > 0 if total else None,
            "terminal_success": first_terminal,
            "complete_success": first_complete,
            "successful_required_write_label_available": (
                bool(valid_terminal_indices)
                and all(
                    successful_write_labels[index] is not None
                    for index in valid_terminal_indices
                )
            ),
            "successful_required_write_rollouts": sum(
                successful_write_labels[index] is True
                for index in valid_terminal_indices
            ),
        }

    valid_terminal_indices = [
        index
        for index in sorted(valid_raw_indices)
        if terminal_labels[index] is not None
    ]
    terminal_available = bool(valid_terminal_indices) and all(
        terminal_labels[index] is not None for index in valid_raw_indices
    )
    terminal_success_count = sum(
        terminal_labels[index] is True for index in valid_terminal_indices
    )

    valid_groups: list[list[dict[str, Any]]] = []
    valid_group_diagnostics: list[list[dict[str, Any]]] = []
    for group, diagnostics in zip(
        bound_groups, bound_group_diagnostics, strict=True
    ):
        if all(
            item.get("trajectory_transport_complete") is True
            and _terminal_success(row) is not None
            for row, item in zip(group, diagnostics, strict=True)
        ):
            valid_groups.append(group)
            valid_group_diagnostics.append(diagnostics)

    staged_group_stds = [
        float(_population_std([_reward_value(row) for row in group]) or 0.0)
        for group in valid_groups
    ]
    terminal_group_stds = [
        float(
            _population_std(
                [1.0 if _terminal_success(row) is True else 0.0 for row in group]
            )
            or 0.0
        )
        for group in valid_groups
    ]
    terminal_group_kinds: Counter[str] = Counter()
    group_any_success = 0
    for group in valid_groups:
        labels = [_terminal_success(row) is True for row in group]
        if all(labels):
            terminal_group_kinds["all_correct"] += 1
        elif not any(labels):
            terminal_group_kinds["all_wrong"] += 1
        else:
            terminal_group_kinds["mixed"] += 1
        group_any_success += int(any(labels))
    ordering = _same_group_terminal_ordering(valid_groups)

    common_pass_keys = sorted(
        {
            key
            for row in per_task.values()
            for key in (row.get("sampled_terminal_pass_at_k") or {})
        },
        key=int,
    )
    macro_pass_curve = {
        key: _mean(
            [
                row["sampled_terminal_pass_at_k"][key]
                for row in per_task.values()
                if key in (row.get("sampled_terminal_pass_at_k") or {})
            ]
        )
        for key in common_pass_keys
    }

    sampling_request = config.get("sampling") or {}
    manifest_decode_contract = manifest.get("decode_contract") or {}
    requested_mode = sampling_request.get("mode")
    configured_group_size = (config.get("grpo") or {}).get("num_generations")
    greedy_request_matches = (
        requested_mode == "TRUE_GREEDY"
        and sampling_request.get("do_sample") is False
        and configured_group_size == 1
        and true_greedy
    )
    stochastic_request_matches = (
        requested_mode == "STOCHASTIC_GROUP_SAMPLING"
        and sampling_request.get("do_sample") is True
        and type(configured_group_size) is int
        and configured_group_size >= 2
        and effective_do_sample is True
        and effective_num_generations == configured_group_size
        and decode_contract.get("effective_temperature")
        == sampling_request.get("temperature")
        and decode_contract.get("effective_top_p")
        == sampling_request.get("top_p")
        and decode_contract.get("effective_top_k")
        == sampling_request.get("top_k")
    )
    runtime_contract_keys = set(sampling_request) - {"configured_temperature"}
    runtime_decode_contract_bound = all(
        decode_contract.get(key) == sampling_request.get(key)
        for key in runtime_contract_keys
    )
    manifest_decode_contract_bound = manifest_decode_contract == sampling_request
    task_group_counts = Counter(group_task_ids)
    group_ids = [str(row.get("group_id") or "") for row in sampling_groups]
    expected_group_ids = {
        f"{task_id}:{repeat}"
        for repeat in range(groups_per_task)
        for task_id in expected_task_ids
    }
    release_criteria = {
        "manifest_is_completed_pure_sampling": (
            manifest.get("status") == "COMPLETED"
            and manifest.get("execution_mode") == "PURE_SAMPLING"
        ),
        "no_training_or_update_path": all(
            manifest.get(name) is False
            for name in (
                "optimization_enabled",
                "backward_called",
                "optimizer_created",
                "loss_computed",
                "training_eligible",
            )
        ),
        "runtime_sampling_mode_bound": (
            greedy_request_matches or stochastic_request_matches
        ),
        "frozen_sampling_request_matches": (
            greedy_request_matches or stochastic_request_matches
        ),
        "runtime_decode_contract_bound": runtime_decode_contract_bound,
        "manifest_decode_contract_bound": manifest_decode_contract_bound,
        "starting_model_hash_matches": (
            manifest.get("starting_model_sha256")
            == (config.get("model") or {}).get("expected_sha256")
        ),
        "direct_merged_checkpoint_without_peft": (
            sampling_adapter.get("direct_merged_checkpoint_inference") is True
            and (runtime.get("model_loading") or {}).get("peft_adapter_applied")
            is False
            and (runtime.get("model_loading") or {}).get("quantized") is False
        ),
        "requested_rollout_shape_matches": (
            groups_per_task > 0
            and len(sampling_groups) == expected_groups
            and len(raw_rollouts) == expected_rollouts
            and all(
                task_group_counts[task_id] == groups_per_task
                and raw_task_ids.count(task_id) == groups_per_task * group_size
                for task_id in expected_task_ids
            )
        ),
        "group_ids_unique_and_complete": (
            len(group_ids) == len(set(group_ids))
            and set(group_ids) == expected_group_ids
        ),
        "frozen_task_ids_unique": len(expected_task_ids) == len(expected_task_set),
        "task_coverage_matches": (
            set(raw_task_ids) == expected_task_set
            and set(evidence_task_ids) == expected_task_set
            and set(group_task_ids) == expected_task_set
        ),
        "manifest_counts_match": (
            manifest.get("groups") == len(sampling_groups)
            and manifest.get("rollouts") == len(raw_rollouts)
            and len(raw_rollouts) == expected_rollouts
        ),
        "diagnostic_raw_candidate_indices_bound": diagnostic_index_binding_valid,
        "raw_evidence_identity_bound": identity_bound,
        "opening_prompt_and_seed_bound": opening_bindings,
        "evidence_hashes_bound": evidence_hashes_bound,
        "observational_rewards_bound": observational_rewards_bound,
        "group_prompt_and_completion_bound": (
            group_structure_bound and completion_bindings
        ),
        "all_trajectory_transport_complete": all_transport_complete,
        "no_censored_rollouts": not censored_indices,
        "terminal_labels_available_for_valid_rollouts": terminal_available,
        "complete_trajectory_artifacts": complete_trajectory_artifacts,
    }
    successful_required_write_labels_available = bool(valid_terminal_indices) and all(
        successful_write_labels[index] is not None
        for index in valid_terminal_indices
    )
    grounded_terminal_success_rollouts = sum(
        terminal_labels[index] is True and successful_write_labels[index] is True
        for index in valid_terminal_indices
    )
    write_task_prescreen_criteria = {
        "base_sampling_audit_passed": all(release_criteria.values()),
        "stochastic_group_sampling": sampling_mode == "stochastic",
        "all_rollouts_transport_valid_and_uncensored": (
            len(valid_raw_indices) == len(raw_rollouts)
            and not transport_invalid_indices
            and not censored_indices
        ),
        "mixed_terminal_group_observed": terminal_group_kinds["mixed"] > 0,
        "required_write_execution_labels_available": (
            successful_required_write_labels_available
        ),
        "grounded_terminal_success_observed": grounded_terminal_success_rollouts > 0,
    }
    write_task_prescreen_applicable = sampling_mode == "stochastic"
    return {
        "schema_version": SAMPLING_SCHEMA_VERSION,
        "scope": "PURE_SAMPLING_CAPABILITY_AND_TRAJECTORY_AUDIT_ONLY",
        "status": (
            "PASSED" if all(release_criteria.values()) else "REJECTED"
        ),
        "source": source,
        "bindings": {
            "project_commit": (manifest.get("git") or {}).get("commit"),
            "starting_model_sha256": manifest.get("starting_model_sha256"),
            "config_sha256": manifest.get("config_sha256"),
            "raw_rollouts_sha256": (
                (manifest.get("artifacts") or {}).get("raw_rollouts.jsonl") or {}
            ).get("sha256"),
        },
        "configuration": {
            "task_ids": expected_task_ids,
            "configured_num_generations": (config.get("grpo") or {}).get(
                "num_generations"
            ),
            "groups_per_task": groups_per_task,
            "effective_num_generations": effective_num_generations,
            "effective_do_sample": effective_do_sample,
            "effective_temperature": decode_contract.get("effective_temperature"),
            "effective_top_p": decode_contract.get("effective_top_p"),
            "effective_top_k": decode_contract.get("effective_top_k"),
        },
        "capability": {
            "sampling_mode": sampling_mode,
            "greedy_pass_at_1_assessed": true_greedy and terminal_available,
            "greedy_terminal_pass_at_1": (
                terminal_success_count / len(valid_terminal_indices)
                if true_greedy and terminal_available
                else None
            ),
            "valid_terminal_rollouts": len(valid_terminal_indices),
            "terminal_success_rollouts": terminal_success_count,
            "complete_success_rollouts": sum(
                complete_labels[index] is True for index in valid_terminal_indices
            ),
            "correct_write_rollouts": sum(
                write_labels[index] is True for index in valid_terminal_indices
            ),
            "successful_required_write_label_available": (
                bool(valid_terminal_indices)
                and all(
                    successful_write_labels[index] is not None
                    for index in valid_terminal_indices
                )
            ),
            "successful_required_write_rollouts": sum(
                successful_write_labels[index] is True
                for index in valid_terminal_indices
            ),
            "successful_required_write_scope": (
                "all required write matches have unique later non-error tool "
                "results and no unexpected write is present; "
                "does not prove final state, authorization, or policy compliance"
            ),
            "grounded_terminal_success_rollouts": grounded_terminal_success_rollouts,
            "sampled_terminal_success_rate": (
                terminal_success_count / len(valid_terminal_indices)
                if sampling_mode == "stochastic" and valid_terminal_indices
                else None
            ),
            "macro_sampled_terminal_pass_at_k": (
                macro_pass_curve if sampling_mode == "stochastic" else None
            ),
            "observed_any_terminal_success_across_all_samples": (
                terminal_success_count > 0
                if sampling_mode == "stochastic" and valid_terminal_indices
                else None
            ),
            "actual_group_any_terminal_success": {
                "valid_groups": len(valid_groups),
                "groups_with_any_success": group_any_success,
                "rate": group_any_success / len(valid_groups) if valid_groups else None,
            },
            "same_group_terminal_reward_ordering": ordering,
            "per_task": per_task,
        },
        "group_signal": {
            "groups": len(sampling_groups),
            "valid_groups": len(valid_groups),
            "invalid_groups": len(sampling_groups) - len(valid_groups),
            "terminal_outcome_group_counts": {
                "all_correct": terminal_group_kinds["all_correct"],
                "all_wrong": terminal_group_kinds["all_wrong"],
                "mixed": terminal_group_kinds["mixed"],
            },
            "staged_reward": {
                "group_population_std": staged_group_stds,
                "mean_group_population_std": _mean(staged_group_stds),
                "zero_std_groups": sum(
                    math.isclose(value, 0.0, abs_tol=1e-12)
                    for value in staged_group_stds
                ),
            },
            "terminal_binary": {
                "group_population_std": terminal_group_stds,
                "mean_group_population_std": _mean(terminal_group_stds),
                "zero_std_groups": sum(
                    math.isclose(value, 0.0, abs_tol=1e-12)
                    for value in terminal_group_stds
                ),
            },
        },
        "trajectory": {
            "rollouts": len(raw_rollouts),
            "valid_transport_rollouts": len(valid_raw_indices),
            "transport_invalid_rollouts": len(transport_invalid_indices),
            "censored_rollouts": len(censored_indices),
            "model_eos_before_user_stop_valid_failures": sum(
                diagnostic.get("trajectory_transport_complete") is True
                and diagnostic.get("stop_reason") == "MODEL_EOS_BEFORE_USER_STOP"
                and terminal_labels[index] is False
                for index, diagnostic in diagnostic_by_raw_index.items()
            ),
            "stop_reason_counts": dict(
                Counter(
                    str(row.get("stop_reason") or "<missing>")
                    for row in diagnostic_by_raw_index.values()
                ).most_common()
            ),
            "completion_tokens": _numeric_summary(
                [
                    float(row["completion_tokens"])
                    for row in diagnostic_by_raw_index.values()
                    if type(row.get("completion_tokens")) in (int, float)
                ]
            ),
            "model_tokens_retained": _numeric_summary(
                [
                    float(row["model_tokens_retained"])
                    for row in diagnostic_by_raw_index.values()
                    if type(row.get("model_tokens_retained")) in (int, float)
                ]
            ),
            "observation_tokens_retained": _numeric_summary(
                [
                    float(row["observation_tokens_retained"])
                    for row in diagnostic_by_raw_index.values()
                    if type(row.get("observation_tokens_retained")) in (int, float)
                ]
            ),
            "unique_action_pattern_count": len(set(action_patterns)),
            "action_pattern_counts": dict(Counter(action_patterns).most_common()),
            "mean_tool_calls_per_rollout": _mean(
                [float(value) for value in tool_call_counts]
            ),
            "final_answer_available": all(
                _final_answer(row) is not None for row in raw_rollouts
            ),
            "group_completion_messages_not_in_environment_log": (
                trailing_group_completion_messages
            ),
            "allowed_unconsumed_terminal_model_messages": sum(
                allowed_terminal_suffixes.values()
            ),
            "allowed_unconsumed_terminal_model_message_rules": dict(
                allowed_terminal_suffixes
            ),
            "rejected_completion_suffixes": rejected_completion_suffixes,
            "reward_components_available": all(
                isinstance(row.get("reward"), dict) for row in raw_rollouts
            ),
            "terminal_evaluator_available": all(
                isinstance(row.get("terminal_evaluator"), dict)
                for row in evidence_rows
            ),
        },
        "training": {
            "applicable": False,
            "reward_std": None,
            "kl": None,
            "grad_norm": None,
            "optimizer_steps": 0,
        },
        "write_task_grpo_prescreen": {
            "applicable": write_task_prescreen_applicable,
            "status": (
                "PASSED"
                if write_task_prescreen_applicable
                and all(write_task_prescreen_criteria.values())
                else "BLOCKED"
                if write_task_prescreen_applicable
                else "NOT_APPLICABLE"
            ),
            "criteria": write_task_prescreen_criteria,
            "scope": (
                "sampling and evidence eligibility only; does not authorize "
                "training or establish policy compliance"
            ),
        },
        "release_criteria": release_criteria,
        "claim_limits": {
            "parameter_update_observed": False,
            "behavior_improvement_assessed": False,
            "business_improvement_claim_allowed": False,
            "same_group_reward_ordering_assessed": ordering["assessed"],
        },
    }


def build_training_audit(
    *,
    config: dict[str, Any],
    manifest: dict[str, Any],
    log_history: list[dict[str, Any]],
    raw_rollouts: list[dict[str, Any]],
    train_metrics: dict[str, Any],
    source: dict[str, Any],
    optimization_evidence: dict[str, Any] | None = None,
) -> dict[str, Any]:
    grpo = dict(config.get("grpo") or {})
    learning_rate = float(grpo.get("learning_rate") or 0.0)
    optimization_enabled = bool(
        manifest.get("optimization_enabled", learning_rate > 0.0)
    )
    num_generations = int(grpo.get("num_generations") or 0)
    sampling_runtime = (
        ((manifest.get("runtime") or {}).get("sampling_adapter") or {}).get(
            "decode_contract"
        )
        or {}
    )
    effective_do_sample = sampling_runtime.get("effective_do_sample")
    effective_num_generations = sampling_runtime.get("effective_num_generations")
    if effective_do_sample is False and effective_num_generations == 1:
        sampling_mode = "greedy"
        greedy_pass_at_1_assessed = True
    elif effective_do_sample is True:
        sampling_mode = "stochastic"
        greedy_pass_at_1_assessed = False
    else:
        sampling_mode = "decode_mode_not_execution_bound"
        greedy_pass_at_1_assessed = False
    groups = _group_rollouts(raw_rollouts, num_generations)
    step_logs = [row for row in log_history if "reward_std" in row]
    if not step_logs:
        raise ValueError("log_history contains no rows with reward_std")

    group_kinds: Counter[str] = Counter()
    effective_task_ids: set[str] = set()
    task_groups: dict[str, list[list[dict[str, Any]]]] = {}
    for group in groups:
        rewards = [_reward_value(row) for row in group]
        task_id = str(group[0].get("task_id"))
        task_groups.setdefault(task_id, []).append(group)
        if len(set(rewards)) > 1:
            group_kinds["mixed"] += 1
            effective_task_ids.add(task_id)
        elif all(value == 0.0 for value in rewards):
            group_kinds["all_zero"] += 1
        elif all(value > 0.0 for value in rewards):
            group_kinds["all_positive"] += 1
        else:
            group_kinds["uniform_nonzero"] += 1

    reward_stds = [float(row["reward_std"]) for row in step_logs]
    mean_lengths = [
        float(row["completions/mean_length"])
        for row in step_logs
        if "completions/mean_length" in row
    ]
    clipped_ratios = [
        float(row["completions/clipped_ratio"])
        for row in step_logs
        if "completions/clipped_ratio" in row
    ]
    entropies = [
        float(row["entropy"]) for row in step_logs if row.get("entropy") is not None
    ]
    kl_values = [
        float(row["kl"])
        for row in step_logs
        if row.get("kl") is not None and math.isfinite(float(row["kl"]))
    ]
    nonfinite_training_values = [
        {"step": row.get("step"), "key": key, "value": repr(row[key])}
        for row in step_logs
        for key in ("loss", "grad_norm", "reward", "reward_std", "kl")
        if row.get(key) is not None
        and not math.isfinite(float(row[key]))
    ]
    step_times = [
        float(row["step_time"])
        for row in step_logs
        if row.get("step_time") is not None
    ]
    zero_std_count = sum(math.isclose(value, 0.0, abs_tol=1e-12) for value in reward_stds)
    nonzero_grad_rows = [
        row
        for row in step_logs
        if not math.isclose(float(row.get("grad_norm") or 0.0), 0.0, abs_tol=1e-12)
    ]
    effective_steps = [
        int(row.get("step") or index + 1)
        for index, row in enumerate(step_logs)
        if not math.isclose(float(row["reward_std"]), 0.0, abs_tol=1e-12)
    ]
    clipped_step_count = sum(value > 0.0 for value in clipped_ratios)
    positive_rollouts = sum(_reward_value(row) > 0.0 for row in raw_rollouts)
    terminal_labels = [_terminal_success(row) for row in raw_rollouts]
    complete_labels = [_complete_success(row) for row in raw_rollouts]
    write_labels = [_correct_write(row) for row in raw_rollouts]
    terminal_success_rollouts = sum(value is True for value in terminal_labels)
    complete_success_rollouts = sum(value is True for value in complete_labels)
    correct_write_rollouts = sum(value is True for value in write_labels)
    positive_reward_terminal_failure_rollouts = sum(
        _reward_value(row) > 0.0 and terminal is False
        for row, terminal in zip(raw_rollouts, terminal_labels, strict=True)
    )
    action_patterns = [_action_pattern(row) for row in raw_rollouts]
    exact_action_signatures = [
        _exact_action_signature(row) for row in raw_rollouts
    ]
    action_pattern_counts = Counter(action_patterns)
    positive_action_pattern_counts = Counter(
        _action_pattern(row) for row in raw_rollouts if _reward_value(row) > 0.0
    )
    tool_call_counts = [len(_tool_calls(row)) for row in raw_rollouts]
    rollout_rows_with_tool_calls = sum(value > 0 for value in tool_call_counts)
    boundary_violation_rows = [
        row for row in raw_rollouts if _identity_stage_boundary_violation(row)
    ]
    per_task: dict[str, dict[str, Any]] = {}
    for task_id, grouped in sorted(task_groups.items(), key=lambda item: item[0]):
        task_rows = [row for group in grouped for row in group]
        task_rewards = [_reward_value(row) for row in task_rows]
        task_terminal = [_terminal_success(row) for row in task_rows]
        observed_task_terminal = [
            value for value in task_terminal if value is not None
        ]
        task_complete = [_complete_success(row) for row in task_rows]
        task_writes = [_correct_write(row) for row in task_rows]
        task_terminal_successes = sum(value is True for value in task_terminal)
        task_complete_successes = sum(value is True for value in task_complete)
        task_group_kinds: Counter[str] = Counter()
        group_stds: list[float] = []
        for group in grouped:
            rewards = [_reward_value(row) for row in group]
            group_stds.append(float(_population_std(rewards) or 0.0))
            if len(set(rewards)) > 1:
                task_group_kinds["mixed"] += 1
            elif all(value == 0.0 for value in rewards):
                task_group_kinds["all_zero"] += 1
            elif all(value > 0.0 for value in rewards):
                task_group_kinds["all_positive"] += 1
            else:
                task_group_kinds["uniform_nonzero"] += 1
        per_task[task_id] = {
            "rollouts": len(task_rows),
            "groups": len(grouped),
            "positive_rollouts": sum(value > 0.0 for value in task_rewards),
            "terminal_success_rollouts": task_terminal_successes,
            "complete_success_rollouts": task_complete_successes,
            "correct_write_rollouts": sum(value is True for value in task_writes),
            "sampled_terminal_pass_at_k": _sampled_pass_curve(
                len(task_rows), task_terminal_successes
            ),
            "sampled_complete_pass_at_k": _sampled_pass_curve(
                len(task_rows), task_complete_successes
            ),
            "group_size_signal_forecast": _group_size_signal_forecast(
                len(observed_task_terminal), task_terminal_successes
            ),
            "reward_mean_across_all_rollouts": _mean(task_rewards),
            "reward_std_across_all_rollouts": _population_std(task_rewards),
            "mean_in_group_reward_std": _mean(group_stds),
            "nonzero_std_group_count": sum(value > 0.0 for value in group_stds),
            "group_counts": {
                "all_zero": task_group_kinds["all_zero"],
                "mixed": task_group_kinds["mixed"],
                "all_positive": task_group_kinds["all_positive"],
                "uniform_nonzero": task_group_kinds["uniform_nonzero"],
            },
            "tool_use_by_terminal_outcome": _tool_use_by_terminal_outcome(
                task_rows
            ),
            "terminal_failure_behavior": _failure_behavior_summary(task_rows),
        }

    warnings: list[dict[str, str]] = []
    zero_std_fraction = zero_std_count / len(step_logs)
    if zero_std_fraction >= 0.5:
        warnings.append(
            {
                "code": "SPARSE_RELATIVE_ADVANTAGE",
                "evidence": f"zero reward std on {zero_std_count}/{len(step_logs)} steps",
            }
        )
    mean_clipped_ratio = _mean(clipped_ratios)
    if mean_clipped_ratio is not None and mean_clipped_ratio >= 0.1:
        warnings.append(
            {
                "code": "HIGH_COMPLETION_CLIPPING",
                "evidence": f"mean clipped ratio is {mean_clipped_ratio:.6f}",
            }
        )
    if optimization_enabled and float(grpo.get("beta") or 0.0) == 0.0:
        warnings.append(
            {
                "code": "NO_KL_CONSTRAINT",
                "evidence": "GRPO beta is 0; no KL regularization curve is available",
            }
        )
    if (
        optimization_enabled
        and float(grpo.get("beta") or 0.0) > 0.0
        and not kl_values
    ):
        warnings.append(
            {
                "code": "KL_CONFIGURED_BUT_CURVE_MISSING",
                "evidence": "GRPO beta is positive but no finite KL value is in log_history",
            }
        )
    if nonfinite_training_values:
        warnings.append(
            {
                "code": "NONFINITE_TRAINING_SIGNAL",
                "evidence": f"{len(nonfinite_training_values)} monitored log values are nonfinite",
            }
        )
    if positive_reward_terminal_failure_rollouts:
        warnings.append(
            {
                "code": "PARTIAL_REWARD_WITHOUT_TERMINAL_SUCCESS",
                "evidence": (
                    f"{positive_reward_terminal_failure_rollouts}/"
                    f"{len(raw_rollouts)} rollouts have positive training reward "
                    "but fail the terminal evaluator"
                ),
            }
        )

    ordering = _same_group_terminal_ordering(groups)

    return {
        "schema_version": SCHEMA_VERSION,
        "scope": "TRAINING_SIGNAL_AUDIT_ONLY",
        "source": source,
        "bindings": {
            "run_status": manifest.get("status"),
            "project_commit": (manifest.get("git") or {}).get("commit"),
            "starting_model_sha256": (manifest.get("bindings") or {}).get(
                "starting_model_sha256"
            ),
            "raw_rollouts_sha256": (
                (manifest.get("artifacts") or {}).get("raw_rollouts") or {}
            ).get("sha256"),
        },
        "configuration": {
            "task_ids": list((config.get("data") or {}).get("task_ids") or []),
            "max_steps": grpo.get("max_steps"),
            "num_generations": num_generations,
            "max_completion_length": grpo.get("max_completion_length"),
            "per_device_train_batch_size": grpo.get("per_device_train_batch_size"),
            "gradient_accumulation_steps": grpo.get("gradient_accumulation_steps"),
            "learning_rate": grpo.get("learning_rate"),
            "temperature": grpo.get("temperature"),
            "sampling_contract": dict(config.get("sampling") or {}),
            "beta": grpo.get("beta"),
            "loss_type": grpo.get("loss_type"),
        },
        "training_signal": {
            "training_steps": len(step_logs),
            "rollouts": len(raw_rollouts),
            "groups": len(groups),
            "positive_rollouts": positive_rollouts,
            "grouping_method": "sequential_rows_by_num_generations",
            "group_counts": {
                "all_zero": group_kinds["all_zero"],
                "mixed": group_kinds["mixed"],
                "all_positive": group_kinds["all_positive"],
                "uniform_nonzero": group_kinds["uniform_nonzero"],
            },
            "effective_task_ids": sorted(effective_task_ids),
            "zero_reward_std_steps": zero_std_count,
            "zero_reward_std_fraction": zero_std_fraction,
            "effective_steps": effective_steps,
            "nonzero_grad_steps": len(nonzero_grad_rows),
            "grad_norm_range": _range(
                [float(row["grad_norm"]) for row in nonzero_grad_rows]
            ),
            "per_task": per_task,
        },
        "capability": {
            "sampling_mode": sampling_mode,
            "greedy_pass_at_1_assessed": greedy_pass_at_1_assessed,
            "effective_do_sample": effective_do_sample,
            "effective_num_generations": effective_num_generations,
            "terminal_success_label_available": all(
                value is not None for value in terminal_labels
            ),
            "complete_success_label_available": all(
                value is not None for value in complete_labels
            ),
            "correct_write_label_available": all(
                value is not None for value in write_labels
            ),
            "terminal_success_rollouts": terminal_success_rollouts,
            "complete_success_rollouts": complete_success_rollouts,
            "correct_write_rollouts": correct_write_rollouts,
            "positive_reward_terminal_failure_rollouts": (
                positive_reward_terminal_failure_rollouts
            ),
            "same_group_terminal_reward_ordering": ordering,
        },
        "behavior": {
            "action_pattern_counts": dict(action_pattern_counts.most_common()),
            "positive_reward_action_pattern_counts": dict(
                positive_action_pattern_counts.most_common()
            ),
            "unique_action_pattern_count": len(action_pattern_counts),
            "action_pattern_entropy_nats": _categorical_entropy(
                action_pattern_counts
            ),
            "action_pattern_duplicate_ratio": _duplicate_ratio(action_patterns),
            "exact_action_signature_duplicate_ratio": _duplicate_ratio(
                exact_action_signatures
            ),
            "mean_tool_calls_per_rollout": _mean(
                [float(value) for value in tool_call_counts]
            ),
            "mean_tool_calls_per_rollout_is_quality_metric": False,
            "tool_call_count_range": _range(
                [float(value) for value in tool_call_counts]
            ),
            "tool_use_by_terminal_outcome": _tool_use_by_terminal_outcome(
                raw_rollouts
            ),
            "terminal_failure_behavior": _failure_behavior_summary(raw_rollouts),
            "no_tool_rollout_count": sum(value == 0 for value in tool_call_counts),
            "identity_stage_boundary_violation_count": len(
                boundary_violation_rows
            ),
            "identity_stage_boundary_violation_fraction": (
                len(boundary_violation_rows) / len(raw_rollouts)
                if raw_rollouts
                else None
            ),
            "identity_stage_boundary_violation_task_ids": sorted(
                {str(row.get("task_id")) for row in boundary_violation_rows}
            ),
            "semantic_trajectory_diversity_assessed": False,
        },
        "generation": {
            "mean_completion_length": _mean(mean_lengths),
            "first_step_mean_length": mean_lengths[0] if mean_lengths else None,
            "last_step_mean_length": mean_lengths[-1] if mean_lengths else None,
            "mean_length_range": _range(mean_lengths),
            "steps_with_clipping": clipped_step_count,
            "mean_clipped_ratio": mean_clipped_ratio,
            "max_clipped_ratio": max(clipped_ratios) if clipped_ratios else None,
        },
        "trajectory_artifact_coverage": {
            "prompt_in_raw_rollout": all("prompt" in row for row in raw_rollouts),
            "environment_messages_in_raw_rollout": all(
                isinstance(row.get("messages"), list) for row in raw_rollouts
            ),
            "tool_calls_in_raw_rollout": rollout_rows_with_tool_calls
            == len(raw_rollouts),
            "rollout_rows_with_tool_calls": rollout_rows_with_tool_calls,
            "reward_components_in_raw_rollout": all(
                isinstance(row.get("reward"), dict) for row in raw_rollouts
            ),
            "final_answer_in_raw_rollout": all(
                row.get("final_answer") is not None for row in raw_rollouts
            ),
            "completion_parquet_retained": bool(
                (((manifest.get("artifacts") or {}).get("completion_logs") or {}).get("files") or 0)
            ),
        },
        "optimization": {
            "optimization_enabled": optimization_enabled,
            "train_runtime_seconds": train_metrics.get("train_runtime"),
            "train_loss": train_metrics.get("train_loss"),
            "mean_step_time_seconds": _mean(step_times),
            "rollout_vs_update_time_breakdown_available": False,
            "entropy_first": entropies[0] if entropies else None,
            "entropy_last": entropies[-1] if entropies else None,
            "entropy_range": _range(entropies),
            "kl_curve_available": bool(kl_values),
            "kl_values": kl_values,
            "monitored_log_values_finite": not nonfinite_training_values,
            "nonfinite_training_values": nonfinite_training_values,
            "optimization_evidence_status": (
                optimization_evidence or {}
            ).get("status"),
        },
        "warnings": warnings,
        "claim_limits": {
            "parameter_update_observed": bool(
                optimization_enabled
                and optimization_evidence
                and optimization_evidence.get("status") == "PASSED"
                and optimization_evidence.get(
                    "trainable_parameter_change_detected"
                )
                and (
                    optimization_evidence.get("final_trainable_parameters") or {}
                ).get("all_finite")
            ),
            "parameter_update_inferred_from_gradients": False,
            "behavior_improvement_assessed": False,
            "causal_failure_diagnosis_proven": False,
            "action_pattern_diversity_is_not_trajectory_quality": True,
            "stochastic_pass_at_1_is_not_greedy_pass_at_1": True,
            "reward_alignment_requires_terminal_success_pairs": not ordering[
                "assessed"
            ],
        },
    }


def _audit_pure_sampling_run(
    run_dir: Path,
    config_path: Path,
    config: dict[str, Any],
    manifest: dict[str, Any],
) -> dict[str, Any]:
    required_artifacts = (
        "raw_rollouts.jsonl",
        "rollout_evidence.jsonl",
        "generation_events.jsonl",
        "sampling_groups.jsonl",
        "effective_config.json",
        "command.json",
        "user_simulator_preflight.json",
    )
    artifact_paths = {name: run_dir / name for name in required_artifacts}
    run_state_path = run_dir / "run_state.json"
    missing = [
        str(path)
        for path in [config_path, *artifact_paths.values(), run_state_path]
        if not path.is_file()
    ]
    if missing:
        raise FileNotFoundError(f"Missing pure-sampling audit inputs: {missing}")

    actual_config_sha = _sha256(config_path)
    if str(manifest.get("config_sha256") or "").upper() != actual_config_sha:
        raise ValueError("Config SHA-256 does not match pure-sampling manifest")
    artifacts = manifest.get("artifacts") or {}
    for name, path in artifact_paths.items():
        expected_sha = str(((artifacts.get(name) or {}).get("sha256") or "")).upper()
        if not expected_sha:
            raise ValueError(f"Pure-sampling artifact is not hash-bound: {name}")
        if _sha256(path) != expected_sha:
            raise ValueError(f"Pure-sampling artifact SHA-256 mismatch: {name}")

    run_state = _read_json(run_state_path)
    if not isinstance(run_state, dict):
        raise TypeError("run_state.json must be a JSON object")
    if (
        run_state.get("status") != "COMPLETED"
        or run_state.get("run_manifest_sha256") != _sha256(run_dir / "run_manifest.json")
    ):
        raise ValueError("run_state.json does not bind the completed manifest")
    effective_config = _read_json(artifact_paths["effective_config.json"])
    if not isinstance(effective_config, dict):
        raise TypeError("effective_config.json must be a JSON object")
    _validate_effective_config_matches_frozen(config, effective_config)
    if (effective_config.get("data") or {}).get("task_ids") != (
        config.get("data") or {}
    ).get("task_ids"):
        raise ValueError("Effective task IDs drifted from the frozen config")
    if (effective_config.get("model") or {}).get("expected_sha256") != (
        config.get("model") or {}
    ).get("expected_sha256"):
        raise ValueError("Effective starting model drifted from the frozen config")
    user_simulator = (manifest.get("runtime") or {}).get("user_simulator") or {}
    if (
        not str(user_simulator.get("model") or "").strip()
        or not str(user_simulator.get("llm_args_sha256") or "").strip()
        or user_simulator.get("preflight_status") != "PASSED"
        or user_simulator.get("external_api_called") is not True
    ):
        raise ValueError("Pure-sampling user simulator binding is incomplete")

    return build_sampling_audit(
        config=effective_config,
        manifest=manifest,
        raw_rollouts=_read_jsonl(artifact_paths["raw_rollouts.jsonl"]),
        evidence_rows=_read_jsonl(artifact_paths["rollout_evidence.jsonl"]),
        sampling_groups=_read_jsonl(artifact_paths["sampling_groups.jsonl"]),
        source={
            "run_dir": str(run_dir),
            "config_path": str(config_path),
            "config_sha256": actual_config_sha,
            "manifest_path": str(run_dir / "run_manifest.json"),
            "manifest_sha256": _sha256(run_dir / "run_manifest.json"),
            "run_state_path": str(run_state_path),
            "run_state_sha256": _sha256(run_state_path),
            "artifacts": {
                name: {"path": str(path), "sha256": _sha256(path)}
                for name, path in artifact_paths.items()
            },
        },
    )


def audit_run(run_dir: Path, config_path: Path) -> dict[str, Any]:
    run_dir = run_dir.resolve()
    config_path = config_path.resolve()
    manifest_path = run_dir / "run_manifest.json"
    initial_required = [config_path, manifest_path]
    missing = [str(path) for path in initial_required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing audit inputs: {missing}")
    config = _read_json(config_path)
    manifest = _read_json(manifest_path)
    if not all(isinstance(value, dict) for value in [config, manifest]):
        raise TypeError("Config and manifest must be JSON objects")
    if manifest.get("execution_mode") == "PURE_SAMPLING":
        return _audit_pure_sampling_run(
            run_dir, config_path, config, manifest
        )

    log_history_path = run_dir / "log_history.json"
    raw_rollouts_path = run_dir / "raw_rollouts.jsonl"
    train_metrics_path = run_dir / "train_metrics.json"
    required = [
        config_path,
        manifest_path,
        log_history_path,
        raw_rollouts_path,
        train_metrics_path,
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing audit inputs: {missing}")

    log_history = _read_json(log_history_path)
    train_metrics = _read_json(train_metrics_path)
    if not all(isinstance(value, dict) for value in [config, manifest, train_metrics]):
        raise TypeError("Config, manifest, and train metrics must be JSON objects")
    if not isinstance(log_history, list):
        raise TypeError("log_history must be a JSON array")

    optimization_enabled = bool(
        manifest.get(
            "optimization_enabled",
            float(((config.get("grpo") or {}).get("learning_rate") or 0.0))
            > 0.0,
        )
    )

    actual_config_sha = _sha256(config_path)
    expected_config_sha = str(
        ((manifest.get("bindings") or {}).get("config_sha256") or "")
    ).upper()
    if expected_config_sha and actual_config_sha != expected_config_sha:
        raise ValueError("Config SHA-256 does not match run_manifest.json")
    actual_rollout_sha = _sha256(raw_rollouts_path)
    expected_rollout_sha = str(
        (((manifest.get("artifacts") or {}).get("raw_rollouts") or {}).get("sha256") or "")
    ).upper()
    if expected_rollout_sha and actual_rollout_sha != expected_rollout_sha:
        raise ValueError("Raw rollout SHA-256 does not match run_manifest.json")

    for artifact_name, artifact_path, display_name in [
        ("log_history", log_history_path, "Log history"),
        ("train_metrics", train_metrics_path, "Train metrics"),
    ]:
        expected_sha = str(
            (
                ((manifest.get("artifacts") or {}).get(artifact_name) or {}).get(
                    "sha256"
                )
                or ""
            )
        ).upper()
        if optimization_enabled and not expected_sha:
            raise ValueError(
                f"{display_name} SHA-256 is not bound in run_manifest.json"
            )
        if expected_sha and _sha256(artifact_path) != expected_sha:
            raise ValueError(
                f"{display_name} SHA-256 does not match run_manifest.json"
            )

    optimization_evidence = None
    optimization_evidence_path = run_dir / "optimization_evidence.json"
    expected_evidence_sha = str(
        (
            (manifest.get("artifacts") or {}).get("optimization_evidence")
            or {}
        ).get("sha256")
        or ""
    ).upper()
    if optimization_enabled and not optimization_evidence_path.is_file():
        raise FileNotFoundError(
            "Optimization-enabled run is missing optimization_evidence.json"
        )
    if optimization_enabled and not expected_evidence_sha:
        raise ValueError(
            "Optimization evidence SHA-256 is not bound in run_manifest.json"
        )
    if optimization_evidence_path.is_file():
        optimization_evidence = _read_json(optimization_evidence_path)
        if not isinstance(optimization_evidence, dict):
            raise TypeError("optimization_evidence must be a JSON object")
        actual_evidence_sha = _sha256(optimization_evidence_path)
        if expected_evidence_sha and actual_evidence_sha != expected_evidence_sha:
            raise ValueError(
                "Optimization evidence SHA-256 does not match run_manifest.json"
            )

    return build_training_audit(
        config=config,
        manifest=manifest,
        log_history=log_history,
        raw_rollouts=_read_jsonl(raw_rollouts_path),
        train_metrics=train_metrics,
        optimization_evidence=optimization_evidence,
        source={
            "run_dir": str(run_dir),
            "config_path": str(config_path),
            "config_sha256": actual_config_sha,
            "manifest_path": str(manifest_path),
            "manifest_sha256": _sha256(manifest_path),
            "log_history_path": str(log_history_path),
            "log_history_sha256": _sha256(log_history_path),
            "raw_rollouts_path": str(raw_rollouts_path),
            "raw_rollouts_sha256": actual_rollout_sha,
            "train_metrics_path": str(train_metrics_path),
            "train_metrics_sha256": _sha256(train_metrics_path),
            "optimization_evidence_path": (
                str(optimization_evidence_path)
                if optimization_evidence_path.is_file()
                else None
            ),
            "optimization_evidence_sha256": (
                _sha256(optimization_evidence_path)
                if optimization_evidence_path.is_file()
                else None
            ),
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit a completed Agentic GRPO run")
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    report = audit_run(args.run_dir, args.config)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
