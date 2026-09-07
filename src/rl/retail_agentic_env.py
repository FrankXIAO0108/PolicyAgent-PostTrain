from __future__ import annotations

import hashlib
import json
import os
import re
import sys
import time
from collections import Counter
from copy import deepcopy
from pathlib import Path
from typing import Any, Callable

from src.guards.retail_pre_action import (
    WRITE_TOOLS,
    ToolProposal,
    context_from_messages,
    evaluate_retail_actions,
)
from src.rl.user_simulator_fail_fast import generate_with_fail_fast


REWARD_CONFIG_ENV = "POLICYAGENT_REWARD_CONFIG_JSON"
ROLLOUT_LOG_ENV = "POLICYAGENT_ROLLOUT_LOG"
ROLLOUT_EVIDENCE_LOG_ENV = "POLICYAGENT_ROLLOUT_EVIDENCE_LOG"
ROLLOUT_STAGE_ENV = "POLICYAGENT_ROLLOUT_STAGE"
REQUIRE_TRANSPORT_COMPLETE_ENV = "POLICYAGENT_REQUIRE_TRANSPORT_COMPLETE_GROUPS"
TOOL_ITERATION_LIMIT_AS_TERMINAL_FAILURE_ENV = (
    "POLICYAGENT_TOOL_ITERATION_LIMIT_AS_TERMINAL_FAILURE"
)
COMPLETION_BUDGET_AS_TERMINAL_FAILURE_ENV = (
    "POLICYAGENT_COMPLETION_BUDGET_AS_TERMINAL_FAILURE"
)
FULL_TASK_STAGE = "FULL_TASK"
IDENTITY_AUTHENTICATION_STAGE = "IDENTITY_AUTHENTICATION"
TIERED_TERMINAL_PROCESS_MODE = "tiered_terminal_process_v2"
SUPPORTED_ROLLOUT_STAGES = {
    FULL_TASK_STAGE,
    IDENTITY_AUTHENTICATION_STAGE,
}
IDENTITY_AUTHENTICATION_ACTIONS = {
    "find_user_id_by_email",
    "find_user_id_by_name_zip",
}
DEFAULT_REWARD_CONFIG: dict[str, Any] = {
    "process_reward_mode": "one_to_one_required_action_progress",
    "environment_state_action_progress_gate": "multiply",
    "environment_state_weight": 0.70,
    "required_action_weight": 0.20,
    "communication_weight": 0.10,
    "tool_error_penalty_each": 0.05,
    "tool_error_penalty_cap": 0.20,
    "repeated_call_penalty_each": 0.03,
    "repeated_call_penalty_cap": 0.15,
    "unexpected_write_penalty_each": 0.05,
    "unexpected_write_penalty_cap": 0.20,
    "unfinished_interaction_penalty": 0.10,
    "llm_judge_used": False,
    "policy_guard_used_as_reward": False,
    "confirmation_signal_used_as_reward": False,
}
TERMINAL_ONLY_REWARD_CONFIG: dict[str, Any] = {
    "process_reward_mode": "terminal_environment_state",
    "environment_state_action_progress_gate": "none",
    "environment_state_weight": 1.0,
    "required_action_weight": 0.0,
    "communication_weight": 0.0,
    "tool_error_penalty_each": 0.0,
    "tool_error_penalty_cap": 0.0,
    "repeated_call_penalty_each": 0.0,
    "repeated_call_penalty_cap": 0.0,
    "unexpected_write_penalty_each": 0.0,
    "unexpected_write_penalty_cap": 0.0,
    "unfinished_interaction_penalty": 0.0,
    "llm_judge_used": False,
    "policy_guard_used_as_reward": False,
    "confirmation_signal_used_as_reward": False,
}

GUARDED_STOP_FLAGS = {
    "TOOL_RESULT_BUDGET_EXCEEDED",
    "CONTEXT_LIMIT",
    "COMPLETION_BUDGET_EXHAUSTED",
    "CUSTOMER_TURN_LIMIT",
    "TOOL_CALL_LIMIT",
    "PARSE_OR_PROTOCOL_ERROR",
    "TOOL_ITERATION_LIMIT",
    "UNRESOLVED_TOOL_CALL",
    "USER_STOP_AND_MODEL_EOS",
    "MODEL_EOS_BEFORE_USER_STOP",
    "MODEL_END_WITHOUT_EOS",
}
GUARDED_PRIMARY_STOP_REASONS = GUARDED_STOP_FLAGS - {
    "CUSTOMER_TURN_LIMIT",
    "TOOL_CALL_LIMIT",
    "PARSE_OR_PROTOCOL_ERROR",
}
GUARDED_COMPLETION_TELEMETRY_KEYS = {
    "stop_reason",
    "stop_reason_source",
    "stop_flags",
    "model_ended",
    "model_eos_observed",
    "completion_token_budget_exhausted",
    "context_limit_reached",
    "tool_iteration_limit_reached",
    "unresolved_tool_call",
    "framework_loop_abnormal_end",
    "prompt_tokens",
    "completion_tokens",
    "model_tokens_retained",
    "observation_tokens_retained",
    "model_completion_truncated",
    "model_completion_truncation_source",
}
TRANSPORT_INVALID_COMPLETION_FIELDS = (
    "completion_token_budget_exhausted",
    "context_limit_reached",
    "tool_iteration_limit_reached",
    "unresolved_tool_call",
    "framework_loop_abnormal_end",
    "model_completion_truncated",
)


def transport_invalid_reasons(payload: dict[str, Any] | None) -> list[str]:
    """Return transport failures that make a rollout unsafe for optimization."""

    if payload is None:
        return ["missing_completion_telemetry"]
    return [field for field in TRANSPORT_INVALID_COMPLETION_FIELDS if payload[field]]


def _canonical_sha256(payload: Any) -> str:
    serialized = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")
    return hashlib.sha256(serialized).hexdigest().upper()


def _environment_state(environment: Any) -> dict[str, Any]:
    state: dict[str, Any] = {}
    if environment.tools is not None and environment.tools.db is not None:
        state["agent"] = environment.tools.db.model_dump(mode="json")
    if environment.user_tools is not None and environment.user_tools.db is not None:
        state["user"] = environment.user_tools.db.model_dump(mode="json")
    return state


def _state_diff(before: Any, after: Any, path: str = "") -> list[dict[str, Any]]:
    if isinstance(before, dict) and isinstance(after, dict):
        rows: list[dict[str, Any]] = []
        for key in sorted(set(before) | set(after)):
            child = f"{path}.{key}" if path else str(key)
            if key not in before:
                rows.append({"path": child, "before": None, "after": after[key]})
            elif key not in after:
                rows.append({"path": child, "before": before[key], "after": None})
            else:
                rows.extend(_state_diff(before[key], after[key], child))
        return rows
    if isinstance(before, list) and isinstance(after, list):
        return (
            []
            if before == after
            else [{"path": path, "before": before, "after": after}]
        )
    return [] if before == after else [{"path": path, "before": before, "after": after}]


def _tool_trace(messages: list[Any]) -> list[dict[str, Any]]:
    results: dict[str, Any] = {}
    for message in messages:
        message_id = getattr(message, "id", None)
        if getattr(message, "role", None) == "tool" and message_id:
            results[str(message_id)] = message

    trace: list[dict[str, Any]] = []
    for message_index, message in enumerate(messages):
        for call in list(getattr(message, "tool_calls", None) or []):
            result = results.get(str(call.id))
            trace.append(
                {
                    "message_index": message_index,
                    "call_id": str(call.id),
                    "name": str(call.name),
                    "arguments": deepcopy(call.arguments),
                    "result": (
                        {
                            "content": getattr(result, "content", None),
                            "error": bool(getattr(result, "error", False)),
                        }
                        if result is not None
                        else None
                    ),
                }
            )
    return trace


def _message_payload(message: Any) -> dict[str, Any]:
    if hasattr(message, "model_dump"):
        return dict(message.model_dump(mode="json"))
    return {
        "role": getattr(message, "role", None),
        "content": getattr(message, "content", None),
        "id": getattr(message, "id", None),
        "error": bool(getattr(message, "error", False)),
        "tool_calls": [
            {
                "id": str(call.id),
                "name": str(call.name),
                "arguments": deepcopy(call.arguments),
            }
            for call in list(getattr(message, "tool_calls", None) or [])
        ],
    }


def load_reward_config() -> dict[str, Any]:
    """Load the frozen reward specification supplied by the training runner."""

    raw = os.environ.get(REWARD_CONFIG_ENV)
    configured = json.loads(raw) if raw else {}
    if not isinstance(configured, dict):
        raise RuntimeError(f"{REWARD_CONFIG_ENV} must contain a JSON object")
    allowed_keys = set(DEFAULT_REWARD_CONFIG)
    if configured.get("process_reward_mode") == TIERED_TERMINAL_PROCESS_MODE:
        allowed_keys.add("staged_reward_spec")
    unknown = sorted(set(configured) - allowed_keys)
    if unknown:
        raise RuntimeError(f"Unknown Agentic RL reward keys: {unknown}")
    reward = {**DEFAULT_REWARD_CONFIG, **configured}
    if reward["process_reward_mode"] not in {
        "one_to_one_required_action_progress",
        "terminal_environment_state",
        TIERED_TERMINAL_PROCESS_MODE,
    }:
        raise RuntimeError("Unsupported Agentic RL process_reward_mode")
    expected_gate = (
        "none"
        if reward["process_reward_mode"]
        in {"terminal_environment_state", TIERED_TERMINAL_PROCESS_MODE}
        else "multiply"
    )
    if reward["environment_state_action_progress_gate"] != expected_gate:
        raise RuntimeError("Unsupported environment-state action-progress gate")
    for key in (
        "environment_state_weight",
        "required_action_weight",
        "communication_weight",
        "tool_error_penalty_each",
        "tool_error_penalty_cap",
        "repeated_call_penalty_each",
        "repeated_call_penalty_cap",
        "unexpected_write_penalty_each",
        "unexpected_write_penalty_cap",
        "unfinished_interaction_penalty",
    ):
        reward[key] = float(reward[key])
        if reward[key] < 0:
            raise RuntimeError(f"Reward field {key} must be non-negative")
    if (
        reward["process_reward_mode"] != TIERED_TERMINAL_PROCESS_MODE
        and sum(
            reward[key]
            for key in (
                "environment_state_weight",
                "required_action_weight",
                "communication_weight",
            )
        )
        <= 0
    ):
        raise RuntimeError("At least one positive reward component weight is required")
    if reward["process_reward_mode"] == TIERED_TERMINAL_PROCESS_MODE:
        _validate_staged_reward_spec(reward.get("staged_reward_spec"))
    for key in ("llm_judge_used", "policy_guard_used_as_reward"):
        if reward[key] is not False:
            raise RuntimeError(f"Agentic RL v1 requires {key}=false")
    expected_confirmation_reward = (
        reward["process_reward_mode"] == TIERED_TERMINAL_PROCESS_MODE
        and _staged_reward_uses_confirmation(reward.get("staged_reward_spec"))
    )
    if reward["confirmation_signal_used_as_reward"] is not expected_confirmation_reward:
        raise RuntimeError(
            "confirmation_signal_used_as_reward must match the staged reward "
            "composition mode"
        )
    return reward


def _validate_staged_reward_spec(spec: Any) -> None:
    if not isinstance(spec, dict):
        raise RuntimeError("tiered_terminal_process_v2 requires staged_reward_spec")
    if "semantic_assistance" in spec:
        from src.evaluation.task44_hybrid_reward import validate_settings

        validate_settings(spec["semantic_assistance"])
        if (
            set(spec.get("tasks", {})) != {"44"}
            or spec.get("evidence_rules_version") != "task44_evidence_v2"
            or spec.get("reward", {}).get("composition_mode")
            != "hierarchical_state_authorization_review_v6"
        ):
            raise RuntimeError("Semantic assistance is scoped to Task44 evidence-v2/v6")
    if not isinstance(spec.get("reward"), dict) or not isinstance(
        spec.get("tasks"), dict
    ):
        raise RuntimeError("staged_reward_spec requires reward and tasks objects")
    composition_mode = str(spec["reward"].get("composition_mode") or "")
    required_reward_fields = {
        "terminal_incomplete_communication_cap",
        "nonterminal_component_weights",
        "normalize_active_nonterminal_weights_to",
        "unexpected_write_hard_cap",
    }
    if composition_mode in {
        "hierarchical_state_authorization_v5",
        "hierarchical_state_authorization_review_v6",
        "hierarchical_state_authorization_claim_v7",
    }:
        required_reward_fields.update(
            {
                "additive_component_weights",
                "no_verified_write_cap",
            }
        )
        if composition_mode == "hierarchical_state_authorization_v5":
            required_reward_fields.add("unauthorized_write_hard_cap")
        else:
            required_reward_fields.update(
                {
                    "authorization_review_value",
                    "authorization_review_cap",
                    "authorization_fail_hard_cap",
                }
            )
            if composition_mode == "hierarchical_state_authorization_claim_v7":
                required_reward_fields.update(
                    {
                        "claim_evidence_fail_cap",
                        "claim_evidence_review_cap",
                    }
                )
    elif composition_mode in {
        "additive_terminal_process_v3",
        "additive_terminal_process_confirmation_v4",
    }:
        required_reward_fields.update(
            {"additive_component_weights", "no_correct_write_cap"}
        )
    else:
        required_reward_fields.update(
            {
                "no_correct_write_cap",
                "terminal_complete_success_score",
                "complete_success_efficiency_penalty_cap",
            }
        )
    missing = sorted(required_reward_fields - set(spec["reward"]))
    if missing:
        raise RuntimeError(f"staged_reward_spec reward fields missing: {missing}")
    if composition_mode == "hierarchical_state_authorization_claim_v7":
        missing_claim_rules = sorted(
            str(task_id)
            for task_id, task_spec in spec["tasks"].items()
            if not isinstance(task_spec, dict)
            or not list(task_spec.get("claim_evidence_rules") or [])
        )
        if missing_claim_rules:
            raise RuntimeError(
                "hierarchical_state_authorization_claim_v7 requires "
                f"claim_evidence_rules for tasks: {missing_claim_rules}"
            )


def _staged_reward_uses_confirmation(spec: Any) -> bool:
    return bool(
        isinstance(spec, dict)
        and isinstance(spec.get("reward"), dict)
        and spec["reward"].get("composition_mode")
        in {
            "additive_terminal_process_confirmation_v4",
            "hierarchical_state_authorization_v5",
            "hierarchical_state_authorization_review_v6",
            "hierarchical_state_authorization_claim_v7",
        }
    )


def is_tiered_reward_config(reward: Any) -> bool:
    if not isinstance(reward, dict):
        return False
    if reward.get("process_reward_mode") != TIERED_TERMINAL_PROCESS_MODE:
        return False
    if set(reward) - (set(DEFAULT_REWARD_CONFIG) | {"staged_reward_spec"}):
        return False
    try:
        configured = {**DEFAULT_REWARD_CONFIG, **reward}
        if configured["environment_state_action_progress_gate"] != "none":
            return False
        if any(
            configured[key] is not False
            for key in ("llm_judge_used", "policy_guard_used_as_reward")
        ):
            return False
        if configured["confirmation_signal_used_as_reward"] is not (
            _staged_reward_uses_confirmation(configured.get("staged_reward_spec"))
        ):
            return False
        _validate_staged_reward_spec(configured.get("staged_reward_spec"))
    except RuntimeError:
        return False
    return True


def terminal_environment_reward(
    environment_state_reward: float, *, user_stopped: bool
) -> float:
    """Return the binary terminal outcome for the controlled GRPO arm.

    The signal intentionally ignores reference actions, tool-path quality,
    policy findings, and communication checks.  It is valid only for tasks
    whose business outcome is represented by the final environment state.
    """

    return 1.0 if user_stopped and float(environment_state_reward) == 1.0 else 0.0


def tiered_terminal_process_reward(
    *,
    task_id: str,
    messages: list[Any],
    action_progress: dict[str, Any],
    environment_payload: dict[str, Any],
    communication_payload: dict[str, Any],
    environment_state_reward: float,
    user_stopped: bool,
    completion: dict[str, Any],
    staged_reward_spec: dict[str, Any],
    skip_semantics: bool = False,
) -> dict[str, Any]:
    """Score one online rollout with the frozen v2 offline formula."""

    from src.evaluation.staged_reward_shadow import score_rollout

    task_id = str(task_id)
    if task_id not in staged_reward_spec["tasks"]:
        raise RuntimeError(f"No staged reward specification for task {task_id}")
    terminal_value = terminal_environment_reward(
        environment_state_reward,
        user_stopped=user_stopped,
    )
    evidence = {
        "task_id": task_id,
        "tool_trace": _tool_trace(messages),
        "terminal_evaluator": {
            "reward": terminal_value,
            "user_stopped": user_stopped,
            "action_progress": deepcopy(action_progress),
            "tau2": {
                "environment": deepcopy(environment_payload),
                "communication": deepcopy(communication_payload),
            },
        },
        "completion": deepcopy(completion),
    }
    evidence_sha256 = _canonical_sha256(evidence)
    evidence["evidence_sha256"] = evidence_sha256
    raw = {
        "task_id": task_id,
        "messages": [_message_payload(message) for message in messages],
        "evidence_sha256": evidence_sha256,
    }
    confirmation = (
        confirmation_diagnostics(messages)
        if _staged_reward_uses_confirmation(staged_reward_spec)
        else None
    )
    score = score_rollout(
        raw,
        evidence,
        staged_reward_spec,
        confirmation_diagnostic=confirmation,
    )
    score["terminal_environment_reward"] = terminal_value
    if "semantic_assistance" in staged_reward_spec and not skip_semantics:
        from src.evaluation.task44_hybrid_reward import hybrid_score

        policy_path = Path(os.environ["POLICYAGENT_TAU2_ROOT"]) / "data/tau2/domains/retail/policy.md"
        result = hybrid_score(
            score, raw, staged_reward_spec,
            policy=policy_path.read_text(encoding="utf-8"),
            directory=Path(os.environ[ROLLOUT_LOG_ENV]).resolve().parent,
        )
        # Preserve rule-only diagnostics; never relabel them as the new score.
        score["rule_only_reward"] = score["staged_reward"]
        score["staged_reward"] = result["offline_reward"]
        score["semantic_assistance"] = result
    return score


def _normalized_action_arguments(arguments: dict[str, Any]) -> dict[str, Any]:
    """Normalize only Retail action fields whose list order is not semantic.

    ``item_ids`` is a set for return/cancel-style actions.  For modification or
    exchange, ``item_ids`` and ``new_item_ids`` are positional pairs, so the
    pair order may change while the old-to-new mapping must remain intact.
    Other list-valued arguments are intentionally left untouched.
    """

    normalized = deepcopy(arguments)
    item_ids = normalized.get("item_ids")
    new_item_ids = normalized.get("new_item_ids")
    if isinstance(item_ids, list) and isinstance(new_item_ids, list):
        if len(item_ids) == len(new_item_ids):
            normalized.pop("item_ids")
            normalized.pop("new_item_ids")
            normalized["item_pairs"] = sorted(
                (
                    [str(item_id), str(new_item_id)]
                    for item_id, new_item_id in zip(item_ids, new_item_ids, strict=True)
                ),
                key=lambda pair: (pair[0], pair[1]),
            )
    elif isinstance(item_ids, list):
        normalized["item_ids"] = sorted(str(item_id) for item_id in item_ids)
    return normalized


def _tool_signature(name: str, arguments: dict[str, Any]) -> str:
    normalized = _normalized_action_arguments(arguments)
    return f"{name}:{json.dumps(normalized, ensure_ascii=False, sort_keys=True)}"


def _action_matches(action: Any, call: Any) -> bool:
    if action.compare_with_tool_call(call):
        return True
    return action.name == call.name and _normalized_action_arguments(
        dict(action.arguments)
    ) == _normalized_action_arguments(dict(call.arguments))


def gate_environment_state_reward(
    environment_reward: float, action_recall: float | None
) -> tuple[float, float]:
    """Prevent a satisfied initial DB state from rewarding a no-op trajectory."""

    gate = 1.0 if action_recall is None else action_recall
    return environment_reward * gate, gate


def one_to_one_action_progress(
    task: Any,
    messages: list[Any],
    *,
    expected_action_names: set[str] | None = None,
) -> dict[str, Any]:
    """Measure required-action progress without reusing one call twice.

    Tau2's benchmark evaluator intentionally checks whether each golden action is
    present. For process shaping we need stricter credit accounting: once a model
    tool call satisfies one expected action, it cannot satisfy another duplicate.
    """

    criteria = task.evaluation_criteria
    expected = (
        [
            action
            for action in (criteria.actions or [])
            if action.requestor == "assistant"
            and (expected_action_names is None or action.name in expected_action_names)
        ]
        if criteria is not None
        else []
    )
    predicted = [
        call
        for message in messages
        if getattr(message, "role", None) == "assistant"
        for call in (getattr(message, "tool_calls", None) or [])
        if getattr(call, "requestor", "assistant") == "assistant"
    ]
    unused = set(range(len(predicted)))
    matches: list[dict[str, Any]] = []
    for action in expected:
        match_index = next(
            (
                index
                for index in sorted(unused)
                if _action_matches(action, predicted[index])
            ),
            None,
        )
        matched = match_index is not None
        if matched:
            unused.remove(match_index)
        matches.append(
            {
                "action_id": action.action_id,
                "name": action.name,
                "matched": matched,
                "matched_call_index": match_index,
            }
        )

    expected_exact = Counter(
        _tool_signature(action.name, action.arguments) for action in expected
    )
    predicted_exact = Counter(
        _tool_signature(call.name, call.arguments) for call in predicted
    )
    duplicate_excess = sum(
        max(0, count - max(1, expected_exact.get(signature, 0)))
        for signature, count in predicted_exact.items()
    )
    matched_count = sum(item["matched"] for item in matches)
    unexpected_writes = [
        {
            "name": predicted[index].name,
            "arguments": dict(predicted[index].arguments),
        }
        for index in sorted(unused)
        if predicted[index].name in WRITE_TOOLS
    ]
    return {
        "expected_count": len(expected),
        "predicted_count": len(predicted),
        "matched_count": matched_count,
        "recall": matched_count / len(expected) if expected else None,
        "matches": matches,
        "duplicate_excess_count": duplicate_excess,
        "unexpected_write_count": len(unexpected_writes),
        "unexpected_writes": unexpected_writes,
    }


_CONFIRM_QUESTION = re.compile(
    r"(?:^|[.!?]\s*)confirm(?:ing)?\b"
    r"|\b(?:do|can|could|would)\s+you\s+confirm\b"
    r"|\bplease\s+confirm\b"
    r"|\bbefore\s+i\s+proceed[\s\S]{0,500}?\bconfirm\b"
    r"|\bwould\s+you\s+like\s+me\s+to\s+proceed\b"
    r"|\bis\s+(?:that|this)\s+correct\b"
    r"|\byes\s*/\s*no\b"
    r"|请确认|是否确认|是否继续",
    re.IGNORECASE,
)
_AFFIRMATIVE = re.compile(
    r"\b(yes|confirm(?:ed)?|proceed|go ahead|sure|do it)\b|确认|同意|继续",
    re.IGNORECASE,
)

_CONFIRMATION_PARAMETER_FIELDS = {
    "modify_user_address": (
        "address1",
        "address2",
        "city",
        "state",
        "country",
        "zip",
    ),
    "cancel_pending_order": ("order_id", "reason"),
    "exchange_delivered_order_items": (
        "order_id",
        "item_pairs",
        "payment_method_id",
    ),
    "modify_pending_order_items": (
        "order_id",
        "item_pairs",
        "payment_method_id",
    ),
    "modify_pending_order_payment": ("order_id", "payment_method_id"),
    "return_delivered_order_items": (
        "order_id",
        "item_ids",
        "payment_method_id",
    ),
}


def _literal_occurrences(text: str, value: Any) -> int:
    literal = str(value or "").strip().lower()
    if not literal:
        return 0
    return len(
        re.findall(
            rf"(?<![\w]){re.escape(literal)}(?![\w])",
            text,
            flags=re.IGNORECASE,
        )
    )


def _payment_aliases_before_write(
    messages: list[Any],
    write_index: int,
    arguments: dict[str, Any],
) -> dict[str, list[str]]:
    """Resolve payment aliases only from tool-observed user/order state."""

    expected = str(arguments.get("payment_method_id") or "")
    order_id = str(arguments.get("order_id") or "")
    if not expected:
        return {}

    aliases: set[str] = set()
    original_payment_ids: set[str] = set()
    ids_by_brand: dict[str, set[str]] = {}
    ids_by_source: dict[str, set[str]] = {}
    expected_brand = ""
    expected_source = ""
    for message in messages[:write_index]:
        if getattr(message, "role", None) != "tool":
            continue
        try:
            payload = json.loads(str(getattr(message, "content", "") or ""))
        except (TypeError, ValueError, json.JSONDecodeError):
            continue
        if not isinstance(payload, dict):
            continue

        methods = payload.get("payment_methods")
        if isinstance(methods, dict):
            for method_id, details in methods.items():
                if not isinstance(details, dict):
                    continue
                method_id = str(method_id)
                source = str(details.get("source") or "").replace("_", " ")
                brand = str(details.get("brand") or "")
                if brand:
                    ids_by_brand.setdefault(brand.lower(), set()).add(method_id)
                if source:
                    ids_by_source.setdefault(source.lower(), set()).add(method_id)
                if method_id != expected:
                    continue
                expected_brand = brand.lower()
                expected_source = source.lower()
                last_four = str(details.get("last_four") or "")
                suffix = expected.rsplit("_", 1)[-1]
                display_number = last_four or suffix
                labels = {source, brand}
                if source == "paypal":
                    labels.add("paypal account")
                for label in labels:
                    if label and display_number:
                        aliases.add(f"{label} ending in {display_number}")
                        aliases.add(f"{label} ending {display_number}")

        if str(payload.get("order_id") or "") == order_id:
            for payment in payload.get("payment_history") or []:
                if (
                    isinstance(payment, dict)
                    and payment.get("transaction_type") == "payment"
                ):
                    original_payment_ids.add(
                        str(payment.get("payment_method_id") or "")
                    )

    if expected in original_payment_ids:
        aliases.add("original payment method")
    if expected_brand and ids_by_brand.get(expected_brand) == {expected}:
        aliases.add(expected_brand)
    if expected_source and ids_by_source.get(expected_source) == {expected}:
        aliases.add(expected_source)
    return {expected: sorted(aliases)} if aliases else {}


def confirmation_parameter_binding(
    tool: str,
    arguments: dict[str, Any],
    confirmation_text: str,
    *,
    value_aliases: dict[str, list[str]] | None = None,
) -> dict[str, Any]:
    """Check literal parameter binding in an explicit confirmation summary.

    PASS is deliberately high precision: every supported material parameter must
    be present, and item replacements must preserve old-to-new pairing. Missing
    evidence is REVIEW rather than FAIL because users may use aliases that this
    deterministic checker cannot resolve. This remains diagnostic-only.
    """

    required_fields = _CONFIRMATION_PARAMETER_FIELDS.get(tool)
    if not required_fields:
        return {
            "verdict": "NOT_EVALUABLE",
            "tool": tool,
            "field_checks": [],
            "unsupported_reason": "tool_parameter_contract_not_defined",
            "used_as_reward": False,
        }

    text = re.sub(r"\s+", " ", str(confirmation_text or "").lower()).strip()
    value_aliases = value_aliases or {}
    checks: list[dict[str, Any]] = []
    for field in required_fields:
        if field == "item_pairs":
            old_items = list(arguments.get("item_ids") or [])
            new_items = list(arguments.get("new_item_ids") or [])
            pairs = list(zip(old_items, new_items))
            pair_checks = []
            for old_item, new_item in dict.fromkeys(pairs):
                old = re.escape(str(old_item).lower())
                new = re.escape(str(new_item).lower())
                relation = re.compile(
                    rf"(?<![\w]){old}(?![\w])\s*"
                    rf"(?:to|for|with|->|→|换成|替换为)\s*(?:item\s+)?"
                    rf"(?<![\w]){new}(?![\w])",
                    re.IGNORECASE,
                )
                required_count = pairs.count((old_item, new_item))
                found_count = len(relation.findall(text))
                pair_checks.append(
                    {
                        "old_item_id": str(old_item),
                        "new_item_id": str(new_item),
                        "required_count": required_count,
                        "found_count": found_count,
                        "covered": found_count >= required_count,
                    }
                )
            if len(pairs) == 1:
                old_item, new_item = pairs[0]
                pair_checks[0]["covered"] = bool(
                    _literal_occurrences(text, old_item)
                    and _literal_occurrences(text, new_item)
                )
            elif pairs and len(set(old_items + new_items)) == len(
                old_items + new_items
            ):
                positions = []
                for old_item, new_item in pairs:
                    for role, value in (("old", old_item), ("new", new_item)):
                        match = re.search(
                            rf"(?<![\w]){re.escape(str(value).lower())}(?![\w])",
                            text,
                            re.IGNORECASE,
                        )
                        if match:
                            positions.append((match.start(), role, str(value)))
                if len(positions) == 2 * len(pairs):
                    ordered = sorted(positions)
                    observed_pairs = [
                        (ordered[index][2], ordered[index + 1][2])
                        for index in range(0, len(ordered), 2)
                        if ordered[index][1] == "old" and ordered[index + 1][1] == "new"
                    ]
                    if Counter(observed_pairs) == Counter(
                        (str(old), str(new)) for old, new in pairs
                    ):
                        for item in pair_checks:
                            item["covered"] = True
            covered = (
                bool(pairs)
                and len(old_items) == len(new_items)
                and all(item["covered"] for item in pair_checks)
            )
            checks.append(
                {
                    "field": field,
                    "covered": covered,
                    "pairs": pair_checks,
                    "length_match": len(old_items) == len(new_items),
                }
            )
            continue

        raw_value = arguments.get(field)
        values = list(raw_value or []) if field == "item_ids" else [raw_value]
        value_checks = []
        for value in values:
            required_count = values.count(value)
            found_count = _literal_occurrences(text, value)
            matched_aliases = [
                alias
                for alias in value_aliases.get(str(value), [])
                if _literal_occurrences(text, alias)
            ]
            effective_count = max(found_count, 1 if matched_aliases else 0)
            value_checks.append(
                {
                    "value": str(value),
                    "required_count": required_count,
                    "found_count": found_count,
                    "matched_aliases": matched_aliases,
                    "covered": effective_count >= required_count,
                }
            )
        checks.append(
            {
                "field": field,
                "covered": bool(value_checks)
                and all(item["covered"] for item in value_checks),
                "values": value_checks,
            }
        )

    missing_fields = [item["field"] for item in checks if not item["covered"]]
    return {
        "verdict": "PASS" if not missing_fields else "REVIEW",
        "tool": tool,
        "field_checks": checks,
        "missing_fields": missing_fields,
        "used_as_reward": False,
    }


def confirmation_diagnostics(messages: list[Any]) -> dict[str, Any]:
    """Conservatively detect explicit confirmation before Retail write calls.

    This signal is diagnostic-only until validated against independent human gold.
    """

    confirmed_after = -1
    confirmation_text = ""
    confirmation_prompt_text = ""
    confirmation_user_index = -1
    authorized_write_count = 0
    authorized_signatures: set[str] = set()
    last_write = -1
    checks: list[dict[str, Any]] = []
    for index, message in enumerate(messages):
        role = getattr(message, "role", None)
        content = str(getattr(message, "content", "") or "")
        if role == "user":
            if index > confirmation_user_index:
                confirmed_after = -1
                confirmation_text = ""
                confirmation_prompt_text = ""
                confirmation_user_index = -1
                authorized_write_count = 0
                authorized_signatures.clear()
            affirmative = _AFFIRMATIVE.search(content)
            prior_asks = [
                prior
                for prior in range(last_write + 1, index)
                if getattr(messages[prior], "role", None) == "assistant"
                and _CONFIRM_QUESTION.search(
                    str(getattr(messages[prior], "content", "") or "")
                )
            ]
            if affirmative and prior_asks:
                confirmed_after = max(prior_asks)
                confirmation_user_index = index
                confirmation_prompt_text = str(
                    getattr(messages[confirmed_after], "content", "") or ""
                ).lower()
                confirmation_text = (
                    confirmation_prompt_text + " " + content
                ).lower()
                authorized_signatures.clear()
        if role != "assistant":
            continue
        write_calls = [
            call
            for call in (getattr(message, "tool_calls", None) or [])
            if call.name in WRITE_TOOLS
        ]
        has_confirmation = bool(write_calls) and confirmation_user_index >= 0
        order_ids = [
            str(getattr(call, "arguments", {}).get("order_id") or "").lower()
            for call in write_calls
        ]
        for call in write_calls:
            signature = _tool_signature(call.name, dict(call.arguments))
            order_id = str(getattr(call, "arguments", {}).get("order_id") or "").lower()
            confirmed = (
                has_confirmation
                and (
                    (len(write_calls) == 1 and authorized_write_count == 0)
                    or (all(order_ids) and order_id in confirmation_text)
                )
                and signature not in authorized_signatures
            )
            checks.append(
                {
                    "tool_call_id": str(call.id),
                    "tool": call.name,
                    "confirmed": confirmed,
                    "parameter_binding": confirmation_parameter_binding(
                        call.name,
                        dict(call.arguments),
                        confirmation_prompt_text,
                        value_aliases=_payment_aliases_before_write(
                            messages,
                            index,
                            dict(call.arguments),
                        ),
                    ),
                }
            )
            if has_confirmation:
                authorized_signatures.add(signature)
        if write_calls:
            last_write = index
            authorized_write_count += len(write_calls)
    return {
        "write_count": len(checks),
        "confirmed_write_count": sum(item["confirmed"] for item in checks),
        "missing_confirmation_count": sum(not item["confirmed"] for item in checks),
        "checks": checks,
        "diagnostic_version": "v3_prompt_bound_confirmation_scope",
        "used_as_reward": False,
    }


def identity_authentication_stage_reward(
    task: Any,
    messages: list[Any],
    reward_config: dict[str, Any],
) -> dict[str, Any]:
    """Score only the hidden gold identity-authentication action.

    This is a staged process reward, not a full Retail task reward.  The policy
    sees the stage contract but never the expected action name or arguments.
    """

    action_progress = one_to_one_action_progress(
        task,
        messages,
        expected_action_names=IDENTITY_AUTHENTICATION_ACTIONS,
    )
    if action_progress["expected_count"] != 1:
        raise RuntimeError(
            "IDENTITY_AUTHENTICATION requires exactly one hidden expected "
            "authentication action"
        )
    tool_errors = sum(bool(getattr(message, "error", False)) for message in messages)
    action_recall = float(action_progress["recall"] or 0.0)
    stage_complete = (
        action_recall == 1.0
        and action_progress["predicted_count"] == 1
        and tool_errors == 0
    )
    stage_value = 1.0 if stage_complete else 0.0
    error_penalty = min(
        reward_config["tool_error_penalty_cap"],
        reward_config["tool_error_penalty_each"] * tool_errors,
    )
    repeat_penalty = min(
        reward_config["repeated_call_penalty_cap"],
        reward_config["repeated_call_penalty_each"]
        * action_progress["duplicate_excess_count"],
    )
    unexpected_write_penalty = min(
        reward_config["unexpected_write_penalty_cap"],
        reward_config["unexpected_write_penalty_each"]
        * action_progress["unexpected_write_count"],
    )
    unfinished_penalty = (
        0.0 if stage_complete else reward_config["unfinished_interaction_penalty"]
    )
    reward = max(
        0.0,
        stage_value
        - error_penalty
        - repeat_penalty
        - unexpected_write_penalty
        - unfinished_penalty,
    )
    return {
        "reward": reward,
        "components": {
            "identity_authentication_action": {
                "weight": 1.0,
                "value": stage_value,
            }
        },
        "rollout_stage": IDENTITY_AUTHENTICATION_STAGE,
        "stage_complete": stage_complete,
        "termination_basis": "stage_target_complete",
        "tool_error_count": tool_errors,
        "tool_error_penalty": error_penalty,
        "repeated_call_penalty": repeat_penalty,
        "unexpected_write_penalty": unexpected_write_penalty,
        "unfinished_interaction_penalty": unfinished_penalty,
        "user_stopped": None,
        "action_progress": action_progress,
        "environment_state_diagnostics": None,
        "confirmation_diagnostics": confirmation_diagnostics(messages),
        "reward_config": deepcopy(reward_config),
        "nl_assertions_used": False,
        "policy_guard_used_as_reward": False,
        "tau2": {"environment": None, "communication": None},
    }


def _ensure_tau2_importable() -> None:
    """Add the pinned upstream checkout to sys.path when it is not installed."""

    try:
        import tau2  # noqa: F401

        return
    except ImportError:
        pass

    root = os.environ.get("POLICYAGENT_TAU2_ROOT")
    if not root:
        raise RuntimeError(
            "tau2 is not importable. Install the pinned upstream checkout or set "
            "POLICYAGENT_TAU2_ROOT to its repository root."
        )
    src = Path(root).expanduser().resolve() / "src"
    if not src.is_dir():
        raise RuntimeError(f"Invalid POLICYAGENT_TAU2_ROOT: {root}")
    sys.path.insert(0, str(src))
    try:
        import tau2  # noqa: F401
    except ImportError as exc:
        raise RuntimeError(f"Unable to import tau2 from {src}") from exc


class RetailAgenticEnvironment:
    """TRL-compatible stateful wrapper around the pinned tau2 Retail domain.

    The class deliberately keeps hidden ``task.user_scenario`` content inside the
    user simulator. ``reset`` requires a separately frozen opening utterance, so
    the policy model only observes what a real customer would have said.

    Public methods other than ``reset`` and ``get_reward`` are exposed by modern
    TRL as tools. Private helpers remain invisible to the policy model.
    """

    def __init__(
        self,
        *,
        environment_factory: Callable[[], Any] | None = None,
        tasks_loader: Callable[[str], list[Any]] | None = None,
        user_factory: Callable[[Any, Any, list[Any], int], tuple[Any, Any]]
        | None = None,
        evaluator: Callable[[Any, Any], Any] | None = None,
    ) -> None:
        _ensure_tau2_importable()
        from tau2.registry import registry

        self._environment_factory = environment_factory or registry.get_env_constructor(
            "retail"
        )
        self._tasks_loader = tasks_loader or registry.get_tasks_loader("retail")
        self._user_factory = user_factory
        self._evaluator = evaluator
        self._environment: Any | None = None
        self._initial_environment_state: dict[str, Any] = {}
        self._task: Any | None = None
        self._user: Any | None = None
        self._user_state: Any | None = None
        self._messages: list[Any] = []
        self._user_dialogue: list[Any] = []
        self._seed = 0
        self._started_at = 0.0
        self._tool_counter = 0
        self._user_stopped = False
        self._policy_findings: list[dict[str, Any]] = []
        self._last_reward_info: dict[str, Any] | None = None
        self._reward_config = load_reward_config()
        transport_gate = os.environ.get(REQUIRE_TRANSPORT_COMPLETE_ENV, "0").strip()
        if transport_gate not in {"0", "1"}:
            raise RuntimeError(
                f"{REQUIRE_TRANSPORT_COMPLETE_ENV} must be exactly '0' or '1'"
            )
        self._require_transport_complete = transport_gate == "1"
        tool_iteration_policy = os.environ.get(
            TOOL_ITERATION_LIMIT_AS_TERMINAL_FAILURE_ENV, "0"
        ).strip()
        if tool_iteration_policy not in {"0", "1"}:
            raise RuntimeError(
                f"{TOOL_ITERATION_LIMIT_AS_TERMINAL_FAILURE_ENV} must be exactly "
                "'0' or '1'"
            )
        self._tool_iteration_limit_as_terminal_failure = (
            tool_iteration_policy == "1"
        )
        completion_budget_policy = os.environ.get(
            COMPLETION_BUDGET_AS_TERMINAL_FAILURE_ENV, "0"
        ).strip()
        if completion_budget_policy not in {"0", "1"}:
            raise RuntimeError(
                f"{COMPLETION_BUDGET_AS_TERMINAL_FAILURE_ENV} must be exactly '0' or '1'"
            )
        self._completion_budget_as_terminal_failure = completion_budget_policy == "1"
        self._rollout_stage = os.environ.get(ROLLOUT_STAGE_ENV, FULL_TASK_STAGE).strip()
        if self._rollout_stage not in SUPPORTED_ROLLOUT_STAGES:
            raise RuntimeError(
                f"Unsupported Agentic RL rollout stage: {self._rollout_stage}"
            )
        self._customer_turns = 0
        self._max_customer_turns = int(
            os.environ.get("POLICYAGENT_MAX_CUSTOMER_TURNS", "8")
        )
        self._max_tool_calls = int(os.environ.get("POLICYAGENT_MAX_TOOL_CALLS", "32"))
        self._reward_persisted = False
        self._trainer_completion_telemetry: dict[str, Any] | None = None
        self._runtime_stop_signals: list[str] = []

    def reset(
        self,
        task_id: str,
        initial_user_message: str,
        task_split: str = "train",
        user_seed: int = 0,
        **_: Any,
    ) -> None:
        """Reset one rollout to a deterministic tau2 task state.

        Args:
            task_id: Retail task ID from the frozen RL split manifest.
            initial_user_message: Pre-generated customer opening utterance. It
                must not contain the hidden user-simulator instructions.
            task_split: Upstream split containing the task. RL uses ``train``.
            user_seed: Shared seed used by all generations in one GRPO group.
        """

        if not str(initial_user_message).strip():
            raise ValueError("initial_user_message must be frozen and non-empty")
        tasks = {str(task.id): task for task in self._tasks_loader(task_split)}
        if str(task_id) not in tasks:
            raise KeyError(f"Retail task {task_id!r} is not in split {task_split!r}")

        from tau2.data_model.message import AssistantMessage, UserMessage

        task = tasks[str(task_id)]
        environment = self._environment_factory()
        initial_state = task.initial_state
        history = (
            deepcopy(initial_state.message_history)
            if initial_state is not None and initial_state.message_history
            else []
        )
        if history:
            raise ValueError(
                "RetailAgenticEnvironment v1 only accepts tasks without an existing "
                "message history; supporting it requires a separately audited prompt path."
            )
        environment.set_state(
            initialization_data=(
                initial_state.initialization_data if initial_state is not None else None
            ),
            initialization_actions=(
                initial_state.initialization_actions
                if initial_state is not None
                else None
            ),
            message_history=[],
        )

        hello = AssistantMessage(
            role="assistant", content="Hi! How can I help you today?", cost=0.0
        )
        opening = UserMessage(role="user", content=str(initial_user_message).strip())
        self._environment = environment
        self._initial_environment_state = _environment_state(environment)
        self._task = task
        self._messages = [hello, opening]
        self._user_dialogue = [hello, opening]
        self._seed = int(user_seed)
        self._started_at = time.perf_counter()
        self._tool_counter = 0
        self._user_stopped = False
        self._policy_findings = []
        self._last_reward_info = None
        self._reward_persisted = False
        self._trainer_completion_telemetry = None
        self._rejected_snapshot_saved = False
        self._runtime_stop_signals = []
        self._customer_turns = 0
        self._user, self._user_state = self._build_user()
        return None

    def get_reward(self) -> float:
        """Score with deterministic tau2 components and opt-in local semantics.

        Returns:
            Weighted terminal-state, required-action, and communication reward.
            LLM-judged natural-language assertions and diagnostic policy-guard
            findings do not alter the default v1 reward. Task44 hybrid mode
            additionally validates candidate-bound semantic extraction.
        """

        self._require_ready()
        if self._uses_semantic_reward() and self._last_reward_info is not None:
            # The pre-advantage barrier scores once; native TRL consumes the cache.
            return float(self._last_reward_info["reward"])
        if self._is_completion_budget_terminal_failure():
            # Budget exhaustion is the bounded-task outcome, not a guessed
            # evaluator score for a complete trajectory. Never parse/evaluate
            # the unfinished response, and never dispatch its unfinished tool call.
            payload = {
                "reward": 0.0,
                "reward_mode": "completion_budget_terminal_failure_v1",
                "evaluator_called": False,
                "reward_override": {
                    "reason": "completion_token_budget_exhausted",
                    "policy": "bounded_completion_terminal_failure_v1",
                },
                "diagnostic_policy_findings": deepcopy(self._policy_findings),
                "policy_findings_are_reward_authority": False,
            }
            self._last_reward_info = payload
            self._persist_rollout(payload)
            return 0.0
        tool_iteration_terminal_failure = False
        if self._require_transport_complete:
            invalid_reasons = transport_invalid_reasons(
                self._trainer_completion_telemetry
            )
            if (
                self._tool_iteration_limit_as_terminal_failure
                and invalid_reasons == ["tool_iteration_limit_reached"]
            ):
                tool_iteration_terminal_failure = True
                invalid_reasons = []
            if invalid_reasons:
                self._persist_rejected_rollout()
                raise RuntimeError(
                    "Transport-invalid rollout rejected before reward: "
                    + ", ".join(invalid_reasons)
                )
        if self._evaluator is not None:
            reward_info = self._evaluator(self._task, deepcopy(self._messages))
        else:
            reward_info = self._calculate_programmatic_reward()

        if hasattr(reward_info, "model_dump"):
            payload = reward_info.model_dump(mode="json")
            reward = float(reward_info.reward)
        elif isinstance(reward_info, dict):
            payload = dict(reward_info)
            reward = float(payload["reward"])
        else:
            reward = float(reward_info)
            payload = {"reward": reward}
        if tool_iteration_terminal_failure:
            payload["underlying_evaluator_reward"] = reward
            payload["reward"] = 0.0
            payload["reward_override"] = {
                "reason": "tool_iteration_limit_reached",
                "policy": "bounded_agent_loop_terminal_failure_v1",
            }
            reward = 0.0
        payload["diagnostic_policy_findings"] = deepcopy(self._policy_findings)
        payload["policy_findings_are_reward_authority"] = False
        self._last_reward_info = payload
        self._persist_rollout(payload)
        return reward

    def _uses_semantic_reward(self) -> bool:
        return "semantic_assistance" in self._reward_config.get("staged_reward_spec", {})

    def _semantic_pending_snapshot(self) -> dict[str, Any]:
        self._require_ready()
        return {
            "task_id": str(self._task.id), "user_seed": self._seed,
            "messages": [_message_payload(m) for m in self._messages],
            "initial_state": deepcopy(self._initial_environment_state),
            "final_state": _environment_state(self._environment),
            "completion": deepcopy(self._trainer_completion_telemetry),
        }

    def _blocking_transport_reasons(self) -> list[str]:
        if not self._require_transport_complete:
            return []
        reasons = transport_invalid_reasons(self._trainer_completion_telemetry)
        if self._is_completion_budget_terminal_failure():
            return []
        if self._tool_iteration_limit_as_terminal_failure and reasons == [
            "tool_iteration_limit_reached"
        ]:
            return []
        return reasons

    def _is_completion_budget_terminal_failure(self) -> bool:
        """Opt-in for model-output exhaustion only, never mixed infra failure."""
        if not (
            self._require_transport_complete
            and getattr(self, "_completion_budget_as_terminal_failure", False)
            and not self._uses_semantic_reward()
        ):
            return False
        telemetry = self._trainer_completion_telemetry
        return bool(
            telemetry
            and telemetry["stop_reason"] == "COMPLETION_BUDGET_EXHAUSTED"
            and "TOOL_RESULT_BUDGET_EXCEEDED" not in telemetry["stop_flags"]
            and telemetry["model_tokens_retained"] > 0
            and set(transport_invalid_reasons(telemetry)) == {
                "completion_token_budget_exhausted", "model_completion_truncated"
            }
        )

    def _persist_rejected_rollout(self, *, trainer_evidence=None) -> None:
        """Quarantine evidence without evaluating it or assigning a reward."""
        path_value = os.environ.get(ROLLOUT_LOG_ENV)
        if not path_value:
            return
        # A group snapshot is richer than a standalone get_reward failure.
        if getattr(self, "_rejected_snapshot_saved", False):
            return
        record = {
            "schema_version": "retail-agentic-rejected-rollout-v1",
            "task_id": str(self._task.id),
            "user_seed": self._seed,
            "training_eligible": False,
            "reward_eligible": False,
            "reward": None,
            "reasons": self._blocking_transport_reasons(),
            "messages": [m.model_dump(mode="json") for m in self._messages],
            "initial_state": deepcopy(self._initial_environment_state),
            "final_state": _environment_state(self._environment),
            "completion": deepcopy(self._trainer_completion_telemetry),
            "trainer_evidence": deepcopy(trainer_evidence),
            "hidden_user_scenario_persisted": False,
        }
        record["evidence_sha256"] = _canonical_sha256(record)
        path = Path(path_value).resolve().with_name("rejected_rollouts.jsonl")
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
        self._rejected_snapshot_saved = True

    def _set_trainer_completion_telemetry(self, payload: dict[str, Any]) -> None:
        """Bind trainer-observed termination evidence before reward persistence."""

        self._require_ready()
        if self._last_reward_info is not None or self._reward_persisted:
            raise RuntimeError("Completion telemetry must be bound before reward")
        if self._trainer_completion_telemetry is not None:
            raise RuntimeError("Completion telemetry is already bound")
        if (
            not isinstance(payload, dict)
            or set(payload) != GUARDED_COMPLETION_TELEMETRY_KEYS
        ):
            raise ValueError("Unexpected guarded completion telemetry schema")
        if payload["stop_reason"] not in GUARDED_PRIMARY_STOP_REASONS:
            raise ValueError("Unsupported guarded primary stop reason")
        if payload["stop_reason_source"] != "guarded_grpo_trainer_v1":
            raise ValueError("Unsupported guarded stop-reason source")
        if payload["model_completion_truncation_source"] != "guarded_grpo_trainer_v1":
            raise ValueError("Unsupported guarded truncation source")
        flags = payload["stop_flags"]
        if (
            not isinstance(flags, list)
            or any(not isinstance(flag, str) or not flag for flag in flags)
            or len(flags) != len(set(flags))
        ):
            raise ValueError("stop_flags must be unique non-empty strings")
        if not set(flags).issubset(GUARDED_STOP_FLAGS):
            raise ValueError("stop_flags contain an unsupported guarded stop reason")
        if not set(self._runtime_stop_signals).issubset(flags):
            raise ValueError(
                "Trainer telemetry omitted a causal environment stop signal"
            )
        bool_fields = {
            "model_ended",
            "model_eos_observed",
            "completion_token_budget_exhausted",
            "context_limit_reached",
            "tool_iteration_limit_reached",
            "unresolved_tool_call",
            "framework_loop_abnormal_end",
            "model_completion_truncated",
        }
        if any(type(payload[name]) is not bool for name in bool_fields):
            raise ValueError("Guarded completion booleans must be explicit bool values")
        count_fields = {
            "prompt_tokens",
            "completion_tokens",
            "model_tokens_retained",
            "observation_tokens_retained",
        }
        if any(
            type(payload[name]) is not int or payload[name] < 0 for name in count_fields
        ):
            raise ValueError(
                "Guarded completion token counts must be non-negative ints"
            )
        if (
            payload["model_tokens_retained"] + payload["observation_tokens_retained"]
            != payload["completion_tokens"]
        ):
            raise ValueError("Guarded completion token counts do not conserve length")
        expected_flags = {
            "completion_token_budget_exhausted": {
                "TOOL_RESULT_BUDGET_EXCEEDED",
                "COMPLETION_BUDGET_EXHAUSTED",
            },
            "context_limit_reached": {"CONTEXT_LIMIT"},
            "tool_iteration_limit_reached": {"TOOL_ITERATION_LIMIT"},
            "unresolved_tool_call": {"UNRESOLVED_TOOL_CALL"},
            "model_completion_truncated": {"COMPLETION_BUDGET_EXHAUSTED"},
        }
        for field, causes in expected_flags.items():
            if payload[field] is not bool(causes.intersection(flags)):
                raise ValueError(f"Guarded completion {field} contradicts stop_flags")
        stop_reason = payload["stop_reason"]
        primary_flag_reasons = {
            "TOOL_RESULT_BUDGET_EXCEEDED",
            "CONTEXT_LIMIT",
            "COMPLETION_BUDGET_EXHAUSTED",
            "TOOL_ITERATION_LIMIT",
            "UNRESOLVED_TOOL_CALL",
            "MODEL_END_WITHOUT_EOS",
        }
        if stop_reason in primary_flag_reasons and stop_reason not in flags:
            raise ValueError("Guarded primary stop reason is absent from stop_flags")
        if stop_reason == "USER_STOP_AND_MODEL_EOS" and (
            not payload["model_eos_observed"] or not self._user_stopped
        ):
            raise ValueError("USER_STOP_AND_MODEL_EOS contradicts runtime state")
        if stop_reason == "MODEL_EOS_BEFORE_USER_STOP" and (
            not payload["model_eos_observed"] or self._user_stopped
        ):
            raise ValueError("MODEL_EOS_BEFORE_USER_STOP contradicts runtime state")
        if stop_reason == "MODEL_END_WITHOUT_EOS" and payload["model_eos_observed"]:
            raise ValueError("MODEL_END_WITHOUT_EOS contradicts EOS telemetry")
        self._trainer_completion_telemetry = deepcopy(payload)

    def _persist_rollout(self, reward_payload: dict[str, Any]) -> None:
        """Append one raw record and its rollout-bound evidence sidecar."""

        path_value = os.environ.get(ROLLOUT_LOG_ENV)
        if not path_value or self._reward_persisted:
            return
        evidence_path_value = os.environ.get(ROLLOUT_EVIDENCE_LOG_ENV)
        if not evidence_path_value:
            raise RuntimeError(
                f"{ROLLOUT_EVIDENCE_LOG_ENV} is required when {ROLLOUT_LOG_ENV} is set"
            )
        path = Path(path_value).expanduser().resolve()
        evidence_path = Path(evidence_path_value).expanduser().resolve()
        path.parent.mkdir(parents=True, exist_ok=True)
        evidence_path.parent.mkdir(parents=True, exist_ok=True)
        messages = [
            message.model_dump(mode="json")
            if hasattr(message, "model_dump")
            else str(message)
            for message in self._messages
        ]
        final_state = _environment_state(self._environment)
        state_evidence = {
            "initial_state": deepcopy(self._initial_environment_state),
            "final_state": final_state,
            "state_diff": _state_diff(self._initial_environment_state, final_state),
            "state_hashes": {
                "initial_sha256": _canonical_sha256(self._initial_environment_state),
                "final_sha256": _canonical_sha256(final_state),
            },
        }
        completion = {
            "user_stopped": self._user_stopped,
            "customer_turn_limit_reached": (
                self._customer_turns >= self._max_customer_turns
                and not self._user_stopped
            ),
            "tool_call_limit_reached": self._tool_counter >= self._max_tool_calls,
            "model_completion_truncated": None,
            "model_completion_truncation_source": "trainer_metrics_only",
        }
        if self._trainer_completion_telemetry is not None:
            completion.update(deepcopy(self._trainer_completion_telemetry))
        terminal_evaluator = deepcopy(reward_payload)
        if reward_payload.get("reward_mode") == TIERED_TERMINAL_PROCESS_MODE:
            terminal_evaluator["reward"] = reward_payload["terminal_environment_reward"]
        evidence = {
            "schema_version": "retail-agentic-rollout-evidence-v1",
            "task_id": str(self._task.id),
            "rollout_stage": self._rollout_stage,
            "user_seed": self._seed,
            **state_evidence,
            "tool_trace": _tool_trace(self._messages),
            "terminal_evaluator": terminal_evaluator,
            "completion": completion,
            "hidden_user_scenario_persisted": False,
        }
        evidence_sha256 = _canonical_sha256(evidence)
        evidence["evidence_sha256"] = evidence_sha256
        record = {
            "schema_version": "retail-agentic-rollout-v2",
            "task_id": str(self._task.id),
            "rollout_stage": self._rollout_stage,
            "user_seed": self._seed,
            "elapsed_seconds": time.perf_counter() - self._started_at,
            "customer_turns": self._customer_turns,
            "tool_calls": self._tool_counter,
            "messages": messages,
            "reward": deepcopy(reward_payload),
            "completion": completion,
            "evidence_sha256": evidence_sha256,
            "hidden_user_scenario_persisted": False,
        }
        with evidence_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(evidence, ensure_ascii=False, default=str) + "\n")
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
        self._reward_persisted = True

    def _calculate_programmatic_reward(self) -> dict[str, Any]:
        """Compose rule scores; optional local extractor never assigns totals."""

        trajectory = deepcopy(self._messages)
        if self._rollout_stage == IDENTITY_AUTHENTICATION_STAGE:
            return identity_authentication_stage_reward(
                self._task,
                trajectory,
                self._reward_config,
            )

        from tau2.evaluator.evaluator_communicate import CommunicateEvaluator
        from tau2.evaluator.evaluator_env import EnvironmentEvaluator

        env_info = EnvironmentEvaluator.calculate_reward(
            environment_constructor=self._environment_factory,
            task=self._task,
            full_trajectory=trajectory,
            solo_mode=False,
            env_kwargs={},
        )
        communication_info = CommunicateEvaluator.calculate_reward(
            task=self._task,
            full_trajectory=trajectory,
        )

        criteria = self._task.evaluation_criteria
        required_communication = (
            list(criteria.communicate_info or []) if criteria is not None else []
        )
        action_progress = one_to_one_action_progress(self._task, trajectory)
        communication_checks = list(communication_info.communicate_checks or [])
        action_recall = action_progress["recall"]
        communication_recall = (
            sum(float(check.met) for check in communication_checks)
            / len(communication_checks)
            if required_communication
            else None
        )

        reward_config = self._reward_config
        environment_state_raw = float(env_info.reward)
        if reward_config["process_reward_mode"] == "terminal_environment_state":
            reward = terminal_environment_reward(
                environment_state_raw,
                user_stopped=self._user_stopped,
            )
            return {
                "reward": reward,
                "components": {
                    "terminal_environment_state": {
                        "weight": 1.0,
                        "value": reward,
                    }
                },
                "tool_error_count": sum(
                    bool(getattr(message, "error", False)) for message in trajectory
                ),
                "tool_error_penalty": 0.0,
                "repeated_call_penalty": 0.0,
                "unexpected_write_penalty": 0.0,
                "unfinished_interaction_penalty": 0.0,
                "user_stopped": self._user_stopped,
                "rollout_stage": FULL_TASK_STAGE,
                "stage_complete": None,
                "termination_basis": "user_stopped_and_environment_state",
                "action_progress": action_progress,
                "environment_state_diagnostics": {
                    "raw_value": environment_state_raw,
                    "action_progress_gate": None,
                    "gated_value": environment_state_raw,
                    "gate_mode": "none",
                },
                "confirmation_diagnostics": confirmation_diagnostics(trajectory),
                "reward_config": deepcopy(reward_config),
                "nl_assertions_used": False,
                "communication_used_as_reward": False,
                "policy_guard_used_as_reward": False,
                "tau2": {
                    "environment": env_info.model_dump(mode="json"),
                    "communication": communication_info.model_dump(mode="json"),
                },
            }
        if reward_config["process_reward_mode"] == TIERED_TERMINAL_PROCESS_MODE:
            environment_payload = env_info.model_dump(mode="json")
            communication_payload = communication_info.model_dump(mode="json")
            completion = {
                "user_stopped": self._user_stopped,
                "customer_turn_limit_reached": (
                    self._customer_turns >= self._max_customer_turns
                    and not self._user_stopped
                ),
                "tool_call_limit_reached": self._tool_counter >= self._max_tool_calls,
            }
            score = tiered_terminal_process_reward(
                task_id=str(self._task.id),
                messages=trajectory,
                action_progress=action_progress,
                environment_payload=environment_payload,
                communication_payload=communication_payload,
                environment_state_reward=environment_state_raw,
                user_stopped=self._user_stopped,
                completion=completion,
                staged_reward_spec=reward_config["staged_reward_spec"],
                skip_semantics=bool(
                    self._tool_iteration_limit_as_terminal_failure
                    and transport_invalid_reasons(self._trainer_completion_telemetry)
                    == ["tool_iteration_limit_reached"]
                ),
            )
            return {
                "reward": score["staged_reward"],
                **({
                    "rule_only_reward": score["rule_only_reward"],
                    "semantic_assistance": deepcopy(score["semantic_assistance"]),
                    "hybrid_additive_components": deepcopy(score["semantic_assistance"]["additive_components"]),
                    "reward_component_source": "hybrid_additive_components",
                    "llm_semantic_assistance_used": True,
                } if "semantic_assistance" in score else {}),
                "reward_mode": TIERED_TERMINAL_PROCESS_MODE,
                "terminal_environment_reward": score["terminal_environment_reward"],
                "components": deepcopy(score["components"]),
                "effective_component_values": deepcopy(
                    score["effective_component_values"]
                ),
                "normalized_weights": deepcopy(score["normalized_weights"]),
                "process_score_before_penalties": score[
                    "process_score_before_penalties"
                ],
                "penalties": deepcopy(score["penalties"]),
                "total_penalty_applied": score["total_penalty_applied"],
                "terminal_success": score["terminal_success"],
                "complete_success": score["complete_success"],
                "communication_complete": score["communication_complete"],
                # Preserve the historical rollout-evidence field while the scorer
                # uses the more precise name: a bound non-error write is verified,
                # not its business or policy correctness.
                "no_correct_write_cap_applied": score[
                    "no_verified_write_cap_applied"
                ],
                "no_verified_write_cap_applied": score[
                    "no_verified_write_cap_applied"
                ],
                "tool_error_count": score["tool_error_count"],
                "repeated_call_count": score["repeated_call_count"],
                "unexpected_write_count": score["unexpected_write_count"],
                "user_stopped": self._user_stopped,
                "rollout_stage": FULL_TASK_STAGE,
                "stage_complete": None,
                "termination_basis": "tiered_terminal_plus_process_v2",
                "action_progress": action_progress,
                "environment_state_diagnostics": {
                    "raw_value": environment_state_raw,
                    "terminal_value": score["terminal_environment_reward"],
                    "gate_mode": "tiered_formula",
                },
                "confirmation_diagnostics": confirmation_diagnostics(trajectory),
                "reward_config": deepcopy(reward_config),
                "nl_assertions_used": False,
                "communication_used_as_reward": True,
                "claim_evidence_used_as_reward": bool(
                    (
                        score["components"].get("claim_evidence_consistency")
                        or {}
                    ).get("used_as_reward")
                ),
                "policy_guard_used_as_reward": False,
                "tau2": {
                    "environment": environment_payload,
                    "communication": communication_payload,
                },
            }
        environment_state_value, environment_state_gate = gate_environment_state_reward(
            environment_state_raw, action_recall
        )
        weighted: list[tuple[str, float, float]] = [
            (
                "environment_state",
                reward_config["environment_state_weight"],
                environment_state_value,
            )
        ]
        if action_recall is not None:
            weighted.append(
                (
                    "required_action_recall",
                    reward_config["required_action_weight"],
                    action_recall,
                )
            )
        if communication_recall is not None:
            weighted.append(
                (
                    "communication_recall",
                    reward_config["communication_weight"],
                    communication_recall,
                )
            )
        weighted = [item for item in weighted if item[1] > 0]
        weight_sum = sum(weight for _, weight, _ in weighted)
        raw_reward = sum(weight * value for _, weight, value in weighted) / weight_sum
        tool_errors = sum(
            bool(getattr(message, "error", False)) for message in trajectory
        )
        error_penalty = min(
            reward_config["tool_error_penalty_cap"],
            reward_config["tool_error_penalty_each"] * tool_errors,
        )
        repeat_penalty = min(
            reward_config["repeated_call_penalty_cap"],
            reward_config["repeated_call_penalty_each"]
            * action_progress["duplicate_excess_count"],
        )
        unexpected_write_penalty = min(
            reward_config["unexpected_write_penalty_cap"],
            reward_config["unexpected_write_penalty_each"]
            * action_progress["unexpected_write_count"],
        )
        unfinished_penalty = (
            0.0
            if self._user_stopped
            else reward_config["unfinished_interaction_penalty"]
        )
        reward = max(
            0.0,
            raw_reward
            - error_penalty
            - repeat_penalty
            - unexpected_write_penalty
            - unfinished_penalty,
        )
        return {
            "reward": reward,
            "components": {
                name: {"weight": weight / weight_sum, "value": value}
                for name, weight, value in weighted
            },
            "tool_error_count": tool_errors,
            "tool_error_penalty": error_penalty,
            "repeated_call_penalty": repeat_penalty,
            "unexpected_write_penalty": unexpected_write_penalty,
            "unfinished_interaction_penalty": unfinished_penalty,
            "user_stopped": self._user_stopped,
            "rollout_stage": FULL_TASK_STAGE,
            "stage_complete": None,
            "termination_basis": "user_stopped",
            "action_progress": action_progress,
            "environment_state_diagnostics": {
                "raw_value": environment_state_raw,
                "action_progress_gate": environment_state_gate,
                "gated_value": environment_state_value,
                "gate_mode": reward_config["environment_state_action_progress_gate"],
            },
            "confirmation_diagnostics": confirmation_diagnostics(trajectory),
            "reward_config": deepcopy(reward_config),
            "nl_assertions_used": False,
            "policy_guard_used_as_reward": False,
            "tau2": {
                "environment": env_info.model_dump(mode="json"),
                "communication": communication_info.model_dump(mode="json"),
            },
        }

    def respond_to_user(self, message: str) -> str:
        """Send a customer-facing message and receive the simulated reply.

        Args:
            message: The exact text to send to the customer. Use this before a
                state-changing action when explicit confirmation is required.

        Returns:
            The next customer utterance generated from the hidden tau2 user
            scenario, or a stop marker when the customer ends the interaction.
        """

        self._require_ready()
        from tau2.data_model.message import AssistantMessage
        from tau2.user.user_simulator import UserSimulator

        if self._customer_turns >= self._max_customer_turns:
            if "CUSTOMER_TURN_LIMIT" not in self._runtime_stop_signals:
                self._runtime_stop_signals.append("CUSTOMER_TURN_LIMIT")
            raise RuntimeError("Maximum customer turns reached for this rollout")
        assistant = AssistantMessage(role="assistant", content=str(message).strip())
        if not assistant.content:
            raise ValueError("Customer-facing message cannot be empty")
        user_message, self._user_state = generate_with_fail_fast(
            self._user.generate_next_message,
            assistant,
            self._user_state,
            task_id=str(self._task.id),
            user_seed=self._seed,
        )
        self._messages.extend([assistant, user_message])
        self._user_dialogue.extend([assistant, user_message])
        self._customer_turns += 1
        self._user_stopped = UserSimulator.is_stop(user_message)
        return str(user_message.content or "")

    def calculate(self, expression: str) -> str:
        """Calculate a mathematical expression.

        Args:
            expression: Numbers and arithmetic operators to evaluate.

        Returns:
            The calculated value or a structured tool error.
        """

        return self._call_tool("calculate", expression=expression)

    def cancel_pending_order(self, order_id: str, reason: str) -> str:
        """Cancel a pending order after explicit customer confirmation.

        Args:
            order_id: Order identifier including the leading ``#``.
            reason: Either ``no longer needed`` or ``ordered by mistake``.

        Returns:
            Updated order state or a structured tool error.
        """

        return self._call_tool("cancel_pending_order", order_id=order_id, reason=reason)

    def exchange_delivered_order_items(
        self,
        order_id: str,
        item_ids: list[str],
        new_item_ids: list[str],
        payment_method_id: str,
    ) -> str:
        """Exchange delivered items for different variants after confirmation.

        Args:
            order_id: Delivered order identifier.
            item_ids: Existing item IDs to exchange.
            new_item_ids: Replacement variant IDs aligned with ``item_ids``.
            payment_method_id: Payment method for any price difference.

        Returns:
            Updated order state or a structured tool error.
        """

        return self._call_tool(
            "exchange_delivered_order_items",
            order_id=order_id,
            item_ids=item_ids,
            new_item_ids=new_item_ids,
            payment_method_id=payment_method_id,
        )

    def find_user_id_by_name_zip(
        self, first_name: str, last_name: str, zip: str
    ) -> str:
        """Find a customer by name and postal code.

        Args:
            first_name: Customer first name.
            last_name: Customer last name.
            zip: Customer postal code.

        Returns:
            User ID or a structured tool error.
        """

        return self._call_tool(
            "find_user_id_by_name_zip",
            first_name=first_name,
            last_name=last_name,
            zip=zip,
        )

    def find_user_id_by_email(self, email: str) -> str:
        """Find a customer by email.

        Args:
            email: Customer email address.

        Returns:
            User ID or a structured tool error.
        """

        return self._call_tool("find_user_id_by_email", email=email)

    def get_order_details(self, order_id: str) -> str:
        """Read an order's current state.

        Args:
            order_id: Order identifier including the leading ``#``.

        Returns:
            Serialized order details or a structured tool error.
        """

        return self._call_tool("get_order_details", order_id=order_id)

    def get_product_details(self, product_id: str) -> str:
        """Read product variants and availability.

        Args:
            product_id: Product identifier, not an item ID.

        Returns:
            Serialized product details or a structured tool error.
        """

        return self._call_tool("get_product_details", product_id=product_id)

    def get_item_details(self, item_id: str) -> str:
        """Read one item variant.

        Args:
            item_id: Item or variant identifier.

        Returns:
            Serialized item details or a structured tool error.
        """

        return self._call_tool("get_item_details", item_id=item_id)

    def get_user_details(self, user_id: str) -> str:
        """Read a customer profile, orders, and payment methods.

        Args:
            user_id: Authenticated customer identifier.

        Returns:
            Serialized user details or a structured tool error.
        """

        return self._call_tool("get_user_details", user_id=user_id)

    def list_all_product_types(self) -> str:
        """List product names and product identifiers.

        Returns:
            JSON mapping of product names to product IDs.
        """

        return self._call_tool("list_all_product_types")

    def modify_pending_order_address(
        self,
        order_id: str,
        address1: str,
        address2: str,
        city: str,
        state: str,
        country: str,
        zip: str,
    ) -> str:
        """Change a pending order address after explicit confirmation.

        Args:
            order_id: Pending order identifier.
            address1: First address line.
            address2: Second address line, or an empty string.
            city: City.
            state: State or region.
            country: Country.
            zip: Postal code.

        Returns:
            Updated order state or a structured tool error.
        """

        return self._call_tool(
            "modify_pending_order_address",
            order_id=order_id,
            address1=address1,
            address2=address2,
            city=city,
            state=state,
            country=country,
            zip=zip,
        )

    def modify_pending_order_items(
        self,
        order_id: str,
        item_ids: list[str],
        new_item_ids: list[str],
        payment_method_id: str,
    ) -> str:
        """Replace variants in a pending order after confirmation.

        Args:
            order_id: Pending order identifier.
            item_ids: Existing item IDs to replace.
            new_item_ids: Replacement IDs aligned with ``item_ids``.
            payment_method_id: Payment method for the price difference.

        Returns:
            Updated order state or a structured tool error.
        """

        return self._call_tool(
            "modify_pending_order_items",
            order_id=order_id,
            item_ids=item_ids,
            new_item_ids=new_item_ids,
            payment_method_id=payment_method_id,
        )

    def modify_pending_order_payment(
        self, order_id: str, payment_method_id: str
    ) -> str:
        """Change a pending order payment method after confirmation.

        Args:
            order_id: Pending order identifier.
            payment_method_id: New customer-owned payment method ID.

        Returns:
            Updated order state or a structured tool error.
        """

        return self._call_tool(
            "modify_pending_order_payment",
            order_id=order_id,
            payment_method_id=payment_method_id,
        )

    def modify_user_address(
        self,
        user_id: str,
        address1: str,
        address2: str,
        city: str,
        state: str,
        country: str,
        zip: str,
    ) -> str:
        """Change the customer's default address after confirmation.

        Args:
            user_id: Authenticated customer identifier.
            address1: First address line.
            address2: Second address line, or an empty string.
            city: City.
            state: State or region.
            country: Country.
            zip: Postal code.

        Returns:
            Updated customer state or a structured tool error.
        """

        return self._call_tool(
            "modify_user_address",
            user_id=user_id,
            address1=address1,
            address2=address2,
            city=city,
            state=state,
            country=country,
            zip=zip,
        )

    def return_delivered_order_items(
        self, order_id: str, item_ids: list[str], payment_method_id: str
    ) -> str:
        """Request a delivered-item return after explicit confirmation.

        Args:
            order_id: Delivered order identifier.
            item_ids: Item IDs to return.
            payment_method_id: Original payment method or a gift card.

        Returns:
            Updated order state or a structured tool error.
        """

        return self._call_tool(
            "return_delivered_order_items",
            order_id=order_id,
            item_ids=item_ids,
            payment_method_id=payment_method_id,
        )

    def transfer_to_human_agents(self, summary: str) -> str:
        """Transfer an unsolvable or explicitly escalated request to a human.

        Args:
            summary: Concise issue summary for the receiving human agent.

        Returns:
            Transfer result.
        """

        return self._call_tool("transfer_to_human_agents", summary=summary)

    def _build_user(self) -> tuple[Any, Any]:
        if self._user_factory is not None:
            return self._user_factory(
                self._environment,
                self._task,
                deepcopy(self._user_dialogue),
                self._seed,
            )

        from tau2.runner.build import build_user

        model = os.environ.get("POLICYAGENT_USER_MODEL")
        if not model:
            raise RuntimeError(
                "POLICYAGENT_USER_MODEL is required for dynamic Retail user simulation"
            )
        raw_args = os.environ.get("POLICYAGENT_USER_LLM_ARGS_JSON", "{}")
        try:
            llm_args = json.loads(raw_args)
        except json.JSONDecodeError as exc:
            raise RuntimeError(
                "POLICYAGENT_USER_LLM_ARGS_JSON is invalid JSON"
            ) from exc
        user = build_user(
            "user_simulator",
            self._environment,
            self._task,
            llm=model,
            llm_args=llm_args,
            solo_mode=False,
        )
        user.set_seed(self._seed)
        state = user.get_init_state(message_history=deepcopy(self._user_dialogue))
        return user, state

    def _call_tool(self, name: str, **arguments: Any) -> str:
        self._require_ready()
        from tau2.data_model.message import AssistantMessage, ToolCall

        if self._tool_counter >= self._max_tool_calls:
            if "TOOL_CALL_LIMIT" not in self._runtime_stop_signals:
                self._runtime_stop_signals.append("TOOL_CALL_LIMIT")
            raise RuntimeError("Maximum Retail tool calls reached for this rollout")
        self._tool_counter += 1
        call = ToolCall(
            id=f"rl-tool-{self._tool_counter:04d}",
            name=name,
            arguments=arguments,
            requestor="assistant",
        )
        proposal = ToolProposal(id=call.id, name=name, arguments=arguments)
        guard = evaluate_retail_actions(
            [proposal], context_from_messages(self._messages)
        )
        for finding in guard.findings:
            payload = finding.to_dict()
            payload["tool_call_id"] = call.id
            self._policy_findings.append(payload)

        assistant = AssistantMessage(role="assistant", tool_calls=[call])
        result = self._environment.get_response(call)
        self._messages.extend([assistant, result])
        return str(result.content)

    def _require_ready(self) -> None:
        if self._environment is None or self._task is None:
            raise RuntimeError("reset() must be called before using the environment")
