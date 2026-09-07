from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import statistics
from collections import Counter
from pathlib import Path
from types import SimpleNamespace
from typing import Any


SCHEMA_VERSION = "retail-staged-reward-shadow-v1"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _strict_contains(text: Any, value: Any) -> bool:
    haystack = str(text or "")
    needle = str(value or "").strip()
    if not needle:
        return False
    return bool(
        re.search(
            rf"(?<![A-Za-z0-9]){re.escape(needle)}(?![A-Za-z0-9])",
            haystack,
            flags=re.IGNORECASE,
        )
    )


def _grounded_value_contains(text: Any, value: Any) -> bool:
    """Match communicated values while tolerating serialized float noise.

    Ordinary strings keep the strict token-boundary behavior.  A value that is
    itself a decimal number may also match a numeric literal whose difference
    is below half of the value's displayed least-significant decimal place.
    This covers tool JSON such as ``17.98999999999998`` for the communicated
    amount ``17.99`` without turning unrelated natural-language text into a
    fuzzy match.
    """

    if _strict_contains(text, value):
        return True
    needle = str(value or "").strip()
    numeric_match = re.fullmatch(r"[-+]?(?:\d+(?:\.\d+)?|\.\d+)", needle)
    if numeric_match is None:
        return False
    decimals = len(needle.partition(".")[2])
    tolerance = 0.5 * (10.0 ** -decimals) if decimals else 0.0
    expected = float(needle)
    for candidate in re.findall(
        r"(?<![A-Za-z0-9])[-+]?(?:\d+(?:\.\d+)?|\.\d+)(?![A-Za-z0-9])",
        str(text or ""),
    ):
        actual = float(candidate)
        if math.isclose(actual, expected, rel_tol=0.0, abs_tol=tolerance):
            return True
    return False


def _canonical_arguments(arguments: dict[str, Any]) -> str:
    return json.dumps(
        arguments,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _no_verified_write_cap(reward_spec: dict[str, Any]) -> float:
    if "no_verified_write_cap" in reward_spec:
        return float(reward_spec["no_verified_write_cap"])
    if "no_required_action_match_cap" in reward_spec:
        return float(reward_spec["no_required_action_match_cap"])
    return float(reward_spec["no_correct_write_cap"])


def _successful_exact_call(
    trace: list[dict[str, Any]], expected: dict[str, Any]
) -> bool:
    expected_args = _canonical_arguments(dict(expected["arguments"]))
    return any(
        str(call.get("name")) == str(expected["name"])
        and _canonical_arguments(dict(call.get("arguments") or {})) == expected_args
        and not bool((call.get("result") or {}).get("error"))
        for call in trace
    )


def _successful_json_payload(
    trace: list[dict[str, Any]], source_call: dict[str, Any]
) -> tuple[dict[str, Any] | None, str | None]:
    expected_args = _canonical_arguments(dict(source_call["arguments"]))
    matches = [
        call
        for call in trace
        if str(call.get("name")) == str(source_call["name"])
        and _canonical_arguments(dict(call.get("arguments") or {})) == expected_args
        and not bool((call.get("result") or {}).get("error"))
    ]
    if not matches:
        return None, "required successful source call is missing"
    try:
        payload = json.loads(str((matches[-1].get("result") or {}).get("content") or ""))
    except json.JSONDecodeError:
        return None, "required source result is not valid JSON"
    if not isinstance(payload, dict):
        return None, "required source result is not a JSON object"
    return payload, None


def _lookup_path(value: Any, path: list[Any]) -> tuple[bool, Any]:
    current = value
    for part in path:
        if isinstance(current, dict) and isinstance(part, str) and part in current:
            current = current[part]
        elif (
            isinstance(current, list)
            and isinstance(part, int)
            and 0 <= part < len(current)
        ):
            current = current[part]
        else:
            return False, None
    return True, current


def _pattern_matches(text: str, patterns: list[Any]) -> list[dict[str, Any]]:
    matches = []
    for raw_pattern in patterns:
        if not isinstance(raw_pattern, str) or not raw_pattern:
            raise ValueError("Claim-evidence trigger patterns must be non-empty strings")
        for match in re.finditer(raw_pattern, text):
            matches.append(
                {
                    "pattern": raw_pattern,
                    "matched_text": match.group(0)[:500],
                }
            )
    return matches


def _claim_evidence_diagnostic(
    messages: list[dict[str, Any]],
    trace: list[dict[str, Any]],
    task_spec: dict[str, Any],
) -> dict[str, Any] | None:
    """Detect only configured, high-confidence claim/evidence conflicts.

    This is a deterministic development diagnostic.  A PASS means that no
    configured rule fired; it is not proof that every natural-language claim
    is grounded.  The caller determines whether a configured reward mode uses it.
    """

    rules = list(task_spec.get("claim_evidence_rules") or [])
    if not rules:
        return None
    assistant_text = "\n".join(
        str(message.get("content") or "")
        for message in messages
        if isinstance(message, dict) and message.get("role") == "assistant"
    )
    tool_text = "\n".join(
        str((call.get("result") or {}).get("content") or "")
        for call in trace
        if not bool((call.get("result") or {}).get("error"))
    )
    findings: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    seen_rule_ids: set[str] = set()
    for raw_rule in rules:
        if not isinstance(raw_rule, dict):
            raise ValueError("Claim-evidence rules must be objects")
        rule = dict(raw_rule)
        rule_id = rule.get("rule_id")
        if not isinstance(rule_id, str) or not rule_id or rule_id in seen_rule_ids:
            raise ValueError("Claim-evidence rule IDs must be unique non-empty strings")
        seen_rule_ids.add(rule_id)
        verdict = str(rule.get("verdict") or "")
        if verdict not in {"FAIL", "REVIEW"}:
            raise ValueError("Claim-evidence rule verdict must be FAIL or REVIEW")
        if rule.get("rule_type") in {"refund_timing_by_payment_v1", "refund_timing_by_payment_v2"}:
            from src.evaluation.refund_timing import refund_timing_diagnostic

            diagnostic = refund_timing_diagnostic(messages, trace, rule)
            base = {"rule_id": rule_id, "rule_type": rule["rule_type"],
                    "configured_verdict": verdict, "claim_checks": diagnostic["findings"]}
            if diagnostic["errors"]:
                errors.extend({**base, **error} for error in diagnostic["errors"])
            elif diagnostic["verdict"] != "PASS":
                findings.append({**base, "verdict": diagnostic["verdict"]})
            continue
        # Exclude only explicitly specified local spans, not the entire message:
        # a valid gift-card conditional must not hide a separate false promise.
        trigger_text = assistant_text
        exclusion_patterns = list(rule.get("trigger_exclusion_patterns") or [])
        excluded_triggers = _pattern_matches(trigger_text, exclusion_patterns)
        for pattern in exclusion_patterns:
            trigger_text = re.sub(pattern, lambda match: " " * len(match.group(0)), trigger_text)
        triggers = _pattern_matches(
            trigger_text,
            list(rule.get("trigger_patterns") or []),
        )
        if not triggers:
            continue
        rule_type = str(rule.get("rule_type") or "")
        base = {
            "rule_id": rule_id,
            "rule_type": rule_type,
            "configured_verdict": verdict,
            "trigger_matches": triggers,
            "excluded_trigger_spans": excluded_triggers,
        }
        if rule_type == "denial_conflicts_with_nonempty_path":
            payload, error = _successful_json_payload(
                trace, dict(rule.get("source_call") or {})
            )
            if error:
                errors.append({**base, "error": error})
                continue
            path = list(rule.get("source_path") or [])
            found, observed = _lookup_path(payload, path)
            if not found:
                errors.append({**base, "error": "configured source path is missing"})
                continue
            if observed:
                findings.append(
                    {
                        **base,
                        "verdict": verdict,
                        "source_path": path,
                        "observed_evidence": observed,
                    }
                )
        elif rule_type == "denial_conflicts_with_matching_collection_item":
            payload, error = _successful_json_payload(
                trace, dict(rule.get("source_call") or {})
            )
            if error:
                errors.append({**base, "error": error})
                continue
            collection_path = list(rule.get("collection_path") or [])
            found, collection = _lookup_path(payload, collection_path)
            if not found or not isinstance(collection, (dict, list)):
                errors.append(
                    {**base, "error": "configured source collection is missing"}
                )
                continue
            items = list(collection.values()) if isinstance(collection, dict) else collection
            matching_items = []
            for item in items:
                option_found, option = _lookup_path(
                    item, list(rule.get("option_path") or [])
                )
                availability_found, availability = _lookup_path(
                    item, list(rule.get("availability_path") or [])
                )
                if (
                    option_found
                    and availability_found
                    and option == rule.get("expected_option")
                    and availability == rule.get("expected_availability")
                ):
                    matching_items.append(item)
            if matching_items:
                findings.append(
                    {
                        **base,
                        "verdict": verdict,
                        "collection_path": collection_path,
                        "matching_item_count": len(matching_items),
                    }
                )
        elif rule_type == "unsupported_literal":
            literals = []
            for extraction in _pattern_matches(
                assistant_text,
                [rule.get("extract_pattern")],
            ):
                literals.append(extraction["matched_text"])
            unsupported = [
                literal
                for literal in literals
                if literal.casefold() not in tool_text.casefold()
            ]
            if unsupported:
                findings.append(
                    {
                        **base,
                        "verdict": verdict,
                        "unsupported_literals": unsupported,
                    }
                )
        elif rule_type == "pattern_without_evidence_terms":
            payload, error = _successful_json_payload(
                trace, dict(rule.get("source_call") or {})
            )
            if error:
                errors.append({**base, "error": error})
                continue
            source_text = json.dumps(payload, ensure_ascii=False).casefold()
            support_terms = list(rule.get("support_terms") or [])
            if not support_terms or any(
                not isinstance(term, str) or not term for term in support_terms
            ):
                raise ValueError("Claim-evidence support terms must be non-empty strings")
            if not any(term.casefold() in source_text for term in support_terms):
                findings.append(
                    {
                        **base,
                        "verdict": verdict,
                        "support_terms_checked": support_terms,
                    }
                )
        else:
            raise ValueError(f"Unsupported claim-evidence rule type: {rule_type}")
    if errors:
        aggregate = "ERROR"
    elif any(item["verdict"] == "FAIL" for item in findings):
        aggregate = "FAIL"
    elif findings:
        aggregate = "REVIEW"
    else:
        aggregate = "PASS"
    return {
        "verdict": aggregate,
        "configured_rule_count": len(rules),
        "triggered_rule_count": len(findings),
        "findings": findings,
        "errors": errors,
        "used_as_reward": False,
        "pass_semantics": "No configured conflict was detected; not all claims were verified.",
    }


def _identity_link(
    trace: list[dict[str, Any]], task_spec: dict[str, Any]
) -> dict[str, Any]:
    identity_spec = dict(task_spec["identity_link"])
    required_user_id = str(identity_spec["required_user_id"])
    required_orders = {str(value) for value in identity_spec["required_order_ids"]}
    successful_find_index: int | None = None
    successful_profile_index: int | None = None
    for index, call in enumerate(trace):
        result = dict(call.get("result") or {})
        if bool(result.get("error")):
            continue
        if (
            str(call.get("name"))
            in {"find_user_id_by_email", "find_user_id_by_name_zip"}
            and str(result.get("content") or "").strip() == required_user_id
        ):
            successful_find_index = index
            continue
        if (
            successful_find_index is not None
            and index > successful_find_index
            and str(call.get("name")) == "get_user_details"
            and str((call.get("arguments") or {}).get("user_id")) == required_user_id
        ):
            try:
                profile = json.loads(str(result.get("content") or ""))
            except json.JSONDecodeError:
                continue
            profile_orders = {str(value) for value in profile.get("orders") or []}
            if required_orders.issubset(profile_orders):
                successful_profile_index = index
                break
    complete = successful_profile_index is not None
    return {
        "value": 1.0 if complete else 0.0,
        "complete": complete,
        "find_call_index": successful_find_index,
        "profile_call_index": successful_profile_index,
    }


def _target_evidence(
    trace: list[dict[str, Any]], task_spec: dict[str, Any]
) -> dict[str, Any]:
    checks = []
    for expected in task_spec["target_evidence_calls"]:
        matched = _successful_exact_call(trace, expected)
        checks.append(
            {
                "evidence_id": str(expected["evidence_id"]),
                "matched": matched,
            }
        )
    value = sum(float(check["matched"]) for check in checks) / len(checks)
    return {"value": value, "checks": checks}


def _required_write_progress(
    messages: list[dict[str, Any]],
    terminal: dict[str, Any],
    task_spec: dict[str, Any],
) -> dict[str, Any]:
    """Bind benchmark action matches to unique, later non-error tool results.

    The benchmark match establishes that call arguments satisfy a required action.
    The serialized message trace independently establishes that the selected call
    actually received a usable tool result.  Neither signal proves final-state or
    policy correctness.
    """

    required_ids = {str(value) for value in task_spec["required_write_action_ids"]}
    match_rows = (terminal.get("action_progress") or {}).get("matches") or []
    if any(not isinstance(row, dict) for row in match_rows):
        raise ValueError("Required write action matches are malformed")
    matches = {str(row.get("action_id")): row for row in match_rows}
    if len(matches) != len(match_rows):
        raise ValueError("Required write action IDs must be unique")
    missing_ids = sorted(required_ids - set(matches))
    if missing_ids:
        raise ValueError(f"Required write action IDs absent from evidence: {missing_ids}")

    calls: list[tuple[int, dict[str, Any]]] = []
    call_id_counts: Counter[str] = Counter()
    results: dict[str, list[tuple[int, dict[str, Any]]]] = {}
    for position, message in enumerate(messages):
        if not isinstance(message, dict):
            raise ValueError("Serialized message evidence is malformed")
        if message.get("role") in {"assistant", "user"}:
            tool_calls = message.get("tool_calls") or []
            if not isinstance(tool_calls, list):
                raise ValueError("Serialized tool-call evidence is malformed")
            for tool_call in tool_calls:
                if not isinstance(tool_call, dict):
                    raise ValueError("Serialized tool-call evidence is malformed")
                call_id = tool_call.get("id")
                if isinstance(call_id, str) and call_id:
                    call_id_counts[call_id] += 1
                if (
                    message.get("role") == "assistant"
                    and tool_call.get("requestor", "assistant") == "assistant"
                ):
                    calls.append((position, tool_call))
        elif message.get("role") == "tool":
            call_id = message.get("id")
            if isinstance(call_id, str) and call_id:
                results.setdefault(call_id, []).append((position, message))

    checks = []
    selected_call_indices: list[int] = []
    for action_id in sorted(required_ids):
        match = matches[action_id]
        if match.get("matched") is not True:
            checks.append(
                {
                    "action_id": action_id,
                    "status": "FAIL",
                    "reason": "NO_BENCHMARK_ACTION_MATCH",
                    "matched_call_index": None,
                    "call_id": None,
                }
            )
            continue
        index = match.get("matched_call_index")
        if type(index) is not int or not 0 <= index < len(calls):
            raise ValueError(f"Invalid matched call index for action {action_id}")
        if index in selected_call_indices:
            raise ValueError("One tool call was bound to multiple required actions")
        selected_call_indices.append(index)
        call_position, tool_call = calls[index]
        if tool_call.get("name") != match.get("name"):
            raise ValueError(f"Matched tool name differs for action {action_id}")
        call_id = tool_call.get("id")
        if (
            not isinstance(call_id, str)
            or not call_id
            or call_id_counts[call_id] != 1
            or len(results.get(call_id, [])) != 1
        ):
            raise ValueError(f"Tool result binding is ambiguous for action {action_id}")
        result_position, result = results[call_id][0]
        if (
            result_position <= call_position
            or result.get("requestor", "assistant") != "assistant"
            or type(result.get("error")) is not bool
            or (
                result.get("name") is not None
                and result.get("name") != tool_call.get("name")
            )
            or not isinstance(result.get("content"), str)
            or not result["content"].strip()
        ):
            raise ValueError(f"Tool result evidence is invalid for action {action_id}")
        status = "FAIL" if result["error"] else "PASS"
        checks.append(
            {
                "action_id": action_id,
                "status": status,
                "reason": (
                    "TOOL_REPORTED_ERROR"
                    if result["error"]
                    else "BOUND_NONERROR_RESPONSE"
                ),
                "matched_call_index": index,
                "call_id": call_id,
            }
        )

    verified_ids = sorted(
        row["action_id"] for row in checks if row["status"] == "PASS"
    )
    return {
        "value": len(verified_ids) / len(required_ids),
        "required_action_ids": sorted(required_ids),
        "verified_action_ids": verified_ids,
        "checks": checks,
        "final_state_verified": False,
        "policy_verified": False,
    }


def _post_write_response(
    messages: list[dict[str, Any]], write: dict[str, Any]
) -> dict[str, Any]:
    """Require a non-empty assistant response after the verified write call."""

    call_ids = {
        str(check.get("call_id"))
        for check in list(write.get("checks") or [])
        if check.get("status") == "PASS" and check.get("call_id")
    }
    write_indices = []
    result_indices = []
    for index, message in enumerate(messages):
        if message.get("role") == "assistant" and any(
                str(call.get("id")) in call_ids
                for call in list(message.get("tool_calls") or [])
                if isinstance(call, dict)
            ):
                write_indices.append(index)
        if message.get("role") == "tool" and str(message.get("id")) in call_ids:
            result_indices.append(index)
    if not write_indices or not result_indices:
        return {
            "value": 0.0,
            "write_message_index": None,
            "result_message_index": None,
            "response_message_index": None,
        }
    last_write_index = max(write_indices)
    last_result_index = max(result_indices)
    response_index = next(
        (
            index
            for index, message in enumerate(
                messages[last_result_index + 1 :], start=last_result_index + 1
            )
            if message.get("role") == "assistant"
            and str(message.get("content") or "").strip()
        ),
        None,
    )
    return {
        "value": 1.0 if response_index is not None else 0.0,
        "write_message_index": last_write_index,
        "result_message_index": last_result_index,
        "response_message_index": response_index,
    }


def _successful_transfer(trace: list[dict[str, Any]]) -> bool:
    return any(
        str(call.get("name")) == "transfer_to_human_agents"
        and not bool((call.get("result") or {}).get("error"))
        for call in trace
    )


def _grounded_communication(
    messages: list[dict[str, Any]], terminal: dict[str, Any]
) -> dict[str, Any] | None:
    checks = (
        (((terminal.get("tau2") or {}).get("communication") or {}).get(
            "communicate_checks"
        ))
        or []
    )
    if not checks:
        return None
    results = []
    for check in checks:
        info = str(check.get("info") or "")
        observed = False
        grounded = False
        for message in messages:
            role = str(message.get("role") or "")
            if role == "tool" and not bool(message.get("error")):
                observed = observed or _grounded_value_contains(
                    message.get("content"), info
                )
            elif role == "assistant" and observed:
                grounded = grounded or _grounded_value_contains(
                    message.get("content"), info
                )
        results.append({"info": info, "observed": observed, "grounded": grounded})
    return {
        "value": sum(float(row["grounded"]) for row in results) / len(results),
        "checks": results,
    }


def _repeat_count(trace: list[dict[str, Any]]) -> int:
    signatures = Counter(
        (
            str(call.get("name")),
            _canonical_arguments(dict(call.get("arguments") or {})),
        )
        for call in trace
    )
    return sum(max(0, count - 1) for count in signatures.values())


def score_rollout(
    raw: dict[str, Any],
    evidence: dict[str, Any],
    spec: dict[str, Any],
    confirmation_diagnostic: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if str(raw.get("task_id")) != str(evidence.get("task_id")):
        raise ValueError("Raw rollout and evidence task IDs differ")
    expected_evidence_sha = str(raw.get("evidence_sha256") or "").upper()
    actual_evidence_sha = str(evidence.get("evidence_sha256") or "").upper()
    if expected_evidence_sha != actual_evidence_sha:
        raise ValueError("Raw rollout and evidence SHA binding differs")

    task_id = str(raw["task_id"])
    task_spec = dict(spec["tasks"][task_id])
    reward_spec = dict(spec["reward"])
    trace = list(evidence.get("tool_trace") or [])
    terminal = dict(evidence.get("terminal_evaluator") or {})

    evidence_version = spec.get("evidence_rules_version")
    if evidence_version not in {None, "task44_evidence_v2"}:
        raise ValueError("Unsupported reward evidence rules version")
    if evidence_version == "task44_evidence_v2":
        if task_id != "44":
            raise ValueError("task44_evidence_v2 is scoped to Task44")
        from src.evaluation.task44_reward_evidence import (
            identity_evidence, confirmation_evidence,
        )
        identity = identity_evidence(list(raw.get("messages") or []), task_spec["identity_link"])
        confirmation_diagnostic = confirmation_evidence(list(raw.get("messages") or []))
    else:
        identity = _identity_link(trace, task_spec)
    target = _target_evidence(trace, task_spec)
    messages = list(raw.get("messages") or [])
    write = _required_write_progress(messages, terminal, task_spec)
    communication = _grounded_communication(
        messages, terminal
    )
    claim_evidence = _claim_evidence_diagnostic(messages, trace, task_spec)
    terminal_success = bool(terminal.get("user_stopped")) and math.isclose(
        float(terminal.get("reward") or 0.0), 1.0, abs_tol=1e-12
    )

    raw_component_values: dict[str, float] = {
        "identity_link": float(identity["value"]),
        "target_evidence": float(target["value"]),
        "required_write_progress": float(write["value"]),
    }
    if communication is not None:
        raw_component_values["grounded_communication"] = float(
            communication["value"]
        )
    identity_gate = raw_component_values["identity_link"]
    component_values = {
        name: (
            value
            if name == "identity_link"
            else value * identity_gate
        )
        for name, value in raw_component_values.items()
    }
    configured_weights = dict(reward_spec["nonterminal_component_weights"])
    active_weight_sum = sum(configured_weights[name] for name in component_values)
    normalized_total = float(reward_spec["normalize_active_nonterminal_weights_to"])
    normalized_weights = {
        name: float(configured_weights[name]) / active_weight_sum * normalized_total
        for name in component_values
    }
    process_score = sum(
        normalized_weights[name] * value
        for name, value in component_values.items()
    )
    if float(write["value"]) == 0.0:
        process_score = min(
            process_score,
            _no_verified_write_cap(reward_spec),
        )

    tool_error_count = sum(
        bool((call.get("result") or {}).get("error")) for call in trace
    )
    repeat_count = _repeat_count(trace)
    unexpected_write_count = int(
        ((terminal.get("action_progress") or {}).get("unexpected_write_count") or 0)
    )
    completion = dict(evidence.get("completion") or {})
    limit_reached = bool(completion.get("customer_turn_limit_reached")) or bool(
        completion.get("tool_call_limit_reached")
    )
    penalties = {
        "tool_error": min(
            float(reward_spec["tool_error_penalty_cap"]),
            tool_error_count * float(reward_spec["tool_error_penalty_each"]),
        ),
        "repeated_call": min(
            float(reward_spec["repeated_call_penalty_cap"]),
            repeat_count * float(reward_spec["repeated_call_penalty_each"]),
        ),
        "limit_reached": float(reward_spec["limit_reached_penalty"])
        if limit_reached
        else 0.0,
    }

    communication_complete = communication is None or math.isclose(
        float(communication["value"]), 1.0, abs_tol=1e-12
    )
    write_complete = math.isclose(float(write["value"]), 1.0, abs_tol=1e-12)
    complete_success = (
        terminal_success
        and bool(identity_gate)
        and write_complete
        and communication_complete
    )
    terminal_incomplete_communication = (
        terminal_success and bool(identity_gate) and not communication_complete
    )
    composition_mode = str(reward_spec.get("composition_mode") or "")
    additive_components: dict[str, float] | None = None
    confirmation_component: dict[str, Any] | None = None
    authorization_applicable = False
    claim_evidence_cap_applied = False
    if composition_mode in {
        "additive_terminal_process_v3",
        "additive_terminal_process_confirmation_v4",
        "hierarchical_state_authorization_v5",
        "hierarchical_state_authorization_review_v6",
        "hierarchical_state_authorization_claim_v7",
    }:
        additive_weights = {
            name: float(value)
            for name, value in dict(reward_spec["additive_component_weights"]).items()
        }
        required_weight_names = (
            {
                "environment_state",
                "interaction_complete",
                "identity_link",
                "target_evidence",
                "authorized_write",
                "post_write_communication",
            }
            if composition_mode == "hierarchical_state_authorization_v5"
            else {
                "environment_state",
                "interaction_complete",
                "identity_link",
                "target_evidence",
                "write_authorization",
                "post_write_communication",
            }
            if composition_mode
            in {
                "hierarchical_state_authorization_review_v6",
                "hierarchical_state_authorization_claim_v7",
            }
            else {
                "terminal_environment",
                "identity_link",
                "target_evidence",
                "required_write_progress",
                "grounded_communication",
                *(
                    {"confirmation_binding"}
                    if composition_mode == "additive_terminal_process_confirmation_v4"
                    else set()
                ),
            }
        )
        if set(additive_weights) != required_weight_names:
            raise ValueError(
                f"{composition_mode} requires exactly the declared component weights"
            )
        if any(value < 0.0 for value in additive_weights.values()):
            raise ValueError("Additive reward weights must be non-negative")
        if not math.isclose(sum(additive_weights.values()), 1.0, abs_tol=1e-12):
            raise ValueError("Additive reward weights must sum to 1.0")
        if composition_mode == "hierarchical_state_authorization_v5":
            additive_components = {}
        else:
            additive_components = {
                "terminal_environment": float(terminal_success),
                "identity_link": component_values["identity_link"],
                "target_evidence": component_values["target_evidence"],
                "required_write_progress": component_values["required_write_progress"],
                "grounded_communication": (
                    component_values["grounded_communication"]
                    if "grounded_communication" in component_values
                    else 1.0
                ),
            }
        if composition_mode in {
            "additive_terminal_process_confirmation_v4",
            "hierarchical_state_authorization_v5",
            "hierarchical_state_authorization_review_v6",
            "hierarchical_state_authorization_claim_v7",
        }:
            if not isinstance(confirmation_diagnostic, dict):
                raise ValueError(
                    "additive_terminal_process_confirmation_v4 requires bound "
                    "confirmation diagnostics"
                )
            checks = list(confirmation_diagnostic.get("checks") or [])
            scope = reward_spec.get("authorization_scope", "complete_required_writes")
            if scope not in {"complete_required_writes", "every_observed_write"}:
                raise ValueError("Unknown authorization_scope")
            authorization_applicable = write_complete or (
                scope == "every_observed_write"
                and int(confirmation_diagnostic.get("write_count") or 0) > 0
            )
            confirmation_passed = bool(authorization_applicable and checks) and all(
                item.get("confirmed") is True
                and (item.get("parameter_binding") or {}).get("verdict") == "PASS"
                for item in checks
            )
            if not authorization_applicable:
                confirmation_verdict = "NOT_APPLICABLE"
            elif not checks or any(item.get("confirmed") is not True for item in checks):
                confirmation_verdict = "FAIL"
            elif any(
                (item.get("parameter_binding") or {}).get("verdict") == "FAIL"
                for item in checks
            ):
                confirmation_verdict = "FAIL"
            elif confirmation_passed:
                confirmation_verdict = "PASS"
            else:
                confirmation_verdict = "REVIEW"
            if evidence_version == "task44_evidence_v2" and write_complete:
                verdicts = [item["verified_verdict"] for item in checks]
                confirmation_verdict = (
                    "FAIL" if not verdicts or "FAIL" in verdicts
                    else "PASS" if all(v == "PASS" for v in verdicts) else "REVIEW"
                )
            confirmation_value = (
                1.0
                if confirmation_verdict == "PASS"
                else float(reward_spec.get("authorization_review_value", 0.0))
                if confirmation_verdict == "REVIEW"
                else 0.0
            )
            confirmation_component = {
                "value": confirmation_value,
                "verdict": confirmation_verdict,
                "write_complete": write_complete,
                "write_count": int(confirmation_diagnostic.get("write_count") or 0),
                "confirmed_write_count": int(
                    confirmation_diagnostic.get("confirmed_write_count") or 0
                ),
                "all_parameter_bindings_pass": bool(checks)
                and all(
                    (item.get("parameter_binding") or {}).get("verdict") == "PASS"
                    for item in checks
                ),
                "diagnostic_version": confirmation_diagnostic.get(
                    "diagnostic_version"
                ),
                "used_as_reward": True,
            }
            if composition_mode == "additive_terminal_process_confirmation_v4":
                additive_components["confirmation_binding"] = float(
                    confirmation_component["value"]
                )
        if composition_mode in {
            "hierarchical_state_authorization_v5",
            "hierarchical_state_authorization_review_v6",
            "hierarchical_state_authorization_claim_v7",
        }:
            tau2_environment = dict((terminal.get("tau2") or {}).get("environment") or {})
            environment_state = 1.0 if math.isclose(
                float(tau2_environment.get("reward") or 0.0),
                1.0,
                abs_tol=1e-12,
            ) else 0.0
            authorized_write = float(
                write_complete
                and bool(identity_gate)
                and confirmation_component is not None
                and math.isclose(
                    float(confirmation_component["value"]), 1.0, abs_tol=1e-12
                )
            )
            post_write = _post_write_response(messages, write)
            if composition_mode == "hierarchical_state_authorization_v5":
                post_write_communication = float(
                    bool(authorized_write)
                    and communication_complete
                    and math.isclose(float(post_write["value"]), 1.0, abs_tol=1e-12)
                )
                additive_components = {
                    "environment_state": environment_state,
                    "interaction_complete": float(bool(terminal.get("user_stopped"))),
                    "identity_link": component_values["identity_link"],
                    "target_evidence": component_values["target_evidence"],
                    "authorized_write": authorized_write,
                    "post_write_communication": post_write_communication,
                }
            else:
                write_authorization = float(
                    write_complete
                    and bool(identity_gate)
                    and confirmation_component is not None
                ) * float(confirmation_component["value"])
                post_write_communication = float(
                    write_complete
                    and communication_complete
                    and math.isclose(float(post_write["value"]), 1.0, abs_tol=1e-12)
                )
                additive_components = {
                    "environment_state": environment_state,
                    "interaction_complete": float(bool(terminal.get("user_stopped"))),
                    "identity_link": component_values["identity_link"],
                    "target_evidence": component_values["target_evidence"],
                    "write_authorization": write_authorization,
                    "post_write_communication": post_write_communication,
                }
        unpenalized = sum(
            additive_weights[name] * value
            for name, value in additive_components.items()
        )
        if composition_mode in {
            "hierarchical_state_authorization_v5",
            "hierarchical_state_authorization_review_v6",
            "hierarchical_state_authorization_claim_v7",
        }:
            normalized_weights = additive_weights
            process_score = unpenalized
        if float(write["value"]) == 0.0:
            unpenalized = min(unpenalized, _no_verified_write_cap(reward_spec))
        if (
            composition_mode == "hierarchical_state_authorization_v5"
            and write_complete
            and not bool(additive_components["authorized_write"])
        ):
            unpenalized = min(
                unpenalized,
                float(reward_spec["unauthorized_write_hard_cap"]),
            )
        if composition_mode in {
            "hierarchical_state_authorization_review_v6",
            "hierarchical_state_authorization_claim_v7",
        }:
            confirmation_verdict = str(
                (confirmation_component or {}).get("verdict") or "FAIL"
            )
            if authorization_applicable and (
                not bool(identity_gate) or confirmation_verdict == "FAIL"
            ):
                unpenalized = min(
                    unpenalized,
                    float(reward_spec["authorization_fail_hard_cap"]),
                )
            elif authorization_applicable and confirmation_verdict == "REVIEW":
                unpenalized = min(
                    unpenalized,
                    float(reward_spec["authorization_review_cap"]),
                )
        if composition_mode == "hierarchical_state_authorization_claim_v7":
            if not isinstance(claim_evidence, dict):
                raise ValueError(
                    "hierarchical_state_authorization_claim_v7 requires "
                    "claim-evidence diagnostics"
                )
            claim_evidence = dict(claim_evidence)
            claim_evidence["used_as_reward"] = True
            claim_verdict = str(claim_evidence.get("verdict") or "ERROR")
            if claim_verdict in {"FAIL", "ERROR"}:
                claim_cap = float(reward_spec["claim_evidence_fail_cap"])
                claim_evidence_cap_applied = unpenalized > claim_cap
                unpenalized = min(unpenalized, claim_cap)
            elif claim_verdict == "REVIEW":
                claim_cap = float(reward_spec["claim_evidence_review_cap"])
                claim_evidence_cap_applied = unpenalized > claim_cap
                unpenalized = min(unpenalized, claim_cap)
        if terminal_incomplete_communication:
            unpenalized = min(
                unpenalized,
                float(reward_spec["terminal_incomplete_communication_cap"]),
            )
        total_penalty = sum(penalties.values())
        terminal_override_applied = False
    elif "terminal_complete_success_score" in reward_spec:
        if complete_success:
            unpenalized = float(reward_spec["terminal_complete_success_score"])
            total_penalty = min(
                sum(penalties.values()),
                float(reward_spec["complete_success_efficiency_penalty_cap"]),
            )
        else:
            unpenalized = process_score
            if terminal_incomplete_communication:
                unpenalized = min(
                    unpenalized,
                    float(reward_spec["terminal_incomplete_communication_cap"]),
                )
            total_penalty = sum(penalties.values())
        terminal_override_applied = complete_success
    else:
        terminal_override_applied = terminal_success and bool(identity_gate)
        unpenalized = (
            float(reward_spec["terminal_success_override"])
            if terminal_override_applied
            else process_score
        )
        total_penalty = sum(penalties.values())
    reward = unpenalized - total_penalty
    transfer_rule = task_spec.get("premature_transfer_rule") or {}
    premature_transfer = False
    if transfer_rule:
        target_complete = math.isclose(float(target["value"]), 1.0, abs_tol=1e-12)
        premature_transfer = (
            bool(transfer_rule.get("enabled"))
            and target_complete
            and not write_complete
            and tool_error_count == 0
            and _successful_transfer(trace)
        )
        if premature_transfer:
            reward = min(reward, float(transfer_rule["hard_cap"]))
    if unexpected_write_count:
        reward = min(reward, float(reward_spec["unexpected_write_hard_cap"]))
    reward = min(
        float(reward_spec["maximum"]),
        max(float(reward_spec["minimum"]), reward),
    )
    return {
        "task_id": task_id,
        "terminal_reward": float(terminal.get("reward") or 0.0),
        "staged_reward": round(reward, 12),
        "terminal_success": terminal_success,
        "complete_success": complete_success,
        "communication_complete": communication_complete,
        "terminal_incomplete_communication": terminal_incomplete_communication,
        "terminal_override_applied": terminal_override_applied,
        "composition_mode": composition_mode or "legacy_terminal_override",
        "evidence_rules_version": evidence_version,
        "confirmation_evidence": confirmation_diagnostic if evidence_version else None,
        "additive_components": additive_components,
        "write_complete": write_complete,
        "premature_transfer": premature_transfer,
        "components": {
            "identity_link": identity,
            "target_evidence": target,
            "required_write_progress": write,
            "grounded_communication": communication,
            "confirmation_binding": confirmation_component,
            "post_write_response": (
                _post_write_response(messages, write)
                if composition_mode
                in {
                    "hierarchical_state_authorization_v5",
                    "hierarchical_state_authorization_review_v6",
                    "hierarchical_state_authorization_claim_v7",
                }
                else None
            ),
            "claim_evidence_consistency": claim_evidence,
        },
        "effective_component_values": component_values,
        "normalized_weights": normalized_weights,
        "process_score_before_penalties": process_score,
        "penalties": penalties,
        "total_penalty_applied": total_penalty,
        "tool_error_count": tool_error_count,
        "repeated_call_count": repeat_count,
        "unexpected_write_count": unexpected_write_count,
        "limit_reached": limit_reached,
        "authorization_gate_applied": bool(
            composition_mode == "hierarchical_state_authorization_v5"
            and write_complete
            and not bool((additive_components or {}).get("authorized_write"))
        ),
        "authorization_review_cap_applied": bool(
            composition_mode
            in {
                "hierarchical_state_authorization_review_v6",
                "hierarchical_state_authorization_claim_v7",
            }
            and authorization_applicable
            and (confirmation_component or {}).get("verdict") == "REVIEW"
        ),
        "authorization_fail_cap_applied": bool(
            composition_mode
            in {
                "hierarchical_state_authorization_review_v6",
                "hierarchical_state_authorization_claim_v7",
            }
            and authorization_applicable
            and (
                not bool(identity_gate)
                or (confirmation_component or {}).get("verdict") == "FAIL"
            )
        ),
        "claim_evidence_cap_applied": claim_evidence_cap_applied,
        "no_verified_write_cap_applied": float(write["value"]) == 0.0,
    }


def _confirmation_diagnostic_from_serialized_messages(
    messages: list[dict[str, Any]],
) -> dict[str, Any]:
    from src.rl.retail_agentic_env import confirmation_diagnostics

    converted = []
    for message in messages:
        if not isinstance(message, dict):
            raise ValueError("Serialized confirmation message is malformed")
        converted.append(
            SimpleNamespace(
                **{
                    **message,
                    "tool_calls": [
                        SimpleNamespace(**call)
                        for call in list(message.get("tool_calls") or [])
                    ],
                }
            )
        )
    return confirmation_diagnostics(converted)


def _group_statistics(scores: list[float], num_generations: int) -> dict[str, Any]:
    if len(scores) % num_generations:
        raise ValueError("Score count is not divisible by num_generations")
    groups = [
        scores[start : start + num_generations]
        for start in range(0, len(scores), num_generations)
    ]
    kinds: Counter[str] = Counter()
    stds = []
    for group in groups:
        std = statistics.pstdev(group)
        stds.append(std)
        if std > 1e-12:
            kinds["mixed"] += 1
        elif all(math.isclose(value, 0.0, abs_tol=1e-12) for value in group):
            kinds["all_zero"] += 1
        else:
            kinds["uniform_nonzero"] += 1
    return {
        "groups": len(groups),
        "group_counts": {
            "mixed": kinds["mixed"],
            "all_zero": kinds["all_zero"],
            "uniform_nonzero": kinds["uniform_nonzero"],
        },
        "zero_reward_std_fraction": sum(std <= 1e-12 for std in stds) / len(stds),
        "unique_reward_values": len({round(value, 12) for value in scores}),
        "reward_mean": statistics.mean(scores),
        "reward_std": statistics.pstdev(scores),
    }


def _sampling_group_bindings(
    run_dir: Path,
    source: dict[str, Any],
    raw_rows: list[dict[str, Any]],
    evidence_rows: list[dict[str, Any]],
) -> tuple[Path | None, dict[int, dict[str, Any]], list[dict[str, Any]]]:
    expected_rollouts = int(source["expected_rollouts"])
    expected_groups = int(source["expected_groups"])
    num_generations = int(source["num_generations"])
    sampling_sha = source.get("sampling_groups_sha256")
    if sampling_sha is None:
        bindings = {
            index: {
                "group_id": str(index // num_generations),
                "group_index": index // num_generations,
                "generation_index": index % num_generations,
                "raw_row_index": index,
                "trajectory_transport_complete": True,
                "censored": False,
                "row_transport_eligible": True,
                "eligible_for_group_statistics": True,
                "group_eligible_for_statistics": True,
                "exclusion_reasons": [],
            }
            for index in range(expected_rollouts)
        }
        groups = [
            {
                "group_id": str(group_index),
                "group_index": group_index,
                "eligible_for_statistics": True,
            }
            for group_index in range(expected_groups)
        ]
        return None, bindings, groups

    sampling_path = run_dir / "sampling_groups.jsonl"
    if _sha256(sampling_path) != str(sampling_sha).upper():
        raise ValueError("Sampling-groups SHA-256 does not match frozen spec")
    sampling_rows = _read_jsonl(sampling_path)
    if len(sampling_rows) != expected_groups:
        raise ValueError("Sampling-group count does not match frozen spec")

    bindings: dict[int, dict[str, Any]] = {}
    groups: list[dict[str, Any]] = []
    seen_group_ids: set[str] = set()
    diagnostic_flags = (
        "completion_token_budget_exhausted",
        "context_limit_reached",
        "customer_turn_limit_reached",
        "tool_call_limit_reached",
        "tool_iteration_limit_reached",
        "unresolved_tool_call",
        "framework_loop_abnormal_end",
        "model_completion_truncated",
    )
    transport_invalid_flags = {
        "completion_token_budget_exhausted",
        "context_limit_reached",
        "tool_iteration_limit_reached",
        "unresolved_tool_call",
        "framework_loop_abnormal_end",
        "model_completion_truncated",
    }
    for group_index, group in enumerate(sampling_rows):
        group_id = str(group.get("group_id") or "").strip()
        if not group_id or group_id in seen_group_ids:
            raise ValueError("Sampling group IDs must be non-empty and unique")
        seen_group_ids.add(group_id)
        if str(group.get("status")) != "COMPLETED":
            raise ValueError(f"Sampling group {group_id} is not COMPLETED")
        diagnostics = list(group.get("diagnostics") or [])
        if len(diagnostics) != num_generations:
            raise ValueError(
                f"Sampling group {group_id} does not contain num_generations diagnostics"
            )
        candidate_indices = {row.get("candidate_index") for row in diagnostics}
        if candidate_indices != set(range(num_generations)):
            raise ValueError(
                f"Sampling group {group_id} candidate indices do not cover the group"
            )
        declared_complete = group.get("all_candidates_transport_complete")
        if not isinstance(declared_complete, bool):
            raise ValueError(
                f"Sampling group {group_id} has invalid transport-complete aggregate"
            )
        row_completeness = []
        group_task_id = str(group.get("task_id"))
        for diagnostic in diagnostics:
            raw_row_index = diagnostic.get("raw_row_index")
            if not isinstance(raw_row_index, int):
                raise ValueError("Sampling diagnostic raw_row_index must be an integer")
            if raw_row_index < 0 or raw_row_index >= expected_rollouts:
                raise ValueError("Sampling diagnostic raw_row_index is out of range")
            if raw_row_index in bindings:
                raise ValueError("Sampling diagnostic raw_row_index must be unique")
            transport_complete = diagnostic.get("trajectory_transport_complete")
            if not isinstance(transport_complete, bool):
                raise ValueError(
                    "Sampling diagnostic trajectory_transport_complete must be boolean"
                )
            if (
                str(raw_rows[raw_row_index].get("task_id")) != group_task_id
                or str(evidence_rows[raw_row_index].get("task_id")) != group_task_id
            ):
                raise ValueError("Sampling group, raw rollout, and evidence task IDs differ")
            true_flags = []
            for flag in diagnostic_flags:
                value = diagnostic.get(flag)
                if value is not None and not isinstance(value, bool):
                    raise ValueError(f"Sampling diagnostic {flag} must be boolean or null")
                if value is True:
                    true_flags.append(flag.upper())
            contradictory_flags = [
                flag
                for flag in transport_invalid_flags
                if diagnostic.get(flag) is True
            ]
            if transport_complete and contradictory_flags:
                raise ValueError(
                    "Sampling diagnostic marks transport complete while a transport-invalid "
                    f"flag is true: {sorted(contradictory_flags)}"
                )
            exclusion_reasons = []
            if not transport_complete:
                exclusion_reasons.append("TRANSPORT_INVALID")
                exclusion_reasons.extend(true_flags)
            row_completeness.append(transport_complete)
            bindings[raw_row_index] = {
                "group_id": group_id,
                "group_index": group_index,
                "generation_index": int(diagnostic["candidate_index"]),
                "raw_row_index": raw_row_index,
                "trajectory_transport_complete": transport_complete,
                "censored": not transport_complete,
                "row_transport_eligible": transport_complete,
                "eligible_for_group_statistics": False,
                "group_eligible_for_statistics": False,
                "exclusion_reasons": exclusion_reasons,
            }
        actual_complete = all(row_completeness)
        if declared_complete != actual_complete:
            raise ValueError(
                f"Sampling group {group_id} transport aggregate disagrees with diagnostics"
            )
        for diagnostic in diagnostics:
            binding = bindings[int(diagnostic["raw_row_index"])]
            binding["eligible_for_group_statistics"] = actual_complete
            binding["group_eligible_for_statistics"] = actual_complete
            if binding["row_transport_eligible"] and not actual_complete:
                binding["exclusion_reasons"].append(
                    "PEER_TRANSPORT_INVALID_GROUP_EXCLUDED"
                )
        groups.append(
            {
                "group_id": group_id,
                "group_index": group_index,
                "eligible_for_statistics": actual_complete,
            }
        )
    if set(bindings) != set(range(expected_rollouts)):
        raise ValueError("Sampling diagnostics do not cover every raw rollout exactly once")
    return sampling_path, bindings, groups


def _eligible_group_statistics(
    trajectories: list[dict[str, Any]],
    groups: list[dict[str, Any]],
    score_field: str,
) -> dict[str, Any]:
    eligible_groups = [row for row in groups if row["eligible_for_statistics"]]
    excluded_groups = [row for row in groups if not row["eligible_for_statistics"]]
    grouped_scores: list[list[float]] = []
    for group in eligible_groups:
        scores = [
            float(row[score_field])
            for row in sorted(
                (
                    row
                    for row in trajectories
                    if row["group_index"] == group["group_index"]
                ),
                key=lambda row: row["generation_index"],
            )
        ]
        grouped_scores.append(scores)
    flat_scores = [score for group in grouped_scores for score in group]
    kinds: Counter[str] = Counter()
    stds = []
    for group in grouped_scores:
        std = statistics.pstdev(group)
        stds.append(std)
        if std > 1e-12:
            kinds["mixed"] += 1
        elif all(math.isclose(value, 0.0, abs_tol=1e-12) for value in group):
            kinds["all_zero"] += 1
        else:
            kinds["uniform_nonzero"] += 1
    return {
        "groups": len(eligible_groups),
        "total_groups": len(groups),
        "eligible_groups": len(eligible_groups),
        "excluded_groups": len(excluded_groups),
        "excluded_group_ids": [row["group_id"] for row in excluded_groups],
        "total_rollouts": len(trajectories),
        "eligible_group_rollouts": len(flat_scores),
        "group_counts": {
            "mixed": kinds["mixed"],
            "all_zero": kinds["all_zero"],
            "uniform_nonzero": kinds["uniform_nonzero"],
        },
        "zero_reward_std_fraction": (
            sum(std <= 1e-12 for std in stds) / len(stds) if stds else None
        ),
        "unique_reward_values": len({round(value, 12) for value in flat_scores}),
        "reward_mean": statistics.mean(flat_scores) if flat_scores else None,
        "reward_std": statistics.pstdev(flat_scores) if flat_scores else None,
    }


def build_report(run_dir: Path, spec_path: Path) -> dict[str, Any]:
    spec = _read_json(spec_path)
    raw_path = run_dir / "raw_rollouts.jsonl"
    evidence_path = run_dir / "rollout_evidence.jsonl"
    source = dict(spec["source_run"])
    if _sha256(raw_path) != str(source["raw_rollouts_sha256"]).upper():
        raise ValueError("Raw rollout SHA-256 does not match frozen spec")
    if _sha256(evidence_path) != str(source["rollout_evidence_sha256"]).upper():
        raise ValueError("Evidence SHA-256 does not match frozen spec")
    raw_rows = _read_jsonl(raw_path)
    evidence_rows = _read_jsonl(evidence_path)
    if len(raw_rows) != len(evidence_rows):
        raise ValueError("Raw rollout and evidence row counts differ")
    if len(raw_rows) != int(source["expected_rollouts"]):
        raise ValueError("Rollout count does not match frozen spec")

    sampling_path, sampling_bindings, sampling_groups = _sampling_group_bindings(
        run_dir, source, raw_rows, evidence_rows
    )

    confirmation_mode = (spec.get("reward") or {}).get("composition_mode") in {
        "additive_terminal_process_confirmation_v4",
        "hierarchical_state_authorization_v5",
        "hierarchical_state_authorization_review_v6",
        "hierarchical_state_authorization_claim_v7",
    }
    trajectories = []
    for raw, evidence in zip(raw_rows, evidence_rows, strict=True):
        confirmation = (
            _confirmation_diagnostic_from_serialized_messages(
                list(raw.get("messages") or [])
            )
            if confirmation_mode
            else None
        )
        trajectories.append(
            score_rollout(
                raw,
                evidence,
                spec,
                confirmation_diagnostic=confirmation,
            )
        )
    for index, row in enumerate(trajectories):
        row["rollout_index"] = index
        row.update(sampling_bindings[index])

    terminal_stats = _eligible_group_statistics(
        trajectories,
        sampling_groups,
        "terminal_reward",
    )
    staged_stats = _eligible_group_statistics(
        trajectories,
        sampling_groups,
        "staged_reward",
    )
    claim_verdict_counts = Counter(
        str(claim.get("verdict"))
        for row in trajectories
        if isinstance(
            claim := row["components"].get("claim_evidence_consistency"), dict
        )
    )
    claim_evidence_used_as_reward = any(
        bool(claim.get("used_as_reward"))
        for row in trajectories
        if isinstance(
            claim := row["components"].get("claim_evidence_consistency"), dict
        )
    )
    gate_spec = dict(spec["shadow_gate"])
    separability_checks = {
        "minimum_mixed_groups": staged_stats["group_counts"]["mixed"]
        >= int(gate_spec["minimum_mixed_groups"]),
        "maximum_zero_reward_std_fraction": staged_stats[
            "zero_reward_std_fraction"
        ] is not None
        and staged_stats["zero_reward_std_fraction"]
        <= float(gate_spec["maximum_zero_reward_std_fraction"]),
        "minimum_unique_reward_values": staged_stats["unique_reward_values"]
        >= int(gate_spec["minimum_unique_reward_values"]),
        "no_verified_write_cap_observed": all(
            row["staged_reward"] <= _no_verified_write_cap(spec["reward"])
            + 1e-12
            for row in trajectories
            if row["no_verified_write_cap_applied"]
        ),
    }
    verified_write_rollouts_all = sum(
        row["components"]["required_write_progress"]["value"] > 0.0
        for row in trajectories
    )
    verified_write_rollouts_eligible_groups = sum(
        row["components"]["required_write_progress"]["value"] > 0.0
        and row["eligible_for_group_statistics"]
        for row in trajectories
    )
    claim_safe_authorized_rollouts = [
        row
        for row in trajectories
        if float(
            (row.get("additive_components") or {}).get(
                "authorized_write",
                (row.get("additive_components") or {}).get(
                    "write_authorization", 0.0
                ),
            )
        )
        > 0.0
        and float(
            (row.get("additive_components") or {}).get(
                "post_write_communication", 0.0
            )
        )
        > 0.0
        and (
            row["components"].get("claim_evidence_consistency") or {}
        ).get("verdict")
        == "PASS"
    ]
    claim_safe_authorized_group_ids = {
        str(row["group_id"])
        for row in claim_safe_authorized_rollouts
        if row["eligible_for_group_statistics"]
    }
    minimum_verified_writes = int(
        gate_spec.get(
            "minimum_verified_write_rollouts_for_online_promotion",
            gate_spec.get(
                "minimum_required_action_match_rollouts_for_online_promotion",
                gate_spec.get("minimum_correct_write_rollouts_for_online_promotion", 0),
            ),
        )
    )
    promotion_checks = {
        "separability_gate": all(separability_checks.values()),
        "minimum_verified_write_rollouts": (
            verified_write_rollouts_eligible_groups >= minimum_verified_writes
        ),
    }
    minimum_claim_safe_authorized = int(
        gate_spec.get(
            "minimum_claim_safe_authorized_rollouts_for_online_promotion", 0
        )
    )
    if minimum_claim_safe_authorized:
        promotion_checks["minimum_claim_safe_authorized_rollouts"] = (
            len(claim_safe_authorized_rollouts) >= minimum_claim_safe_authorized
        )
    if gate_spec.get("require_no_excluded_groups_for_online_promotion", False):
        promotion_checks["no_excluded_groups"] = staged_stats["excluded_groups"] == 0
    if not gate_spec.get("online_promotion_allowed", True):
        promotion_checks["online_promotion_allowed_by_spec"] = False
    source_rows = {
        "spec": {"path": str(spec_path), "sha256": _sha256(spec_path)},
        "raw_rollouts": {"path": str(raw_path), "sha256": _sha256(raw_path)},
        "rollout_evidence": {
            "path": str(evidence_path),
            "sha256": _sha256(evidence_path),
        },
    }
    if sampling_path is not None:
        source_rows["sampling_groups"] = {
            "path": str(sampling_path),
            "sha256": _sha256(sampling_path),
        }
    if verified_write_rollouts_all:
        write_limitation = (
            "At least one trajectory has a benchmark-matched required write bound to a "
            "unique later non-error tool result. This still does not prove final-state, "
            "confirmation, authorization, or policy correctness."
        )
    else:
        write_limitation = (
            "No trajectory in this batch has a benchmark-matched required write bound to "
            "a unique later non-error tool result."
        )
    return {
        "schema_version": SCHEMA_VERSION,
        "spec_id": spec["spec_id"],
        "scope": {
            "offline_shadow_only": True,
            "same_frozen_rollouts": True,
            "changes_online_reward": False,
            "calls_external_api": False,
            "uses_cloud_gpu": False,
        },
        "sources": source_rows,
        "comparison": {
            "terminal_only": terminal_stats,
            "staged_shadow": staged_stats,
            "claim_evidence_diagnostic": {
                "verdict_counts": dict(sorted(claim_verdict_counts.items())),
                "used_as_reward": claim_evidence_used_as_reward,
            },
        },
        "gate": {
            "separability_checks": separability_checks,
            "separability_passed": all(separability_checks.values()),
            "component_coverage": {
                "verified_write_rollouts_all": verified_write_rollouts_all,
                "verified_write_rollouts_eligible_groups": (
                    verified_write_rollouts_eligible_groups
                ),
                "terminal_success_rollouts": sum(
                    row["terminal_success"] for row in trajectories
                ),
                "claim_safe_authorized_rollouts": len(
                    claim_safe_authorized_rollouts
                ),
                "groups_with_claim_safe_authorized_rollout": len(
                    claim_safe_authorized_group_ids
                ),
            },
            "online_promotion_checks": promotion_checks,
            "online_promotion_ready": all(promotion_checks.values()),
            "interpretation": gate_spec["interpretation"],
        },
        "trajectories": trajectories,
        "limitations": [
            "The staged signal uses hidden benchmark references and is valid only for benchmark training experiments.",
            "The same frozen rollouts isolate reward separability, not policy-learning effectiveness.",
            "Transport-invalid groups are excluded from group-level separability statistics while their per-trajectory diagnostic scores are retained.",
            write_limitation,
            "Positive prefix credit does not make a failed full trajectory safe for online optimization; the Task95 premature-transfer cap is benchmark-specific and does not generalize to production escalation policy.",
            (
                "Prompt-bound confirmation is included in this offline scalar candidate; "
                "its REVIEW/PASS labels remain developmental and are not independent gold."
                if confirmation_mode
                else "Confirmation and policy diagnostics remain excluded from the scalar reward."
            ),
            "Transport completeness and truncation diagnostics are bound from the frozen sampling-groups artifact; valid tool/customer limit failures are not automatically treated as transport corruption.",
            "Claim-evidence PASS means only that no configured deterministic conflict rule fired; it is not proof that all natural-language claims are grounded.",
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Offline staged reward shadow audit")
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = build_report(args.run_dir.resolve(), args.spec.resolve())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report["comparison"], ensure_ascii=False, indent=2))
    print(json.dumps(report["gate"], ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
