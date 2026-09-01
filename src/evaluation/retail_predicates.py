from __future__ import annotations

import hashlib
import json
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Sequence

from src.evaluation.task_rubric import (
    AtomicVerdict,
    PredicateResult,
    PredicateSpec,
)
from src.guards.retail_pre_action import WRITE_TOOLS


PathPart = str | int
JsonPath = tuple[PathPart, ...]


@dataclass(frozen=True, slots=True)
class ToolCallRecord:
    call_id: str
    name: str
    arguments: Mapping[str, Any]
    message_index: int
    result_message_index: int | None = None
    result_success: bool | None = None
    result_content: Any = None

    @property
    def action_class(self) -> str:
        return "write" if self.name in WRITE_TOOLS else "read"


@dataclass(frozen=True, slots=True)
class RetailPredicateContext:
    """Evidence supplied to deterministic Retail predicates.

    Semantic facts that cannot be derived safely from raw text are supplied as
    explicit bindings.  In particular, confirmations are bound to call IDs and
    final-answer claim checks are bound to the exact answer hash.
    """

    initial_state: Mapping[str, Any] | None = None
    final_state: Mapping[str, Any] | None = None
    messages: tuple[Mapping[str, Any], ...] | None = None
    final_answer: str | None = None
    confirmation_message_index_by_call_id: Mapping[str, int] | None = None
    latest_intent_revision_message_index_by_call_id: Mapping[str, int] | None = None
    claim_checks: Mapping[str, Mapping[str, Any]] | None = None
    stopped: bool | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)


def final_answer_sha256(answer: str) -> str:
    return hashlib.sha256(answer.encode("utf-8")).hexdigest()


def _result(
    spec: PredicateSpec,
    verdict: AtomicVerdict,
    reason: str,
    evidence: Sequence[Mapping[str, Any]],
    *,
    error: str | None = None,
) -> PredicateResult:
    return PredicateResult(
        predicate_id=spec.predicate_id,
        capability_group=spec.capability_group,
        verdict=verdict,
        reason=reason,
        evidence=tuple(evidence),
        error=error,
    )


def _error(
    spec: PredicateSpec,
    detail: str,
    evidence: Sequence[Mapping[str, Any]],
) -> PredicateResult:
    return _result(
        spec,
        AtomicVerdict.ERROR,
        "Evaluation evidence is incomplete or invalid.",
        evidence,
        error=detail,
    )


def _parameter(spec: PredicateSpec, key: str) -> tuple[bool, Any]:
    return key in spec.parameters, spec.parameters.get(key)


def _path(value: Any, location: str) -> tuple[JsonPath | None, str | None]:
    if not isinstance(value, list) or not value:
        return None, f"{location} must be a non-empty JSON path array"
    if any(not isinstance(part, (str, int)) or isinstance(part, bool) for part in value):
        return None, f"{location} path parts must be strings or integers"
    return tuple(value), None


def _paths(value: Any, location: str) -> tuple[tuple[JsonPath, ...] | None, str | None]:
    if not isinstance(value, list):
        return None, f"{location} must be an array of JSON paths"
    parsed: list[JsonPath] = []
    for index, raw_path in enumerate(value):
        path, error = _path(raw_path, f"{location}[{index}]")
        if error:
            return None, error
        assert path is not None
        parsed.append(path)
    return tuple(parsed), None


def _lookup(root: Any, path: JsonPath) -> tuple[bool, Any]:
    current = root
    for part in path:
        if isinstance(part, str) and isinstance(current, Mapping) and part in current:
            current = current[part]
        elif (
            isinstance(part, int)
            and isinstance(current, Sequence)
            and not isinstance(current, (str, bytes, bytearray))
            and 0 <= part < len(current)
        ):
            current = current[part]
        else:
            return False, None
    return True, current


def _state(context: RetailPredicateContext, name: Any) -> Mapping[str, Any] | None:
    if name == "initial":
        return context.initial_state
    if name == "final":
        return context.final_state
    return None


def _extract_tool_calls(
    context: RetailPredicateContext,
) -> tuple[tuple[ToolCallRecord, ...] | None, str | None]:
    if context.messages is None:
        return None, "messages evidence is missing"
    pending: dict[str, dict[str, Any]] = {}
    order: list[str] = []
    for message_index, message in enumerate(context.messages):
        if not isinstance(message, Mapping):
            return None, f"messages[{message_index}] is not a mapping"
        role = message.get("role")
        if role == "assistant":
            raw_calls = message.get("tool_calls", [])
            if raw_calls is None:
                raw_calls = []
            if not isinstance(raw_calls, list):
                return None, f"messages[{message_index}].tool_calls is not a list"
            for call_index, raw_call in enumerate(raw_calls):
                if not isinstance(raw_call, Mapping):
                    return None, f"tool call {message_index}:{call_index} is not a mapping"
                call_id = raw_call.get("id")
                name = raw_call.get("name")
                arguments = raw_call.get("arguments", {})
                if not isinstance(call_id, str) or not call_id:
                    return None, f"tool call {message_index}:{call_index} lacks a call id"
                if call_id in pending:
                    return None, f"duplicate tool call id: {call_id}"
                if not isinstance(name, str) or not name:
                    return None, f"tool call {call_id} lacks a tool name"
                if not isinstance(arguments, Mapping):
                    return None, f"tool call {call_id} arguments are not a mapping"
                pending[call_id] = {
                    "call_id": call_id,
                    "name": name,
                    "arguments": dict(arguments),
                    "message_index": message_index,
                    "result_message_index": None,
                    "result_success": None,
                    "result_content": None,
                }
                order.append(call_id)
        elif role == "tool":
            call_id = message.get("tool_call_id", message.get("id"))
            if not isinstance(call_id, str) or call_id not in pending:
                return None, f"messages[{message_index}] has an unknown tool call id"
            record = pending[call_id]
            if record["result_message_index"] is not None:
                return None, f"tool call {call_id} has multiple result messages"
            success = message.get("success")
            if not isinstance(success, bool):
                error_flag = message.get("error")
                if not isinstance(error_flag, bool):
                    error_flag = message.get("tool_error")
                success = not error_flag if isinstance(error_flag, bool) else None
            record["result_message_index"] = message_index
            record["result_success"] = success
            record["result_content"] = message.get("content")
    return tuple(ToolCallRecord(**pending[call_id]) for call_id in order), None


def _matching_calls(
    spec: PredicateSpec,
    context: RetailPredicateContext,
) -> tuple[tuple[ToolCallRecord, ...] | None, PredicateResult | None]:
    calls, error = _extract_tool_calls(context)
    if error:
        return None, _error(spec, error, ({"source": "messages", "status": "INVALID"},))
    assert calls is not None
    present, name = _parameter(spec, "name")
    if not present or not isinstance(name, str) or not name:
        return None, _error(
            spec,
            "predicate parameter 'name' must be a non-empty string",
            ({"source": "parameters.name", "status": "INVALID"},),
        )
    return tuple(call for call in calls if call.name == name), None


def _call_evidence(call: ToolCallRecord) -> Mapping[str, Any]:
    return {
        "source": "tool_call",
        "call_id": call.call_id,
        "name": call.name,
        "action_class": call.action_class,
        "arguments": dict(call.arguments),
        "message_index": call.message_index,
        "result_message_index": call.result_message_index,
        "result_success": call.result_success,
    }


def _field_equals(spec: PredicateSpec, context: RetailPredicateContext) -> PredicateResult:
    state_name = spec.parameters.get("state")
    if not isinstance(state_name, str) or state_name not in {"initial", "final"}:
        return _error(spec, "state must be 'initial' or 'final'", ({"source": "parameters.state", "status": "INVALID"},))
    state = _state(context, state_name)
    if state is None:
        return _error(spec, f"{state_name}_state evidence is missing", ({"source": f"{state_name}_state", "status": "MISSING"},))
    path, path_error = _path(spec.parameters.get("path"), "parameters.path")
    if path_error:
        return _error(spec, path_error, ({"source": "parameters.path", "status": "INVALID"},))
    has_expected, expected = _parameter(spec, "expected")
    if not has_expected:
        return _error(spec, "expected parameter is missing", ({"source": "parameters.expected", "status": "MISSING"},))
    assert path is not None
    found, observed = _lookup(state, path)
    evidence = ({"source": f"{state_name}_state", "path": list(path), "found": found, "observed": observed, "expected": expected},)
    if not found:
        return _error(spec, "state path is missing", evidence)
    verdict = AtomicVerdict.PASS if observed == expected else AtomicVerdict.FAIL
    return _result(spec, verdict, "State field matches expected value." if verdict is AtomicVerdict.PASS else "State field does not match expected value.", evidence)


def _field_unchanged(spec: PredicateSpec, context: RetailPredicateContext) -> PredicateResult:
    if context.initial_state is None or context.final_state is None:
        return _error(spec, "initial_state and final_state are both required", ({"source": "state_pair", "initial_present": context.initial_state is not None, "final_present": context.final_state is not None},))
    path, path_error = _path(spec.parameters.get("path"), "parameters.path")
    if path_error:
        return _error(spec, path_error, ({"source": "parameters.path", "status": "INVALID"},))
    assert path is not None
    initial_found, initial = _lookup(context.initial_state, path)
    final_found, final = _lookup(context.final_state, path)
    evidence = ({"source": "state_pair", "path": list(path), "initial_found": initial_found, "final_found": final_found, "initial": initial, "final": final},)
    if not initial_found or not final_found:
        return _error(spec, "state path is missing from initial or final state", evidence)
    verdict = AtomicVerdict.PASS if initial == final else AtomicVerdict.FAIL
    return _result(spec, verdict, "State field is unchanged." if verdict is AtomicVerdict.PASS else "State field changed.", evidence)


def _required_action(spec: PredicateSpec, context: RetailPredicateContext) -> PredicateResult:
    calls, failure = _matching_calls(spec, context)
    if failure:
        return failure
    assert calls is not None
    evidence = tuple(_call_evidence(call) for call in calls) or ({"source": "tool_calls", "name": spec.parameters["name"], "count": 0},)
    verdict = AtomicVerdict.PASS if calls else AtomicVerdict.FAIL
    return _result(spec, verdict, "Required action was observed." if calls else "Required action was not observed.", evidence)


def _forbidden_action(spec: PredicateSpec, context: RetailPredicateContext) -> PredicateResult:
    calls, failure = _matching_calls(spec, context)
    if failure:
        return failure
    assert calls is not None
    evidence = tuple(_call_evidence(call) for call in calls) or ({"source": "tool_calls", "name": spec.parameters["name"], "count": 0},)
    verdict = AtomicVerdict.FAIL if calls else AtomicVerdict.PASS
    return _result(spec, verdict, "Forbidden action was observed." if calls else "Forbidden action was not observed.", evidence)


def _action_cardinality(spec: PredicateSpec, context: RetailPredicateContext) -> PredicateResult:
    calls, failure = _matching_calls(spec, context)
    if failure:
        return failure
    assert calls is not None
    expected_class = spec.parameters.get("action_class")
    if expected_class not in {"read", "write"}:
        return _error(spec, "action_class must explicitly be 'read' or 'write'", ({"source": "parameters.action_class", "status": "INVALID"},))
    actual_class = "write" if spec.parameters["name"] in WRITE_TOOLS else "read"
    if expected_class != actual_class:
        return _error(spec, "declared action_class disagrees with Retail tool classification", ({"source": "tool_classification", "name": spec.parameters["name"], "declared": expected_class, "actual": actual_class},))
    minimum = spec.parameters.get("min_count", 0)
    maximum = spec.parameters.get("max_count")
    if not isinstance(minimum, int) or isinstance(minimum, bool) or minimum < 0:
        return _error(spec, "min_count must be a non-negative integer", ({"source": "parameters.min_count", "status": "INVALID"},))
    if maximum is not None and (not isinstance(maximum, int) or isinstance(maximum, bool) or maximum < minimum):
        return _error(spec, "max_count must be an integer greater than or equal to min_count", ({"source": "parameters.max_count", "status": "INVALID"},))
    count = len(calls)
    passed = count >= minimum and (maximum is None or count <= maximum)
    evidence = ({"source": "tool_calls", "name": spec.parameters["name"], "action_class": actual_class, "count": count, "min_count": minimum, "max_count": maximum, "call_ids": [call.call_id for call in calls]},)
    return _result(spec, AtomicVerdict.PASS if passed else AtomicVerdict.FAIL, "Action count is within the explicit range." if passed else "Action count is outside the explicit range.", evidence)


def _action_argument_equals(spec: PredicateSpec, context: RetailPredicateContext) -> PredicateResult:
    calls, failure = _matching_calls(spec, context)
    if failure:
        return failure
    assert calls is not None
    if not calls:
        return _error(spec, "no matching action exists for argument evaluation", ({"source": "tool_calls", "name": spec.parameters["name"], "count": 0},))
    path, path_error = _path(spec.parameters.get("argument_path"), "parameters.argument_path")
    if path_error:
        return _error(spec, path_error, ({"source": "parameters.argument_path", "status": "INVALID"},))
    has_expected, expected = _parameter(spec, "expected")
    if not has_expected:
        return _error(spec, "expected parameter is missing", ({"source": "parameters.expected", "status": "MISSING"},))
    quantifier = spec.parameters.get("quantifier", "all")
    if quantifier not in {"all", "any"}:
        return _error(spec, "quantifier must be 'all' or 'any'", ({"source": "parameters.quantifier", "status": "INVALID"},))
    assert path is not None
    comparisons = []
    matches = []
    for call in calls:
        found, observed = _lookup(call.arguments, path)
        comparisons.append({"source": "tool_argument", "call_id": call.call_id, "name": call.name, "path": list(path), "found": found, "observed": observed, "expected": expected})
        matches.append(found and observed == expected)
    passed = all(matches) if quantifier == "all" else any(matches)
    return _result(spec, AtomicVerdict.PASS if passed else AtomicVerdict.FAIL, "Action argument satisfies the comparison." if passed else "Action argument does not satisfy the comparison.", comparisons)


def _json_token(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _expected_pairs(
    value: Any,
    location: str,
) -> tuple[tuple[tuple[Any, Any], ...] | None, str | None]:
    if not isinstance(value, list) or not value:
        return None, f"{location} must be a non-empty array of two-item arrays"
    pairs: list[tuple[Any, Any]] = []
    for index, raw_pair in enumerate(value):
        if not isinstance(raw_pair, list) or len(raw_pair) != 2:
            return None, f"{location}[{index}] must be a two-item array"
        try:
            _json_token(raw_pair)
        except (TypeError, ValueError):
            return None, f"{location}[{index}] must contain JSON-compatible values"
        pairs.append((raw_pair[0], raw_pair[1]))
    return tuple(pairs), None


def _observed_argument_pairs(
    arguments: Mapping[str, Any],
    left_path: JsonPath,
    right_path: JsonPath,
) -> tuple[tuple[tuple[Any, Any], ...] | None, Mapping[str, Any]]:
    left_found, left = _lookup(arguments, left_path)
    right_found, right = _lookup(arguments, right_path)
    valid_lists = (
        left_found
        and right_found
        and isinstance(left, list)
        and isinstance(right, list)
        and len(left) == len(right)
    )
    evidence = {
        "left_path": list(left_path),
        "right_path": list(right_path),
        "left_found": left_found,
        "right_found": right_found,
        "left_observed": left,
        "right_observed": right,
        "equal_length_lists": valid_lists,
    }
    if not valid_lists:
        return None, evidence
    return tuple(zip(left, right)), evidence


def _pair_multiset(pairs: Sequence[tuple[Any, Any]]) -> Counter[str]:
    return Counter(_json_token([left, right]) for left, right in pairs)


def _paired_action_arguments_equal(
    spec: PredicateSpec,
    context: RetailPredicateContext,
) -> PredicateResult:
    calls, failure = _matching_calls(spec, context)
    if failure:
        return failure
    assert calls is not None
    if not calls:
        return _error(
            spec,
            "no matching action exists for paired-argument evaluation",
            ({"source": "tool_calls", "name": spec.parameters["name"], "count": 0},),
        )
    left_path, left_error = _path(
        spec.parameters.get("left_argument_path"),
        "parameters.left_argument_path",
    )
    right_path, right_error = _path(
        spec.parameters.get("right_argument_path"),
        "parameters.right_argument_path",
    )
    if left_error or right_error:
        return _error(
            spec,
            left_error or right_error or "invalid paired argument path",
            ({"source": "paired_argument_paths", "status": "INVALID"},),
        )
    expected, expected_error = _expected_pairs(
        spec.parameters.get("expected_pairs"),
        "parameters.expected_pairs",
    )
    if expected_error:
        return _error(
            spec,
            expected_error,
            ({"source": "parameters.expected_pairs", "status": "INVALID"},),
        )
    quantifier = spec.parameters.get("quantifier", "all")
    if quantifier not in {"all", "any"}:
        return _error(
            spec,
            "quantifier must be 'all' or 'any'",
            ({"source": "parameters.quantifier", "status": "INVALID"},),
        )
    assert left_path is not None and right_path is not None and expected is not None
    expected_counter = _pair_multiset(expected)
    comparisons = []
    matches = []
    for call in calls:
        observed, pair_evidence = _observed_argument_pairs(
            call.arguments,
            left_path,
            right_path,
        )
        matched = observed is not None and _pair_multiset(observed) == expected_counter
        comparisons.append(
            {
                "source": "paired_tool_arguments",
                "call_id": call.call_id,
                "name": call.name,
                **dict(pair_evidence),
                "observed_pairs": [list(pair) for pair in observed] if observed is not None else None,
                "expected_pairs": [list(pair) for pair in expected],
                "matched_as_multiset": matched,
            }
        )
        matches.append(matched)
    passed = all(matches) if quantifier == "all" else any(matches)
    return _result(
        spec,
        AtomicVerdict.PASS if passed else AtomicVerdict.FAIL,
        "Paired action arguments match the expected pair multiset."
        if passed
        else "Paired action arguments do not match the expected pair multiset.",
        comparisons,
    )


def _argument_checks(
    value: Any,
    location: str,
) -> tuple[tuple[tuple[JsonPath, Any], ...] | None, str | None]:
    if not isinstance(value, list):
        return None, f"{location} must be an array"
    checks: list[tuple[JsonPath, Any]] = []
    for index, raw_check in enumerate(value):
        if not isinstance(raw_check, Mapping) or set(raw_check) != {"path", "expected"}:
            return None, f"{location}[{index}] must contain exactly path and expected"
        path, error = _path(raw_check["path"], f"{location}[{index}].path")
        if error:
            return None, error
        try:
            _json_token(raw_check["expected"])
        except (TypeError, ValueError):
            return None, f"{location}[{index}].expected must be JSON-compatible"
        assert path is not None
        checks.append((path, raw_check["expected"]))
    return tuple(checks), None


def _action_state_transition_consistent(
    spec: PredicateSpec,
    context: RetailPredicateContext,
) -> PredicateResult:
    calls, failure = _matching_calls(spec, context)
    if failure:
        return failure
    assert calls is not None
    selector_checks: tuple[tuple[JsonPath, Any], ...] | None = None
    selector_evidence: list[Mapping[str, Any]] = []
    if "selector_checks" in spec.parameters:
        selector_checks, selector_error = _argument_checks(
            spec.parameters.get("selector_checks"),
            "parameters.selector_checks",
        )
        if selector_error or not selector_checks:
            return _error(
                spec,
                selector_error or "selector_checks must be non-empty when supplied",
                ({"source": "parameters.selector_checks", "status": "INVALID"},),
            )
        selected_calls: list[ToolCallRecord] = []
        for call in calls:
            comparisons = []
            selected = True
            for path, expected in selector_checks:
                found, observed = _lookup(call.arguments, path)
                matched = found and observed == expected
                selected = selected and matched
                comparisons.append(
                    {
                        "path": list(path),
                        "found": found,
                        "observed": observed,
                        "expected": expected,
                        "matched": matched,
                    }
                )
            selector_evidence.append(
                {
                    "source": "tool_call_selector",
                    "call_id": call.call_id,
                    "name": call.name,
                    "comparisons": comparisons,
                    "selected": selected,
                }
            )
            if selected:
                selected_calls.append(call)
        calls = tuple(selected_calls)
        if not calls:
            return _result(
                spec,
                AtomicVerdict.FAIL,
                "No state-changing action matches the required call selector.",
                selector_evidence,
            )
    if not calls:
        return _error(
            spec,
            "no matching action exists for action-state transition evaluation",
            ({"source": "tool_calls", "name": spec.parameters["name"], "count": 0},),
        )
    if len(calls) > 1:
        return _result(
            spec,
            AtomicVerdict.FAIL,
            "The state-changing action was called more than once.",
            ({"source": "tool_calls", "name": spec.parameters["name"], "count": len(calls), "call_ids": [call.call_id for call in calls]},),
        )
    argument_checks, argument_error = _argument_checks(
        spec.parameters.get("argument_checks"),
        "parameters.argument_checks",
    )
    initial_checks, initial_error = _argument_checks(
        spec.parameters.get("required_initial_fields"),
        "parameters.required_initial_fields",
    )
    final_checks, final_error = _argument_checks(
        spec.parameters.get("expected_final_fields"),
        "parameters.expected_final_fields",
    )
    paired = spec.parameters.get("paired_arguments")
    if argument_error or initial_error or final_error:
        return _error(
            spec,
            argument_error or initial_error or final_error or "invalid transition checks",
            ({"source": "transition_parameters", "status": "INVALID"},),
        )
    if not initial_checks or not final_checks:
        return _error(
            spec,
            "required_initial_fields and expected_final_fields must both be non-empty",
            ({"source": "transition_parameters", "status": "INVALID"},),
        )
    if not argument_checks and paired is None:
        return _error(
            spec,
            "argument_checks must be non-empty when paired_arguments is absent",
            ({"source": "transition_parameters", "status": "INVALID"},),
        )
    left_path: JsonPath | None = None
    right_path: JsonPath | None = None
    expected_pairs: tuple[tuple[Any, Any], ...] | None = None
    if paired is not None:
        if not isinstance(paired, Mapping) or set(paired) != {
            "left_argument_path",
            "right_argument_path",
            "expected_pairs",
        }:
            return _error(
                spec,
                "paired_arguments must contain exactly left_argument_path, right_argument_path, and expected_pairs",
                ({"source": "parameters.paired_arguments", "status": "INVALID"},),
            )
        left_path, left_error = _path(
            paired["left_argument_path"],
            "parameters.paired_arguments.left_argument_path",
        )
        right_path, right_error = _path(
            paired["right_argument_path"],
            "parameters.paired_arguments.right_argument_path",
        )
        expected_pairs, pairs_error = _expected_pairs(
            paired["expected_pairs"],
            "parameters.paired_arguments.expected_pairs",
        )
        if left_error or right_error or pairs_error:
            return _error(
                spec,
                left_error or right_error or pairs_error or "invalid paired transition parameters",
                ({"source": "parameters.paired_arguments", "status": "INVALID"},),
            )
    if context.initial_state is None or context.final_state is None:
        return _error(
            spec,
            "initial_state and final_state evidence are both required",
            ({"source": "state_pair", "initial_present": context.initial_state is not None, "final_present": context.final_state is not None},),
        )
    assert argument_checks is not None and initial_checks is not None and final_checks is not None
    call = calls[0]
    evidence: list[Mapping[str, Any]] = list(selector_evidence)
    initial_matches = True
    for path, expected in initial_checks:
        found, observed = _lookup(context.initial_state, path)
        matched = found and observed == expected
        initial_matches = initial_matches and matched
        evidence.append(
            {
                "source": "initial_state",
                "path": list(path),
                "found": found,
                "observed": observed,
                "expected": expected,
                "matched": matched,
            }
        )
    if not initial_matches:
        return _error(
            spec,
            "the supplied initial state does not satisfy the bound transition preconditions",
            evidence,
        )
    arguments_match = True
    for path, expected in argument_checks:
        found, observed = _lookup(call.arguments, path)
        matched = found and observed == expected
        arguments_match = arguments_match and matched
        evidence.append(
            {
                "source": "tool_argument",
                "call_id": call.call_id,
                "path": list(path),
                "found": found,
                "observed": observed,
                "expected": expected,
                "matched": matched,
            }
        )
    if paired is not None:
        assert left_path is not None and right_path is not None and expected_pairs is not None
        observed_pairs, pair_evidence = _observed_argument_pairs(
            call.arguments,
            left_path,
            right_path,
        )
        pairs_match = (
            observed_pairs is not None
            and _pair_multiset(observed_pairs) == _pair_multiset(expected_pairs)
        )
        arguments_match = arguments_match and pairs_match
        evidence.append(
            {
                "source": "paired_tool_arguments",
                "call_id": call.call_id,
                **dict(pair_evidence),
                "observed_pairs": [list(pair) for pair in observed_pairs]
                if observed_pairs is not None
                else None,
                "expected_pairs": [list(pair) for pair in expected_pairs],
                "matched_as_multiset": pairs_match,
            }
        )
    evidence.append(_call_evidence(call))
    if not arguments_match:
        return _result(
            spec,
            AtomicVerdict.FAIL,
            "The action arguments do not describe the required state transition.",
            evidence,
        )
    if call.result_message_index is None or call.result_success is None:
        return _error(spec, "explicit tool-result status is missing", evidence)
    if call.result_success is not True:
        return _error(
            spec,
            "a correctly specified action was rejected by the environment",
            evidence,
        )
    final_matches = True
    for path, expected in final_checks:
        found, observed = _lookup(context.final_state, path)
        matched = found and observed == expected
        final_matches = final_matches and matched
        evidence.append(
            {
                "source": "final_state",
                "path": list(path),
                "found": found,
                "observed": observed,
                "expected": expected,
                "matched": matched,
            }
        )
    if not final_matches:
        return _error(
            spec,
            "a correctly specified successful action produced an inconsistent final state",
            evidence,
        )
    return _result(
        spec,
        AtomicVerdict.PASS,
        "The correctly specified successful action produced the expected final state.",
        evidence,
    )


def _write_after_confirmation(spec: PredicateSpec, context: RetailPredicateContext) -> PredicateResult:
    calls, failure = _matching_calls(spec, context)
    if failure:
        return failure
    assert calls is not None
    if spec.parameters["name"] not in WRITE_TOOLS:
        return _error(spec, "write_after_confirmation only accepts a classified write tool", ({"source": "tool_classification", "name": spec.parameters["name"], "actual": "read"},))
    if not calls:
        return _error(spec, "no matching write exists for confirmation evaluation", ({"source": "tool_calls", "name": spec.parameters["name"], "count": 0},))
    bindings = context.confirmation_message_index_by_call_id
    if bindings is None:
        return _error(spec, "confirmation bindings are missing", ({"source": "confirmation_bindings", "status": "MISSING"},))
    require_after_latest_revision = spec.parameters.get(
        "require_after_latest_revision",
        False,
    )
    if not isinstance(require_after_latest_revision, bool):
        return _error(
            spec,
            "require_after_latest_revision must be a boolean",
            ({"source": "parameters.require_after_latest_revision", "status": "INVALID"},),
        )
    revision_bindings = context.latest_intent_revision_message_index_by_call_id
    if require_after_latest_revision and revision_bindings is None:
        return _error(
            spec,
            "latest-intent revision bindings are missing",
            ({"source": "latest_intent_revision_bindings", "status": "MISSING"},),
        )
    evidence = []
    passed = True
    for call in calls:
        confirmation_index = bindings.get(call.call_id)
        confirmation_is_bound = confirmation_index is not None
        if confirmation_is_bound and (
            type(confirmation_index) is not int
            or confirmation_index < 0
            or confirmation_index >= len(context.messages)
            or context.messages[confirmation_index].get("role") != "user"
        ):
            return _error(
                spec,
                f"confirmation binding is invalid for call id {call.call_id}",
                ({"source": "confirmation_binding", "call_id": call.call_id, "message_index": confirmation_index, "status": "INVALID"},),
            )
        confirmed_before = (
            type(confirmation_index) is int
            and confirmation_index < call.message_index
        )
        revision_index = (
            revision_bindings.get(call.call_id)
            if revision_bindings is not None
            else None
        )
        if require_after_latest_revision and (
            type(revision_index) is not int
            or revision_index < 0
            or revision_index >= len(context.messages)
            or context.messages[revision_index].get("role") != "user"
        ):
            return _error(
                spec,
                f"latest-intent revision binding is missing or invalid for call id {call.call_id}",
                ({"source": "latest_intent_revision_binding", "call_id": call.call_id, "message_index": revision_index, "status": "MISSING_OR_INVALID"},),
            )
        confirmed_after_latest_revision = (
            not require_after_latest_revision
            or (
                isinstance(revision_index, int)
                and type(confirmation_index) is int
                and revision_index < confirmation_index
            )
        )
        valid = confirmed_before and confirmed_after_latest_revision
        evidence.append(
            {
                "source": "confirmation_binding",
                "call_id": call.call_id,
                "latest_intent_revision_message_index": revision_index,
                "confirmation_message_index": confirmation_index,
                "write_message_index": call.message_index,
                "confirmed_after_latest_revision": confirmed_after_latest_revision,
                "confirmed_before_write": confirmed_before,
            }
        )
        passed = passed and valid
    return _result(
        spec,
        AtomicVerdict.PASS if passed else AtomicVerdict.FAIL,
        "Every write is bound to confirmation after the latest intent revision."
        if passed and require_after_latest_revision
        else "Every write is bound to prior confirmation."
        if passed
        else "At least one write lacks confirmation after the latest intent revision and before execution."
        if require_after_latest_revision
        else "At least one write lacks prior bound confirmation.",
        evidence,
    )


def _diff_paths(initial: Any, final: Any, path: JsonPath = ()) -> list[JsonPath]:
    if isinstance(initial, Mapping) and isinstance(final, Mapping):
        output: list[JsonPath] = []
        for key in sorted(set(initial) | set(final), key=str):
            if key not in initial or key not in final:
                output.append(path + (key,))
            else:
                output.extend(_diff_paths(initial[key], final[key], path + (key,)))
        return output
    if isinstance(initial, list) and isinstance(final, list):
        output = []
        for index in range(max(len(initial), len(final))):
            if index >= len(initial) or index >= len(final):
                output.append(path + (index,))
            else:
                output.extend(_diff_paths(initial[index], final[index], path + (index,)))
        return output
    return [] if initial == final else [path]


def _no_unrelated_mutation(spec: PredicateSpec, context: RetailPredicateContext) -> PredicateResult:
    if context.initial_state is None or context.final_state is None:
        return _error(spec, "initial_state and final_state are both required", ({"source": "state_pair", "initial_present": context.initial_state is not None, "final_present": context.final_state is not None},))
    allowed, paths_error = _paths(spec.parameters.get("allowed_paths"), "parameters.allowed_paths")
    if paths_error:
        return _error(spec, paths_error, ({"source": "parameters.allowed_paths", "status": "INVALID"},))
    assert allowed is not None
    changed = tuple(_diff_paths(context.initial_state, context.final_state))
    unrelated = tuple(path for path in changed if not any(path[: len(prefix)] == prefix for prefix in allowed))
    evidence = ({"source": "state_diff", "changed_paths": [list(path) for path in changed], "allowed_paths": [list(path) for path in allowed], "unrelated_paths": [list(path) for path in unrelated]},)
    verdict = AtomicVerdict.PASS if not unrelated else AtomicVerdict.FAIL
    return _result(spec, verdict, "No mutation occurred outside the allowed paths." if verdict is AtomicVerdict.PASS else "Mutation occurred outside the allowed paths.", evidence)


def _tool_result_success(spec: PredicateSpec, context: RetailPredicateContext) -> PredicateResult:
    calls, failure = _matching_calls(spec, context)
    if failure:
        return failure
    assert calls is not None
    if not calls:
        return _error(spec, "no matching action exists for result evaluation", ({"source": "tool_calls", "name": spec.parameters["name"], "count": 0},))
    missing = [call.call_id for call in calls if call.result_message_index is None or call.result_success is None]
    evidence = tuple(_call_evidence(call) for call in calls)
    if missing:
        return _error(spec, f"explicit tool-result status missing for call ids: {missing}", evidence)
    passed = all(call.result_success is True for call in calls)
    return _result(spec, AtomicVerdict.PASS if passed else AtomicVerdict.FAIL, "All matching tool results report success." if passed else "At least one matching tool result reports failure.", evidence)


def _final_claim_matches_state(spec: PredicateSpec, context: RetailPredicateContext) -> PredicateResult:
    claim_id = spec.parameters.get("claim_id")
    if not isinstance(claim_id, str) or not claim_id:
        return _error(spec, "claim_id must be a non-empty string", ({"source": "parameters.claim_id", "status": "INVALID"},))
    if context.final_answer is None:
        return _error(spec, "final_answer evidence is missing", ({"source": "final_answer", "status": "MISSING"},))
    if context.claim_checks is None or claim_id not in context.claim_checks:
        return _error(spec, "bound claim check is missing", ({"source": "claim_checks", "claim_id": claim_id, "status": "MISSING"},))
    check = context.claim_checks[claim_id]
    if not isinstance(check, Mapping):
        return _error(spec, "claim check is not a mapping", ({"source": "claim_checks", "claim_id": claim_id, "status": "INVALID"},))
    expected_hash = final_answer_sha256(context.final_answer)
    observed_hash = check.get("answer_sha256")
    if observed_hash != expected_hash:
        return _error(spec, "claim check is not bound to the supplied final answer", ({"source": "claim_check_binding", "claim_id": claim_id, "expected_answer_sha256": expected_hash, "observed_answer_sha256": observed_hash},))
    raw_verdict = check.get("verdict")
    try:
        verdict = AtomicVerdict(raw_verdict)
    except (TypeError, ValueError):
        return _error(spec, "claim check verdict is invalid", ({"source": "claim_check", "claim_id": claim_id, "verdict": raw_verdict},))
    evidence_item = dict(check)
    evidence_item.update({"source": "claim_check", "claim_id": claim_id})
    evidence = (evidence_item,)
    if verdict is AtomicVerdict.ERROR:
        detail = check.get("error")
        return _error(spec, str(detail or "claim check reported an unspecified error"), evidence)
    reasons = {
        AtomicVerdict.PASS: "Final claim matches bound state evidence.",
        AtomicVerdict.FAIL: "Final claim conflicts with bound state evidence.",
        AtomicVerdict.REVIEW: "Final claim requires human review.",
        AtomicVerdict.NOT_APPLICABLE: "The configured final claim is not applicable.",
    }
    return _result(spec, verdict, reasons[verdict], evidence)


def _transfer_and_stop(spec: PredicateSpec, context: RetailPredicateContext) -> PredicateResult:
    parameters = dict(spec.parameters)
    parameters.setdefault("name", "transfer_to_human_agents")
    effective_spec = PredicateSpec(
        predicate_id=spec.predicate_id,
        predicate_type=spec.predicate_type,
        capability_group=spec.capability_group,
        required=spec.required,
        parameters=parameters,
        description=spec.description,
    )
    calls, failure = _matching_calls(effective_spec, context)
    if failure:
        return failure
    assert calls is not None
    if not calls:
        return _result(spec, AtomicVerdict.FAIL, "Required transfer action was not observed.", ({"source": "tool_calls", "name": parameters["name"], "count": 0},))
    if context.stopped is None:
        return _error(spec, "explicit stopped evidence is missing", ({"source": "termination", "status": "MISSING"},))
    transfer = calls[-1]
    if transfer.result_message_index is None or transfer.result_success is None:
        return _error(spec, "transfer result status is missing", (_call_evidence(transfer),))
    assert context.messages is not None
    later_assistant_activity = []
    boundary = transfer.message_index
    for index, message in enumerate(context.messages):
        if index <= boundary or message.get("role") != "assistant":
            continue
        if message.get("tool_calls") or str(message.get("content") or "").strip():
            later_assistant_activity.append(
                {
                    "message_index": index,
                    "content": message.get("content"),
                    "tool_call_count": len(message.get("tool_calls") or []),
                }
            )
    required_terminal_message = parameters.get("required_terminal_message")
    if required_terminal_message is not None and (
        not isinstance(required_terminal_message, str)
        or not required_terminal_message.strip()
    ):
        return _error(
            spec,
            "required_terminal_message must be a non-empty string when supplied",
            ({"source": "parameters.required_terminal_message", "status": "INVALID"},),
        )
    if required_terminal_message is None:
        terminal_protocol_passed = not later_assistant_activity
    else:
        terminal_protocol_passed = (
            len(later_assistant_activity) == 1
            and later_assistant_activity[0]["tool_call_count"] == 0
            and later_assistant_activity[0]["message_index"]
            > transfer.result_message_index
            and isinstance(later_assistant_activity[0]["content"], str)
            and later_assistant_activity[0]["content"].strip()
            == required_terminal_message.strip()
        )
    passed = (
        transfer.result_success is True
        and context.stopped is True
        and terminal_protocol_passed
    )
    evidence = (
        _call_evidence(transfer),
        {
            "source": "termination",
            "stopped": context.stopped,
            "required_terminal_message": required_terminal_message,
            "later_assistant_activity": later_assistant_activity,
            "terminal_protocol_passed": terminal_protocol_passed,
        },
    )
    return _result(
        spec,
        AtomicVerdict.PASS if passed else AtomicVerdict.FAIL,
        "Transfer succeeded, the terminal-message contract was satisfied, and the agent stopped."
        if passed
        else "Transfer failed, the terminal-message contract was violated, or the agent did not stop.",
        evidence,
    )


def _assistant_turn_evidence(
    spec: PredicateSpec,
    context: RetailPredicateContext,
) -> tuple[tuple[Mapping[str, Any], ...] | None, PredicateResult | None]:
    if context.messages is None:
        return None, _error(
            spec,
            "messages evidence is missing",
            ({"source": "messages", "status": "MISSING"},),
        )
    turns: list[Mapping[str, Any]] = []
    for message_index, message in enumerate(context.messages):
        if not isinstance(message, Mapping):
            return None, _error(
                spec,
                f"messages[{message_index}] is not a mapping",
                ({"source": "messages", "message_index": message_index, "status": "INVALID"},),
            )
        if message.get("role") != "assistant":
            continue
        raw_calls = message.get("tool_calls", [])
        if raw_calls is None:
            raw_calls = []
        if not isinstance(raw_calls, list):
            return None, _error(
                spec,
                f"messages[{message_index}].tool_calls is not a list",
                ({"source": "assistant_turn", "message_index": message_index, "status": "INVALID"},),
            )
        tool_names = []
        for call_index, raw_call in enumerate(raw_calls):
            if not isinstance(raw_call, Mapping):
                return None, _error(
                    spec,
                    f"tool call {message_index}:{call_index} is not a mapping",
                    ({"source": "assistant_turn", "message_index": message_index, "call_index": call_index, "status": "INVALID"},),
                )
            name = raw_call.get("name")
            tool_names.append(name if isinstance(name, str) else None)
        content = message.get("content")
        has_content = bool(content.strip()) if isinstance(content, str) else content is not None
        turns.append(
            {
                "source": "assistant_turn",
                "message_index": message_index,
                "tool_call_count": len(raw_calls),
                "tool_names": tool_names,
                "has_user_facing_content": has_content,
            }
        )
    if not turns:
        turns.append({"source": "assistant_turns", "count": 0})
    return tuple(turns), None


def _one_tool_call_per_turn(spec: PredicateSpec, context: RetailPredicateContext) -> PredicateResult:
    turns, failure = _assistant_turn_evidence(spec, context)
    if failure:
        return failure
    assert turns is not None
    maximum = spec.parameters.get("max_calls", 1)
    if not isinstance(maximum, int) or isinstance(maximum, bool) or maximum < 0:
        return _error(
            spec,
            "max_calls must be a non-negative integer",
            ({"source": "parameters.max_calls", "status": "INVALID"},),
        )
    violating = [
        turn["message_index"]
        for turn in turns
        if "message_index" in turn and turn["tool_call_count"] > maximum
    ]
    evidence = tuple(
        {**dict(turn), "max_calls": maximum, "violates": turn.get("message_index") in violating}
        for turn in turns
    )
    verdict = AtomicVerdict.PASS if not violating else AtomicVerdict.FAIL
    return _result(
        spec,
        verdict,
        "Every assistant turn stays within the tool-call limit."
        if verdict is AtomicVerdict.PASS
        else "At least one assistant turn exceeds the tool-call limit.",
        evidence,
    )


def _message_tool_exclusivity(spec: PredicateSpec, context: RetailPredicateContext) -> PredicateResult:
    turns, failure = _assistant_turn_evidence(spec, context)
    if failure:
        return failure
    assert turns is not None
    violating = [
        turn["message_index"]
        for turn in turns
        if "message_index" in turn
        and turn["tool_call_count"] > 0
        and turn["has_user_facing_content"]
    ]
    evidence = tuple(
        {**dict(turn), "violates": turn.get("message_index") in violating}
        for turn in turns
    )
    verdict = AtomicVerdict.PASS if not violating else AtomicVerdict.FAIL
    return _result(
        spec,
        verdict,
        "Assistant messages and tool calls are separated."
        if verdict is AtomicVerdict.PASS
        else "At least one assistant turn combines user-facing content with tool calls.",
        evidence,
    )


PredicateFunction = Callable[[PredicateSpec, RetailPredicateContext], PredicateResult]

PREDICATE_REGISTRY: Mapping[str, PredicateFunction] = {
    "field_equals": _field_equals,
    "field_unchanged": _field_unchanged,
    "required_action": _required_action,
    "forbidden_action": _forbidden_action,
    "action_cardinality": _action_cardinality,
    "action_argument_equals": _action_argument_equals,
    "paired_action_arguments_equal": _paired_action_arguments_equal,
    "action_state_transition_consistent": _action_state_transition_consistent,
    "write_after_confirmation": _write_after_confirmation,
    "no_unrelated_mutation": _no_unrelated_mutation,
    "tool_result_success": _tool_result_success,
    "final_claim_matches_state": _final_claim_matches_state,
    "transfer_and_stop": _transfer_and_stop,
    "one_tool_call_per_turn": _one_tool_call_per_turn,
    "message_tool_exclusivity": _message_tool_exclusivity,
}


def evaluate_retail_predicate(
    spec: PredicateSpec,
    context: RetailPredicateContext,
) -> PredicateResult:
    function = PREDICATE_REGISTRY.get(spec.predicate_type)
    if function is None:
        return _error(
            spec,
            f"unsupported Retail predicate type: {spec.predicate_type}",
            ({"source": "predicate_registry", "predicate_type": spec.predicate_type, "status": "UNSUPPORTED"},),
        )
    return function(spec, context)
