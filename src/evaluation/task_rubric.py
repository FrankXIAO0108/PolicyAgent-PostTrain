from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Mapping


SCHEMA_VERSION = "retail-strict-task-rubric-v1"


class RubricSchemaError(ValueError):
    """Raised when a strict-evaluation rubric violates its data contract."""


class AtomicVerdict(str, Enum):
    PASS = "PASS"
    FAIL = "FAIL"
    REVIEW = "REVIEW"
    NOT_APPLICABLE = "NOT_APPLICABLE"
    ERROR = "ERROR"

    @property
    def counts_as_pass(self) -> bool:
        return self is AtomicVerdict.PASS

    @property
    def counts_as_model_failure(self) -> bool:
        return self is AtomicVerdict.FAIL

    @property
    def invalidates_evaluation(self) -> bool:
        return self is AtomicVerdict.ERROR


class CapabilityGroup(str, Enum):
    EVALUATION_INTEGRITY = "evaluation_integrity"
    FINAL_STATE_CORRECTNESS = "final_state_correctness"
    REQUIRED_TASK_EXECUTION = "required_task_execution"
    INVALID_ACTION_AVOIDANCE = "invalid_action_avoidance"
    PROTOCOL_COMPLIANCE = "protocol_compliance"
    INTENT_ALIGNMENT = "intent_alignment"
    EVIDENCE_CONSISTENCY = "evidence_consistency"


@dataclass(frozen=True, slots=True)
class PredicateSpec:
    predicate_id: str
    predicate_type: str
    capability_group: CapabilityGroup
    required: bool
    parameters: Mapping[str, Any] = field(default_factory=dict)
    description: str | None = None


@dataclass(frozen=True, slots=True)
class TaskRubric:
    schema_version: str
    task_id: str
    domain: str
    required_capability_groups: tuple[CapabilityGroup, ...]
    predicates: tuple[PredicateSpec, ...]
    description: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class PredicateResult:
    """Structured result returned later by an atomic predicate implementation."""

    predicate_id: str
    capability_group: CapabilityGroup
    verdict: AtomicVerdict
    reason: str
    evidence: tuple[Mapping[str, Any], ...] = ()
    error: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.predicate_id, str) or not self.predicate_id.strip():
            raise RubricSchemaError("predicate result predicate_id must be non-empty")
        if not isinstance(self.capability_group, CapabilityGroup):
            raise RubricSchemaError(
                "predicate result capability_group must be a CapabilityGroup"
            )
        if not isinstance(self.verdict, AtomicVerdict):
            raise RubricSchemaError(
                "predicate result verdict must be an AtomicVerdict"
            )
        if not isinstance(self.reason, str) or not self.reason.strip():
            raise RubricSchemaError("predicate result reason must be non-empty")
        if not isinstance(self.evidence, tuple) or any(
            not isinstance(item, Mapping) for item in self.evidence
        ):
            raise RubricSchemaError(
                "predicate result evidence must be a tuple of mappings"
            )
        if self.verdict is AtomicVerdict.ERROR:
            if not isinstance(self.error, str) or not self.error.strip():
                raise RubricSchemaError("ERROR predicate result requires error detail")
        elif self.error is not None:
            raise RubricSchemaError("Only ERROR predicate results may carry error detail")

    @property
    def counts_as_pass(self) -> bool:
        return self.verdict.counts_as_pass

    @property
    def counts_as_model_failure(self) -> bool:
        return self.verdict.counts_as_model_failure


ROOT_REQUIRED_KEYS = {
    "schema_version",
    "task_id",
    "domain",
    "required_capability_groups",
    "predicates",
}
ROOT_OPTIONAL_KEYS = {"description", "metadata"}
PREDICATE_REQUIRED_KEYS = {
    "predicate_id",
    "predicate_type",
    "capability_group",
    "required",
    "parameters",
}
PREDICATE_OPTIONAL_KEYS = {"description"}


def _object(value: Any, location: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise RubricSchemaError(f"{location} must be a JSON object")
    if any(not isinstance(key, str) for key in value):
        raise RubricSchemaError(f"{location} keys must be strings")
    return value


def _exact_keys(
    value: Mapping[str, Any],
    *,
    required: set[str],
    optional: set[str],
    location: str,
) -> None:
    missing = sorted(required - set(value))
    unknown = sorted(set(value) - required - optional)
    if missing:
        raise RubricSchemaError(f"{location} missing required keys: {missing}")
    if unknown:
        raise RubricSchemaError(f"{location} contains unknown keys: {unknown}")


def _nonempty_string(value: Any, location: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise RubricSchemaError(f"{location} must be a non-empty string")
    return value


def _optional_string(value: Any, location: str) -> str | None:
    if value is None:
        return None
    return _nonempty_string(value, location)


def _json_value(value: Any, location: str) -> Any:
    if isinstance(value, float) and not math.isfinite(value):
        raise RubricSchemaError(f"{location} must not contain NaN or Infinity")
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, list):
        return [_json_value(item, f"{location}[]") for item in value]
    if isinstance(value, dict):
        if any(not isinstance(key, str) for key in value):
            raise RubricSchemaError(f"{location} keys must be strings")
        return {
            key: _json_value(item, f"{location}.{key}")
            for key, item in value.items()
        }
    raise RubricSchemaError(f"{location} must contain JSON-compatible values")


def _capability_group(value: Any, location: str) -> CapabilityGroup:
    text = _nonempty_string(value, location)
    try:
        return CapabilityGroup(text)
    except ValueError as exc:
        raise RubricSchemaError(
            f"{location} has unsupported capability group: {text!r}"
        ) from exc


def parse_task_rubric(payload: Any) -> TaskRubric:
    root = _object(payload, "rubric")
    _exact_keys(
        root,
        required=ROOT_REQUIRED_KEYS,
        optional=ROOT_OPTIONAL_KEYS,
        location="rubric",
    )
    schema_version = _nonempty_string(root["schema_version"], "schema_version")
    if schema_version != SCHEMA_VERSION:
        raise RubricSchemaError(
            f"Unsupported schema_version {schema_version!r}; expected {SCHEMA_VERSION!r}"
        )

    raw_groups = root["required_capability_groups"]
    if not isinstance(raw_groups, list) or not raw_groups:
        raise RubricSchemaError(
            "required_capability_groups must be a non-empty JSON array"
        )
    groups = tuple(
        _capability_group(value, f"required_capability_groups[{index}]")
        for index, value in enumerate(raw_groups)
    )
    if len(set(groups)) != len(groups):
        raise RubricSchemaError("required_capability_groups contains duplicates")

    raw_predicates = root["predicates"]
    if not isinstance(raw_predicates, list) or not raw_predicates:
        raise RubricSchemaError("predicates must be a non-empty JSON array")
    predicates: list[PredicateSpec] = []
    predicate_ids: set[str] = set()
    for index, raw_predicate in enumerate(raw_predicates):
        location = f"predicates[{index}]"
        predicate = _object(raw_predicate, location)
        _exact_keys(
            predicate,
            required=PREDICATE_REQUIRED_KEYS,
            optional=PREDICATE_OPTIONAL_KEYS,
            location=location,
        )
        predicate_id = _nonempty_string(
            predicate["predicate_id"], f"{location}.predicate_id"
        )
        if predicate_id in predicate_ids:
            raise RubricSchemaError(f"duplicate predicate_id: {predicate_id!r}")
        predicate_ids.add(predicate_id)
        required = predicate["required"]
        if not isinstance(required, bool):
            raise RubricSchemaError(f"{location}.required must be a boolean")
        capability_group = _capability_group(
            predicate["capability_group"], f"{location}.capability_group"
        )
        if required and capability_group not in groups:
            raise RubricSchemaError(
                f"required predicate {predicate_id!r} belongs to a non-required "
                f"capability group {capability_group.value!r}"
            )
        parameters = _object(predicate["parameters"], f"{location}.parameters")
        predicates.append(
            PredicateSpec(
                predicate_id=predicate_id,
                predicate_type=_nonempty_string(
                    predicate["predicate_type"], f"{location}.predicate_type"
                ),
                capability_group=capability_group,
                required=required,
                parameters=_json_value(parameters, f"{location}.parameters"),
                description=_optional_string(
                    predicate.get("description"), f"{location}.description"
                ),
            )
        )

    uncovered = [
        group.value
        for group in groups
        if not any(
            predicate.required and predicate.capability_group is group
            for predicate in predicates
        )
    ]
    if uncovered:
        raise RubricSchemaError(
            f"required capability groups lack a required predicate: {uncovered}"
        )

    metadata = _object(root.get("metadata", {}), "metadata")
    return TaskRubric(
        schema_version=schema_version,
        task_id=_nonempty_string(root["task_id"], "task_id"),
        domain=_nonempty_string(root["domain"], "domain"),
        required_capability_groups=groups,
        predicates=tuple(predicates),
        description=_optional_string(root.get("description"), "description"),
        metadata=_json_value(metadata, "metadata"),
    )


def load_task_rubric(path: str | Path) -> TaskRubric:
    source = Path(path)
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise RubricSchemaError(f"Unable to load rubric {source}: {exc}") from exc
    return parse_task_rubric(payload)
