from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any


class Verdict(str, Enum):
    PASS = "PASS"
    FAIL = "FAIL"
    REVIEW = "REVIEW"
    NOT_EVALUATED = "NOT_EVALUATED"


class Severity(str, Enum):
    INFO = "INFO"
    MINOR = "MINOR"
    MAJOR = "MAJOR"


class Dimension(str, Enum):
    BENCHMARK_REWARD = "benchmark_reward"
    LATEST_INTENT = "latest_intent"
    EXPLICIT_CONFIRMATION = "explicit_confirmation"
    POLICY_COMPLIANCE = "policy_compliance"
    ACTION_RESULT_TRUTHFULNESS = "action_result_truthfulness"


@dataclass(slots=True)
class ToolCall:
    id: str
    name: str
    arguments: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class MessageEvent:
    index: int
    role: str
    content: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    tool_call_id: str | None = None
    tool_error: bool = False
    source: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            **asdict(self),
            "tool_calls": [call.to_dict() for call in self.tool_calls],
        }


@dataclass(slots=True)
class ArtifactBundle:
    task_dir: Path
    events: list[MessageEvent]
    task: dict[str, Any] = field(default_factory=dict)
    summary: dict[str, Any] = field(default_factory=dict)
    source_path: Path | None = None

    @property
    def task_id(self) -> str:
        value = self.task.get("id", self.summary.get("task_id", self.task_dir.name))
        return str(value)


@dataclass(slots=True)
class Finding:
    code: str
    dimension: Dimension
    severity: Severity
    message: str
    event_indices: list[int] = field(default_factory=list)
    evidence: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["dimension"] = self.dimension.value
        value["severity"] = self.severity.value
        return value


@dataclass(slots=True)
class VerificationResult:
    task_id: str
    verdict: Verdict
    dimensions: dict[Dimension, Verdict]
    findings: list[Finding] = field(default_factory=list)
    metrics: dict[str, Any] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "verdict": self.verdict.value,
            "dimensions": {
                dimension.value: verdict.value
                for dimension, verdict in self.dimensions.items()
            },
            "findings": [finding.to_dict() for finding in self.findings],
            "metrics": self.metrics,
            "notes": self.notes,
        }
