"""Policy-grounding trajectory verification primitives."""

from typing import Any

from .schemas import (
    ArtifactBundle,
    Dimension,
    Finding,
    MessageEvent,
    Severity,
    ToolCall,
    VerificationResult,
    Verdict,
)
from .trajectory_loader import load_task_artifacts, load_trajectory


def verify_artifacts(*args: Any, **kwargs: Any) -> VerificationResult:
    from .policy_grounding_v1 import verify_artifacts as implementation

    return implementation(*args, **kwargs)


def verify_trajectory(*args: Any, **kwargs: Any) -> VerificationResult:
    from .policy_grounding_v1 import verify_trajectory as implementation

    return implementation(*args, **kwargs)

__all__ = [
    "ArtifactBundle",
    "Dimension",
    "Finding",
    "MessageEvent",
    "Severity",
    "ToolCall",
    "VerificationResult",
    "Verdict",
    "load_task_artifacts",
    "load_trajectory",
    "verify_artifacts",
    "verify_trajectory",
]
