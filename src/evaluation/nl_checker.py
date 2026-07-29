from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(slots=True)
class NLCheckResult:
    nl_match: bool | None
    source: str
    assertions: list[dict[str, Any]] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def check_recorded_nl_assertions(task: Any, simulation: Any) -> NLCheckResult:
    """Read the frozen Tau2 NL-judge result without making a new model call."""

    expected = list(task.evaluation_criteria.nl_assertions or [])
    if not expected:
        return NLCheckResult(
            nl_match=True,
            source="no_assertions",
            notes=["Tau2 treats missing/empty nl_assertions as reward 1."],
        )

    reward_info = simulation.reward_info
    recorded = list(reward_info.nl_assertions or []) if reward_info else []
    if recorded:
        assertions = [
            {
                "assertion": item.nl_assertion,
                "met": item.met,
                "justification": item.justification,
            }
            for item in recorded
        ]
        return NLCheckResult(
            nl_match=all(item["met"] for item in assertions),
            source="recorded_official_judge",
            assertions=assertions,
        )

    breakdown = reward_info.reward_breakdown if reward_info else None
    if breakdown:
        for key, value in breakdown.items():
            if getattr(key, "value", key) == "NL_ASSERTION":
                return NLCheckResult(
                    nl_match=float(value) == 1.0,
                    source="recorded_reward_breakdown",
                    notes=[
                        "Per-assertion records were absent; used frozen component reward."
                    ],
                )

    return NLCheckResult(
        nl_match=None,
        source="not_evaluated",
        notes=[
            "Task has NL assertions but the artifact contains no recorded judge result."
        ],
    )
