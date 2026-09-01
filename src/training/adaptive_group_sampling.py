"""Bounded adaptive sampling decisions for GRPO prompt groups.

This module is deliberately sampling-only.  It detects whether an observed
same-prompt group contains a usable reward contrast and, when configured,
at least one independently derived quality-eligible candidate.  It does not
calculate a parameter gradient, normalize advantages, or update a policy.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from statistics import pstdev
from typing import Any, Iterable, Mapping


_BOUND_FIELDS = (
    "task_id",
    "prompt_sha256",
    "initial_state_sha256",
    "policy_snapshot_sha256",
    "decoding_config_sha256",
    "reward_version",
)


@dataclass(frozen=True)
class AdaptiveSamplingPolicy:
    min_samples: int = 4
    increment: int = 2
    max_samples: int = 8
    mode: str = "terminal"
    reward_std_epsilon: float = 1e-6
    minimum_reward_range: float = 0.1
    minimum_quality_positive: int = 0

    def validate(self) -> None:
        if self.min_samples < 2:
            raise ValueError("min_samples must be at least 2")
        if self.increment <= 0:
            raise ValueError("increment must be positive")
        if self.max_samples < self.min_samples:
            raise ValueError("max_samples must be at least min_samples")
        if (self.max_samples - self.min_samples) % self.increment:
            raise ValueError("max_samples must be reachable by fixed increments")
        if self.mode not in {"terminal", "layered"}:
            raise ValueError("mode must be terminal or layered")
        if not math.isfinite(self.reward_std_epsilon) or self.reward_std_epsilon < 0:
            raise ValueError("reward_std_epsilon must be finite and non-negative")
        if not math.isfinite(self.minimum_reward_range) or self.minimum_reward_range < 0:
            raise ValueError("minimum_reward_range must be finite and non-negative")
        if (
            isinstance(self.minimum_quality_positive, bool)
            or not isinstance(self.minimum_quality_positive, int)
            or self.minimum_quality_positive < 0
            or self.minimum_quality_positive > self.max_samples
        ):
            raise ValueError(
                "minimum_quality_positive must be an integer between 0 and max_samples"
            )


def _validated_rows(observations: Iterable[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
    rows = list(observations)
    if not rows:
        raise ValueError("observations must not be empty")
    for field in _BOUND_FIELDS:
        values = []
        for row in rows:
            value = row.get(field)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{field} must be a non-empty string")
            values.append(value)
        if len(set(values)) != 1:
            raise ValueError(f"adaptive group binding drifted at {field}")

    rollout_ids = [row.get("rollout_id") for row in rows]
    environment_ids = [row.get("environment_instance_id") for row in rows]
    for name, values in (
        ("rollout_id", rollout_ids),
        ("environment_instance_id", environment_ids),
    ):
        if any(not isinstance(value, str) or not value.strip() for value in values):
            raise ValueError(f"{name} must be a non-empty string")
        if len(set(values)) != len(values):
            raise ValueError(f"{name} must be unique within an adaptive group")

    for row in rows:
        reward = row.get("reward")
        if isinstance(reward, bool) or not isinstance(reward, (int, float)):
            raise ValueError("reward must be numeric")
        if not math.isfinite(float(reward)):
            raise ValueError("reward must be finite")
    return rows


def evaluate_adaptive_group(
    observations: Iterable[Mapping[str, Any]], policy: AdaptiveSamplingPolicy
) -> dict[str, Any]:
    """Return ACCEPT, CONTINUE, or DROP for one bound same-prompt group."""
    policy.validate()
    rows = _validated_rows(observations)
    sample_count = len(rows)
    if sample_count < policy.min_samples:
        raise ValueError("group is smaller than min_samples")
    if sample_count > policy.max_samples:
        raise ValueError("group exceeds max_samples")
    if (sample_count - policy.min_samples) % policy.increment:
        raise ValueError("group size is not on the configured sampling schedule")

    rewards = [float(row["reward"]) for row in rows]
    reward_std = pstdev(rewards)
    reward_range = max(rewards) - min(rewards)
    terminal_successes = None
    has_usable_contrast = False
    quality_positive_count = None
    has_quality_coverage = True
    if policy.minimum_quality_positive:
        quality_flags = [row.get("quality_eligible") for row in rows]
        if any(type(value) is not bool for value in quality_flags):
            raise ValueError(
                "quality-gated sampling requires an explicit boolean quality_eligible"
            )
        quality_positive_count = sum(quality_flags)
        has_quality_coverage = (
            quality_positive_count >= policy.minimum_quality_positive
        )

    if policy.mode == "terminal":
        terminal_successes = [row.get("terminal_success") for row in rows]
        if any(type(value) is not bool for value in terminal_successes):
            raise ValueError("terminal mode requires an explicit boolean terminal_success")
        has_usable_contrast = any(terminal_successes) and not all(terminal_successes)
    else:
        has_usable_contrast = (
            reward_std > policy.reward_std_epsilon
            and reward_range >= policy.minimum_reward_range
        )

    if has_usable_contrast and has_quality_coverage:
        decision = "ACCEPT"
        reason = (
            "usable_contrast_and_quality_positive"
            if policy.minimum_quality_positive
            else "usable_within_group_reward_contrast"
        )
        next_sample_count = 0
    elif sample_count < policy.max_samples:
        decision = "CONTINUE"
        if not has_usable_contrast and not has_quality_coverage:
            reason = "no_usable_contrast_or_quality_positive_yet"
        elif not has_quality_coverage:
            reason = "no_quality_positive_yet"
        else:
            reason = "no_usable_contrast_yet"
        next_sample_count = min(policy.increment, policy.max_samples - sample_count)
    else:
        decision = "DROP"
        if not has_usable_contrast and not has_quality_coverage:
            reason = "no_usable_contrast_or_quality_positive_at_sampling_cap"
        elif not has_quality_coverage:
            reason = "no_quality_positive_at_sampling_cap"
        else:
            reason = "no_usable_contrast_at_sampling_cap"
        next_sample_count = 0

    return {
        "schema_version": "adaptive-group-sampling-decision-v1",
        "decision": decision,
        "reason": reason,
        "sample_count": sample_count,
        "next_sample_count": next_sample_count,
        "has_usable_contrast": has_usable_contrast,
        "reward_mean": sum(rewards) / sample_count,
        "reward_std": reward_std,
        "reward_range": reward_range,
        "terminal_success_count": (
            sum(terminal_successes) if terminal_successes is not None else None
        ),
        "minimum_quality_positive": policy.minimum_quality_positive,
        "quality_positive_count": quality_positive_count,
        "has_quality_coverage": has_quality_coverage,
        "signal_semantics": "reward_contrast_proxy_not_parameter_gradient",
        "bindings": {field: rows[0][field] for field in _BOUND_FIELDS},
    }
