from __future__ import annotations

from copy import deepcopy

import pytest

from src.training.adaptive_group_sampling import (
    AdaptiveSamplingPolicy,
    evaluate_adaptive_group,
)


def _rows(successes, rewards=None):
    if rewards is None:
        rewards = [float(value) for value in successes]
    return [
        {
            "rollout_id": f"rollout-{index}",
            "environment_instance_id": f"env-{index}",
            "task_id": "43",
            "prompt_sha256": "prompt-hash",
            "initial_state_sha256": "state-hash",
            "policy_snapshot_sha256": "policy-hash",
            "decoding_config_sha256": "decode-hash",
            "reward_version": "terminal-v1",
            "reward": reward,
            "terminal_success": success,
        }
        for index, (success, reward) in enumerate(zip(successes, rewards, strict=True))
    ]


def test_terminal_mixed_group_is_accepted_at_minimum():
    report = evaluate_adaptive_group(
        _rows([True, True, True, False]), AdaptiveSamplingPolicy()
    )
    assert report["decision"] == "ACCEPT"
    assert report["terminal_success_count"] == 3
    assert report["next_sample_count"] == 0
    assert report["signal_semantics"] == "reward_contrast_proxy_not_parameter_gradient"


def test_terminal_uniform_group_requests_two_more_samples():
    report = evaluate_adaptive_group(
        _rows([True, True, True, True]), AdaptiveSamplingPolicy()
    )
    assert report["decision"] == "CONTINUE"
    assert report["next_sample_count"] == 2


def test_terminal_uniform_group_is_dropped_at_cap():
    report = evaluate_adaptive_group(
        _rows([False] * 8), AdaptiveSamplingPolicy()
    )
    assert report["decision"] == "DROP"
    assert report["terminal_success_count"] == 0


def test_layered_mode_requires_material_not_numerical_contrast():
    policy = AdaptiveSamplingPolicy(mode="layered", minimum_reward_range=0.1)
    tiny_noise = evaluate_adaptive_group(
        _rows([False] * 4, [0.5, 0.500001, 0.5, 0.5]), policy
    )
    assert tiny_noise["decision"] == "CONTINUE"

    useful = evaluate_adaptive_group(
        _rows([False] * 4, [0.3, 0.5, 0.5, 0.5]), policy
    )
    assert useful["decision"] == "ACCEPT"


def test_binding_drift_is_rejected():
    rows = _rows([True, True, True, False])
    rows[3]["policy_snapshot_sha256"] = "different-policy"
    with pytest.raises(ValueError, match="policy_snapshot_sha256"):
        evaluate_adaptive_group(rows, AdaptiveSamplingPolicy())


def test_reused_environment_is_rejected():
    rows = _rows([True, True, True, False])
    rows[3]["environment_instance_id"] = rows[0]["environment_instance_id"]
    with pytest.raises(ValueError, match="environment_instance_id"):
        evaluate_adaptive_group(rows, AdaptiveSamplingPolicy())


def test_off_schedule_group_size_is_rejected():
    rows = _rows([True, True, True, True, False])
    with pytest.raises(ValueError, match="sampling schedule"):
        evaluate_adaptive_group(rows, AdaptiveSamplingPolicy())


def test_terminal_success_must_be_explicit_boolean():
    rows = deepcopy(_rows([True, True, True, False]))
    rows[0]["terminal_success"] = 1
    with pytest.raises(ValueError, match="explicit boolean"):
        evaluate_adaptive_group(rows, AdaptiveSamplingPolicy())


def test_quality_gate_continues_despite_terminal_contrast_without_safe_positive():
    rows = _rows([True, True, True, False])
    for row in rows:
        row["quality_eligible"] = False
    report = evaluate_adaptive_group(
        rows,
        AdaptiveSamplingPolicy(minimum_quality_positive=1),
    )
    assert report["decision"] == "CONTINUE"
    assert report["reason"] == "no_quality_positive_yet"
    assert report["quality_positive_count"] == 0
    assert report["has_usable_contrast"] is True


def test_quality_gate_accepts_contrast_with_safe_positive():
    rows = _rows([True, True, True, False])
    for index, row in enumerate(rows):
        row["quality_eligible"] = index == 1
    report = evaluate_adaptive_group(
        rows,
        AdaptiveSamplingPolicy(minimum_quality_positive=1),
    )
    assert report["decision"] == "ACCEPT"
    assert report["reason"] == "usable_contrast_and_quality_positive"
    assert report["quality_positive_count"] == 1


def test_quality_gate_drops_contrast_only_group_at_sampling_cap():
    rows = _rows([True, True, True, False, True, False, True, False])
    for row in rows:
        row["quality_eligible"] = False
    report = evaluate_adaptive_group(
        rows,
        AdaptiveSamplingPolicy(minimum_quality_positive=1),
    )
    assert report["decision"] == "DROP"
    assert report["reason"] == "no_quality_positive_at_sampling_cap"


def test_quality_gate_requires_explicit_boolean_label():
    rows = _rows([True, True, True, False])
    for row in rows:
        row["quality_eligible"] = False
    rows[0]["quality_eligible"] = 1
    with pytest.raises(ValueError, match="explicit boolean quality_eligible"):
        evaluate_adaptive_group(
            rows,
            AdaptiveSamplingPolicy(minimum_quality_positive=1),
        )
