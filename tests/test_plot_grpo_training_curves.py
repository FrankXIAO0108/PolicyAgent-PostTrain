from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.plot_grpo_training_curves import (
    load_log_history,
    metric_points,
    moving_average,
    plot_curves,
)


def test_metric_points_ignores_missing_and_nonfinite_values() -> None:
    rows = [
        {"step": 1, "reward": 0.25},
        {"step": 2, "reward": None},
        {"step": 3, "reward": "nan"},
        {"step": 4, "loss": 0.1},
    ]

    assert metric_points(rows, "reward") == ([1.0], [0.25])
    assert metric_points(rows, "kl") == ([], [])


def test_moving_average_uses_trailing_window() -> None:
    assert moving_average([1.0, 2.0, 6.0], 2) == [1.0, 1.5, 4.0]


def test_plot_curves_writes_png_when_kl_is_missing(tmp_path: Path) -> None:
    log_path = tmp_path / "log_history.json"
    log_path.write_text(
        json.dumps(
            [
                {"step": 1, "reward": 0.5, "reward_std": 0.5, "loss": -0.1},
                {"step": 2, "reward": 1.0, "reward_std": 0.0, "loss": 0.0},
            ]
        ),
        encoding="utf-8",
    )
    output_path = tmp_path / "curves.png"

    rows = load_log_history(log_path)
    plot_curves(rows, output_path, label="test-run")

    assert output_path.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")


def test_load_log_history_rejects_non_array(tmp_path: Path) -> None:
    log_path = tmp_path / "log_history.json"
    log_path.write_text("{}", encoding="utf-8")

    with pytest.raises(TypeError, match="Expected a JSON array"):
        load_log_history(log_path)
