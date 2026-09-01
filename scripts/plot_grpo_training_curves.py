"""Render the four core GRPO training curves from a TRL log history."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


METRICS = (
    ("reward", "train/reward"),
    ("reward_std", "train/reward_std"),
    ("kl", "train/kl"),
    ("loss", "train/loss"),
)


def load_log_history(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise TypeError(f"Expected a JSON array in {path}")
    rows = [row for row in payload if isinstance(row, dict)]
    if not rows:
        raise ValueError(f"No log rows found in {path}")
    return rows


def metric_points(
    rows: list[dict[str, Any]], metric: str
) -> tuple[list[float], list[float]]:
    steps: list[float] = []
    values: list[float] = []
    for row in rows:
        if row.get("step") is None or row.get(metric) is None:
            continue
        try:
            step = float(row["step"])
            value = float(row[metric])
        except (TypeError, ValueError):
            continue
        if math.isfinite(step) and math.isfinite(value):
            steps.append(step)
            values.append(value)
    return steps, values


def moving_average(values: list[float], window: int) -> list[float]:
    if window <= 1:
        return list(values)
    smoothed: list[float] = []
    for index in range(len(values)):
        start = max(0, index - window + 1)
        segment = values[start : index + 1]
        smoothed.append(sum(segment) / len(segment))
    return smoothed


def plot_curves(
    rows: list[dict[str, Any]],
    output_path: Path,
    *,
    label: str,
    smoothing_window: int = 1,
) -> None:
    if smoothing_window < 1:
        raise ValueError("smoothing_window must be at least 1")

    color = "#249789"
    fig, axes = plt.subplots(2, 2, figsize=(12, 10), constrained_layout=True)
    fig.patch.set_facecolor("white")

    for axis, (metric, title) in zip(axes.flat, METRICS, strict=True):
        steps, values = metric_points(rows, metric)
        axis.set_title(title, loc="left", fontsize=17, fontweight="bold")
        axis.set_xlabel("train/global_step", fontsize=12)
        axis.grid(axis="y", color="#E5E7EB", linewidth=0.8)
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
        axis.spines["left"].set_color("#D1D5DB")
        axis.spines["bottom"].set_color("#D1D5DB")
        axis.tick_params(colors="#5F6470")

        if not values:
            axis.text(
                0.5,
                0.5,
                "not logged",
                ha="center",
                va="center",
                transform=axis.transAxes,
                color="#6B7280",
                fontsize=13,
            )
            continue

        plotted_values = moving_average(values, smoothing_window)
        if smoothing_window > 1:
            axis.plot(steps, values, color=color, alpha=0.25, linewidth=1.0)
        axis.plot(
            steps,
            plotted_values,
            color=color,
            linewidth=2.0,
            marker="o" if len(steps) <= 12 else None,
            markersize=4,
            label=label,
        )
        axis.scatter(
            [steps[-1]], [plotted_values[-1]], color=color, s=28, zorder=3
        )
        axis.legend(frameon=False, loc="upper left", fontsize=11)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot train/reward, reward_std, kl, and loss from GRPO logs."
    )
    parser.add_argument(
        "run_dir", type=Path, help="Run directory containing log_history.json"
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="PNG output path (default: <run_dir>/grpo_training_curves.png)",
    )
    parser.add_argument("--label", help="Legend label (default: run directory name)")
    parser.add_argument(
        "--smoothing-window",
        type=int,
        default=1,
        help="Trailing moving-average window; 1 preserves raw values",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    log_path = args.run_dir / "log_history.json"
    output_path = args.output or args.run_dir / "grpo_training_curves.png"
    rows = load_log_history(log_path)
    plot_curves(
        rows,
        output_path,
        label=args.label or args.run_dir.name,
        smoothing_window=args.smoothing_window,
    )
    print(output_path.resolve())


if __name__ == "__main__":
    main()
