from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any


SCHEMA_VERSION = "retail-agentic-grpo-training-audit-v1"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def _read_json(path: Path) -> dict[str, Any] | list[dict[str, Any]]:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not rows:
        raise ValueError(f"No JSONL rows found in {path}")
    return rows


def _mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def _range(values: list[float]) -> dict[str, float | None]:
    return {
        "min": min(values) if values else None,
        "max": max(values) if values else None,
    }


def _reward_value(row: dict[str, Any]) -> float:
    reward = row.get("reward")
    if not isinstance(reward, dict) or "reward" not in reward:
        raise ValueError("Rollout row is missing reward.reward")
    return float(reward["reward"])


def _group_rollouts(
    rows: list[dict[str, Any]], num_generations: int
) -> list[list[dict[str, Any]]]:
    if num_generations < 2:
        raise ValueError("num_generations must be at least 2 for a GRPO audit")
    if len(rows) % num_generations:
        raise ValueError(
            f"Rollout rows ({len(rows)}) are not divisible by num_generations "
            f"({num_generations})"
        )
    groups: list[list[dict[str, Any]]] = []
    for start in range(0, len(rows), num_generations):
        group = rows[start : start + num_generations]
        task_ids = {str(row.get("task_id")) for row in group}
        if len(task_ids) != 1:
            raise ValueError(
                "Sequential rollout group contains multiple task IDs: "
                f"rows {start}-{start + num_generations - 1}: {sorted(task_ids)}"
            )
        groups.append(group)
    return groups


def build_training_audit(
    *,
    config: dict[str, Any],
    manifest: dict[str, Any],
    log_history: list[dict[str, Any]],
    raw_rollouts: list[dict[str, Any]],
    train_metrics: dict[str, Any],
    source: dict[str, Any],
) -> dict[str, Any]:
    grpo = dict(config.get("grpo") or {})
    num_generations = int(grpo.get("num_generations") or 0)
    groups = _group_rollouts(raw_rollouts, num_generations)
    step_logs = [row for row in log_history if "reward_std" in row]
    if not step_logs:
        raise ValueError("log_history contains no rows with reward_std")

    group_kinds: Counter[str] = Counter()
    effective_task_ids: set[str] = set()
    for group in groups:
        rewards = [_reward_value(row) for row in group]
        if len(set(rewards)) > 1:
            group_kinds["mixed"] += 1
            effective_task_ids.add(str(group[0].get("task_id")))
        elif all(value == 0.0 for value in rewards):
            group_kinds["all_zero"] += 1
        elif all(value > 0.0 for value in rewards):
            group_kinds["all_positive"] += 1
        else:
            group_kinds["uniform_nonzero"] += 1

    reward_stds = [float(row["reward_std"]) for row in step_logs]
    grad_norms = [float(row.get("grad_norm") or 0.0) for row in step_logs]
    mean_lengths = [
        float(row["completions/mean_length"])
        for row in step_logs
        if "completions/mean_length" in row
    ]
    clipped_ratios = [
        float(row["completions/clipped_ratio"])
        for row in step_logs
        if "completions/clipped_ratio" in row
    ]
    entropies = [
        float(row["entropy"]) for row in step_logs if row.get("entropy") is not None
    ]
    step_times = [
        float(row["step_time"])
        for row in step_logs
        if row.get("step_time") is not None
    ]
    zero_std_count = sum(math.isclose(value, 0.0, abs_tol=1e-12) for value in reward_stds)
    nonzero_grad_rows = [
        row
        for row in step_logs
        if not math.isclose(float(row.get("grad_norm") or 0.0), 0.0, abs_tol=1e-12)
    ]
    effective_steps = [
        int(row.get("step") or index + 1)
        for index, row in enumerate(step_logs)
        if not math.isclose(float(row["reward_std"]), 0.0, abs_tol=1e-12)
    ]
    clipped_step_count = sum(value > 0.0 for value in clipped_ratios)
    positive_rollouts = sum(_reward_value(row) > 0.0 for row in raw_rollouts)

    warnings: list[dict[str, str]] = []
    zero_std_fraction = zero_std_count / len(step_logs)
    if zero_std_fraction >= 0.5:
        warnings.append(
            {
                "code": "SPARSE_RELATIVE_ADVANTAGE",
                "evidence": f"zero reward std on {zero_std_count}/{len(step_logs)} steps",
            }
        )
    mean_clipped_ratio = _mean(clipped_ratios)
    if mean_clipped_ratio is not None and mean_clipped_ratio >= 0.1:
        warnings.append(
            {
                "code": "HIGH_COMPLETION_CLIPPING",
                "evidence": f"mean clipped ratio is {mean_clipped_ratio:.6f}",
            }
        )
    if float(grpo.get("beta") or 0.0) == 0.0:
        warnings.append(
            {
                "code": "NO_KL_CONSTRAINT",
                "evidence": "GRPO beta is 0; no KL regularization curve is available",
            }
        )

    return {
        "schema_version": SCHEMA_VERSION,
        "scope": "TRAINING_SIGNAL_AUDIT_ONLY",
        "source": source,
        "bindings": {
            "run_status": manifest.get("status"),
            "project_commit": (manifest.get("git") or {}).get("commit"),
            "starting_model_sha256": (manifest.get("bindings") or {}).get(
                "starting_model_sha256"
            ),
            "raw_rollouts_sha256": (
                (manifest.get("artifacts") or {}).get("raw_rollouts") or {}
            ).get("sha256"),
        },
        "configuration": {
            "task_ids": list((config.get("data") or {}).get("task_ids") or []),
            "max_steps": grpo.get("max_steps"),
            "num_generations": num_generations,
            "max_completion_length": grpo.get("max_completion_length"),
            "per_device_train_batch_size": grpo.get("per_device_train_batch_size"),
            "gradient_accumulation_steps": grpo.get("gradient_accumulation_steps"),
            "learning_rate": grpo.get("learning_rate"),
            "temperature": grpo.get("temperature"),
            "beta": grpo.get("beta"),
            "loss_type": grpo.get("loss_type"),
        },
        "training_signal": {
            "training_steps": len(step_logs),
            "rollouts": len(raw_rollouts),
            "groups": len(groups),
            "positive_rollouts": positive_rollouts,
            "grouping_method": "sequential_rows_by_num_generations",
            "group_counts": {
                "all_zero": group_kinds["all_zero"],
                "mixed": group_kinds["mixed"],
                "all_positive": group_kinds["all_positive"],
                "uniform_nonzero": group_kinds["uniform_nonzero"],
            },
            "effective_task_ids": sorted(effective_task_ids),
            "zero_reward_std_steps": zero_std_count,
            "zero_reward_std_fraction": zero_std_fraction,
            "effective_steps": effective_steps,
            "nonzero_grad_steps": len(nonzero_grad_rows),
            "grad_norm_range": _range(
                [float(row["grad_norm"]) for row in nonzero_grad_rows]
            ),
        },
        "generation": {
            "mean_completion_length": _mean(mean_lengths),
            "first_step_mean_length": mean_lengths[0] if mean_lengths else None,
            "last_step_mean_length": mean_lengths[-1] if mean_lengths else None,
            "mean_length_range": _range(mean_lengths),
            "steps_with_clipping": clipped_step_count,
            "mean_clipped_ratio": mean_clipped_ratio,
            "max_clipped_ratio": max(clipped_ratios) if clipped_ratios else None,
        },
        "optimization": {
            "train_runtime_seconds": train_metrics.get("train_runtime"),
            "train_loss": train_metrics.get("train_loss"),
            "mean_step_time_seconds": _mean(step_times),
            "rollout_vs_update_time_breakdown_available": False,
            "entropy_first": entropies[0] if entropies else None,
            "entropy_last": entropies[-1] if entropies else None,
            "entropy_range": _range(entropies),
            "kl_curve_available": float(grpo.get("beta") or 0.0) != 0.0,
        },
        "warnings": warnings,
        "claim_limits": {
            "parameter_update_observed": bool(nonzero_grad_rows and effective_steps),
            "behavior_improvement_assessed": False,
            "causal_failure_diagnosis_proven": False,
        },
    }


def audit_run(run_dir: Path, config_path: Path) -> dict[str, Any]:
    run_dir = run_dir.resolve()
    config_path = config_path.resolve()
    manifest_path = run_dir / "run_manifest.json"
    log_history_path = run_dir / "log_history.json"
    raw_rollouts_path = run_dir / "raw_rollouts.jsonl"
    train_metrics_path = run_dir / "train_metrics.json"
    required = [
        config_path,
        manifest_path,
        log_history_path,
        raw_rollouts_path,
        train_metrics_path,
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing audit inputs: {missing}")

    config = _read_json(config_path)
    manifest = _read_json(manifest_path)
    log_history = _read_json(log_history_path)
    train_metrics = _read_json(train_metrics_path)
    if not all(isinstance(value, dict) for value in [config, manifest, train_metrics]):
        raise TypeError("Config, manifest, and train metrics must be JSON objects")
    if not isinstance(log_history, list):
        raise TypeError("log_history must be a JSON array")

    actual_config_sha = _sha256(config_path)
    expected_config_sha = str(
        ((manifest.get("bindings") or {}).get("config_sha256") or "")
    ).upper()
    if expected_config_sha and actual_config_sha != expected_config_sha:
        raise ValueError("Config SHA-256 does not match run_manifest.json")
    actual_rollout_sha = _sha256(raw_rollouts_path)
    expected_rollout_sha = str(
        (((manifest.get("artifacts") or {}).get("raw_rollouts") or {}).get("sha256") or "")
    ).upper()
    if expected_rollout_sha and actual_rollout_sha != expected_rollout_sha:
        raise ValueError("Raw rollout SHA-256 does not match run_manifest.json")

    return build_training_audit(
        config=config,
        manifest=manifest,
        log_history=log_history,
        raw_rollouts=_read_jsonl(raw_rollouts_path),
        train_metrics=train_metrics,
        source={
            "run_dir": str(run_dir),
            "config_path": str(config_path),
            "config_sha256": actual_config_sha,
            "manifest_path": str(manifest_path),
            "manifest_sha256": _sha256(manifest_path),
            "log_history_path": str(log_history_path),
            "log_history_sha256": _sha256(log_history_path),
            "raw_rollouts_path": str(raw_rollouts_path),
            "raw_rollouts_sha256": actual_rollout_sha,
            "train_metrics_path": str(train_metrics_path),
            "train_metrics_sha256": _sha256(train_metrics_path),
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit a completed Agentic GRPO run")
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    report = audit_run(args.run_dir, args.config)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
