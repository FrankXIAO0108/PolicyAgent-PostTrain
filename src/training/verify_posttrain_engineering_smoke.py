from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

from src.training.run_posttrain_engineering_smoke import directory_sha256, sha256


REQUIRED_STAGES = ("SFT", "DPO", "GRPO")
REQUIRED_EVALUATIONS = ("base", "sft", "dpo", "grpo")


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def verify_run(run_dir: Path) -> dict[str, Any]:
    manifest_path = run_dir / "run_manifest.json"
    reasons: list[str] = []
    if not manifest_path.is_file():
        return {
            "schema_version": "posttrain-engineering-smoke-verification-v1",
            "verified_complete": False,
            "completion_claim_allowed": False,
            "reasons": [f"Missing run manifest: {manifest_path}"],
        }

    manifest = _load(manifest_path)
    if manifest.get("scope") != "ISOLATED_ENGINEERING_SMOKE":
        reasons.append("Run scope is not ISOLATED_ENGINEERING_SMOKE.")
    if manifest.get("status") != "COMPLETED":
        reasons.append("Run status is not COMPLETED.")
    if manifest.get("git", {}).get("dirty_at_start"):
        reasons.append("Training started from a dirty worktree.")
    if manifest.get("formal_retail_readiness_gate_opened") is not False:
        reasons.append("Formal Retail readiness gate boundary is missing or invalid.")
    if manifest.get("business_improvement_claim_allowed") is not False:
        reasons.append("Business-improvement claim must remain prohibited.")

    stages = {row.get("stage"): row for row in manifest.get("stages", [])}
    for stage in REQUIRED_STAGES:
        row = stages.get(stage)
        if row is None:
            reasons.append(f"Missing stage: {stage}.")
            continue
        if row.get("status") != "COMPLETED":
            reasons.append(f"Stage {stage} is not COMPLETED.")
        artifact_path = Path(str(row.get("artifact_path", "")))
        if not artifact_path.is_absolute():
            artifact_path = run_dir / artifact_path
        if not artifact_path.is_dir():
            reasons.append(f"Missing stage artifact directory: {artifact_path}.")
        elif directory_sha256(artifact_path) != row.get("artifact_sha256"):
            reasons.append(f"Stage artifact hash mismatch: {stage}.")
        for artifact_name in ("adapter", "merged_model", "checkpoint"):
            artifact = row.get(artifact_name, {})
            bound_path = Path(str(artifact.get("path", "")))
            if not bound_path.is_dir():
                reasons.append(f"Missing {stage} {artifact_name}: {bound_path}.")
            elif directory_sha256(bound_path) != artifact.get("sha256"):
                reasons.append(f"{stage} {artifact_name} hash mismatch.")
        loss_history = row.get("loss_history", {})
        loss_path = Path(str(loss_history.get("path", "")))
        if not loss_path.is_file():
            reasons.append(f"Missing {stage} loss history: {loss_path}.")
        elif sha256(loss_path) != loss_history.get("sha256"):
            reasons.append(f"{stage} loss-history hash mismatch.")
        if (loss_history.get("rows") or 0) <= 0:
            reasons.append(f"Stage {stage} loss history is empty.")
        metrics = row.get("train_metrics", {})
        loss = metrics.get("train_loss")
        if not isinstance(loss, (int, float)) or not math.isfinite(loss):
            reasons.append(f"Stage {stage} lacks a finite train_loss.")

    evaluations = manifest.get("holdout_evaluations", {})
    for name in REQUIRED_EVALUATIONS:
        metrics = evaluations.get(name)
        if not isinstance(metrics, dict):
            reasons.append(f"Missing holdout evaluation: {name}.")
            continue
        if metrics.get("rows") != 8:
            reasons.append(f"Holdout evaluation {name} must contain 8 rows.")
        for metric_name in (
            "valid_json_rate",
            "tool_match_rate",
            "arguments_match_rate",
            "exact_action_match_rate",
        ):
            value = metrics.get(metric_name)
            if not isinstance(value, (int, float)) or not 0.0 <= value <= 1.0:
                reasons.append(f"Invalid {name}.{metric_name}: {value!r}.")

    bindings = manifest.get("bindings", {})
    for path_key, hash_key in (
        ("config_path", "config_sha256"),
        ("data_manifest_path", "data_manifest_sha256"),
    ):
        bound_path = Path(str(bindings.get(path_key, "")))
        expected_hash = bindings.get(hash_key)
        if not bound_path.is_file():
            reasons.append(f"Missing bound input: {bound_path}.")
        elif sha256(bound_path) != expected_hash:
            reasons.append(f"Bound input hash mismatch: {bound_path}.")

    environment = manifest.get("environment", {})
    for package in ("torch", "transformers", "trl", "datasets", "peft", "accelerate"):
        if not environment.get(package):
            reasons.append(f"Missing runtime package version: {package}.")

    verified = not reasons
    return {
        "schema_version": "posttrain-engineering-smoke-verification-v1",
        "run_manifest_path": str(manifest_path.resolve()),
        "run_manifest_sha256": sha256(manifest_path),
        "verified_complete": verified,
        "completion_claim_allowed": verified,
        "allowed_claim_zh": (
            "已完成隔离合成数据上的 SFT→DPO→GRPO 工程闭环实操，并形成可复现的训练、"
            "checkpoint 与冻结评测证据；该结果不代表 Retail 正式业务指标提升。"
            if verified
            else None
        ),
        "prohibited_claims": [
            "已完成正式 Retail SFT/DPO/GRPO 训练闭环",
            "SFT/DPO/GRPO 已提升冻结 Retail 业务成功率",
            "GRPO reward 已通过独立人工 gold 验证",
        ],
        "reasons": reasons,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = verify_run(args.run_dir.resolve())
    text = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        args.output.write_text(text, encoding="utf-8")
    print(text, end="")
    raise SystemExit(0 if report["verified_complete"] else 1)


if __name__ == "__main__":
    main()
