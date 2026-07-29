from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def _load(path: Path | None) -> dict[str, Any] | None:
    if path is None:
        return None
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _input(path: Path | None) -> dict[str, str] | None:
    if path is None:
        return None
    return {"path": str(path), "sha256": sha256(path)}


def _verify_sibling(
    manifest_path: Path,
    filename: str,
    expected_hash: str,
    reasons: list[str],
) -> None:
    artifact = manifest_path.parent / filename
    if not artifact.is_file():
        reasons.append(f"Missing artifact bound by manifest: {artifact}")
        return
    actual = sha256(artifact)
    if actual != expected_hash.upper():
        reasons.append(
            f"Hash mismatch for {artifact}: expected={expected_hash}, actual={actual}"
        )


def evaluate_readiness(
    *,
    policy_validation_path: Path | None = None,
    sft_dataset_manifest_path: Path | None = None,
    sft_run_manifest_path: Path | None = None,
    comparison_manifest_path: Path | None = None,
    preference_manifest_path: Path | None = None,
    reward_validation_path: Path | None = None,
    reward_min_precision: float = 0.90,
    reward_min_recall: float = 0.90,
    critical_min_recall: float = 0.95,
) -> dict[str, Any]:
    policy = _load(policy_validation_path)
    dataset = _load(sft_dataset_manifest_path)
    run = _load(sft_run_manifest_path)
    comparison = _load(comparison_manifest_path)
    preference = _load(preference_manifest_path)
    reward = _load(reward_validation_path)

    sft_reasons: list[str] = []
    if policy is None:
        sft_reasons.append("Missing adjudicated-only policy validation.")
    elif not policy.get("release_gate", {}).get("official_metrics_allowed"):
        sft_reasons.append("Policy-gold official metric release gate is closed.")
    if dataset is None or sft_dataset_manifest_path is None:
        sft_reasons.append("Missing released SFT dataset manifest.")
    else:
        counts = dataset.get("counts", {})
        if (
            counts.get("released", 0) <= 0
            or counts.get("train", 0) <= 0
            or counts.get("validation", 0) <= 0
        ):
            sft_reasons.append(
                "SFT dataset requires non-empty released/train/validation counts."
            )
        dataset_hash = str(dataset.get("dataset_sha256", ""))
        if not dataset_hash:
            sft_reasons.append("SFT dataset manifest lacks dataset_sha256.")
        else:
            _verify_sibling(
                sft_dataset_manifest_path,
                "sft_dataset.jsonl",
                dataset_hash,
                sft_reasons,
            )
    sft_start_ready = not sft_reasons

    evaluation_reasons: list[str] = list(sft_reasons)
    if run is None or sft_run_manifest_path is None:
        evaluation_reasons.append("Missing completed SFT run manifest.")
    else:
        if run.get("stage") != "SFT" or run.get("status") != "COMPLETED":
            evaluation_reasons.append("SFT run must have stage SFT and status COMPLETED.")
        if not run.get("smoke_passed"):
            evaluation_reasons.append("SFT local smoke is not confirmed.")
        checkpoint = run.get("checkpoint", {})
        checkpoint_path_value = checkpoint.get("path")
        checkpoint_hash = str(checkpoint.get("sha256", ""))
        if not checkpoint_path_value or not checkpoint_hash:
            evaluation_reasons.append("SFT checkpoint path/hash is missing.")
        else:
            checkpoint_path = Path(str(checkpoint_path_value))
            if not checkpoint_path.is_file():
                evaluation_reasons.append(f"Missing SFT checkpoint: {checkpoint_path}")
            elif sha256(checkpoint_path) != checkpoint_hash.upper():
                evaluation_reasons.append("SFT checkpoint hash mismatch.")
        if dataset is not None and (
            str(run.get("input_dataset_sha256", "")).upper()
            != str(dataset.get("dataset_sha256", "")).upper()
        ):
            evaluation_reasons.append("SFT run is not bound to the released dataset.")
    if comparison is None:
        evaluation_reasons.append("Missing Base-vs-SFT comparison manifest.")
    else:
        if comparison.get("stage") != "BASE_VS_SFT":
            evaluation_reasons.append("Comparison stage must be BASE_VS_SFT.")
        if comparison.get("status") != "COMPLETED":
            evaluation_reasons.append("Base-vs-SFT comparison is not complete.")
        if not comparison.get("comparable_protocol"):
            evaluation_reasons.append("Base and SFT evaluations are not comparable.")
        if not comparison.get("frozen_protocol"):
            evaluation_reasons.append("Comparison protocol is not frozen.")
        if not comparison.get("no_posthoc_tuning"):
            evaluation_reasons.append("No-post-hoc-tuning boundary is not confirmed.")
        bindings = comparison.get("bindings", {})
        required_bindings = {
            "sft_run_sha256",
            "sft_checkpoint_sha256",
            "task_set_sha256",
            "runtime_config_sha256",
        }
        missing_bindings = sorted(required_bindings - bindings.keys())
        if missing_bindings:
            evaluation_reasons.append(
                f"Comparison bindings are incomplete: {missing_bindings}"
            )
        if sft_run_manifest_path is not None and (
            str(bindings.get("sft_run_sha256", "")).upper()
            != sha256(sft_run_manifest_path)
        ):
            evaluation_reasons.append("Comparison is not bound to the SFT run.")
        if run is not None and (
            str(bindings.get("sft_checkpoint_sha256", "")).upper()
            != str(run.get("checkpoint", {}).get("sha256", "")).upper()
        ):
            evaluation_reasons.append("Comparison is not bound to the SFT checkpoint.")
    sft_evaluation_complete = not evaluation_reasons

    dpo_reasons: list[str] = list(evaluation_reasons)
    residual = (
        comparison.get("residual_systematic_failures", [])
        if comparison is not None
        else []
    )
    if not residual:
        dpo_reasons.append("No residual systematic failures justify preference work.")
    if preference is None or preference_manifest_path is None:
        dpo_reasons.append("Missing adjudicated preference-pair manifest.")
    else:
        if preference.get("status") != "READY":
            dpo_reasons.append("Preference dataset status is not READY.")
        if not preference.get("fully_adjudicated"):
            dpo_reasons.append("Preference pairs are not fully adjudicated.")
        if preference.get("pair_count", 0) <= 0:
            dpo_reasons.append("Preference dataset contains no pairs.")
        if preference.get("group_leakage_detected"):
            dpo_reasons.append("Preference dataset has group leakage.")
        if comparison_manifest_path is not None and (
            str(preference.get("source_comparison_sha256", "")).upper()
            != sha256(comparison_manifest_path)
        ):
            dpo_reasons.append("Preference data is not bound to the SFT comparison.")
    dpo_ready = not dpo_reasons

    rl_reasons: list[str] = list(evaluation_reasons)
    if comparison is None or not comparison.get("rl_justified"):
        rl_reasons.append("Comparable SFT evaluation does not justify RL.")
    if reward is None:
        rl_reasons.append("Missing held-out reward validation.")
    else:
        if not reward.get("held_out"):
            rl_reasons.append("Reward validation is not held out.")
        if not reward.get("release_gate", {}).get("official_metrics_allowed"):
            rl_reasons.append("Reward-validation official release gate is closed.")
        if not reward.get("reward_spec_sha256"):
            rl_reasons.append("Reward validation lacks a frozen reward-spec hash.")
        if policy_validation_path is not None and (
            str(reward.get("source_policy_validation_sha256", "")).upper()
            != sha256(policy_validation_path)
        ):
            rl_reasons.append(
                "Reward validation is not bound to adjudicated policy gold."
            )
        metrics = reward.get("metrics", {})
        if (metrics.get("precision") or 0.0) < reward_min_precision:
            rl_reasons.append("Reward precision is below threshold.")
        if (metrics.get("recall") or 0.0) < reward_min_recall:
            rl_reasons.append("Reward recall is below threshold.")
        if (metrics.get("critical_recall") or 0.0) < critical_min_recall:
            rl_reasons.append("Critical-risk reward recall is below threshold.")
        if reward.get("unresolved_fp_task_ids") or reward.get(
            "unresolved_fn_task_ids"
        ):
            rl_reasons.append("Reward validation has unresolved FP/FN cases.")
    rl_ready = not rl_reasons

    return {
        "schema_version": "policy-agent-post-training-readiness-v0.1",
        "inputs": {
            "policy_validation": _input(policy_validation_path),
            "sft_dataset_manifest": _input(sft_dataset_manifest_path),
            "sft_run_manifest": _input(sft_run_manifest_path),
            "comparison_manifest": _input(comparison_manifest_path),
            "preference_manifest": _input(preference_manifest_path),
            "reward_validation": _input(reward_validation_path),
        },
        "thresholds": {
            "reward_min_precision": reward_min_precision,
            "reward_min_recall": reward_min_recall,
            "critical_min_recall": critical_min_recall,
        },
        "gates": {
            "sft_start": {"ready": sft_start_ready, "reasons": sft_reasons},
            "sft_evaluation": {
                "ready": sft_evaluation_complete,
                "reasons": evaluation_reasons,
            },
            "dpo": {"ready": dpo_ready, "reasons": dpo_reasons},
            "rlhf_grpo": {"ready": rl_ready, "reasons": rl_reasons},
        },
    }


def write_report(result: dict[str, Any], output_dir: Path) -> None:
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty output: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "readiness_report.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate evidence-backed SFT, DPO, and RLHF/GRPO readiness."
    )
    parser.add_argument("--policy-validation", type=Path)
    parser.add_argument("--sft-dataset-manifest", type=Path)
    parser.add_argument("--sft-run-manifest", type=Path)
    parser.add_argument("--comparison-manifest", type=Path)
    parser.add_argument("--preference-manifest", type=Path)
    parser.add_argument("--reward-validation", type=Path)
    parser.add_argument(
        "--require-stage",
        choices=("SFT_START", "SFT_EVALUATION", "DPO", "RLHF_GRPO"),
        help="Exit 2 only when the selected stage is not ready.",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = evaluate_readiness(
        policy_validation_path=args.policy_validation,
        sft_dataset_manifest_path=args.sft_dataset_manifest,
        sft_run_manifest_path=args.sft_run_manifest,
        comparison_manifest_path=args.comparison_manifest,
        preference_manifest_path=args.preference_manifest,
        reward_validation_path=args.reward_validation,
    )
    write_report(result, args.output)
    print(
        json.dumps(
            {
                gate: value["ready"]
                for gate, value in result["gates"].items()
            },
            ensure_ascii=False,
        )
    )
    stage_keys = {
        "SFT_START": "sft_start",
        "SFT_EVALUATION": "sft_evaluation",
        "DPO": "dpo",
        "RLHF_GRPO": "rlhf_grpo",
    }
    if (
        args.require_stage
        and not result["gates"][stage_keys[args.require_stage]]["ready"]
    ):
        raise SystemExit(2)


if __name__ == "__main__":
    main()
