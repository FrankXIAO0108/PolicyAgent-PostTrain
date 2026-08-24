from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from src.rl.retail_agentic_env import _ensure_tau2_importable


REPO_ROOT = Path(__file__).resolve().parents[2]
WRITE_ACTIONS = {
    "cancel_pending_order",
    "exchange_delivered_order_items",
    "modify_pending_order_address",
    "modify_pending_order_items",
    "modify_pending_order_payment",
    "modify_user_address",
    "return_delivered_order_items",
}


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest().upper()


def sha256_file(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def canonical_sha256(value: object) -> str:
    return sha256_bytes(
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode(
            "utf-8"
        )
    )


def task_stratum(task: Any) -> str:
    criteria = task.evaluation_criteria
    names = {
        action.name
        for action in (criteria.actions or [])
        if action.requestor == "assistant"
    } if criteria is not None else set()
    writes = names & WRITE_ACTIONS
    families: list[str] = []
    if "cancel_pending_order" in writes:
        families.append("cancel")
    if "exchange_delivered_order_items" in writes:
        families.append("exchange")
    if "return_delivered_order_items" in writes:
        families.append("return")
    if writes & {
        "modify_pending_order_address",
        "modify_pending_order_items",
        "modify_pending_order_payment",
        "modify_user_address",
    }:
        families.append("modify")
    if len(families) > 1:
        return "mixed_" + "_".join(sorted(families))
    if families:
        return families[0]
    if "transfer_to_human_agents" in names:
        return "handoff"
    return "query"


def _stable_order(task_ids: list[str], seed: int) -> list[str]:
    return sorted(
        task_ids,
        key=lambda task_id: hashlib.sha256(f"{seed}:{task_id}".encode()).hexdigest(),
    )


def stratified_split(
    tasks: list[Any],
    development_ids: set[str],
    *,
    validation_size: int,
    seed: int,
) -> dict[str, Any]:
    task_by_id = {str(task.id): task for task in tasks}
    missing = sorted(development_ids - set(task_by_id))
    if missing:
        raise ValueError(f"Development IDs are not in upstream train split: {missing}")
    candidates = [task for task in tasks if str(task.id) not in development_ids]
    if validation_size <= 0 or validation_size >= len(candidates):
        raise ValueError("validation_size must leave non-empty train and validation sets")

    grouped: dict[str, list[str]] = defaultdict(list)
    for task in candidates:
        grouped[task_stratum(task)].append(str(task.id))
    for stratum in grouped:
        grouped[stratum] = _stable_order(grouped[stratum], seed)

    strata = sorted(grouped)
    quotas = {stratum: 0 for stratum in strata}
    if len(strata) <= validation_size:
        for stratum in strata:
            quotas[stratum] = 1
    remaining = validation_size - sum(quotas.values())
    candidate_count = len(candidates)
    ideal = {
        stratum: len(grouped[stratum]) * validation_size / candidate_count
        for stratum in strata
    }
    while remaining > 0:
        eligible = [
            stratum
            for stratum in strata
            if quotas[stratum] < len(grouped[stratum])
        ]
        chosen = max(
            eligible,
            key=lambda stratum: (
                ideal[stratum] - quotas[stratum],
                len(grouped[stratum]) - quotas[stratum],
                stratum,
            ),
        )
        quotas[chosen] += 1
        remaining -= 1

    validation_ids = {
        task_id
        for stratum, quota in quotas.items()
        for task_id in grouped[stratum][:quota]
    }
    train_ids = {str(task.id) for task in candidates} - validation_ids
    return {
        "rl_train": sorted(train_ids, key=int),
        "rl_validation": sorted(validation_ids, key=int),
        "development_audit": sorted(development_ids, key=int),
        "strata": {
            "candidate_counts": dict(
                sorted(Counter(task_stratum(task) for task in candidates).items())
            ),
            "validation_counts": dict(
                sorted(
                    Counter(task_stratum(task_by_id[item]) for item in validation_ids).items()
                )
            ),
        },
    }


def protect_validation_from_sft_tasks(
    split: dict[str, Any],
    *,
    tasks: list[Any],
    sft_task_ids: set[str],
) -> list[str]:
    """Move SFT-seen tasks out of RL validation while preserving the partition."""
    overlap = sorted(set(split["rl_validation"]) & sft_task_ids, key=int)
    if not overlap:
        return []
    split["rl_validation"] = sorted(
        set(split["rl_validation"]) - set(overlap), key=int
    )
    split["rl_train"] = sorted(set(split["rl_train"]) | set(overlap), key=int)
    task_by_id = {str(task.id): task for task in tasks}
    split["strata"]["validation_counts"] = dict(
        sorted(
            Counter(
                task_stratum(task_by_id[task_id])
                for task_id in split["rl_validation"]
            ).items()
        )
    )
    return overlap


def git_commit(path: Path) -> str:
    return subprocess.run(
        ["git", "-c", f"safe.directory={path.as_posix()}", "rev-parse", "HEAD"],
        cwd=path,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def build_manifest(
    *,
    baseline_config: Path,
    upstream_root: Path,
    validation_size: int,
    seed: int,
    sft_split_plan: Path | None = None,
    sft_data_manifest: Path | None = None,
) -> dict[str, Any]:
    _ensure_tau2_importable()
    from tau2.registry import registry

    baseline = json.loads(baseline_config.read_text(encoding="utf-8-sig"))
    development_ids = {str(item) for item in baseline["task_ids"]}
    loader = registry.get_tasks_loader("retail")
    train_tasks = loader("train")
    test_tasks = loader("test")
    split = stratified_split(
        train_tasks,
        development_ids,
        validation_size=validation_size,
        seed=seed,
    )
    sft_task_ids: set[str] = set()
    reassigned_validation_ids: list[str] = []
    if (sft_split_plan is None) != (sft_data_manifest is None):
        raise ValueError(
            "sft_split_plan and sft_data_manifest must be provided together"
        )
    if sft_split_plan is not None and sft_data_manifest is not None:
        if not sft_split_plan.is_file() or not sft_data_manifest.is_file():
            raise FileNotFoundError("SFT split plan or data manifest is missing")
        sft_rows = [
            json.loads(line)
            for line in sft_split_plan.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        sft_task_ids = {str(row["task_id"]) for row in sft_rows}
        reassigned_validation_ids = protect_validation_from_sft_tasks(
            split,
            tasks=train_tasks,
            sft_task_ids=sft_task_ids,
        )
        if set(split["rl_validation"]) & sft_task_ids:
            raise RuntimeError("SFT-seen task remains in RL validation")
    all_selected = (
        set(split["rl_train"])
        | set(split["rl_validation"])
        | set(split["development_audit"])
    )
    test_ids = {str(task.id) for task in test_tasks}
    if all_selected & test_ids:
        raise RuntimeError("Official Retail test leakage detected")
    if len(all_selected) != len(train_tasks):
        raise RuntimeError("RL split does not partition the upstream train split")

    upstream_commit = git_commit(upstream_root)
    expected_upstream = baseline["upstream"]["commit"]
    if upstream_commit != expected_upstream:
        raise RuntimeError(
            f"Upstream commit mismatch: expected {expected_upstream}, got {upstream_commit}"
        )
    manifest = {
        "schema_version": (
            "retail-agentic-rl-task-split-v2"
            if sft_split_plan is not None
            else "retail-agentic-rl-task-split-v1"
        ),
        "scope": "ISOLATED_AGENTIC_RL_ENGINEERING",
        "seed": seed,
        "claims": {
            "formal_retail_gate_unchanged": True,
            "business_improvement_claim_allowed": False,
            "official_test_reserved_for_final_evaluation": True,
        },
        "upstream": {
            "repository": "sierra-research/tau2-bench",
            "commit": upstream_commit,
            "train_count": len(train_tasks),
            "test_count": len(test_tasks),
            "official_test_ids_sha256": canonical_sha256(sorted(test_ids, key=int)),
        },
        "source": {
            "development_config": str(baseline_config.relative_to(REPO_ROOT)).replace(
                "\\", "/"
            ),
            "development_config_sha256": sha256_file(baseline_config),
        },
        "splits": split,
        "counts": {
            "rl_train": len(split["rl_train"]),
            "rl_validation": len(split["rl_validation"]),
            "development_audit": len(split["development_audit"]),
        },
        "leakage_checks": {
            "pairwise_disjoint": True,
            "covers_all_upstream_train_tasks": True,
            "official_test_overlap_count": 0,
            "passed": True,
        },
    }
    if sft_split_plan is not None and sft_data_manifest is not None:
        manifest["source"]["sft_split_plan"] = str(
            sft_split_plan.relative_to(REPO_ROOT)
        ).replace("\\", "/")
        manifest["source"]["sft_split_plan_sha256"] = sha256_file(sft_split_plan)
        manifest["source"]["sft_data_manifest"] = str(
            sft_data_manifest.relative_to(REPO_ROOT)
        ).replace("\\", "/")
        manifest["source"]["sft_data_manifest_sha256"] = sha256_file(
            sft_data_manifest
        )
        manifest["sft_task_isolation"] = {
            "sft_task_count": len(sft_task_ids),
            "reassigned_from_rl_validation_to_rl_train": reassigned_validation_ids,
            "rl_train_overlap_count": len(set(split["rl_train"]) & sft_task_ids),
            "rl_validation_overlap_count": len(
                set(split["rl_validation"]) & sft_task_ids
            ),
            "development_audit_overlap_count": len(
                set(split["development_audit"]) & sft_task_ids
            ),
        }
        manifest["leakage_checks"]["rl_validation_sft_task_overlap_count"] = 0
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--baseline-config",
        type=Path,
        default=REPO_ROOT / "configs" / "baseline_20_tasks.json",
    )
    parser.add_argument(
        "--upstream-root",
        type=Path,
        default=Path(os.environ.get("POLICYAGENT_TAU2_ROOT", r"D:\tau2-bench")),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=REPO_ROOT / "data" / "retail_agentic_rl_v1" / "task_split.json",
    )
    parser.add_argument("--validation-size", type=int, default=10)
    parser.add_argument("--seed", type=int, default=20260810)
    parser.add_argument("--sft-split-plan", type=Path)
    parser.add_argument("--sft-data-manifest", type=Path)
    args = parser.parse_args()
    manifest = build_manifest(
        baseline_config=args.baseline_config.resolve(),
        upstream_root=args.upstream_root.resolve(),
        validation_size=args.validation_size,
        seed=args.seed,
        sft_split_plan=(
            args.sft_split_plan.resolve() if args.sft_split_plan is not None else None
        ),
        sft_data_manifest=(
            args.sft_data_manifest.resolve()
            if args.sft_data_manifest is not None
            else None
        ),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest["counts"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
