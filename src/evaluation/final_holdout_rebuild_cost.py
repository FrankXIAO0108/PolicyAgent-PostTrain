from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from collections import Counter
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
SCHEMA_VERSION = "retail-final-holdout-rebuild-cost-v1.0.0"
WRITE_FAMILIES = {
    "cancel_pending_order": "cancel",
    "exchange_delivered_order_items": "exchange",
    "return_delivered_order_items": "return",
    "modify_pending_order_address": "modify",
    "modify_pending_order_items": "modify",
    "modify_pending_order_payment": "modify",
    "modify_user_address": "modify",
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8-sig").splitlines(), start=1
    ):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError(f"JSONL row {line_number} is not an object")
        rows.append(value)
    if not rows:
        raise ValueError("SFT dataset is empty")
    return rows


def git_commit(path: Path) -> str:
    return subprocess.run(
        ["git", "-c", f"safe.directory={path.as_posix()}", "rev-parse", "HEAD"],
        cwd=path,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def task_stratum(task: dict[str, Any]) -> str:
    criteria = task.get("evaluation_criteria") or {}
    names = {
        str(action.get("name"))
        for action in criteria.get("actions") or []
        if action.get("name")
    }
    families = sorted(
        {WRITE_FAMILIES[name] for name in names if name in WRITE_FAMILIES}
    )
    if len(families) > 1:
        return "mixed_" + "_".join(families)
    if families:
        return families[0]
    if "transfer_to_human_agents" in names:
        return "handoff"
    return "query"


def build_cost_analysis(
    *,
    audit: dict[str, Any],
    sft_rows: list[dict[str, Any]],
    tasks: list[dict[str, Any]],
) -> dict[str, Any]:
    task_by_id = {str(task["id"]): task for task in tasks}
    if len(task_by_id) != len(tasks):
        raise ValueError("Duplicate task ID in upstream tasks")

    candidates = [
        row
        for row in audit.get("rows") or []
        if not row.get("used_in_development_evaluation")
    ]
    if not candidates:
        raise ValueError("Contamination audit contains no development-unseen tasks")

    rows: list[dict[str, Any]] = []
    for candidate in candidates:
        task_id = str(candidate["task_id"])
        if task_id not in task_by_id:
            raise ValueError(f"Candidate task missing from upstream tasks: {task_id}")
        overlapping_groups = set(candidate.get("overlapping_entity_groups") or [])
        affected = [
            row
            for row in sft_rows
            if overlapping_groups & {str(group) for group in row.get("group_ids") or []}
        ]
        affected_tasks = sorted({str(row["task_id"]) for row in affected}, key=int)
        split_counts = Counter(str(row.get("split")) for row in affected)
        rows.append(
            {
                "task_id": task_id,
                "stratum": task_stratum(task_by_id[task_id]),
                "overlapping_entity_groups": sorted(overlapping_groups),
                "affected_sft_row_count": len(affected),
                "affected_train_row_count": split_counts.get("TRAIN", 0),
                "affected_validation_row_count": split_counts.get("VALIDATION", 0),
                "affected_sft_task_count": len(affected_tasks),
                "affected_sft_task_ids": affected_tasks,
                "remaining_sft_row_count": len(sft_rows) - len(affected),
            }
        )

    rows.sort(key=lambda row: (row["affected_sft_row_count"], int(row["task_id"])))
    strata = Counter(row["stratum"] for row in rows)
    costs = Counter(row["affected_sft_row_count"] for row in rows)
    return {
        "schema_version": SCHEMA_VERSION,
        "summary": {
            "candidate_count": len(rows),
            "current_sft_row_count": len(sft_rows),
            "current_train_row_count": sum(
                str(row.get("split")) == "TRAIN" for row in sft_rows
            ),
            "current_validation_row_count": sum(
                str(row.get("split")) == "VALIDATION" for row in sft_rows
            ),
            "minimum_individual_exclusion_rows": min(
                row["affected_sft_row_count"] for row in rows
            ),
            "maximum_individual_exclusion_rows": max(
                row["affected_sft_row_count"] for row in rows
            ),
            "candidate_counts_by_stratum": dict(sorted(strata.items())),
            "candidate_counts_by_individual_cost": {
                str(key): value for key, value in sorted(costs.items())
            },
        },
        "rows": rows,
        "decision": {
            "final_holdout_selected": False,
            "portfolio_cost_requires_set_union": True,
            "selection_must_not_use_model_results": True,
            "next_step": (
                "Choose a capability-balanced portfolio from task definitions, then "
                "compute the union of affected SFT rows before freezing it."
            ),
        },
    }


def run_analysis(
    *, config_path: Path, upstream_root: Path, output_dir: Path
) -> dict[str, Any]:
    config_path = config_path.resolve()
    upstream_root = upstream_root.resolve()
    output_dir = output_dir.resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty output: {output_dir}")
    config = load_json(config_path)
    if config.get("status") != "FROZEN":
        raise ValueError("Rebuild-cost config must be FROZEN")

    actual_commit = git_commit(upstream_root)
    expected_commit = str(config["upstream"]["commit"])
    if actual_commit != expected_commit:
        raise ValueError(
            f"Upstream commit mismatch: {actual_commit} != {expected_commit}"
        )

    tasks_path = upstream_root / config["upstream"]["tasks_path"]
    audit_path = REPO_ROOT / config["contamination_audit"]["path"]
    dataset_path = REPO_ROOT / config["sft_dataset"]["path"]
    bindings = {
        "tasks": (tasks_path, str(config["upstream"]["tasks_sha256"]).upper()),
        "contamination_audit": (
            audit_path,
            str(config["contamination_audit"]["sha256"]).upper(),
        ),
        "sft_dataset": (
            dataset_path,
            str(config["sft_dataset"]["sha256"]).upper(),
        ),
    }
    for name, (path, expected_hash) in bindings.items():
        actual_hash = sha256(path)
        if actual_hash != expected_hash:
            raise ValueError(f"{name} hash mismatch: {actual_hash} != {expected_hash}")

    result = build_cost_analysis(
        audit=load_json(audit_path),
        sft_rows=load_jsonl(dataset_path),
        tasks=load_json(tasks_path),
    )
    runner_path = Path(__file__).resolve()
    result["bindings"] = {
        "project_commit": git_commit(REPO_ROOT),
        "upstream_commit": actual_commit,
        "config": {"path": str(config_path), "sha256": sha256(config_path)},
        "runner": {"path": str(runner_path), "sha256": sha256(runner_path)},
        **{
            name: {"path": str(path), "sha256": sha256(path)}
            for name, (path, _) in bindings.items()
        },
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    analysis_path = output_dir / "rebuild_cost.json"
    analysis_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "analysis": {"path": str(analysis_path), "sha256": sha256(analysis_path)},
        "summary": result["summary"],
        "decision": result["decision"],
        "command": (
            "python -m src.evaluation.final_holdout_rebuild_cost "
            f"--config {config_path} --upstream-root {upstream_root} "
            f"--output {output_dir}"
        ),
        "contains_raw_trajectories": False,
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Measure SFT exclusion cost for final-holdout candidates."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=REPO_ROOT / "configs/retail_final_holdout_rebuild_cost_v1.json",
    )
    parser.add_argument("--upstream-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(
        json.dumps(
            run_analysis(
                config_path=args.config,
                upstream_root=args.upstream_root,
                output_dir=args.output,
            ),
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
