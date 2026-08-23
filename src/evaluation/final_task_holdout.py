from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
SCHEMA_VERSION = "retail-final-task-holdout-v1.0.0"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest().upper()


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def git_commit(path: Path) -> str:
    return subprocess.run(
        ["git", "-c", f"safe.directory={path.as_posix()}", "rev-parse", "HEAD"],
        cwd=path,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def build_holdout_manifest(
    *,
    config: dict[str, Any],
    tasks: list[dict[str, Any]],
    split: dict[str, Any],
    audit: dict[str, Any],
) -> dict[str, Any]:
    selected_ids = [str(task_id) for task_id in config.get("task_ids") or []]
    if not selected_ids or len(selected_ids) != len(set(selected_ids)):
        raise ValueError("Holdout task IDs must be non-empty and unique")
    selected = set(selected_ids)

    by_id = {str(task["id"]): task for task in tasks}
    if len(by_id) != len(tasks):
        raise ValueError("Duplicate task ID in upstream tasks")
    official_test = {str(task_id) for task_id in split.get("test") or []}
    outside_test = selected - official_test
    if outside_test:
        raise ValueError(f"Holdout tasks are outside official test: {outside_test}")
    missing_tasks = selected - set(by_id)
    if missing_tasks:
        raise ValueError(f"Holdout tasks are absent from tasks.json: {missing_tasks}")

    audit_by_id = {str(row["task_id"]): row for row in audit.get("rows") or []}
    missing_audit = selected - set(audit_by_id)
    if missing_audit:
        raise ValueError(
            f"Holdout tasks are absent from contamination audit: {missing_audit}"
        )

    eligible_ids = {
        task_id
        for task_id, row in audit_by_id.items()
        if not row.get("used_in_development_evaluation")
        and not row.get("task_used_in_sft_data")
    }
    if config["selection_rule"].get(
        "uses_all_remaining_task_unseen_official_test_tasks"
    ):
        if selected != eligible_ids:
            raise ValueError(
                "Selected task IDs do not equal all remaining task-unseen test tasks"
            )

    invalid_rows = [
        audit_by_id[task_id]
        for task_id in selected
        if audit_by_id[task_id].get("used_in_development_evaluation")
        or audit_by_id[task_id].get("task_used_in_sft_data")
    ]
    if invalid_rows:
        raise ValueError(
            "Selected holdout contains development-used or SFT-used task IDs"
        )

    entity_overlap_count = sum(
        bool(audit_by_id[task_id].get("overlapping_entity_groups"))
        for task_id in selected
    )
    rows = [
        {
            "task_id": task_id,
            "task_payload_sha256": canonical_sha256(by_id[task_id]),
            "entity_overlap_with_current_sft": bool(
                audit_by_id[task_id].get("overlapping_entity_groups")
            ),
        }
        for task_id in sorted(selected, key=int)
    ]
    return {
        "schema_version": SCHEMA_VERSION,
        "status": "SEALED_NOT_RUN",
        "evaluation_scope": config["evaluation_scope"],
        "summary": {
            "task_count": len(rows),
            "development_used_task_count": 0,
            "sft_used_task_count": 0,
            "entity_overlap_task_count": entity_overlap_count,
            "model_result_count": 0,
        },
        "task_ids_sha256": canonical_sha256([row["task_id"] for row in rows]),
        "rows": rows,
        "prohibited_uses_before_unsealing": config["prohibited_uses_before_unsealing"],
        "unseal_gate": config["unseal_gate"],
        "claims": config["claims"],
    }


def run_validation(
    *, config_path: Path, upstream_root: Path, output_dir: Path
) -> dict[str, Any]:
    config_path = config_path.resolve()
    upstream_root = upstream_root.resolve()
    output_dir = output_dir.resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty output: {output_dir}")
    config = load_json(config_path)
    if config.get("status") != "SEALED_NOT_RUN":
        raise ValueError("Final holdout config must have status SEALED_NOT_RUN")

    actual_commit = git_commit(upstream_root)
    expected_commit = str(config["upstream"]["commit"])
    if actual_commit != expected_commit:
        raise ValueError(
            f"Upstream commit mismatch: {actual_commit} != {expected_commit}"
        )

    tasks_path = upstream_root / config["upstream"]["tasks_path"]
    split_path = upstream_root / config["upstream"]["split_path"]
    audit_path = REPO_ROOT / config["contamination_audit"]["path"]
    bindings = {
        "tasks": (tasks_path, str(config["upstream"]["tasks_sha256"]).upper()),
        "split": (split_path, str(config["upstream"]["split_sha256"]).upper()),
        "contamination_audit": (
            audit_path,
            str(config["contamination_audit"]["sha256"]).upper(),
        ),
    }
    for name, (path, expected_hash) in bindings.items():
        actual_hash = sha256(path)
        if actual_hash != expected_hash:
            raise ValueError(f"{name} hash mismatch: {actual_hash} != {expected_hash}")

    manifest = build_holdout_manifest(
        config=config,
        tasks=load_json(tasks_path),
        split=load_json(split_path),
        audit=load_json(audit_path),
    )
    runner_path = Path(__file__).resolve()
    manifest["bindings"] = {
        "project_commit_before_freeze": git_commit(REPO_ROOT),
        "upstream_commit": actual_commit,
        "config": {"path": str(config_path), "sha256": sha256(config_path)},
        "runner": {"path": str(runner_path), "sha256": sha256(runner_path)},
        **{
            name: {"path": str(path), "sha256": sha256(path)}
            for name, (path, _) in bindings.items()
        },
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "holdout_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    evidence = {
        "schema_version": SCHEMA_VERSION,
        "holdout_manifest": {
            "path": str(manifest_path),
            "sha256": sha256(manifest_path),
        },
        "status": manifest["status"],
        "summary": manifest["summary"],
        "contains_model_results": False,
        "external_api_called": False,
        "command": (
            "python -m src.evaluation.final_task_holdout "
            f"--config {config_path} --upstream-root {upstream_root} "
            f"--output {output_dir}"
        ),
    }
    evidence_path = output_dir / "evidence.json"
    evidence_path.write_text(
        json.dumps(evidence, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return evidence


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate and seal the project-internal final task holdout."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=REPO_ROOT / "configs/retail_final_task_holdout_v1.json",
    )
    parser.add_argument("--upstream-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(
        json.dumps(
            run_validation(
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
