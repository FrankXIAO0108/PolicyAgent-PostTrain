from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
SCHEMA_VERSION = "retail-final-holdout-contamination-v1.0.0"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def git_commit(path: Path) -> str:
    return subprocess.run(
        [
            "git",
            "-c",
            f"safe.directory={path.as_posix()}",
            "rev-parse",
            "HEAD",
        ],
        cwd=path,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _entity_values(groups: dict[str, Any]) -> list[tuple[str, str]]:
    values: list[tuple[str, str]] = []
    for group in groups.get("entity_groups") or []:
        text = str(group)
        if ":" not in text:
            raise ValueError(f"Malformed entity group: {text!r}")
        values.append((text, text.split(":", 1)[1]))
    if not values:
        raise ValueError("Training entity-group list is empty")
    return values


def build_audit(
    *,
    tasks: list[dict[str, Any]],
    split: dict[str, Any],
    training_groups: dict[str, Any],
    development_config: dict[str, Any],
    development_sources: set[str] | None = None,
) -> dict[str, Any]:
    by_id = {str(task["id"]): task for task in tasks}
    if len(by_id) != len(tasks):
        raise ValueError("Duplicate task ID in upstream tasks")
    official_test_ids = {str(task_id) for task_id in split.get("test") or []}
    if not official_test_ids:
        raise ValueError("Official test split is empty")
    missing_test = official_test_ids - set(by_id)
    if missing_test:
        raise ValueError(f"Official test tasks missing from tasks.json: {missing_test}")

    included_sources = development_sources or {"test_clean"}
    development_ids = {
        str(row["task_id"])
        for row in development_config.get("tasks") or []
        if str(row.get("source")) in included_sources
    }
    outside_test = development_ids - official_test_ids
    if outside_test:
        raise ValueError(
            f"Development test rows are outside the official test split: {outside_test}"
        )

    training_task_ids = {
        str(task_id) for task_id in training_groups.get("teacher_task_ids") or []
    }
    entity_values = _entity_values(training_groups)
    rows: list[dict[str, Any]] = []
    for task_id in sorted(official_test_ids, key=int):
        payload = json.dumps(by_id[task_id], ensure_ascii=False, sort_keys=True)
        overlapping_groups = [
            group for group, value in entity_values if value and value in payload
        ]
        blockers: list[str] = []
        if task_id in development_ids:
            blockers.append("USED_IN_DEVELOPMENT_EVALUATION")
        if task_id in training_task_ids:
            blockers.append("TASK_USED_IN_SFT_DATA")
        if overlapping_groups:
            blockers.append("ENTITY_OVERLAP_WITH_SFT_DATA")
        rows.append(
            {
                "task_id": task_id,
                "used_in_development_evaluation": task_id in development_ids,
                "task_used_in_sft_data": task_id in training_task_ids,
                "overlapping_entity_groups": overlapping_groups,
                "strict_holdout_candidate": not blockers,
                "blockers": blockers,
            }
        )

    unused_rows = [row for row in rows if not row["used_in_development_evaluation"]]
    eligible_rows = [row for row in rows if row["strict_holdout_candidate"]]
    summary = {
        "official_test_task_count": len(rows),
        "development_used_test_task_count": sum(
            row["used_in_development_evaluation"] for row in rows
        ),
        "sft_training_task_in_official_test_count": sum(
            row["task_used_in_sft_data"] for row in rows
        ),
        "development_unseen_test_task_count": len(unused_rows),
        "development_unseen_with_sft_entity_overlap_count": sum(
            bool(row["overlapping_entity_groups"]) for row in unused_rows
        ),
        "strict_holdout_candidate_count": len(eligible_rows),
    }
    return {
        "schema_version": SCHEMA_VERSION,
        "summary": summary,
        "rows": rows,
        "decision": {
            "final_holdout_frozen": False,
            "current_checkpoint_has_strict_untouched_holdout": bool(eligible_rows),
            "rebuild_training_split_required": not bool(eligible_rows),
            "reason": (
                "No official test task is both development-unseen and entity-disjoint "
                "from the current SFT data."
                if not eligible_rows
                else "Candidate tasks exist, but explicit final-holdout selection is still required."
            ),
        },
    }


def run_audit(
    *, config_path: Path, upstream_root: Path, output_dir: Path
) -> dict[str, Any]:
    config_path = config_path.resolve()
    upstream_root = upstream_root.resolve()
    output_dir = output_dir.resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty output: {output_dir}")
    config = load_json(config_path)
    if config.get("status") != "FROZEN":
        raise ValueError("Contamination audit config must be FROZEN")

    upstream = config["upstream"]
    actual_commit = git_commit(upstream_root)
    if actual_commit != str(upstream["commit"]):
        raise ValueError(
            f"Upstream commit mismatch: {actual_commit} != {upstream['commit']}"
        )
    tasks_path = upstream_root / upstream["tasks_path"]
    split_path = upstream_root / upstream["split_path"]
    groups_path = REPO_ROOT / config["training_entities"]["path"]
    development_path = REPO_ROOT / config["development_evaluation"]["path"]
    bound_paths = {
        "config": (config_path, sha256(config_path)),
        "tasks": (tasks_path, str(upstream["tasks_sha256"]).upper()),
        "split": (split_path, str(upstream["split_sha256"]).upper()),
        "training_entities": (
            groups_path,
            str(config["training_entities"]["sha256"]).upper(),
        ),
        "development_evaluation": (
            development_path,
            str(config["development_evaluation"]["sha256"]).upper(),
        ),
    }
    for name, (path, expected) in bound_paths.items():
        actual = sha256(path)
        if name != "config" and actual != expected:
            raise ValueError(f"{name} hash mismatch: {actual} != {expected}")

    result = build_audit(
        tasks=load_json(tasks_path),
        split=load_json(split_path),
        training_groups=load_json(groups_path),
        development_config=load_json(development_path),
        development_sources={
            str(source)
            for source in config["development_evaluation"]["included_sources"]
        },
    )
    runner_path = Path(__file__).resolve()
    result["bindings"] = {
        "project_commit": git_commit(REPO_ROOT),
        "upstream_commit": actual_commit,
        "runner": {"path": str(runner_path), "sha256": sha256(runner_path)},
        **{
            name: {"path": str(path), "sha256": sha256(path)}
            for name, (path, _) in bound_paths.items()
        },
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    audit_path = output_dir / "audit.json"
    audit_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "audit": {"path": str(audit_path), "sha256": sha256(audit_path)},
        "summary": result["summary"],
        "decision": result["decision"],
        "command": (
            "python -m src.evaluation.final_holdout_contamination "
            f"--config {config_path} --upstream-root {upstream_root} "
            f"--output {output_dir}"
        ),
        "contains_raw_trajectories": False,
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Audit official Retail test contamination for the current SFT data."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=REPO_ROOT / "configs/retail_final_holdout_contamination_v1.json",
    )
    parser.add_argument("--upstream-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(
        json.dumps(
            run_audit(
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
