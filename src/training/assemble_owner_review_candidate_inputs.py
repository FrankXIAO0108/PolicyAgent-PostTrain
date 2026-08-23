"""Assemble reviewed teacher-candidate inputs for the existing release gates.

This adapter consolidates owner review batches, assistant correction approvals,
explicit assistant HOLDOUT reviews, correction artifacts, final validations and
an already-frozen SFT split.  It does not change labels or release data; it
only emits hash-bound inputs for ``teacher_candidate_decision_builder``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from src.training.sft_release import entity_groups, sha256


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8-sig").splitlines()
        if line.strip()
    ]


def _unique(rows: list[dict[str, Any]], field: str, source: str) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for row in rows:
        key = str(row.get(field, ""))
        if not key or key in result:
            raise ValueError(f"{source}: missing or duplicate {field} {key!r}")
        result[key] = row
    return result


def _load_many(paths: list[Path], field: str) -> dict[str, dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in paths:
        rows.extend(load_jsonl(path))
    return _unique(rows, field, ", ".join(str(path) for path in paths))


def _assistant_approvals(
    paths: list[Path],
) -> dict[str, tuple[dict[str, Any], Path, dict[str, Any]]]:
    result: dict[str, tuple[dict[str, Any], Path, dict[str, Any]]] = {}
    for approval in _load_many(paths, "correction_sha256").values():
        if str(approval.get("verdict", "")).upper() != "APPROVE":
            raise ValueError("Assistant correction registry accepts APPROVE only")
        correction_hash = str(approval["correction_sha256"]).upper()
        corrected_paths = [
            Path(str(value))
            for value in approval.get("evidence_files", [])
            if Path(str(value)).name.startswith("corrected_")
        ]
        if len(corrected_paths) != 1:
            raise ValueError(
                f"Correction {correction_hash}: expected one corrected evidence file"
            )
        correction_path = corrected_paths[0]
        if not correction_path.is_file() or sha256(correction_path) != correction_hash:
            raise ValueError(f"Correction {correction_hash}: file/hash mismatch")
        payload = json.loads(correction_path.read_text(encoding="utf-8-sig"))
        candidate_id = str(payload.get("candidate_id", ""))
        if not candidate_id or candidate_id in result:
            raise ValueError(f"Missing or duplicate corrected candidate {candidate_id!r}")
        result[candidate_id] = (approval, correction_path, payload)
    return result


def _final_validations(root: Path) -> dict[str, Path]:
    by_hash: dict[str, list[Path]] = defaultdict(list)
    for path in root.rglob("correction_validation.json"):
        payload = json.loads(path.read_text(encoding="utf-8-sig"))
        if payload.get("ready") is True and int(payload.get("approval_count") or 0) >= 2:
            by_hash[str(payload.get("correction_sha256", "")).upper()].append(path)
    duplicates = {key: paths for key, paths in by_hash.items() if len(paths) != 1}
    if duplicates:
        detail = {key: [str(path) for path in paths] for key, paths in duplicates.items()}
        raise ValueError(f"Final correction validations are not unique: {detail}")
    return {key: paths[0] for key, paths in by_hash.items()}


def _existing_splits(path: Path) -> dict[str, str]:
    result: dict[str, str] = {}
    for row in load_jsonl(path):
        task_id = str(row["task_id"])
        split = str(row["split"]).upper()
        if split not in {"TRAIN", "VALIDATION"}:
            raise ValueError(f"Existing task {task_id}: invalid split {split!r}")
        if task_id in result and result[task_id] != split:
            raise ValueError(f"Existing task {task_id}: conflicting splits")
        result[task_id] = split
    return result


def _existing_group_splits(path: Path) -> dict[str, str]:
    result: dict[str, str] = {}
    for row in load_jsonl(path):
        split = str(row["split"]).upper()
        for group_id in row.get("group_ids", []):
            group_id = str(group_id)
            if group_id in result and result[group_id] != split:
                raise ValueError(f"Existing group {group_id}: conflicting splits")
            result[group_id] = split
    return result


def _default_split(task_id: str, seed: int) -> str:
    digest = hashlib.sha256(f"{seed}:{task_id}".encode()).digest()
    return "VALIDATION" if int.from_bytes(digest[:4], "big") % 5 == 0 else "TRAIN"


def assemble_inputs(
    *,
    owner_review_paths: list[Path],
    assistant_approval_paths: list[Path],
    assistant_holdouts_path: Path,
    validation_root: Path,
    existing_dataset_path: Path,
    existing_split_plan_path: Path,
    split_seed: int,
) -> dict[str, Any]:
    owners = _load_many(owner_review_paths, "candidate_id")
    holdouts = _unique(
        load_jsonl(assistant_holdouts_path),
        "candidate_id",
        str(assistant_holdouts_path),
    )
    approvals = _assistant_approvals(assistant_approval_paths)
    validations = _final_validations(validation_root)

    owner_corrected = {
        candidate_id
        for candidate_id, row in owners.items()
        if str(row.get("quality_label", "")).upper() == "CORRECTION_REQUIRED"
    }
    owner_holdouts = {
        candidate_id
        for candidate_id, row in owners.items()
        if str(row.get("quality_label", "")).upper() == "HOLDOUT"
    }
    unsupported = set(owners) - owner_corrected - owner_holdouts
    if unsupported:
        raise ValueError(f"Unsupported owner labels for candidates: {sorted(unsupported)}")
    if owner_corrected != set(approvals):
        raise ValueError(
            "Corrected coverage mismatch: "
            f"missing_approvals={sorted(owner_corrected - set(approvals))}, "
            f"extra_approvals={sorted(set(approvals) - owner_corrected)}"
        )
    if owner_holdouts != set(holdouts):
        raise ValueError(
            "HOLDOUT coverage mismatch: "
            f"missing_assistant={sorted(owner_holdouts - set(holdouts))}, "
            f"extra_assistant={sorted(set(holdouts) - owner_holdouts)}"
        )

    reviews_a = [owners[key] for key in sorted(owners)]
    reviews_b: list[dict[str, Any]] = []
    registry: list[dict[str, Any]] = []
    task_groups: dict[str, set[str]] = defaultdict(set)

    for candidate_id in sorted(owners):
        owner = owners[candidate_id]
        task_id = str(owner["task_id"])
        label = str(owner["quality_label"]).upper()
        if label == "HOLDOUT":
            assistant = holdouts[candidate_id]
            if str(assistant.get("task_id")) != task_id:
                raise ValueError(f"Candidate {candidate_id}: HOLDOUT task mismatch")
            reviews_b.append(assistant)
            continue

        approval, correction_path, correction = approvals[candidate_id]
        correction_hash = str(approval["correction_sha256"]).upper()
        if str(correction.get("task_id")) != task_id:
            raise ValueError(f"Candidate {candidate_id}: correction task mismatch")
        validation_path = validations.get(correction_hash)
        if validation_path is None:
            raise ValueError(f"Candidate {candidate_id}: no final validation")
        source = correction.get("source") or {}
        source_path = Path(str(source.get("path", "")))
        source_hash = str(source.get("sha256", "")).upper()
        if not source_path.is_file() or sha256(source_path) != source_hash:
            raise ValueError(f"Candidate {candidate_id}: source file/hash mismatch")
        validation_hash = sha256(validation_path)
        groups = entity_groups(source_path)
        if not groups:
            raise ValueError(f"Candidate {candidate_id}: source has no entity groups")
        task_groups[task_id].update(groups)
        reviews_b.append(
            {
                "candidate_id": candidate_id,
                "task_id": task_id,
                "quality_label": "CORRECTION_REQUIRED",
                "reviewer_id": str(approval["reviewer_id"]),
                "reviewed_at": approval.get("reviewed_at"),
                "rationale": approval.get("rationale"),
                "evidence_files": approval.get("evidence_files", []),
            }
        )
        registry.append(
            {
                "candidate_id": candidate_id,
                "source_path": str(source_path),
                "source_sha256": source_hash,
                "correction_path": str(correction_path),
                "correction_sha256": correction_hash,
                "correction_validation_path": str(validation_path),
                "correction_validation_sha256": validation_hash,
            }
        )

    existing_task_splits = _existing_splits(existing_split_plan_path)
    group_splits = _existing_group_splits(existing_dataset_path)
    assigned_tasks: dict[str, str] = {}
    for task_id in sorted(task_groups, key=lambda value: int(value)):
        forced = {group_splits[group] for group in task_groups[task_id] if group in group_splits}
        if task_id in existing_task_splits:
            forced.add(existing_task_splits[task_id])
        if len(forced) > 1:
            raise ValueError(f"Task {task_id}: existing split conflict {sorted(forced)}")
        split = next(iter(forced)) if forced else _default_split(task_id, split_seed)
        assigned_tasks[task_id] = split
        for group in task_groups[task_id]:
            previous = group_splits.get(group)
            if previous is not None and previous != split:
                raise ValueError(f"Group {group}: split conflict {previous} vs {split}")
            group_splits[group] = split

    split_plan = [
        {
            "candidate_id": row["candidate_id"],
            "task_id": str(owners[row["candidate_id"]]["task_id"]),
            "split": assigned_tasks[str(owners[row["candidate_id"]]["task_id"])],
            "source_split": "TRAIN",
            "rationale": "Preserve the existing task/entity split when bound; otherwise use the frozen deterministic split seed.",
        }
        for row in registry
    ]
    return {
        "reviews_a": reviews_a,
        "reviews_b": reviews_b,
        "correction_registry": registry,
        "split_plan": split_plan,
        "counts": {
            "reviewed": len(owners),
            "corrected": len(registry),
            "holdout": len(holdouts),
            "train": sum(row["split"] == "TRAIN" for row in split_plan),
            "validation": sum(row["split"] == "VALIDATION" for row in split_plan),
            "unique_tasks": len({str(row["task_id"]) for row in owners.values()}),
        },
        "inputs": {
            "owner_reviews": [
                {"path": str(path), "sha256": sha256(path)} for path in owner_review_paths
            ],
            "assistant_approvals": [
                {"path": str(path), "sha256": sha256(path)}
                for path in assistant_approval_paths
            ],
            "assistant_holdouts": {
                "path": str(assistant_holdouts_path),
                "sha256": sha256(assistant_holdouts_path),
            },
            "existing_dataset": {
                "path": str(existing_dataset_path),
                "sha256": sha256(existing_dataset_path),
            },
            "existing_split_plan": {
                "path": str(existing_split_plan_path),
                "sha256": sha256(existing_split_plan_path),
            },
        },
        "split_seed": split_seed,
    }


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def write_outputs(result: dict[str, Any], output_dir: Path) -> None:
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty output: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    for key, filename in (
        ("reviews_a", "owner_reviews.jsonl"),
        ("reviews_b", "assistant_reviews.jsonl"),
        ("correction_registry", "correction_registry.jsonl"),
        ("split_plan", "split_plan.jsonl"),
    ):
        _write_jsonl(output_dir / filename, result[key])
    manifest = {key: value for key, value in result.items() if key not in {"reviews_a", "reviews_b", "correction_registry", "split_plan"}}
    manifest["outputs"] = {
        name: {"path": str(output_dir / name), "sha256": sha256(output_dir / name)}
        for name in (
            "owner_reviews.jsonl",
            "assistant_reviews.jsonl",
            "correction_registry.jsonl",
            "split_plan.jsonl",
        )
    }
    (output_dir / "assembly_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Assemble owner-reviewed candidate release inputs.")
    parser.add_argument("--owner-review", type=Path, action="append", required=True)
    parser.add_argument("--assistant-approval", type=Path, action="append", required=True)
    parser.add_argument("--assistant-holdouts", type=Path, required=True)
    parser.add_argument("--validation-root", type=Path, required=True)
    parser.add_argument("--existing-dataset", type=Path, required=True)
    parser.add_argument("--existing-split-plan", type=Path, required=True)
    parser.add_argument("--split-seed", type=int, default=20260823)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = assemble_inputs(
        owner_review_paths=args.owner_review,
        assistant_approval_paths=args.assistant_approval,
        assistant_holdouts_path=args.assistant_holdouts,
        validation_root=args.validation_root,
        existing_dataset_path=args.existing_dataset,
        existing_split_plan_path=args.existing_split_plan,
        split_seed=args.split_seed,
    )
    write_outputs(result, args.output)
    print(json.dumps(result["counts"], ensure_ascii=False))


if __name__ == "__main__":
    main()
