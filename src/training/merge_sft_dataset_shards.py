"""Merge already-gated SFT dataset shards and recheck global leakage.

Each input directory must contain ``sft_dataset.jsonl`` and
``dataset_manifest.json``. The merge is fail-closed on broken hashes,
duplicate candidates, duplicate selected trajectories, or entity groups that
cross TRAIN/VALIDATION after combining shards.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from src.training.sft_release import sha256
from src.verifiers.gold_validation import load_jsonl


REQUIRED_ROW_FIELDS = {
    "candidate_id",
    "task_id",
    "split",
    "disposition",
    "source_sha256",
    "group_ids",
    "system_policy",
    "messages",
}


def _load_shard(directory: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    dataset = directory / "sft_dataset.jsonl"
    manifest_path = directory / "dataset_manifest.json"
    if not dataset.is_file() or not manifest_path.is_file():
        raise FileNotFoundError(
            f"Shard requires sft_dataset.jsonl and dataset_manifest.json: {directory}"
        )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8-sig"))
    expected = str(manifest.get("dataset_sha256", "")).upper()
    actual = sha256(dataset)
    if not expected or expected != actual:
        raise ValueError(
            f"Shard dataset hash mismatch: {directory}; "
            f"expected={expected}, actual={actual}"
        )
    rows = load_jsonl(dataset)
    released = manifest.get("counts", {}).get("released")
    if released is not None and int(released) != len(rows):
        raise ValueError(
            f"Shard row-count mismatch: {directory}; "
            f"manifest={released}, actual={len(rows)}"
        )
    for index, row in enumerate(rows):
        missing = sorted(REQUIRED_ROW_FIELDS - row.keys())
        if missing:
            raise ValueError(f"{dataset}:{index + 1}: missing fields {missing}")
        if str(row["split"]).upper() not in {"TRAIN", "VALIDATION"}:
            raise ValueError(f"{dataset}:{index + 1}: unsupported split")
        if not isinstance(row["group_ids"], list) or not row["group_ids"]:
            raise ValueError(f"{dataset}:{index + 1}: group_ids are required")
        if not isinstance(row["messages"], list) or not row["messages"]:
            raise ValueError(f"{dataset}:{index + 1}: messages are required")
    return rows, {
        "directory": str(directory),
        "dataset": {"path": str(dataset), "sha256": actual, "rows": len(rows)},
        "manifest": {
            "path": str(manifest_path),
            "sha256": sha256(manifest_path),
            "schema_version": manifest.get("schema_version"),
            "review_mode": manifest.get("review_mode"),
        },
    }


def merge_shards(shard_dirs: list[Path]) -> dict[str, Any]:
    if len(shard_dirs) < 2:
        raise ValueError("At least two SFT shards are required for a merge")
    rows: list[dict[str, Any]] = []
    inputs: list[dict[str, Any]] = []
    candidate_ids: set[str] = set()
    source_hashes: set[str] = set()
    group_splits: dict[str, set[str]] = defaultdict(set)
    task_splits: dict[str, set[str]] = defaultdict(set)

    for directory in shard_dirs:
        shard_rows, binding = _load_shard(directory)
        inputs.append(binding)
        for row in shard_rows:
            candidate_id = str(row["candidate_id"])
            source_hash = str(row["source_sha256"]).upper()
            if candidate_id in candidate_ids:
                raise ValueError(f"Duplicate candidate_id across shards: {candidate_id}")
            if source_hash in source_hashes:
                raise ValueError(
                    f"Duplicate selected trajectory across shards: {source_hash}"
                )
            candidate_ids.add(candidate_id)
            source_hashes.add(source_hash)
            split = str(row["split"]).upper()
            normalized = {**row, "split": split, "source_sha256": source_hash}
            task_splits[str(normalized["task_id"])].add(split)
            for group_id in normalized["group_ids"]:
                group_splits[str(group_id)].add(split)
            rows.append(normalized)

    leakage = sorted(
        group_id for group_id, splits in group_splits.items() if len(splits) > 1
    )
    task_leakage = sorted(
        task_id for task_id, splits in task_splits.items() if len(splits) > 1
    )
    reasons: list[str] = []
    if leakage:
        reasons.append("Group leakage across merged splits: " + ", ".join(leakage))
    if task_leakage:
        reasons.append("Task IDs cross merged splits: " + ", ".join(task_leakage))
    split_plan = [
        {"task_id": task_id, "split": next(iter(splits)), "source_split": "TRAIN"}
        for task_id, splits in sorted(
            task_splits.items(), key=lambda item: (len(item[0]), item[0])
        )
    ]
    return {
        "ready": not reasons,
        "reasons": reasons,
        "records": rows if not reasons else [],
        "split_plan": split_plan if not reasons else [],
        "counts": {
            "released": len(rows) if not reasons else 0,
            "train": sum(row["split"] == "TRAIN" for row in rows)
            if not reasons
            else 0,
            "validation": sum(row["split"] == "VALIDATION" for row in rows)
            if not reasons
            else 0,
            "tasks": len({str(row["task_id"]) for row in rows})
            if not reasons
            else 0,
        },
        "leakage_checks": {
            "entity_groups_across_splits": leakage,
            "task_ids_across_splits": task_leakage,
            "passed": not leakage and not task_leakage,
        },
        "inputs": inputs,
    }


def write_merge(result: dict[str, Any], output_dir: Path) -> None:
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty output: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    report = {
        key: value
        for key, value in result.items()
        if key not in {"records", "split_plan"}
    }
    (output_dir / "merge_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    if result["ready"]:
        dataset = output_dir / "sft_dataset.jsonl"
        with dataset.open("w", encoding="utf-8") as handle:
            for row in result["records"]:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        split_plan = output_dir / "split_plan.jsonl"
        with split_plan.open("w", encoding="utf-8") as handle:
            for row in result["split_plan"]:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        manifest = {
            "schema_version": "policy-agent-merged-sft-v1",
            "dataset_sha256": sha256(dataset),
            "split_plan_sha256": sha256(split_plan),
            "split_plan_rows": len(result["split_plan"]),
            "counts": result["counts"],
            "leakage_checks": result["leakage_checks"],
            "inputs": result["inputs"],
        }
        (output_dir / "dataset_manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge gated SFT dataset shards.")
    parser.add_argument("--shard", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = merge_shards(args.shard)
    write_merge(result, args.output)
    print(json.dumps({"ready": result["ready"], **result["counts"]}))
    if not result["ready"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
