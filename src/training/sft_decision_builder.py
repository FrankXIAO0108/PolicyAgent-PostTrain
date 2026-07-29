from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from src.training.sft_release import QualityDecision, entity_groups, sha256
from src.training.quality_adjudication import QUALITY_LABELS
from src.verifiers.gold_validation import load_annotations, load_jsonl


def _unique_rows(path: Path, field: str = "task_id") -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    for row in load_jsonl(path):
        task_id = str(row[field])
        if task_id in rows:
            raise ValueError(f"{path}: duplicate task ID {task_id}")
        rows[task_id] = row
    return rows


def build_sft_decisions(
    policy_annotations_path: Path,
    *,
    adjudicated_quality_path: Path | None = None,
    corrections_path: Path | None = None,
    split_plan_path: Path | None = None,
) -> dict[str, Any]:
    policy = load_annotations(policy_annotations_path)
    blocked = sorted(
        row.task_id
        for row in policy
        if row.status != "ADJUDICATED" or row.label is None
    )
    reasons: list[str] = []
    if blocked:
        reasons.append(
            f"Policy gold is not fully adjudicated; blocked task IDs: {blocked}"
        )
    if adjudicated_quality_path is None:
        reasons.append("Missing adjudicated trajectory-quality labels.")
    if reasons:
        return {
            "ready": False,
            "reasons": reasons,
            "decisions": [],
            "inputs": {
                "policy_annotations": {
                    "path": str(policy_annotations_path),
                    "sha256": sha256(policy_annotations_path),
                },
                "adjudicated_quality": None,
                "corrections": None,
                "split_plan": None,
            },
        }

    assert adjudicated_quality_path is not None
    quality = _unique_rows(adjudicated_quality_path)
    policy_by_task = {row.task_id: row for row in policy}
    missing_quality = sorted(policy_by_task.keys() - quality.keys())
    unknown_quality = sorted(quality.keys() - policy_by_task.keys())
    if missing_quality or unknown_quality:
        reasons.append(
            f"Quality coverage mismatch: missing={missing_quality}, "
            f"unknown={unknown_quality}"
        )
    for task_id, row in quality.items():
        if task_id not in policy_by_task:
            continue
        if row.get("status") != "ADJUDICATED":
            reasons.append(f"Task {task_id}: quality status is not ADJUDICATED.")
        if str(row.get("quality_label")) not in QUALITY_LABELS:
            reasons.append(f"Task {task_id}: unsupported quality label.")
        if str(row.get("policy_label")) != str(policy_by_task[task_id].label):
            reasons.append(f"Task {task_id}: policy/quality label binding mismatch.")
        if (
            row.get("quality_label") == "RAW_GOLD"
            and policy_by_task[task_id].label != "PASS"
        ):
            reasons.append(f"Task {task_id}: RAW_GOLD requires policy PASS.")
        source = Path(str(row.get("source_path", "")))
        if not source.is_file():
            reasons.append(f"Task {task_id}: missing quality source {source}.")
        elif sha256(source) != str(row.get("source_sha256", "")).upper():
            reasons.append(f"Task {task_id}: quality source hash mismatch.")
    if reasons:
        return {
            "ready": False,
            "reasons": reasons,
            "decisions": [],
            "inputs": {
                "policy_annotations": {
                    "path": str(policy_annotations_path),
                    "sha256": sha256(policy_annotations_path),
                },
                "adjudicated_quality": {
                    "path": str(adjudicated_quality_path),
                    "sha256": sha256(adjudicated_quality_path),
                },
                "corrections": None,
                "split_plan": None,
            },
        }

    released_ids = {
        task_id
        for task_id, row in quality.items()
        if row["quality_label"] in {"RAW_GOLD", "CORRECTION_REQUIRED"}
    }
    correction_ids = {
        task_id
        for task_id, row in quality.items()
        if row["quality_label"] == "CORRECTION_REQUIRED"
    }
    corrections = (
        _unique_rows(corrections_path) if corrections_path is not None else {}
    )
    if correction_ids != set(corrections):
        reasons.append(
            "Correction registry coverage mismatch: "
            f"missing={sorted(correction_ids - corrections.keys())}, "
            f"unknown={sorted(corrections.keys() - correction_ids)}"
        )
    splits = _unique_rows(split_plan_path) if split_plan_path is not None else {}
    if released_ids != set(splits):
        reasons.append(
            "Split-plan coverage mismatch: "
            f"missing={sorted(released_ids - splits.keys())}, "
            f"unknown={sorted(splits.keys() - released_ids)}"
        )
    if reasons:
        return {
            "ready": False,
            "reasons": reasons,
            "decisions": [],
            "inputs": {
                "policy_annotations": {
                    "path": str(policy_annotations_path),
                    "sha256": sha256(policy_annotations_path),
                },
                "adjudicated_quality": {
                    "path": str(adjudicated_quality_path),
                    "sha256": sha256(adjudicated_quality_path),
                },
                "corrections": (
                    {
                        "path": str(corrections_path),
                        "sha256": sha256(corrections_path),
                    }
                    if corrections_path is not None
                    else None
                ),
                "split_plan": (
                    {
                        "path": str(split_plan_path),
                        "sha256": sha256(split_plan_path),
                    }
                    if split_plan_path is not None
                    else None
                ),
            },
        }

    decisions: list[dict[str, Any]] = []
    for task_id in sorted(quality, key=lambda value: (len(value), value)):
        row = quality[task_id]
        label = str(row["quality_label"])
        source = Path(str(row["source_path"]))
        base = {
            "task_id": task_id,
            "status": "ADJUDICATED",
            "source_path": str(source),
            "source_sha256": str(row["source_sha256"]).upper(),
            "group_ids": sorted(entity_groups(source)),
            "rationale": str(row["rationale"]),
        }
        if label in {"HOLDOUT", "SEGMENT_REQUIRED"}:
            decision = {
                **base,
                "disposition": "HOLDOUT",
                "split": None,
                "source_split": "TRAIN",
                "correction_path": None,
                "correction_sha256": None,
                "correction_validation_path": None,
                "correction_validation_sha256": None,
            }
        elif label == "RAW_GOLD":
            split = splits[task_id]
            decision = {
                **base,
                "disposition": "RAW_POSITIVE",
                "split": str(split["split"]).upper(),
                "source_split": str(split["source_split"]).upper(),
                "correction_path": None,
                "correction_sha256": None,
                "correction_validation_path": None,
                "correction_validation_sha256": None,
            }
        else:
            split = splits[task_id]
            correction = corrections[task_id]
            decision = {
                **base,
                "disposition": "CORRECTED_POSITIVE",
                "split": str(split["split"]).upper(),
                "source_split": str(split["source_split"]).upper(),
                "correction_path": str(correction["correction_path"]),
                "correction_sha256": str(
                    correction["correction_sha256"]
                ).upper(),
                "correction_validation_path": str(
                    correction["correction_validation_path"]
                ),
                "correction_validation_sha256": str(
                    correction["correction_validation_sha256"]
                ).upper(),
            }
        QualityDecision.from_dict(decision)
        decisions.append(decision)

    return {
        "ready": True,
        "reasons": [],
        "decisions": decisions,
        "inputs": {
            "policy_annotations": {
                "path": str(policy_annotations_path),
                "sha256": sha256(policy_annotations_path),
            },
            "adjudicated_quality": {
                "path": str(adjudicated_quality_path),
                "sha256": sha256(adjudicated_quality_path),
            },
            "corrections": (
                {
                    "path": str(corrections_path),
                    "sha256": sha256(corrections_path),
                }
                if corrections_path is not None
                else None
            ),
            "split_plan": (
                {
                    "path": str(split_plan_path),
                    "sha256": sha256(split_plan_path),
                }
                if split_plan_path is not None
                else None
            ),
        },
    }


def write_outputs(result: dict[str, Any], output_dir: Path) -> None:
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty output: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    report = {key: value for key, value in result.items() if key != "decisions"}
    (output_dir / "decision_build_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    if result["ready"]:
        with (output_dir / "sft_quality_decisions.jsonl").open(
            "w", encoding="utf-8"
        ) as handle:
            for row in result["decisions"]:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build SFT release decisions from adjudicated quality labels."
    )
    parser.add_argument("--policy-annotations", type=Path, required=True)
    parser.add_argument("--adjudicated-quality", type=Path)
    parser.add_argument("--corrections", type=Path)
    parser.add_argument("--split-plan", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = build_sft_decisions(
        args.policy_annotations,
        adjudicated_quality_path=args.adjudicated_quality,
        corrections_path=args.corrections,
        split_plan_path=args.split_plan,
    )
    write_outputs(result, args.output)
    print(json.dumps({"ready": result["ready"], "count": len(result["decisions"])}))
    if not result["ready"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
