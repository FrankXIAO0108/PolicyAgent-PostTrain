from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from .gold_validation import LABELS, GoldAnnotation, load_annotations, load_jsonl


@dataclass(frozen=True, slots=True)
class ReviewDecision:
    task_id: str
    label: str
    reviewer_id: str
    reviewed_at: str
    rationale: str
    evidence_files: tuple[str, ...]

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> ReviewDecision:
        task_id = str(value["task_id"])
        label = str(value["label"]).upper()
        reviewer_id = str(value.get("reviewer_id", "")).strip()
        reviewed_at = str(value.get("reviewed_at", "")).strip()
        rationale = str(value.get("rationale", "")).strip()
        evidence_files = tuple(
            str(path).strip()
            for path in value.get("evidence_files", [])
            if str(path).strip()
        )
        if label not in LABELS:
            raise ValueError(f"Task {task_id}: unsupported label {label!r}")
        if not reviewer_id:
            raise ValueError(f"Task {task_id}: reviewer_id is required")
        if not reviewed_at:
            raise ValueError(f"Task {task_id}: reviewed_at is required")
        try:
            timestamp = datetime.fromisoformat(reviewed_at.replace("Z", "+00:00"))
        except ValueError as error:
            raise ValueError(
                f"Task {task_id}: reviewed_at must be an ISO-8601 timestamp"
            ) from error
        if timestamp.tzinfo is None:
            raise ValueError(
                f"Task {task_id}: reviewed_at must include a timezone"
            )
        if not rationale:
            raise ValueError(f"Task {task_id}: rationale is required")
        if not evidence_files:
            raise ValueError(f"Task {task_id}: evidence_files is required")
        return cls(
            task_id=task_id,
            label=label,
            reviewer_id=reviewer_id,
            reviewed_at=reviewed_at,
            rationale=rationale,
            evidence_files=evidence_files,
        )


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def _load_review_file(path: Path) -> tuple[str, dict[str, ReviewDecision]]:
    decisions: dict[str, ReviewDecision] = {}
    reviewer_ids: set[str] = set()
    for row in load_jsonl(path):
        decision = ReviewDecision.from_dict(row)
        if decision.task_id in decisions:
            raise ValueError(
                f"{path}: duplicate review task ID {decision.task_id}"
            )
        decisions[decision.task_id] = decision
        reviewer_ids.add(decision.reviewer_id)
    if len(reviewer_ids) != 1:
        raise ValueError(
            f"{path}: exactly one reviewer_id is required, got "
            f"{sorted(reviewer_ids)}"
        )
    return next(iter(reviewer_ids)), decisions


def _validate_coverage(
    *,
    source_task_ids: set[str],
    decisions: dict[str, ReviewDecision],
    path: Path,
) -> None:
    missing = sorted(source_task_ids - decisions.keys())
    unknown = sorted(decisions.keys() - source_task_ids)
    if missing or unknown:
        raise ValueError(
            f"{path}: review coverage mismatch; missing={missing}, unknown={unknown}"
        )


def _adjudicated_row(
    annotation: GoldAnnotation,
    *,
    label: str,
    reviewers: list[ReviewDecision],
    resolution: str,
) -> dict[str, Any]:
    evidence_files: list[str] = list(annotation.evidence_files)
    for review in reviewers:
        for path in review.evidence_files:
            if path not in evidence_files:
                evidence_files.append(path)
    return {
        "task_id": annotation.task_id,
        "label": label,
        "status": "ADJUDICATED",
        "source": "independent_dual_review",
        "rationale": " | ".join(
            f"{review.reviewer_id}: {review.rationale}" for review in reviewers
        ),
        "evidence_files": evidence_files,
        "adjudication": {
            "resolution": resolution,
            "reviewers": [
                {
                    "reviewer_id": review.reviewer_id,
                    "label": review.label,
                    "reviewed_at": review.reviewed_at,
                }
                for review in reviewers
            ],
        },
    }


def adjudicate(
    annotations_path: Path,
    reviewer_a_path: Path,
    reviewer_b_path: Path,
    *,
    resolver_path: Path | None = None,
) -> dict[str, Any]:
    annotations = load_annotations(annotations_path)
    invalid_statuses = sorted(
        annotation.task_id
        for annotation in annotations
        if annotation.status != "PROVISIONAL" or annotation.label is None
    )
    if invalid_statuses:
        raise ValueError(
            "Adjudication input must contain only labeled PROVISIONAL rows; "
            f"invalid task IDs: {invalid_statuses}"
        )

    source_by_task = {annotation.task_id: annotation for annotation in annotations}
    task_ids = set(source_by_task)
    reviewer_a, decisions_a = _load_review_file(reviewer_a_path)
    reviewer_b, decisions_b = _load_review_file(reviewer_b_path)
    if reviewer_a == reviewer_b:
        raise ValueError("reviewer-a and reviewer-b must be independent identities")
    _validate_coverage(
        source_task_ids=task_ids,
        decisions=decisions_a,
        path=reviewer_a_path,
    )
    _validate_coverage(
        source_task_ids=task_ids,
        decisions=decisions_b,
        path=reviewer_b_path,
    )

    conflicts = sorted(
        task_id
        for task_id in task_ids
        if decisions_a[task_id].label != decisions_b[task_id].label
    )
    resolver_id: str | None = None
    resolver_decisions: dict[str, ReviewDecision] = {}
    if resolver_path is not None:
        resolver_id, resolver_decisions = _load_review_file(resolver_path)
        if resolver_id in {reviewer_a, reviewer_b}:
            raise ValueError(
                "Resolver must be independent from reviewer-a and reviewer-b"
            )
        _validate_coverage(
            source_task_ids=set(conflicts),
            decisions=resolver_decisions,
            path=resolver_path,
        )

    unresolved = conflicts if resolver_path is None else []
    adjudicated_rows: list[dict[str, Any]] = []
    if not unresolved:
        for task_id in sorted(task_ids, key=lambda value: (len(value), value)):
            reviews = [decisions_a[task_id], decisions_b[task_id]]
            if task_id in conflicts:
                reviews.append(resolver_decisions[task_id])
                label = resolver_decisions[task_id].label
                resolution = "third_reviewer_resolution"
            else:
                label = decisions_a[task_id].label
                resolution = "dual_reviewer_agreement"
            adjudicated_rows.append(
                _adjudicated_row(
                    source_by_task[task_id],
                    label=label,
                    reviewers=reviews,
                    resolution=resolution,
                )
            )

    return {
        "schema_version": "policy-grounding-adjudication-v0.1",
        "inputs": {
            "annotations": {
                "path": str(annotations_path),
                "sha256": _sha256(annotations_path),
            },
            "reviewer_a": {
                "path": str(reviewer_a_path),
                "sha256": _sha256(reviewer_a_path),
                "reviewer_id": reviewer_a,
            },
            "reviewer_b": {
                "path": str(reviewer_b_path),
                "sha256": _sha256(reviewer_b_path),
                "reviewer_id": reviewer_b,
            },
            "resolver": (
                {
                    "path": str(resolver_path),
                    "sha256": _sha256(resolver_path),
                    "reviewer_id": resolver_id,
                }
                if resolver_path is not None
                else None
            ),
        },
        "coverage": {
            "source_rows": len(annotations),
            "agreement_rows": len(annotations) - len(conflicts),
            "conflict_rows": len(conflicts),
            "adjudicated_rows": len(adjudicated_rows),
        },
        "conflicts": [
            {
                "task_id": task_id,
                "reviewer_a_label": decisions_a[task_id].label,
                "reviewer_b_label": decisions_b[task_id].label,
                "resolved_label": (
                    resolver_decisions[task_id].label
                    if task_id in resolver_decisions
                    else None
                ),
            }
            for task_id in conflicts
        ],
        "adjudicated_annotations": adjudicated_rows,
        "release_gate": {
            "adjudicated_annotations_ready": not unresolved,
            "unresolved_task_ids": unresolved,
            "reason": (
                "Two independent reviews cover every row and all conflicts "
                "have an independent third-reviewer resolution."
                if not unresolved
                else "Independent reviewer conflicts remain unresolved."
            ),
        },
    }


def write_outputs(result: dict[str, Any], output_dir: Path) -> None:
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(
            f"Refusing to overwrite non-empty adjudication output: {output_dir}"
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    report = dict(result)
    adjudicated_rows = report.pop("adjudicated_annotations")
    (output_dir / "adjudication_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    with (output_dir / "conflicts.jsonl").open("w", encoding="utf-8") as handle:
        for conflict in result["conflicts"]:
            handle.write(json.dumps(conflict, ensure_ascii=False) + "\n")
    if result["release_gate"]["adjudicated_annotations_ready"]:
        with (output_dir / "adjudicated_annotations.jsonl").open(
            "w", encoding="utf-8"
        ) as handle:
            for row in adjudicated_rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Build adjudicated policy gold from two independent reviews, "
            "with optional third-reviewer conflict resolution."
        )
    )
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--reviewer-a", type=Path, required=True)
    parser.add_argument("--reviewer-b", type=Path, required=True)
    parser.add_argument("--resolver", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    result = adjudicate(
        args.annotations,
        args.reviewer_a,
        args.reviewer_b,
        resolver_path=args.resolver,
    )
    write_outputs(result, args.output)
    print(json.dumps(result["coverage"], ensure_ascii=False, indent=2))
    if not result["release_gate"]["adjudicated_annotations_ready"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
