from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from src.verifiers.gold_validation import GoldAnnotation, load_annotations, load_jsonl


QUALITY_LABELS = {
    "RAW_GOLD",
    "CORRECTION_REQUIRED",
    "HOLDOUT",
    "SEGMENT_REQUIRED",
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


@dataclass(frozen=True, slots=True)
class QualityReview:
    task_id: str
    quality_label: str
    reviewer_id: str
    reviewed_at: str
    rationale: str
    evidence_files: tuple[str, ...]

    @classmethod
    def from_dict(cls, row: dict[str, Any]) -> QualityReview:
        task_id = str(row["task_id"])
        label = str(row.get("quality_label", "")).upper()
        reviewer = str(row.get("reviewer_id", "")).strip()
        reviewed_at = str(row.get("reviewed_at", "")).strip()
        rationale = str(row.get("rationale", "")).strip()
        evidence = tuple(
            str(value).strip()
            for value in row.get("evidence_files", [])
            if str(value).strip()
        )
        if label not in QUALITY_LABELS:
            raise ValueError(
                f"Task {task_id}: quality_label must be one of "
                f"{sorted(QUALITY_LABELS)}"
            )
        if not reviewer:
            raise ValueError(f"Task {task_id}: reviewer_id is required")
        try:
            timestamp = datetime.fromisoformat(reviewed_at.replace("Z", "+00:00"))
        except ValueError as error:
            raise ValueError(
                f"Task {task_id}: reviewed_at must be ISO-8601"
            ) from error
        if timestamp.tzinfo is None:
            raise ValueError(f"Task {task_id}: reviewed_at requires timezone")
        if not rationale or not evidence:
            raise ValueError(
                f"Task {task_id}: rationale and evidence_files are required"
            )
        return cls(task_id, label, reviewer, reviewed_at, rationale, evidence)


def _load_reviews(path: Path) -> tuple[str, dict[str, QualityReview]]:
    reviews: dict[str, QualityReview] = {}
    reviewers: set[str] = set()
    for row in load_jsonl(path):
        review = QualityReview.from_dict(row)
        if review.task_id in reviews:
            raise ValueError(f"{path}: duplicate task ID {review.task_id}")
        reviews[review.task_id] = review
        reviewers.add(review.reviewer_id)
    if len(reviewers) != 1:
        raise ValueError(f"{path}: exactly one reviewer identity is required")
    return next(iter(reviewers)), reviews


def _coverage(
    expected: set[str], actual: dict[str, QualityReview], path: Path
) -> None:
    missing = sorted(expected - actual.keys())
    unknown = sorted(actual.keys() - expected)
    if missing or unknown:
        raise ValueError(
            f"{path}: review coverage mismatch; missing={missing}, unknown={unknown}"
        )


def _source_paths(experiment_dir: Path, task_id: str) -> tuple[Path, Path]:
    task_dir = experiment_dir / f"task_{task_id}"
    trajectory = task_dir / "returned_results.json"
    summary = task_dir / "summary.json"
    missing = [str(path) for path in (trajectory, summary) if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Task {task_id}: missing sources {missing}")
    return trajectory, summary


def build_review_template(
    annotations: list[GoldAnnotation], experiment_dir: Path
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for annotation in sorted(
        annotations, key=lambda value: (len(value.task_id), value.task_id)
    ):
        trajectory, summary = _source_paths(experiment_dir, annotation.task_id)
        rows.append(
            {
                "task_id": annotation.task_id,
                "adjudicated_policy_label": annotation.label,
                "blind_evidence": {
                    "trajectory": {
                        "path": str(trajectory),
                        "sha256": sha256(trajectory),
                    },
                    "summary": {
                        "path": str(summary),
                        "sha256": sha256(summary),
                    },
                },
                "quality_label": "",
                "reviewer_id": "",
                "reviewed_at": "",
                "rationale": "",
                "evidence_files": [str(trajectory), str(summary)],
            }
        )
    return rows


def evaluate_quality_adjudication(
    annotations_path: Path,
    experiment_dir: Path,
    *,
    reviewer_a_path: Path | None = None,
    reviewer_b_path: Path | None = None,
    resolver_path: Path | None = None,
) -> dict[str, Any]:
    annotations = load_annotations(annotations_path)
    blocked = sorted(
        row.task_id
        for row in annotations
        if row.status != "ADJUDICATED" or row.label is None
    )
    if blocked:
        return {
            "ready": False,
            "stage": "BLOCKED_POLICY_ADJUDICATION",
            "reasons": [
                "Trajectory-quality review requires complete adjudicated policy "
                f"gold; blocked task IDs: {blocked}"
            ],
            "inputs": {
                "annotations": {
                    "path": str(annotations_path),
                    "sha256": sha256(annotations_path),
                }
            },
            "template_rows": [],
            "conflicts": [],
            "adjudicated_quality": [],
        }

    template = build_review_template(annotations, experiment_dir)
    if reviewer_a_path is None or reviewer_b_path is None:
        return {
            "ready": False,
            "stage": "INDEPENDENT_REVIEWS_REQUIRED",
            "reasons": ["Two independent trajectory-quality reviews are required."],
            "inputs": {
                "annotations": {
                    "path": str(annotations_path),
                    "sha256": sha256(annotations_path),
                }
            },
            "template_rows": template,
            "conflicts": [],
            "adjudicated_quality": [],
        }

    annotation_by_task = {row.task_id: row for row in annotations}
    task_ids = set(annotation_by_task)
    reviewer_a, reviews_a = _load_reviews(reviewer_a_path)
    reviewer_b, reviews_b = _load_reviews(reviewer_b_path)
    if reviewer_a == reviewer_b:
        raise ValueError("Quality reviewers must have independent identities")
    _coverage(task_ids, reviews_a, reviewer_a_path)
    _coverage(task_ids, reviews_b, reviewer_b_path)
    conflicts = sorted(
        task_id
        for task_id in task_ids
        if reviews_a[task_id].quality_label != reviews_b[task_id].quality_label
    )

    resolver_id: str | None = None
    resolver_reviews: dict[str, QualityReview] = {}
    if resolver_path is not None:
        resolver_id, resolver_reviews = _load_reviews(resolver_path)
        if resolver_id in {reviewer_a, reviewer_b}:
            raise ValueError("Quality resolver must be a third independent identity")
        _coverage(set(conflicts), resolver_reviews, resolver_path)
    unresolved = conflicts if resolver_path is None else []

    adjudicated: list[dict[str, Any]] = []
    if not unresolved:
        for task_id in sorted(task_ids, key=lambda value: (len(value), value)):
            selected = (
                resolver_reviews[task_id]
                if task_id in conflicts
                else reviews_a[task_id]
            )
            annotation = annotation_by_task[task_id]
            if selected.quality_label == "RAW_GOLD" and annotation.label != "PASS":
                raise ValueError(
                    f"Task {task_id}: RAW_GOLD requires adjudicated policy PASS"
                )
            reviewers = [reviews_a[task_id], reviews_b[task_id]]
            if task_id in conflicts:
                reviewers.append(selected)
            trajectory, summary = _source_paths(experiment_dir, task_id)
            adjudicated.append(
                {
                    "task_id": task_id,
                    "status": "ADJUDICATED",
                    "quality_label": selected.quality_label,
                    "policy_label": annotation.label,
                    "source_path": str(trajectory),
                    "source_sha256": sha256(trajectory),
                    "rationale": " | ".join(
                        f"{review.reviewer_id}: {review.rationale}"
                        for review in reviewers
                    ),
                    "evidence_files": sorted(
                        {
                            str(summary),
                            *(
                                evidence
                                for review in reviewers
                                for evidence in review.evidence_files
                            ),
                        }
                    ),
                    "adjudication": {
                        "resolution": (
                            "third_reviewer_resolution"
                            if task_id in conflicts
                            else "dual_reviewer_agreement"
                        ),
                        "reviewers": [
                            {
                                "reviewer_id": review.reviewer_id,
                                "quality_label": review.quality_label,
                                "reviewed_at": review.reviewed_at,
                            }
                            for review in reviewers
                        ],
                    },
                }
            )

    return {
        "ready": not unresolved,
        "stage": "COMPLETE" if not unresolved else "CONFLICT_RESOLUTION_REQUIRED",
        "reasons": (
            []
            if not unresolved
            else [f"Unresolved quality-review conflicts: {unresolved}"]
        ),
        "inputs": {
            "annotations": {
                "path": str(annotations_path),
                "sha256": sha256(annotations_path),
            },
            "reviewer_a": {
                "path": str(reviewer_a_path),
                "sha256": sha256(reviewer_a_path),
                "reviewer_id": reviewer_a,
            },
            "reviewer_b": {
                "path": str(reviewer_b_path),
                "sha256": sha256(reviewer_b_path),
                "reviewer_id": reviewer_b,
            },
            "resolver": (
                {
                    "path": str(resolver_path),
                    "sha256": sha256(resolver_path),
                    "reviewer_id": resolver_id,
                }
                if resolver_path is not None
                else None
            ),
        },
        "template_rows": [],
        "conflicts": [
            {
                "task_id": task_id,
                "reviewer_a_label": reviews_a[task_id].quality_label,
                "reviewer_b_label": reviews_b[task_id].quality_label,
                "resolved_label": (
                    resolver_reviews[task_id].quality_label
                    if task_id in resolver_reviews
                    else None
                ),
            }
            for task_id in conflicts
        ],
        "adjudicated_quality": adjudicated,
    }


def write_outputs(result: dict[str, Any], output_dir: Path) -> None:
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty output: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    report = {
        key: value
        for key, value in result.items()
        if key not in {"template_rows", "adjudicated_quality"}
    }
    (output_dir / "quality_adjudication_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    if result["template_rows"]:
        with (output_dir / "quality_review_template.jsonl").open(
            "w", encoding="utf-8"
        ) as handle:
            for row in result["template_rows"]:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    if result["ready"]:
        with (output_dir / "adjudicated_quality.jsonl").open(
            "w", encoding="utf-8"
        ) as handle:
            for row in result["adjudicated_quality"]:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build independently adjudicated trajectory-quality labels."
    )
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--experiment", type=Path, required=True)
    parser.add_argument("--reviewer-a", type=Path)
    parser.add_argument("--reviewer-b", type=Path)
    parser.add_argument("--resolver", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = evaluate_quality_adjudication(
        args.annotations,
        args.experiment,
        reviewer_a_path=args.reviewer_a,
        reviewer_b_path=args.reviewer_b,
        resolver_path=args.resolver,
    )
    write_outputs(result, args.output)
    print(json.dumps({"ready": result["ready"], "stage": result["stage"]}))
    if not result["ready"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
