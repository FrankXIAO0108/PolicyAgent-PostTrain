"""Build candidate-level SFT decisions for tau2 teacher trajectories.

The task-level ``sft_decision_builder`` requires adjudicated policy gold and
indexes by task. Teacher candidates are candidate-level and this round has no
human-independent policy adjudication, so they use a dedicated builder:

- two review identities (``reviews_a`` / ``reviews_b``) must both cover every
  candidate and agree on the quality label (conflicts fail closed);
- ``CORRECTION_REQUIRED`` maps to ``CORRECTED_POSITIVE`` and requires a
  validated correction artifact bound by hash plus a TRAIN/VALIDATION split;
- ``HOLDOUT`` maps to ``HOLDOUT`` with no split and no correction data;
- entity groups are read from the frozen source snapshot and checked for
  leakage across splits at decision time.

The output status is ``SECOND_REVIEWED``, not ``ADJUDICATED``: the second
reviewer is an assistant with the same source as the author, so this must not
be presented as human-independent adjudication.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.training.sft_release import entity_groups, sha256


DISPOSITIONS = {"CORRECTED_POSITIVE", "HOLDOUT"}
SPLITS = {"TRAIN", "VALIDATION"}
TEACHER_LABELS = {"CORRECTION_REQUIRED", "HOLDOUT"}


def _unique_rows(path: Path, field: str) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    for row in load_jsonl(path):
        key = str(row[field])
        if key in rows:
            raise ValueError(f"{path}: duplicate {field} {key}")
        rows[key] = row
    return rows


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8-sig").splitlines()
        if line.strip()
    ]


@dataclass(frozen=True, slots=True)
class CandidateDecision:
    candidate_id: str
    task_id: str
    status: str
    quality_label: str
    disposition: str
    split: str | None
    source_split: str
    source_path: Path | None
    source_sha256: str | None
    correction_path: Path | None
    correction_sha256: str | None
    correction_validation_path: Path | None
    correction_validation_sha256: str | None
    group_ids: tuple[str, ...]
    rationale: str

    @classmethod
    def from_dict(cls, row: dict[str, Any]) -> CandidateDecision:
        candidate_id = str(row["candidate_id"])
        task_id = str(row.get("task_id", ""))
        status = str(row.get("status", "")).upper()
        quality_label = str(row.get("quality_label", "")).upper()
        disposition = str(row.get("disposition", "")).upper()
        split_value = row.get("split")
        split = str(split_value).upper() if split_value else None
        source_split = str(row.get("source_split", "")).upper()
        source_value = row.get("source_path")
        source_path = Path(str(source_value)) if source_value else None
        source_hash = (
            str(row["source_sha256"]).upper()
            if row.get("source_sha256")
            else None
        )
        correction_value = row.get("correction_path")
        correction_path = Path(str(correction_value)) if correction_value else None
        correction_hash = (
            str(row["correction_sha256"]).upper()
            if row.get("correction_sha256")
            else None
        )
        validation_value = row.get("correction_validation_path")
        validation_path = (
            Path(str(validation_value)) if validation_value else None
        )
        validation_hash = (
            str(row["correction_validation_sha256"]).upper()
            if row.get("correction_validation_sha256")
            else None
        )
        group_ids = tuple(
            sorted(
                {
                    str(value).strip()
                    for value in row.get("group_ids", [])
                    if str(value).strip()
                }
            )
        )
        rationale = str(row.get("rationale", "")).strip()

        if status != "SECOND_REVIEWED":
            raise ValueError(
                f"Candidate {candidate_id}: status must be SECOND_REVIEWED"
            )
        if quality_label not in TEACHER_LABELS:
            raise ValueError(
                f"Candidate {candidate_id}: unsupported quality label "
                f"{quality_label!r}"
            )
        if disposition not in DISPOSITIONS:
            raise ValueError(
                f"Candidate {candidate_id}: disposition must be one of "
                f"{sorted(DISPOSITIONS)}"
            )
        if disposition == "HOLDOUT":
            if split is not None:
                raise ValueError(
                    f"Candidate {candidate_id}: HOLDOUT must not have a split"
                )
        else:
            if split not in SPLITS:
                raise ValueError(
                    f"Candidate {candidate_id}: released rows require split "
                    f"TRAIN or VALIDATION"
                )
            if source_split != "TRAIN":
                raise ValueError(
                    f"Candidate {candidate_id}: official TEST sources cannot "
                    "enter SFT data"
                )
            if not source_path or not source_hash:
                raise ValueError(
                    f"Candidate {candidate_id}: released rows require source "
                    "path and hash"
                )
            if not group_ids:
                raise ValueError(
                    f"Candidate {candidate_id}: released rows require group_ids"
                )
            if (
                correction_path is None
                or not correction_hash
                or validation_path is None
                or not validation_hash
            ):
                raise ValueError(
                    f"Candidate {candidate_id}: corrected rows require "
                    "correction and validation paths and hashes"
                )
        if (
            disposition == "HOLDOUT"
            and any(
                value is not None
                for value in (
                    correction_path,
                    correction_hash,
                    validation_path,
                    validation_hash,
                )
            )
        ):
            raise ValueError(
                f"Candidate {candidate_id}: HOLDOUT must not carry correction "
                "data"
            )
        if not rationale:
            raise ValueError(f"Candidate {candidate_id}: rationale is required")
        return cls(
            candidate_id=candidate_id,
            task_id=task_id,
            status=status,
            quality_label=quality_label,
            disposition=disposition,
            split=split,
            source_split=source_split,
            source_path=source_path,
            source_sha256=source_hash,
            correction_path=correction_path,
            correction_sha256=correction_hash,
            correction_validation_path=validation_path,
            correction_validation_sha256=validation_hash,
            group_ids=group_ids,
            rationale=rationale,
        )


def _verify_file(path: Path, expected: str, candidate_id: str, field: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(
            f"Candidate {candidate_id}: {field} does not exist: {path}"
        )
    actual = sha256(path)
    if actual != expected:
        raise ValueError(
            f"Candidate {candidate_id}: {field} hash mismatch: "
            f"expected={expected}, actual={actual}"
        )


def build_teacher_candidate_decisions(
    reviews_a_path: Path,
    reviews_b_path: Path,
    *,
    corrections_path: Path | None = None,
    split_plan_path: Path | None = None,
) -> dict[str, Any]:
    reviews_a = _unique_rows(reviews_a_path, "candidate_id")
    reviews_b = _unique_rows(reviews_b_path, "candidate_id")
    reasons: list[str] = []
    if set(reviews_a) != set(reviews_b):
        reasons.append(
            "Review coverage mismatch: "
            f"missing_in_b={sorted(set(reviews_a) - set(reviews_b))}, "
            f"missing_in_a={sorted(set(reviews_b) - set(reviews_a))}"
        )
    labels: dict[str, str] = {}
    for candidate_id in sorted(set(reviews_a) | set(reviews_b)):
        row_a = reviews_a.get(candidate_id)
        row_b = reviews_b.get(candidate_id)
        if row_a is None or row_b is None:
            continue
        label_a = str(row_a.get("quality_label", ""))
        label_b = str(row_b.get("quality_label", ""))
        if label_a != label_b:
            reasons.append(
                f"Candidate {candidate_id}: reviewer label conflict "
                f"{label_a!r} vs {label_b!r}"
            )
            continue
        if label_a not in TEACHER_LABELS:
            reasons.append(
                f"Candidate {candidate_id}: unsupported quality label {label_a!r}"
            )
            continue
        labels[candidate_id] = label_a

    corrections = (
        _unique_rows(corrections_path, "candidate_id")
        if corrections_path is not None
        else {}
    )
    splits = (
        _unique_rows(split_plan_path, "candidate_id")
        if split_plan_path is not None
        else {}
    )
    decisions: list[dict[str, Any]] = []
    group_splits: dict[str, set[str]] = defaultdict(set)

    for candidate_id in sorted(labels):
        label = labels[candidate_id]
        row_a = reviews_a[candidate_id]
        task_id = str(row_a.get("task_id", ""))
        if label == "HOLDOUT":
            if candidate_id in corrections:
                reasons.append(
                    f"Candidate {candidate_id}: HOLDOUT must not have "
                    "correction data"
                )
            if candidate_id in splits:
                reasons.append(
                    f"Candidate {candidate_id}: HOLDOUT must not have a "
                    "split-plan row"
                )
            decisions.append(
                {
                    "candidate_id": candidate_id,
                    "task_id": task_id,
                    "status": "SECOND_REVIEWED",
                    "quality_label": label,
                    "disposition": "HOLDOUT",
                    "split": None,
                    "source_split": "TRAIN",
                    "source_path": None,
                    "source_sha256": None,
                    "correction_path": None,
                    "correction_sha256": None,
                    "correction_validation_path": None,
                    "correction_validation_sha256": None,
                    "group_ids": [],
                    "rationale": "Teacher candidate held out (environment or "
                    "simulator drift); never enters SFT/DPO/RL pools.",
                }
            )
            continue

        if candidate_id not in corrections:
            reasons.append(
                f"Candidate {candidate_id}: CORRECTION_REQUIRED needs a "
                "correction registry row"
            )
            continue
        if candidate_id not in splits:
            reasons.append(
                f"Candidate {candidate_id}: CORRECTION_REQUIRED needs a "
                "split-plan row"
            )
            continue
        correction = corrections[candidate_id]
        split_row = splits[candidate_id]
        try:
            _verify_file(
                Path(correction["source_path"]),
                str(correction["source_sha256"]).upper(),
                candidate_id,
                "source",
            )
            _verify_file(
                Path(correction["correction_path"]),
                str(correction["correction_sha256"]).upper(),
                candidate_id,
                "correction",
            )
            _verify_file(
                Path(correction["correction_validation_path"]),
                str(correction["correction_validation_sha256"]).upper(),
                candidate_id,
                "correction_validation",
            )
        except (FileNotFoundError, ValueError) as error:
            reasons.append(str(error))
            continue
        validation = json.loads(
            Path(correction["correction_validation_path"]).read_text(
                encoding="utf-8-sig"
            )
        )
        if not validation.get("ready"):
            reasons.append(
                f"Candidate {candidate_id}: correction validation is not ready"
            )
            continue
        if (
            str(validation.get("task_id")) != task_id
            or str(validation.get("correction_sha256", "")).upper()
            != str(correction["correction_sha256"]).upper()
        ):
            reasons.append(
                f"Candidate {candidate_id}: correction validation binding "
                "mismatch"
            )
            continue
        reviewer_ids = {
            str(row_a.get("reviewer_id", "")),
            str(reviews_b[candidate_id].get("reviewer_id", "")),
        }
        correction_payload = json.loads(
            Path(correction["correction_path"]).read_text(
                encoding="utf-8-sig"
            )
        )
        author_id = str(correction_payload.get("author_id", ""))
        if author_id in reviewer_ids:
            reasons.append(
                f"Candidate {candidate_id}: correction author {author_id!r} "
                "cannot also be a reviewer"
            )
            continue
        source_path = Path(correction["source_path"])
        group_ids = tuple(sorted(entity_groups(source_path)))
        if not group_ids:
            reasons.append(
                f"Candidate {candidate_id}: no entity groups found in source"
            )
            continue
        split = str(split_row.get("split", "")).upper()
        source_split = str(split_row.get("source_split", "")).upper()
        if split not in SPLITS:
            reasons.append(
                f"Candidate {candidate_id}: split must be TRAIN or VALIDATION"
            )
        if source_split != "TRAIN":
            reasons.append(
                f"Candidate {candidate_id}: source_split must be TRAIN"
            )
        for group_id in group_ids:
            group_splits[group_id].add(split)
        decisions.append(
            {
                "candidate_id": candidate_id,
                "task_id": task_id,
                "status": "SECOND_REVIEWED",
                "quality_label": label,
                "disposition": "CORRECTED_POSITIVE",
                "split": split,
                "source_split": source_split,
                "source_path": str(source_path),
                "source_sha256": str(correction["source_sha256"]).upper(),
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
                "group_ids": list(group_ids),
                "rationale": "Two-reviewer agreement on CORRECTION_REQUIRED; "
                "validated ENVIRONMENT_REPLAY correction bound by hash.",
            }
        )

    leakage = sorted(
        group_id
        for group_id, splits_seen in group_splits.items()
        if len(splits_seen) > 1
    )
    if leakage:
        reasons.append("Group leakage across splits: " + ", ".join(leakage))

    valid_rows = [CandidateDecision.from_dict(row) for row in decisions]
    result: dict[str, Any] = {
        "ready": not reasons,
        "reasons": reasons,
        "decisions": decisions if not reasons else [],
        "counts": {
            "reviewed": len(labels),
            "corrected_positive": sum(
                row.disposition == "CORRECTED_POSITIVE" for row in valid_rows
            ),
            "holdout": sum(row.disposition == "HOLDOUT" for row in valid_rows),
            "train": sum(row.split == "TRAIN" for row in valid_rows),
            "validation": sum(
                row.split == "VALIDATION" for row in valid_rows
            ),
        },
        "inputs": {
            "reviews_a": {
                "path": str(reviews_a_path),
                "sha256": sha256(reviews_a_path),
            },
            "reviews_b": {
                "path": str(reviews_b_path),
                "sha256": sha256(reviews_b_path),
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
    return result


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
        with (output_dir / "teacher_candidate_decisions.jsonl").open(
            "w", encoding="utf-8"
        ) as handle:
            for row in result["decisions"]:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build candidate-level SFT decisions for teacher trajectories."
    )
    parser.add_argument("--reviews-a", type=Path, required=True)
    parser.add_argument("--reviews-b", type=Path, required=True)
    parser.add_argument("--corrections", type=Path)
    parser.add_argument("--split-plan", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = build_teacher_candidate_decisions(
        args.reviews_a,
        args.reviews_b,
        corrections_path=args.corrections,
        split_plan_path=args.split_plan,
    )
    write_outputs(result, args.output)
    print(json.dumps({"ready": result["ready"], "counts": result["counts"]}))
    if not result["ready"]:
        for reason in result["reasons"]:
            print(reason)
        raise SystemExit(2)


if __name__ == "__main__":
    main()
