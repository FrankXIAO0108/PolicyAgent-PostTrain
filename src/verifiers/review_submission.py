from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any

from .adjudication import ReviewDecision
from .gold_validation import load_jsonl


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def _load_submission(path: Path) -> list[dict[str, Any]]:
    if path.suffix.lower() == ".jsonl":
        return load_jsonl(path)
    if path.suffix.lower() != ".csv":
        raise ValueError("Submission must be a .csv or .jsonl file")
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        raw_evidence = str(row.get("evidence_files", "")).strip()
        try:
            evidence = json.loads(raw_evidence)
        except json.JSONDecodeError as error:
            raise ValueError(
                f"Task {row.get('task_id', '')}: evidence_files must remain "
                "the pre-filled JSON list"
            ) from error
        if not isinstance(evidence, list):
            raise ValueError(
                f"Task {row.get('task_id', '')}: evidence_files must be a JSON list"
            )
        row["evidence_files"] = evidence
    return rows


def _safe_evidence_path(packet_dir: Path, relative_path: str) -> Path:
    path = Path(relative_path)
    if path.is_absolute():
        raise ValueError(f"Evidence path must be relative: {relative_path}")
    packet_root = packet_dir.resolve()
    resolved = (packet_dir / path).resolve()
    if packet_root != resolved and packet_root not in resolved.parents:
        raise ValueError(f"Evidence path escapes packet: {relative_path}")
    return resolved


def preflight_submission(
    packet_dir: Path,
    submission_path: Path,
) -> dict[str, Any]:
    template_path = packet_dir / "review_template.jsonl"
    if not template_path.is_file():
        raise FileNotFoundError(f"Missing packet template: {template_path}")
    if not submission_path.is_file():
        raise FileNotFoundError(f"Missing submission: {submission_path}")

    template_rows = load_jsonl(template_path)
    expected = {str(row["task_id"]): row for row in template_rows}
    errors: list[str] = []
    decisions: dict[str, ReviewDecision] = {}
    try:
        submitted_rows = _load_submission(submission_path)
    except (ValueError, KeyError) as error:
        submitted_rows = []
        errors.append(str(error))

    for index, row in enumerate(submitted_rows, start=2):
        try:
            decision = ReviewDecision.from_dict(row)
            if decision.task_id in decisions:
                raise ValueError(f"duplicate task ID {decision.task_id}")
            decisions[decision.task_id] = decision
        except (ValueError, KeyError, TypeError) as error:
            errors.append(f"Row {index}: {error}")

    expected_ids = set(expected)
    submitted_ids = set(decisions)
    missing = sorted(expected_ids - submitted_ids, key=lambda value: (len(value), value))
    unknown = sorted(submitted_ids - expected_ids, key=lambda value: (len(value), value))
    if missing:
        errors.append(f"Missing task IDs: {missing}")
    if unknown:
        errors.append(f"Unknown task IDs: {unknown}")

    reviewer_ids = sorted({decision.reviewer_id for decision in decisions.values()})
    if len(reviewer_ids) != 1:
        errors.append(
            f"Exactly one reviewer_id is required, got {reviewer_ids}"
        )

    for task_id in sorted(expected_ids & submitted_ids, key=lambda value: (len(value), value)):
        decision = decisions[task_id]
        template = expected[task_id]
        expected_files = tuple(str(path) for path in template["evidence_files"])
        if decision.evidence_files != expected_files:
            errors.append(f"Task {task_id}: evidence_files changed from packet")
            continue
        evidence_by_path = {
            str(value["path"]): str(value["sha256"]).upper()
            for value in template["blind_evidence"].values()
        }
        for relative_path in expected_files:
            try:
                evidence_path = _safe_evidence_path(packet_dir, relative_path)
                if not evidence_path.is_file():
                    raise ValueError(f"Evidence file is missing: {relative_path}")
                expected_hash = evidence_by_path.get(relative_path)
                if expected_hash is None:
                    raise ValueError(
                        f"Evidence path is absent from packet manifest: {relative_path}"
                    )
                if _sha256(evidence_path) != expected_hash:
                    raise ValueError(f"Evidence hash mismatch: {relative_path}")
            except ValueError as error:
                errors.append(f"Task {task_id}: {error}")

    normalized_rows = [
        {
            "task_id": decision.task_id,
            "label": decision.label,
            "reviewer_id": decision.reviewer_id,
            "reviewed_at": decision.reviewed_at,
            "rationale": decision.rationale,
            "evidence_files": list(decision.evidence_files),
        }
        for decision in sorted(
            decisions.values(), key=lambda value: (len(value.task_id), value.task_id)
        )
    ]
    ready = not errors
    return {
        "schema_version": "policy-review-submission-preflight-v0.1",
        "status": "READY" if ready else "REJECTED",
        "inputs": {
            "packet": str(packet_dir),
            "template_sha256": _sha256(template_path),
            "submission": str(submission_path),
            "submission_sha256": _sha256(submission_path),
        },
        "coverage": {
            "expected_rows": len(expected),
            "valid_rows": len(decisions),
            "missing_task_ids": missing,
            "unknown_task_ids": unknown,
        },
        "reviewer_id": reviewer_ids[0] if len(reviewer_ids) == 1 else None,
        "errors": errors,
        "normalized_reviews": normalized_rows if ready else [],
    }


def write_outputs(result: dict[str, Any], output_dir: Path) -> None:
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(
            f"Refusing to overwrite non-empty preflight output: {output_dir}"
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    report = dict(result)
    normalized_rows = report.pop("normalized_reviews")
    (output_dir / "preflight_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    if result["status"] == "READY":
        with (output_dir / "normalized_review.jsonl").open(
            "w", encoding="utf-8"
        ) as handle:
            for row in normalized_rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Validate one independent review submission against its frozen "
            "blind-review packet and normalize it for adjudication."
        )
    )
    parser.add_argument("--packet", type=Path, required=True)
    parser.add_argument("--submission", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    result = preflight_submission(args.packet, args.submission)
    write_outputs(result, args.output)
    print(
        json.dumps(
            {
                "status": result["status"],
                "coverage": result["coverage"],
                "errors": result["errors"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    if result["status"] != "READY":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
