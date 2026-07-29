from __future__ import annotations

import argparse
import json
from collections import Counter
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Iterable


LABELS = ("PASS", "REVIEW", "FAIL")
STATUSES = ("ADJUDICATED", "PROVISIONAL", "UNREVIEWED")


@dataclass(frozen=True, slots=True)
class GoldAnnotation:
    task_id: str
    label: str | None
    status: str
    source: str
    rationale: str
    evidence_files: tuple[str, ...]

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> GoldAnnotation:
        task_id = str(value["task_id"])
        label_value = value.get("label")
        label = str(label_value).upper() if label_value is not None else None
        status = str(value["status"]).upper()
        if status not in STATUSES:
            raise ValueError(f"Task {task_id}: unsupported status {status!r}")
        if status == "UNREVIEWED":
            if label is not None:
                raise ValueError(
                    f"Task {task_id}: UNREVIEWED rows must not contain a label"
                )
        elif label not in LABELS:
            raise ValueError(
                f"Task {task_id}: reviewed rows require one of {LABELS}, got {label!r}"
            )
        return cls(
            task_id=task_id,
            label=label,
            status=status,
            source=str(value.get("source", "")),
            rationale=str(value.get("rationale", "")),
            evidence_files=tuple(str(path) for path in value.get("evidence_files", [])),
        )


def load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as error:
                raise ValueError(f"{path}:{line_number}: invalid JSON") from error
    return rows


def load_annotations(path: str | Path) -> list[GoldAnnotation]:
    annotations = [GoldAnnotation.from_dict(row) for row in load_jsonl(path)]
    task_ids = [annotation.task_id for annotation in annotations]
    duplicates = sorted(
        task_id for task_id, count in Counter(task_ids).items() if count > 1
    )
    if duplicates:
        raise ValueError(f"Duplicate annotation task IDs: {duplicates}")
    return annotations


def load_predictions(path: str | Path) -> dict[str, str]:
    # Historical PowerShell-generated artifacts may carry a UTF-8 BOM.
    payload = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    if isinstance(payload, dict):
        candidates: Iterable[dict[str, Any]] = payload.get("results", [])
    elif isinstance(payload, list):
        candidates = payload
    else:
        raise ValueError("Prediction artifact must be a JSON list or results object")

    predictions: dict[str, str] = {}
    for row in candidates:
        task_id = str(row["task_id"])
        verdict = str(row["verdict"]).upper()
        if verdict not in LABELS:
            raise ValueError(f"Task {task_id}: unsupported prediction {verdict!r}")
        if task_id in predictions:
            raise ValueError(f"Duplicate prediction task ID: {task_id}")
        predictions[task_id] = verdict
    return predictions


def _safe_divide(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def _f1(precision: float | None, recall: float | None) -> float | None:
    if precision is None or recall is None or precision + recall == 0:
        return None
    return 2 * precision * recall / (precision + recall)


def evaluate_annotations(
    annotations: list[GoldAnnotation],
    predictions: dict[str, str],
    *,
    include_provisional: bool = False,
) -> dict[str, Any]:
    permitted_statuses = {"ADJUDICATED"}
    if include_provisional:
        permitted_statuses.add("PROVISIONAL")

    status_counts = Counter(annotation.status for annotation in annotations)
    missing_predictions = sorted(
        annotation.task_id
        for annotation in annotations
        if annotation.task_id not in predictions
    )
    unknown_predictions = sorted(
        task_id
        for task_id in predictions
        if task_id not in {annotation.task_id for annotation in annotations}
    )
    evaluated = [
        annotation
        for annotation in annotations
        if annotation.status in permitted_statuses and annotation.label is not None
    ]

    matrix = {
        gold: {prediction: 0 for prediction in LABELS}
        for gold in LABELS
    }
    rows: list[dict[str, Any]] = []
    for annotation in evaluated:
        prediction = predictions.get(annotation.task_id)
        if prediction is None:
            continue
        matrix[annotation.label][prediction] += 1
        rows.append(
            {
                "task_id": annotation.task_id,
                "gold": annotation.label,
                "prediction": prediction,
                "status": annotation.status,
                "match": annotation.label == prediction,
                "source": annotation.source,
                "rationale": annotation.rationale,
            }
        )

    binary_rows = [row for row in rows if row["gold"] in {"PASS", "FAIL"}]
    true_positive = sum(
        row["gold"] == "FAIL" and row["prediction"] == "FAIL"
        for row in binary_rows
    )
    false_positive = sum(
        row["gold"] == "PASS" and row["prediction"] == "FAIL"
        for row in binary_rows
    )
    false_negative = sum(
        row["gold"] == "FAIL" and row["prediction"] != "FAIL"
        for row in binary_rows
    )
    true_negative = sum(
        row["gold"] == "PASS" and row["prediction"] != "FAIL"
        for row in binary_rows
    )
    precision = _safe_divide(true_positive, true_positive + false_positive)
    recall = _safe_divide(true_positive, true_positive + false_negative)
    review_count = sum(row["prediction"] == "REVIEW" for row in rows)

    return {
        "schema_version": "policy-grounding-gold-validation-v0.1",
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "mode": "provisional_included" if include_provisional else "adjudicated_only",
        "coverage": {
            "annotation_rows": len(annotations),
            "prediction_rows": len(predictions),
            "status_counts": {
                status: status_counts.get(status, 0) for status in STATUSES
            },
            "eligible_gold_rows": len(evaluated),
            "evaluated_rows": len(rows),
            "missing_prediction_task_ids": missing_predictions,
            "unknown_prediction_task_ids": unknown_predictions,
        },
        "three_class": {
            "labels": list(LABELS),
            "confusion_matrix": matrix,
            "exact_match_accuracy": _safe_divide(
                sum(row["match"] for row in rows), len(rows)
            ),
            "review_prediction_count": review_count,
            "review_prediction_rate": _safe_divide(review_count, len(rows)),
        },
        "fail_detection": {
            "scope": "Gold PASS/FAIL rows; REVIEW predictions count as abstaining misses",
            "tp": true_positive,
            "fp": false_positive,
            "fn": false_negative,
            "tn": true_negative,
            "precision": precision,
            "recall": recall,
            "f1": _f1(precision, recall),
            "false_positive_task_ids": [
                row["task_id"]
                for row in binary_rows
                if row["gold"] == "PASS" and row["prediction"] == "FAIL"
            ],
            "false_negative_task_ids": [
                row["task_id"]
                for row in binary_rows
                if row["gold"] == "FAIL" and row["prediction"] != "FAIL"
            ],
        },
        "task_results": rows,
        "release_gate": {
            "official_metrics_allowed": (
                not include_provisional
                and status_counts.get("ADJUDICATED", 0) > 0
                and status_counts.get("PROVISIONAL", 0) == 0
                and status_counts.get("UNREVIEWED", 0) == 0
            ),
            "reason": (
                "All rows are adjudicated."
                if (
                    not include_provisional
                    and status_counts.get("ADJUDICATED", 0) > 0
                    and status_counts.get("PROVISIONAL", 0) == 0
                    and status_counts.get("UNREVIEWED", 0) == 0
                )
                else "Human adjudication is incomplete; metrics are diagnostic only."
            ),
        },
    }


def render_markdown(result: dict[str, Any]) -> str:
    coverage = result["coverage"]
    metrics = result["fail_detection"]
    three_class = result["three_class"]
    gate = result["release_gate"]

    def metric(value: float | None) -> str:
        return "N/A" if value is None else f"{value:.3f}"

    lines = [
        "# Programmatic Verifier Gold Validation",
        "",
        f"- Mode: `{result['mode']}`",
        f"- Evaluated rows: {coverage['evaluated_rows']}",
        f"- Adjudicated: {coverage['status_counts']['ADJUDICATED']}",
        f"- Provisional: {coverage['status_counts']['PROVISIONAL']}",
        f"- Unreviewed: {coverage['status_counts']['UNREVIEWED']}",
        f"- Official metric release allowed: `{str(gate['official_metrics_allowed']).lower()}`",
        f"- Gate reason: {gate['reason']}",
        "",
        "## FAIL detection",
        "",
        "| TP | FP | FN | TN | Precision | Recall | F1 |",
        "|---:|---:|---:|---:|---:|---:|---:|",
        (
            f"| {metrics['tp']} | {metrics['fp']} | {metrics['fn']} | "
            f"{metrics['tn']} | {metric(metrics['precision'])} | "
            f"{metric(metrics['recall'])} | {metric(metrics['f1'])} |"
        ),
        "",
        f"- False positives: {metrics['false_positive_task_ids']}",
        f"- False negatives: {metrics['false_negative_task_ids']}",
        f"- REVIEW prediction rate: {metric(three_class['review_prediction_rate'])}",
        "",
        "## Three-class confusion matrix",
        "",
        "| Gold \\ Pred | PASS | REVIEW | FAIL |",
        "|---|---:|---:|---:|",
    ]
    matrix = three_class["confusion_matrix"]
    for gold in LABELS:
        lines.append(
            f"| {gold} | {matrix[gold]['PASS']} | "
            f"{matrix[gold]['REVIEW']} | {matrix[gold]['FAIL']} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation boundary",
            "",
            "- `PROVISIONAL` labels are seeded from existing project audits and are not "
            "a substitute for independent human adjudication.",
            "- `UNREVIEWED` rows do not contribute to metrics.",
            "- A `REVIEW` prediction is an abstention. For FAIL detection it counts as a "
            "miss when the gold label is FAIL.",
            "- These metrics validate policy-grounding rules; they do not replace Tau2 "
            "official reward reconstruction.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate Programmatic Verifier predictions against audited gold."
    )
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--include-provisional",
        action="store_true",
        help="Include provisional audit labels and mark metrics diagnostic-only.",
    )
    args = parser.parse_args()

    result = evaluate_annotations(
        load_annotations(args.annotations),
        load_predictions(args.predictions),
        include_provisional=args.include_provisional,
    )
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "metrics.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (args.output / "analysis.md").write_text(
        render_markdown(result),
        encoding="utf-8",
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
