from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

from src.guards.retail_pre_action import WRITE_TOOLS

from .gold_validation import GoldAnnotation, LABELS, load_annotations
from .trajectory_loader import load_task_artifacts


SCHEMA_VERSION = "policy-grounding-review-queue-v1"
PRIORITY_BY_VERDICT = {"FAIL": 0, "REVIEW": 1, "PASS": 2}


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def load_prediction_rows(path: str | Path) -> dict[str, dict[str, Any]]:
    payload = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    candidates = payload.get("results", []) if isinstance(payload, dict) else payload
    if not isinstance(candidates, list):
        raise ValueError("Prediction artifact must be a JSON list or results object")

    rows: dict[str, dict[str, Any]] = {}
    for raw_row in candidates:
        if not isinstance(raw_row, dict):
            raise ValueError("Every prediction row must be a JSON object")
        task_id = str(raw_row["task_id"])
        verdict = str(raw_row["verdict"]).upper()
        if verdict not in LABELS:
            raise ValueError(f"Task {task_id}: unsupported prediction {verdict!r}")
        if task_id in rows:
            raise ValueError(f"Duplicate prediction task ID: {task_id}")
        rows[task_id] = {**raw_row, "task_id": task_id, "verdict": verdict}
    return rows


def build_review_candidates(
    annotations: list[GoldAnnotation],
    predictions_by_verifier: dict[str, dict[str, dict[str, Any]]],
) -> list[dict[str, Any]]:
    unreviewed = [
        annotation for annotation in annotations if annotation.status == "UNREVIEWED"
    ]
    candidates: list[dict[str, Any]] = []
    for annotation in unreviewed:
        predictions: dict[str, dict[str, Any]] = {}
        for verifier_name, rows in predictions_by_verifier.items():
            prediction = rows.get(annotation.task_id)
            if prediction is None:
                raise ValueError(
                    f"Task {annotation.task_id}: missing {verifier_name} prediction"
                )
            predictions[verifier_name] = prediction

        verdicts = {row["verdict"] for row in predictions.values()}
        disagreement = len(verdicts) > 1
        highest_risk = min(PRIORITY_BY_VERDICT[verdict] for verdict in verdicts)
        candidates.append(
            {
                "task_id": annotation.task_id,
                "annotation_status": annotation.status,
                "annotation_label": annotation.label,
                "queue_reason": annotation.rationale,
                "verifier_disagreement": disagreement,
                "priority": (
                    "P0_VERIFIER_DISAGREEMENT"
                    if disagreement
                    else {
                        0: "P1_PREDICTED_FAIL",
                        1: "P2_PREDICTED_REVIEW",
                        2: "P3_PREDICTED_PASS",
                    }[highest_risk]
                ),
                "predictions": predictions,
            }
        )

    return sorted(
        candidates,
        key=lambda row: (
            0 if row["verifier_disagreement"] else 1,
            min(
                PRIORITY_BY_VERDICT[prediction["verdict"]]
                for prediction in row["predictions"].values()
            ),
            int(row["task_id"]) if row["task_id"].isdigit() else row["task_id"],
        ),
    )


def _relative(path: Path, root: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return str(path.resolve())


def _finding_codes(prediction: dict[str, Any]) -> list[str]:
    return [
        str(finding.get("code", ""))
        for finding in prediction.get("findings", [])
        if isinstance(finding, dict)
    ]


def render_markdown(queue: dict[str, Any]) -> str:
    lines = [
        "# Policy Grounding Targeted Review Queue V1",
        "",
        "This offline queue contains only previously `UNREVIEWED` trajectories.",
        "It does not assign labels or promote any row to `ADJUDICATED`.",
        "",
        f"- Review candidates: {queue['candidate_count']}",
        f"- Predicted FAIL priority cases: {queue['priority_counts'].get('P1_PREDICTED_FAIL', 0)}",
        f"- Verifier disagreement cases: {queue['priority_counts'].get('P0_VERIFIER_DISAGREEMENT', 0)}",
        "",
        "## Queue",
        "",
        "| Priority | Task | V1.2 | V2.0 | Tool calls | Packet |",
        "|---|---:|---|---|---:|---|",
    ]
    for row in queue["tasks"]:
        predictions = row["predictions"]
        lines.append(
            f"| {row['priority']} | {row['task_id']} | "
            f"{predictions['v1.2']['verdict']} | {predictions['v2.0']['verdict']} | "
            f"{row['evidence_summary']['tool_call_count']} | "
            f"[JSON]({row['packet_path']}) |"
        )

    lines.extend(
        [
            "",
            "## Required reviewer decision",
            "",
            "For each packet, inspect the frozen task definition, complete parsed event "
            "sequence, raw source path and hashes. Record:",
            "",
            "1. `PASS`, `REVIEW`, or `FAIL` for policy grounding.",
            "2. A rationale tied to exact event indices and policy clauses.",
            "3. Whether the trajectory is eligible for SFT, requires correction, or must "
            "be quarantined.",
            "4. Reviewer identity and review date outside this generated artifact.",
            "",
            "A generated packet is evidence routing, not independent human gold.",
            "",
        ]
    )
    return "\n".join(lines)


def create_review_queue(
    *,
    annotations_path: Path,
    v1_predictions_path: Path,
    v2_predictions_path: Path,
    experiment_dir: Path,
    output_dir: Path,
    project_root: Path,
    project_commit: str,
    upstream_commit: str,
) -> dict[str, Any]:
    annotations = load_annotations(annotations_path)
    prediction_paths = {
        "v1.2": v1_predictions_path,
        "v2.0": v2_predictions_path,
    }
    prediction_rows = {
        name: load_prediction_rows(path) for name, path in prediction_paths.items()
    }
    candidates = build_review_candidates(annotations, prediction_rows)

    output_dir.mkdir(parents=True, exist_ok=True)
    packets_dir = output_dir / "packets"
    packets_dir.mkdir(parents=True, exist_ok=True)

    queue_rows: list[dict[str, Any]] = []
    for candidate in candidates:
        task_id = candidate["task_id"]
        task_dir = experiment_dir / f"task_{task_id}"
        bundle = load_task_artifacts(task_dir)
        source_path = bundle.source_path
        if source_path is None:
            raise ValueError(f"Task {task_id}: trajectory source path is unavailable")

        tool_call_count = sum(len(event.tool_calls) for event in bundle.events)
        write_call_count = sum(
            call.name in WRITE_TOOLS
            for event in bundle.events
            for call in event.tool_calls
        )
        packet = {
            "schema_version": SCHEMA_VERSION,
            "task_id": task_id,
            "review_state": {
                "status": "PENDING",
                "label": None,
                "rationale": "",
                "event_indices": [],
                "policy_clauses": [],
                "training_disposition": None,
                "reviewer": None,
                "reviewed_at": None,
            },
            "queue_metadata": {
                key: value
                for key, value in candidate.items()
                if key != "predictions"
            },
            "predictions": candidate["predictions"],
            "frozen_evidence": {
                "task_dir": _relative(task_dir, project_root),
                "task_definition": bundle.task,
                "summary": bundle.summary,
                "trajectory_source": _relative(source_path, project_root),
                "trajectory_source_sha256": sha256_file(source_path),
                "task_json_sha256": sha256_file(task_dir / "task.json"),
                "events": [event.to_dict() for event in bundle.events],
            },
        }
        packet_path = packets_dir / f"task_{task_id}_review_packet.json"
        packet_path.write_text(
            json.dumps(packet, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        queue_rows.append(
            {
                **candidate,
                "packet_path": _relative(packet_path, output_dir),
                "packet_sha256": sha256_file(packet_path),
                "evidence_summary": {
                    "event_count": len(bundle.events),
                    "tool_call_count": tool_call_count,
                    "write_call_count": write_call_count,
                    "v1_finding_codes": _finding_codes(candidate["predictions"]["v1.2"]),
                    "v2_finding_codes": _finding_codes(candidate["predictions"]["v2.0"]),
                },
            }
        )

    priority_counts: dict[str, int] = {}
    for row in queue_rows:
        priority = row["priority"]
        priority_counts[priority] = priority_counts.get(priority, 0) + 1

    queue = {
        "schema_version": SCHEMA_VERSION,
        "role": "Route frozen unreviewed trajectories to targeted human review",
        "project_commit_before_change": project_commit,
        "upstream_runtime_commit": upstream_commit,
        "implementation": {
            "path": _relative(Path(__file__), project_root),
            "sha256": sha256_file(Path(__file__)),
        },
        "candidate_count": len(queue_rows),
        "priority_counts": priority_counts,
        "input_artifacts": {
            "annotations": {
                "path": _relative(annotations_path, project_root),
                "sha256": sha256_file(annotations_path),
            },
            **{
                name: {
                    "path": _relative(path, project_root),
                    "sha256": sha256_file(path),
                }
                for name, path in prediction_paths.items()
            },
        },
        "interpretation_boundary": [
            "Queue membership is selected only from annotation status UNREVIEWED.",
            "Verifier predictions prioritize review but do not become gold labels.",
            "Packets preserve generated PENDING review fields and cannot open the metric release gate.",
        ],
        "tasks": queue_rows,
    }
    queue_path = output_dir / "review_queue.json"
    queue_path.write_text(
        json.dumps(queue, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    markdown_path = output_dir / "review_queue.md"
    markdown_path.write_text(render_markdown(queue), encoding="utf-8")

    manifest = {
        "experiment_id": output_dir.name,
        "role": queue["role"],
        "project_commit_before_change": project_commit,
        "upstream_runtime_commit": upstream_commit,
        "new_model_calls": 0,
        "command": [sys.executable, *sys.argv],
        "implementation": queue["implementation"],
        "inputs": queue["input_artifacts"],
        "outputs": {
            "review_queue.json": sha256_file(queue_path),
            "review_queue.md": sha256_file(markdown_path),
            "packets": {
                row["task_id"]: row["packet_sha256"] for row in queue_rows
            },
        },
        "candidate_count": len(queue_rows),
        "interpretation_boundary": queue["interpretation_boundary"],
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return queue


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build frozen evidence packets for unreviewed trajectories."
    )
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--v1-predictions", type=Path, required=True)
    parser.add_argument("--v2-predictions", type=Path, required=True)
    parser.add_argument("--experiment", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--project-commit", required=True)
    parser.add_argument("--upstream-commit", required=True)
    args = parser.parse_args()

    queue = create_review_queue(
        annotations_path=args.annotations.resolve(),
        v1_predictions_path=args.v1_predictions.resolve(),
        v2_predictions_path=args.v2_predictions.resolve(),
        experiment_dir=args.experiment.resolve(),
        output_dir=args.output.resolve(),
        project_root=args.project_root.resolve(),
        project_commit=args.project_commit,
        upstream_commit=args.upstream_commit,
    )
    print(
        json.dumps(
            {
                "candidate_count": queue["candidate_count"],
                "priority_counts": queue["priority_counts"],
                "output": str(args.output.resolve()),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
