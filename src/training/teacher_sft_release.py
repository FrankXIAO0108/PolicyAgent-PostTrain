"""Candidate-level SFT release gate for teacher trajectories.

Consumes the candidate decisions produced by
``teacher_candidate_decision_builder`` and fails closed on:

- missing or hash-mismatched source/correction/validation files;
- correction validation that is not ready or not bound to the correction hash;
- assistant turns mixing text and tool calls, or carrying multiple tool calls;
- entity-group leakage across TRAIN/VALIDATION splits.

Only ``CORRECTED_POSITIVE`` decisions enter the dataset; ``HOLDOUT`` rows are
recorded but never released. The output remains a dry-run artifact until the
data pool is sufficient and the release manifest is formally approved.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from src.training.sft_release import (
    _corrected_messages,
    _normalize_messages,
    _verify_file,
    entity_groups,
    sha256,
)
from src.training.teacher_candidate_decision_builder import (
    CandidateDecision,
    load_jsonl,
)


def load_decisions(path: Path) -> dict[str, CandidateDecision]:
    decisions: dict[str, CandidateDecision] = {}
    for row in load_jsonl(path):
        decision = CandidateDecision.from_dict(row)
        if decision.candidate_id in decisions:
            raise ValueError(
                f"Duplicate candidate decision ID: {decision.candidate_id}"
            )
        decisions[decision.candidate_id] = decision
    return decisions


def assess_candidate_release(decisions_path: Path) -> dict[str, Any]:
    decisions = load_decisions(decisions_path)
    reasons: list[str] = []
    records: list[dict[str, Any]] = []
    group_splits: dict[str, set[str]] = defaultdict(set)

    for candidate_id, decision in decisions.items():
        if decision.disposition == "HOLDOUT":
            continue
        if decision.disposition != "CORRECTED_POSITIVE":
            reasons.append(
                f"Candidate {candidate_id}: unsupported disposition "
                f"{decision.disposition!r}"
            )
            continue
        _verify_file(
            decision.source_path, decision.source_sha256 or "", candidate_id
        )
        assert decision.correction_path is not None
        assert decision.correction_sha256 is not None
        assert decision.correction_validation_path is not None
        assert decision.correction_validation_sha256 is not None
        _verify_file(
            decision.correction_path,
            decision.correction_sha256,
            candidate_id,
        )
        _verify_file(
            decision.correction_validation_path,
            decision.correction_validation_sha256,
            candidate_id,
        )
        validation = json.loads(
            decision.correction_validation_path.read_text(
                encoding="utf-8-sig"
            )
        )
        if not validation.get("ready"):
            reasons.append(
                f"Candidate {candidate_id}: correction validation is not ready"
            )
            continue
        if (
            str(validation.get("task_id")) != decision.task_id
            or str(validation.get("correction_sha256", "")).upper()
            != decision.correction_sha256
        ):
            reasons.append(
                f"Candidate {candidate_id}: correction validation binding "
                "mismatch"
            )
            continue
        messages, policy = _corrected_messages(
            decision.correction_path, candidate_id
        )
        if not policy:
            assert decision.source_path is not None
            payload = json.loads(
                decision.source_path.read_text(encoding="utf-8-sig")
            )
            policy = str(
                payload.get("info", {})
                .get("environment_info", {})
                .get("policy", "")
            )
        try:
            normalized = _normalize_messages(messages, candidate_id)
        except ValueError as error:
            reasons.append(str(error))
            continue
        record_groups = set(decision.group_ids) | entity_groups(
            decision.source_path
        )
        for group_id in record_groups:
            group_splits[group_id].add(str(decision.split))
        records.append(
            {
                "candidate_id": candidate_id,
                "task_id": decision.task_id,
                "split": decision.split,
                "disposition": decision.disposition,
                "corrected": True,
                "source_path": str(decision.correction_path),
                "source_sha256": decision.correction_sha256,
                "group_ids": sorted(record_groups),
                "system_policy": policy,
                "messages": normalized,
            }
        )

    leakage = sorted(
        group_id
        for group_id, splits_seen in group_splits.items()
        if len(splits_seen) > 1
    )
    if leakage:
        reasons.append("Group leakage across splits: " + ", ".join(leakage))
        records = []

    return {
        "ready": not reasons,
        "reasons": reasons,
        "records": records if not reasons else [],
        "counts": {
            "decisions": len(decisions),
            "holdout": sum(
                decision.disposition == "HOLDOUT"
                for decision in decisions.values()
            ),
            "released": len(records),
            "train": sum(row["split"] == "TRAIN" for row in records),
            "validation": sum(row["split"] == "VALIDATION" for row in records),
        },
        "inputs": {
            "decisions": {
                "path": str(decisions_path),
                "sha256": sha256(decisions_path),
            }
        },
    }


def write_release(result: dict[str, Any], output_dir: Path) -> None:
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty output: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    report = {key: value for key, value in result.items() if key != "records"}
    (output_dir / "release_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    if result["ready"]:
        dataset = output_dir / "sft_dataset.jsonl"
        with dataset.open("w", encoding="utf-8") as handle:
            for row in result["records"]:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        (output_dir / "dataset_manifest.json").write_text(
            json.dumps(
                {
                    "schema_version": "tau2-teacher-sft-v0.1",
                    "dataset_sha256": sha256(dataset),
                    "counts": result["counts"],
                    "candidate_ids": [
                        row["candidate_id"] for row in result["records"]
                    ],
                },
                ensure_ascii=False,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Candidate-level SFT release gate for teacher trajectories."
    )
    parser.add_argument("--decisions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = assess_candidate_release(args.decisions)
    write_release(result, args.output)
    print(json.dumps(result["counts"], ensure_ascii=False))
    if not result["ready"]:
        for reason in result["reasons"]:
            print(reason)
        raise SystemExit(2)


if __name__ == "__main__":
    main()
