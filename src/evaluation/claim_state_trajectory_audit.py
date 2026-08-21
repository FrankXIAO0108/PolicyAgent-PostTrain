from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

from src.training.teacher_evidence_pack import claim_state_consistency_v2


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def _messages(path: Path) -> list[dict[str, Any]]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    simulations = raw.get("simulations") or []
    if len(simulations) != 1:
        raise ValueError(f"expected one simulation in {path}")
    return list(simulations[0].get("messages") or [])


def _rank(verdict: str) -> int | None:
    return {"FAIL": 0, "REVIEW": 1, "PASS": 2}.get(verdict)


def audit(process_audit_path: Path) -> dict[str, Any]:
    source = json.loads(process_audit_path.read_text(encoding="utf-8"))
    rows: list[dict[str, Any]] = []
    flip_pairs: list[dict[str, Any]] = []
    for pair in source["pairs"]:
        pair_rows: dict[str, dict[str, Any]] = {}
        for run_name in ("run_a", "run_b"):
            source_row = pair[run_name]
            artifact = Path(source_row["artifact"]["path"])
            if not artifact.is_file():
                raise FileNotFoundError(artifact)
            result = claim_state_consistency_v2(
                _messages(artifact), {"agent": {"orders": {}}}
            )
            row = {
                "task_id": pair["task_id"],
                "run_name": run_name,
                "benchmark_success": bool(source_row["benchmark"]["success"]),
                "source_artifact": {
                    "path": str(artifact),
                    "sha256": source_row["artifact"]["sha256"],
                },
                "claim_state_v2": result,
            }
            rows.append(row)
            pair_rows[run_name] = row
        if pair["cohort"] == "flip":
            success = next(
                row for row in pair_rows.values() if row["benchmark_success"]
            )
            failure = next(
                row for row in pair_rows.values() if not row["benchmark_success"]
            )
            success_rank = _rank(success["claim_state_v2"]["verdict"])
            failure_rank = _rank(failure["claim_state_v2"]["verdict"])
            flip_pairs.append(
                {
                    "task_id": pair["task_id"],
                    "successful_verdict": success["claim_state_v2"]["verdict"],
                    "failed_verdict": failure["claim_state_v2"]["verdict"],
                    "evaluable": success_rank is not None and failure_rank is not None,
                    "successful_run_preferred": (
                        success_rank > failure_rank
                        if success_rank is not None and failure_rank is not None
                        else None
                    ),
                }
            )

    successes = [row for row in rows if row["benchmark_success"]]
    success_failures = [
        row for row in successes if row["claim_state_v2"]["verdict"] == "FAIL"
    ]
    evaluable_flips = [pair for pair in flip_pairs if pair["evaluable"]]
    preferred_flips = [
        pair for pair in evaluable_flips if pair["successful_run_preferred"]
    ]
    return {
        "schema_version": "retail-claim-state-real-trajectory-audit-v2-development",
        "checker_version": "claim-state-v2-development",
        "source_process_audit": {
            "path": str(process_audit_path),
            "sha256": _sha256(process_audit_path),
        },
        "summary": {
            "trajectory_count": len(rows),
            "benchmark_success_count": len(successes),
            "verdict_counts": dict(
                sorted(Counter(row["claim_state_v2"]["verdict"] for row in rows).items())
            ),
            "claim_state_failure_on_success_count": len(success_failures),
            "claim_state_failure_on_success_tasks": [
                {"task_id": row["task_id"], "run_name": row["run_name"]}
                for row in success_failures
            ],
            "flip_count": len(flip_pairs),
            "evaluable_flip_count": len(evaluable_flips),
            "successful_run_preferred_flip_count": len(preferred_flips),
            "successful_run_preferred_flip_task_ids": [
                pair["task_id"] for pair in preferred_flips
            ],
        },
        "gates": {
            "zero_failures_on_frozen_successes": not success_failures,
            "ready_for_reward_penalty": False,
        },
        "flip_pairs": flip_pairs,
        "rows": rows,
        "validity_notes": [
            "This reuses discovery trajectories and cannot estimate generalization.",
            "Benchmark success is only a positive-sanity label, not clean-process gold.",
            "The V2 checker remains outside the scalar reward until a new untouched holdout is evaluated.",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run claim-state V2 on artifacts indexed by a frozen process audit."
    )
    parser.add_argument("process_audit", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = audit(args.process_audit)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report["summary"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
