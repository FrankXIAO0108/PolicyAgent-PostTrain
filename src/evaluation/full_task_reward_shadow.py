from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


SCHEMA_VERSION = "retail-full-task-reward-shadow-v1"


SIGNAL_AUTHORITY = {
    "terminal_environment_state": {
        "authority": "TRAINING_ONLINE_CANDIDATE",
        "confidence": "HIGH",
        "reason": "Environment state is mechanically observed, but the target state is benchmark privileged information.",
    },
    "required_action_progress": {
        "authority": "BENCHMARK_TRAINING_ONLY",
        "confidence": "HIGH",
        "reason": "The matcher uses hidden reference actions and is unavailable in an ordinary production request.",
    },
    "communication_assertions": {
        "authority": "BENCHMARK_TRAINING_ONLY",
        "confidence": "MEDIUM",
        "reason": "Assertions are task-specific benchmark labels and do not cover all final-answer claims.",
    },
    "tool_error_repeat_unexpected_stop": {
        "authority": "TRAINING_ONLINE_CANDIDATE",
        "confidence": "MEDIUM",
        "reason": "These events are observable online, but error category and recovery context must remain separated.",
    },
    "confirmation_parameter_binding": {
        "authority": "REVIEW_ONLY",
        "confidence": "LOW",
        "reason": "Frozen successes still contain REVIEW and NOT_EVALUABLE cases; owner adjudication shows context-sensitive exceptions.",
    },
    "claim_state_consistency": {
        "authority": "DIAGNOSTIC_ONLY",
        "confidence": "MEDIUM",
        "reason": "The checker has narrow coverage and often returns REVIEW or NOT_APPLICABLE.",
    },
    "final_answer_grounding": {
        "authority": "DIAGNOSTIC_ONLY",
        "confidence": "LOW",
        "reason": "The current scalar proxy ties Task 67 despite different terminal answers, and the available claim checker has partial semantic coverage.",
    },
    "policy_compliance": {
        "authority": "REVIEW_ONLY",
        "confidence": "LOW",
        "reason": "Current policy labels are developmental and owner-reviewed, not independent expert gold.",
    },
    "owner_adjudication": {
        "authority": "OFFLINE_ADJUDICATION_ONLY",
        "confidence": "MEDIUM",
        "reason": "Project-owner decisions are useful development evidence but are not an online reward oracle or independent expert gold.",
    },
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object in {path}")
    return value


def _owner_decisions_by_trajectory(
    owner_decisions: dict[str, Any] | None,
) -> dict[tuple[str, str], list[dict[str, Any]]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    if owner_decisions is None:
        return grouped
    for decision in owner_decisions.get("decisions") or []:
        review_id = str(decision.get("review_id") or "")
        parts = review_id.split(":", 2)
        if len(parts) < 2:
            continue
        grouped[(parts[0], parts[1])].append(dict(decision))
    return grouped


def _owner_summary(decisions: list[dict[str, Any]]) -> dict[str, Any] | None:
    if not decisions:
        return None
    return {
        "decision_count": len(decisions),
        "final_labels": sorted(
            {str(row.get("final_label") or "UNSPECIFIED") for row in decisions}
        ),
        "severities": sorted(
            {str(row.get("severity") or "UNASSESSED") for row in decisions}
        ),
        "training_dispositions": sorted(
            {
                str(row.get("training_disposition") or "UNASSESSED")
                for row in decisions
            }
        ),
        "independent_expert_gold": False,
    }


def _trajectory_conflicts(
    trajectory: dict[str, Any], owner_decisions: list[dict[str, Any]]
) -> list[str]:
    conflicts: list[str] = []
    benchmark = trajectory.get("benchmark") or {}
    success = bool(benchmark.get("success", False))
    proxy = trajectory.get("v1_reward_proxy") or {}
    score = float(proxy.get("score") or 0.0)
    errors = trajectory.get("error_recovery") or {}
    confirmation = trajectory.get("confirmation_diagnostics") or {}
    binding_verdicts = {
        str(check.get("parameter_binding", {}).get("verdict") or "")
        for check in confirmation.get("checks") or []
    }

    if success and int(errors.get("tool_error_count") or 0) > 0:
        conflicts.append("BENCHMARK_SUCCESS_WITH_TOOL_ERROR")
    if success and int(confirmation.get("missing_confirmation_count") or 0) > 0:
        conflicts.append("BENCHMARK_SUCCESS_WITH_CONFIRMATION_GAP")
    if success and ({"REVIEW", "NOT_EVALUABLE"} & binding_verdicts):
        conflicts.append("BENCHMARK_SUCCESS_WITH_UNRESOLVED_PARAMETER_BINDING")
    if not success and score >= 0.9:
        conflicts.append("HIGH_V1_PROCESS_PROXY_ON_BENCHMARK_FAILURE")

    dispositions = {
        str(row.get("training_disposition") or "") for row in owner_decisions
    }
    if success and any(
        disposition
        not in {"", "UNASSESSED", "RAW_GOLD_CANDIDATE_SPLIT_CHANGE_REQUIRED"}
        for disposition in dispositions
    ):
        conflicts.append("BENCHMARK_SUCCESS_NOT_OWNER_APPROVED_AS_RAW_GOLD")
    if not success and any(
        str(row.get("benchmark_alignment") or "").endswith("GOLD_CONFLICT")
        or str(row.get("outcome_status") or "")
        in {"USER_FINAL_INTENT_SATISFIED", "BENCHMARK_SUCCESS_WITH_PROCESS_FAILURE"}
        for row in owner_decisions
    ):
        conflicts.append("BENCHMARK_FAILURE_REQUIRES_OWNER_CONTEXT")
    return sorted(set(conflicts))


def build_shadow_report(
    process_audit: dict[str, Any],
    *,
    owner_decisions: dict[str, Any] | None = None,
    source_paths: dict[str, dict[str, str]] | None = None,
) -> dict[str, Any]:
    pairs = process_audit.get("pairs")
    if not isinstance(pairs, list) or not pairs:
        raise ValueError("process audit must contain non-empty pairs")
    owner_by_trajectory = _owner_decisions_by_trajectory(owner_decisions)

    trajectories: list[dict[str, Any]] = []
    conflict_counts: Counter[str] = Counter()
    claim_verdict_counts: Counter[str] = Counter()
    binding_verdict_counts: Counter[str] = Counter()
    communication_component_count = 0
    trajectories_with_write_count = 0
    for pair in pairs:
        task_id = str(pair.get("task_id") or "")
        for run_name in ("run_a", "run_b"):
            trajectory = pair.get(run_name)
            if not isinstance(trajectory, dict):
                raise ValueError(f"Pair {task_id} is missing {run_name}")
            decisions = owner_by_trajectory.get((task_id, run_name), [])
            conflicts = _trajectory_conflicts(trajectory, decisions)
            conflict_counts.update(conflicts)
            components = (trajectory.get("v1_reward_proxy") or {}).get(
                "components"
            ) or {}
            if "communication" in components:
                communication_component_count += 1
            confirmation = trajectory.get("confirmation_diagnostics") or {}
            if int(confirmation.get("write_count") or 0) > 0:
                trajectories_with_write_count += 1
            for check in confirmation.get("checks") or []:
                binding_verdict_counts[
                    str(
                        check.get("parameter_binding", {}).get("verdict")
                        or "UNSPECIFIED"
                    )
                ] += 1
            claim_verdict_counts[
                str(
                    (trajectory.get("claim_state_consistency") or {}).get(
                        "verdict"
                    )
                    or "UNSPECIFIED"
                )
            ] += 1
            trajectories.append(
                {
                    "task_id": task_id,
                    "run_name": run_name,
                    "cohort": pair.get("cohort"),
                    "artifact": trajectory.get("artifact"),
                    "benchmark_success": bool(
                        (trajectory.get("benchmark") or {}).get("success", False)
                    ),
                    "benchmark_reward": (trajectory.get("benchmark") or {}).get(
                        "reward"
                    ),
                    "v1_process_proxy_score": (
                        trajectory.get("v1_reward_proxy") or {}
                    ).get("score"),
                    "tool_error_count": (
                        trajectory.get("error_recovery") or {}
                    ).get("tool_error_count"),
                    "missing_confirmation_count": (
                        trajectory.get("confirmation_diagnostics") or {}
                    ).get("missing_confirmation_count"),
                    "claim_state_verdict": (
                        trajectory.get("claim_state_consistency") or {}
                    ).get("verdict"),
                    "owner_review": _owner_summary(decisions),
                    "conflicts": conflicts,
                }
            )

    flip_ties = [
        str(pair.get("task_id"))
        for pair in pairs
        if pair.get("cohort") == "flip"
        and math.isclose(
            float(
                (pair.get("run_a", {}).get("v1_reward_proxy") or {}).get(
                    "score"
                )
                or 0.0
            ),
            float(
                (pair.get("run_b", {}).get("v1_reward_proxy") or {}).get(
                    "score"
                )
                or 0.0
            ),
            rel_tol=0.0,
            abs_tol=1e-9,
        )
    ]
    reviewed_count = sum(row["owner_review"] is not None for row in trajectories)
    return {
        "schema_version": SCHEMA_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "sources": source_paths or {},
        "scope": {
            "mode": "OFFLINE_SHADOW_AUDIT",
            "changes_online_reward": False,
            "introduces_scalar_v2_reward": False,
            "uses_new_rollouts": False,
            "uses_llm_judge": False,
        },
        "signal_authority": SIGNAL_AUTHORITY,
        "summary": {
            "pair_count": len(pairs),
            "trajectory_count": len(trajectories),
            "owner_reviewed_trajectory_count": reviewed_count,
            "trajectory_with_conflict_count": sum(
                bool(row["conflicts"]) for row in trajectories
            ),
            "conflict_counts": dict(sorted(conflict_counts.items())),
            "flip_score_tie_task_ids": flip_ties,
            "existing_v1_ready_for_grpo": bool(
                (process_audit.get("gates") or {}).get(
                    "ready_to_use_v1_reward_for_grpo", False
                )
            ),
        },
        "component_coverage": {
            "terminal_environment_state": {
                "trajectory_count": len(trajectories),
                "note": "Benchmark reward and DB outcome are present in every audited trajectory.",
            },
            "required_action_progress": {
                "trajectory_count": len(trajectories),
                "note": "Coverage depends on hidden Tau2 reference actions.",
            },
            "communication_assertions": {
                "trajectory_with_component_count": communication_component_count,
                "trajectory_count": len(trajectories),
            },
            "confirmation_parameter_binding": {
                "trajectory_with_write_count": trajectories_with_write_count,
                "write_verdict_counts": dict(sorted(binding_verdict_counts.items())),
            },
            "claim_state_consistency": {
                "trajectory_verdict_counts": dict(sorted(claim_verdict_counts.items())),
            },
            "owner_adjudication": {
                "reviewed_trajectory_count": reviewed_count,
                "trajectory_count": len(trajectories),
            },
        },
        "trajectories": trajectories,
        "decision": {
            "status": "DO_NOT_PROMOTE_TO_ONLINE_REWARD",
            "reasons": [
                "The frozen v1 gate is closed.",
                "Benchmark success and training-data quality disagree on owner-reviewed trajectories.",
                "Confirmation, policy, and claim-state checks are not mature enough to be scalar online reward terms.",
                "Task 67 remains a flip tie, showing that final-answer differences are not resolved by the current scalar proxy.",
            ],
            "next_evidence_required": [
                "Freeze component definitions before choosing scalar weights.",
                "Measure each candidate component on untouched trajectories after freezing.",
                "Run a no-update rollout diagnostic to measure within-prompt reward variance and zero-std groups.",
                "Only then run a short KL-monitored GRPO experiment from the same SFT checkpoint.",
            ],
        },
        "validity_notes": [
            "This report reuses frozen artifacts and does not claim a new evaluation result.",
            "Owner decisions are developmental project-owner review, not independent expert gold.",
            "No scalar v2 reward or reward weight is proposed by this audit.",
            "Hidden benchmark actions may support benchmark training diagnostics but are not production-time signals.",
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build an offline full-task reward authority and conflict report."
    )
    parser.add_argument("--process-audit", type=Path, required=True)
    parser.add_argument("--owner-decisions", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    process_audit = _load_json(args.process_audit)
    owner_decisions = (
        _load_json(args.owner_decisions) if args.owner_decisions is not None else None
    )
    sources = {
        "process_audit": {
            "path": str(args.process_audit.resolve()),
            "sha256": _sha256(args.process_audit),
        }
    }
    if args.owner_decisions is not None:
        sources["owner_decisions"] = {
            "path": str(args.owner_decisions.resolve()),
            "sha256": _sha256(args.owner_decisions),
        }
    report = build_shadow_report(
        process_audit,
        owner_decisions=owner_decisions,
        source_paths=sources,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report["summary"], ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
