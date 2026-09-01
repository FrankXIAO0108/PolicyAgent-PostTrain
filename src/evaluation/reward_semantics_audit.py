from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from src.evaluation.historical_staged_reward_shadow import simulation_to_inputs
from src.evaluation.staged_reward_shadow import (
    _canonical_arguments,
    _read_json,
    _sha256,
    score_rollout,
)


SCHEMA_VERSION = "retail-reward-semantics-audit-v1"


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _lookup(root: Any, path: list[Any]) -> tuple[bool, Any]:
    current = root
    for part in path:
        if isinstance(current, dict) and isinstance(part, str) and part in current:
            current = current[part]
        elif isinstance(current, list) and isinstance(part, int) and 0 <= part < len(current):
            current = current[part]
        else:
            return False, None
    return True, current


def _deep_subset(observed: Any, expected: Any) -> bool:
    if isinstance(expected, dict):
        return isinstance(observed, dict) and all(
            key in observed and _deep_subset(observed[key], value)
            for key, value in expected.items()
        )
    if isinstance(expected, list):
        if not isinstance(observed, list):
            return False
        unmatched = list(observed)
        for expected_item in expected:
            index = next(
                (
                    i
                    for i, observed_item in enumerate(unmatched)
                    if _deep_subset(observed_item, expected_item)
                ),
                None,
            )
            if index is None:
                return False
            unmatched.pop(index)
        return True
    return observed == expected


def _decode_result(call: dict[str, Any]) -> tuple[bool, Any, str | None]:
    result = dict(call.get("result") or {})
    if bool(result.get("error")):
        return False, None, "tool_result_error"
    content = result.get("content")
    if isinstance(content, (dict, list)):
        return True, content, None
    if not isinstance(content, str):
        return False, None, "tool_result_content_missing"
    try:
        return True, json.loads(content), None
    except json.JSONDecodeError:
        return False, None, "tool_result_not_json"


def _exact_calls(
    trace: list[dict[str, Any]], expected: dict[str, Any]
) -> list[dict[str, Any]]:
    expected_args = _canonical_arguments(dict(expected["arguments"]))
    return [
        call
        for call in trace
        if str(call.get("name")) == str(expected["name"])
        and _canonical_arguments(dict(call.get("arguments") or {})) == expected_args
        and not bool((call.get("result") or {}).get("error"))
    ]


def _reference_call_progress(
    trace: list[dict[str, Any]], task_spec: dict[str, Any]
) -> dict[str, Any]:
    checks = [
        {
            "evidence_id": row["evidence_id"],
            "matched": bool(_exact_calls(trace, row)),
        }
        for row in task_spec["reference_calls"]
    ]
    return {
        "value": sum(int(row["matched"]) for row in checks) / len(checks),
        "checks": checks,
    }


def _result_evidence_progress(
    trace: list[dict[str, Any]], task_spec: dict[str, Any]
) -> dict[str, Any]:
    checks = []
    for expected in task_spec["result_evidence"]:
        candidates = _exact_calls(trace, expected)
        candidate_evidence = []
        matched = False
        for call in candidates:
            decoded, content, error = _decode_result(call)
            assertions = []
            if decoded:
                for assertion in expected["assertions"]:
                    found, observed = _lookup(content, list(assertion["path"]))
                    assertion_matched = found and _deep_subset(
                        observed, assertion["expected"]
                    )
                    assertions.append(
                        {
                            "path": assertion["path"],
                            "found": found,
                            "expected": assertion["expected"],
                            "observed": observed,
                            "matched": assertion_matched,
                        }
                    )
                matched = matched or all(row["matched"] for row in assertions)
            candidate_evidence.append(
                {
                    "call_id": call.get("call_id"),
                    "decoded": decoded,
                    "error": error,
                    "assertions": assertions,
                }
            )
        checks.append(
            {
                "evidence_id": expected["evidence_id"],
                "matched": matched,
                "candidates": candidate_evidence,
            }
        )
    return {
        "value": sum(int(row["matched"]) for row in checks) / len(checks),
        "checks": checks,
    }


def _candidate_tier(
    *,
    complete_success: bool,
    write_value: float,
    result_evidence_value: float,
    unexpected_write_count: int,
) -> tuple[int, str]:
    if unexpected_write_count > 0:
        return -1, "unexpected_write_hard_cap"
    if complete_success:
        return 4, "complete_terminal_success"
    if write_value == 1.0:
        return 3, "full_verified_write"
    if write_value > 0.0:
        return 2, "partial_verified_write"
    if result_evidence_value > 0.0:
        return 1, "result_evidence"
    return 0, "reference_or_identity_only"


def _candidate_dimensions(
    *,
    raw: dict[str, Any],
    evidence: dict[str, Any],
    v2_spec: dict[str, Any],
    semantic_spec: dict[str, Any],
) -> dict[str, Any]:
    v2 = score_rollout(raw, evidence, v2_spec)
    trace = list(evidence.get("tool_trace") or [])
    task_spec = dict(semantic_spec["tasks"][str(raw["task_id"])])
    reference = _reference_call_progress(trace, task_spec)
    result_evidence = _result_evidence_progress(trace, task_spec)
    write_value = float(v2["components"]["required_write_progress"]["value"])
    unexpected_write_count = int(v2["unexpected_write_count"])
    tier, tier_name = _candidate_tier(
        complete_success=bool(v2["complete_success"]),
        write_value=write_value,
        result_evidence_value=float(result_evidence["value"]),
        unexpected_write_count=unexpected_write_count,
    )
    identity = float(v2["components"]["identity_link"]["value"])
    return {
        "v2_staged_reward": v2["staged_reward"],
        "terminal_success": v2["terminal_success"],
        "complete_success": v2["complete_success"],
        "write_progress": write_value,
        "unexpected_write_count": unexpected_write_count,
        "reference_call_progress": reference,
        "result_evidence_progress": result_evidence,
        "identity_link": identity,
        "candidate_rank": [tier, result_evidence["value"], reference["value"], identity],
        "candidate_tier": tier_name,
        "future_intent_dependency": task_spec.get("future_intent_dependency"),
    }


def _verify_source(root: Path, source: dict[str, Any]) -> Path:
    path = (root / str(source["path"])).resolve()
    actual = _sha256(path)
    expected = str(source["sha256"]).upper()
    if actual != expected:
        raise ValueError(f"Source hash mismatch: {path}: {actual} != {expected}")
    return path


def build_report(spec_path: Path, *, repo_root: Path) -> dict[str, Any]:
    semantic_spec = _read_json(spec_path)
    sources = semantic_spec["sources"]
    v2_path = _verify_source(repo_root, sources["v2_spec"])
    raw_path = _verify_source(repo_root, sources["s5i_raw_rollouts"])
    evidence_path = _verify_source(repo_root, sources["s5i_rollout_evidence"])
    historical_paths = [
        _verify_source(repo_root, source)
        for source in sources["historical_task_results"]
    ]
    v2_spec = _read_json(v2_path)

    evidence_by_sha = {
        str(row["evidence_sha256"]): row for row in _read_jsonl(evidence_path)
    }
    rows = []
    for index, raw in enumerate(_read_jsonl(raw_path), 1):
        evidence = evidence_by_sha[str(raw["evidence_sha256"])]
        rows.append(
            {
                "source": "s5i",
                "source_row": index,
                "task_id": str(raw["task_id"]),
                **_candidate_dimensions(
                    raw=raw,
                    evidence=evidence,
                    v2_spec=v2_spec,
                    semantic_spec=semantic_spec,
                ),
            }
        )
    for path in historical_paths:
        for simulation in _read_json(path).get("simulations") or []:
            raw, evidence = simulation_to_inputs(simulation, v2_spec)
            rows.append(
                {
                    "source": "historical",
                    "simulation_id": str(simulation["id"]),
                    "task_id": str(simulation["task_id"]),
                    "benchmark_reward": float(
                        (simulation.get("reward_info") or {}).get("reward") or 0.0
                    ),
                    **_candidate_dimensions(
                        raw=raw,
                        evidence=evidence,
                        v2_spec=v2_spec,
                        semantic_spec=semantic_spec,
                    ),
                }
            )

    def rank(row: dict[str, Any]) -> tuple[float, ...]:
        return tuple(float(value) for value in row["candidate_rank"])

    complete_rows = [row for row in rows if row["complete_success"]]
    incomplete_rows = [row for row in rows if not row["complete_success"]]
    full_write_rows = [row for row in rows if row["write_progress"] == 1.0]
    no_write_rows = [row for row in rows if row["write_progress"] == 0.0]
    complete_above_incomplete = all(
        rank(success) > rank(failure)
        for success in complete_rows
        for failure in incomplete_rows
        if success["task_id"] == failure["task_id"]
    )
    full_write_above_no_write = all(
        rank(write) > rank(no_write)
        for write in full_write_rows
        for no_write in no_write_rows
        if write["task_id"] == no_write["task_id"]
    )
    unexpected_write_rows = [row for row in rows if row["unexpected_write_count"] > 0]
    safe_no_write_rows = [
        row
        for row in no_write_rows
        if row["unexpected_write_count"] == 0
        and row["result_evidence_progress"]["value"] > 0.0
    ]
    unexpected_write_below_safe_evidence = all(
        rank(unexpected) < rank(safe)
        for unexpected in unexpected_write_rows
        for safe in safe_no_write_rows
        if unexpected["task_id"] == safe["task_id"]
    )
    s5i_task43 = [
        row for row in rows if row["source"] == "s5i" and row["task_id"] == "43"
    ]
    future_binding_ready = all(
        not task.get("future_intent_dependency")
        or bool(
            task["future_intent_dependency"].get(
                "structured_dialogue_binding_available"
            )
        )
        for task in semantic_spec["tasks"].values()
    )
    checks = {
        "complete_success_above_incomplete_within_task": complete_above_incomplete,
        "full_write_above_no_write_within_task": full_write_above_no_write,
        "task43_wrong_order_prefix_has_no_business_result_evidence": all(
            row["result_evidence_progress"]["value"] == 0.0
            for row in s5i_task43
        ),
        "unexpected_write_below_safe_nonmutating_evidence": (
            bool(unexpected_write_rows)
            and bool(safe_no_write_rows)
            and unexpected_write_below_safe_evidence
        ),
        "task72_future_intent_binding_ready": future_binding_ready,
    }
    return {
        "schema_version": SCHEMA_VERSION,
        "scope": {
            "offline_audit_only": True,
            "changes_online_reward": False,
            "calls_external_api": False,
            "uses_cloud_gpu": False,
        },
        "sources": {
            "candidate_spec": {
                "path": str(spec_path.resolve()),
                "sha256": _sha256(spec_path),
            },
            "verified_source_count": 3 + len(historical_paths),
        },
        "summary": {
            "trajectory_count": len(rows),
            "s5i_trajectory_count": sum(row["source"] == "s5i" for row in rows),
            "historical_trajectory_count": sum(
                row["source"] == "historical" for row in rows
            ),
            "complete_success_count": len(complete_rows),
            "full_write_count": len(full_write_rows),
            "no_write_count": len(no_write_rows),
            "unexpected_write_count": len(unexpected_write_rows),
        },
        "checks": checks,
        "scalar_online_promotion_ready": all(checks.values()),
        "rows": rows,
        "limitations": [
            "Candidate ranks are lexicographic audit tiers, not scalar training rewards.",
            "Historical and S5I trajectories come from different checkpoints and support ordering analysis only.",
            "Task 72 final intent remains unavailable as a structured dialogue-stage binding.",
            "Reference-call progress is not evidence that a business fact is true.",
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit staged Reward semantics")
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = build_report(args.spec.resolve(), repo_root=args.repo_root.resolve())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report["summary"], ensure_ascii=False, indent=2))
    print(json.dumps(report["checks"], ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
