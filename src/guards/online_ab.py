from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PROTOCOL = PROJECT_ROOT / "configs" / "guard_online_ab_v1.json"
DEFAULT_PREFLIGHT_OUTPUT = (
    PROJECT_ROOT / "experiments" / "20260730_guard_online_ab_preflight_v1"
)


def sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git(*args: str) -> str | None:
    try:
        return subprocess.check_output(
            ["git", *args],
            cwd=PROJECT_ROOT,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def load_protocol(path: str | Path) -> dict[str, Any]:
    protocol = json.loads(Path(path).read_text(encoding="utf-8"))
    task_ids = [str(value) for value in protocol.get("task_ids") or []]
    excluded = {
        str(value)
        for value in protocol.get("excluded_quarantined_task_ids") or []
    }
    if not task_ids or len(task_ids) != len(set(task_ids)):
        raise ValueError("A/B protocol task_ids must be non-empty and unique.")
    if set(task_ids) & excluded:
        raise ValueError("A/B task_ids include a quarantined task.")
    if set(protocol.get("arms") or {}) != {"base", "guarded"}:
        raise ValueError("A/B protocol must define exactly base and guarded arms.")
    if protocol.get("interpretation", {}).get(
        "general_success_rate_claim_allowed"
    ) is not False:
        raise ValueError("Failure-selected A/B cannot allow general success claims.")
    return protocol


def _check(
    checks: list[dict[str, Any]],
    check_id: str,
    passed: bool,
    detail: str,
    *,
    blocking: bool = True,
) -> None:
    checks.append(
        {
            "check_id": check_id,
            "passed": passed,
            "blocking": blocking,
            "detail": detail,
        }
    )


def preflight(
    protocol: dict[str, Any],
    *,
    paid_approval: bool,
    api_key_configured: bool | None = None,
    git_dirty: bool | None = None,
) -> dict[str, Any]:
    parent_path = PROJECT_ROOT / protocol["parent_run_config"]
    parent = json.loads(parent_path.read_text(encoding="utf-8"))
    checks: list[dict[str, Any]] = []
    task_ids = [str(value) for value in protocol["task_ids"]]
    parent_task_ids = {str(value) for value in parent.get("task_ids") or []}
    _check(
        checks,
        "tasks_in_frozen_parent",
        set(task_ids).issubset(parent_task_ids),
        "All A/B tasks must be present in the frozen parent run config.",
    )
    _check(
        checks,
        "official_test_unused",
        bool(
            parent.get("protocol", {}).get(
                "official_test_split_must_remain_unused"
            )
            and protocol.get("execution_guardrails", {}).get(
                "official_test_split_must_remain_unused"
            )
        ),
        "The official Tau2 test split must remain unused.",
    )
    controlled = set(protocol.get("controlled_variables") or [])
    required_controls = {
        "task_ids",
        "agent_model",
        "agent_temperature",
        "user_model",
        "user_temperature",
        "nl_judge_model",
        "nl_judge_temperature",
        "seed",
        "evaluation_type",
    }
    _check(
        checks,
        "controlled_variables_complete",
        required_controls.issubset(controlled),
        "Both arms may differ only by Guard implementation and retry behavior.",
    )
    if git_dirty is None:
        git_dirty = bool(_git("status", "--porcelain"))
    _check(
        checks,
        "clean_git_tree",
        not git_dirty,
        "Paid execution requires committed code and protocol hashes.",
    )
    if api_key_configured is None:
        variable = protocol["execution_guardrails"][
            "requires_api_key_environment_variable"
        ]
        api_key_configured = bool(os.getenv(variable))
    _check(
        checks,
        "api_key_configured",
        api_key_configured,
        "The required API key must exist in the environment; its value is never recorded.",
    )
    _check(
        checks,
        "explicit_paid_approval",
        paid_approval,
        "Paid model execution requires the explicit CLI approval flag.",
    )
    _check(
        checks,
        "failure_selected_scope_disclosed",
        bool(
            protocol.get("interpretation", {}).get("failure_selected_subset")
            and not protocol.get("interpretation", {}).get(
                "general_success_rate_claim_allowed"
            )
        ),
        "Results must be reported as a failure-selected subset.",
    )
    blocking_failures = [
        check["check_id"]
        for check in checks
        if check["blocking"] and not check["passed"]
    ]
    return {
        "schema_version": "guard-online-ab-preflight-v1.0.0",
        "status": "READY" if not blocking_failures else "BLOCKED",
        "task_ids": task_ids,
        "arms": protocol["arms"],
        "controlled_configuration": {
            "agent": parent["agent"],
            "user": parent["user"],
            "nl_judge": parent["nl_judge"],
            "evaluation": parent["evaluation"],
            "runtime": parent["runtime"],
        },
        "checks": checks,
        "blocking_failure_ids": blocking_failures,
        "secrets": {
            "api_key_value_recorded": False,
            "api_key_configured": api_key_configured,
        },
        "paid_calls_executed": False,
        "interpretation": protocol["interpretation"],
    }


def _simulation(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    simulations = payload.get("simulations") or []
    if len(simulations) != 1:
        raise ValueError(f"Expected exactly one simulation in {path}.")
    return dict(simulations[0])


def _arm_row(path: str | Path) -> dict[str, Any]:
    simulation = _simulation(path)
    reward_info = dict(simulation.get("reward_info") or {})
    breakdown = dict(reward_info.get("reward_breakdown") or {})
    agent_cost = simulation.get("agent_cost")
    user_cost = simulation.get("user_cost")
    known_costs = [
        float(value)
        for value in (agent_cost, user_cost)
        if isinstance(value, (int, float))
    ]
    return {
        "task_id": str(simulation.get("task_id")),
        "reward": reward_info.get("reward"),
        "db_reward": breakdown.get("DB"),
        "nl_assertion_reward": breakdown.get("NL_ASSERTION"),
        "duration_seconds": simulation.get("duration"),
        "agent_cost_usd": agent_cost,
        "user_cost_usd": user_cost,
        "known_agent_user_cost_usd": sum(known_costs) if known_costs else None,
        "termination_reason": simulation.get("termination_reason"),
        "source": str(Path(path).resolve()),
        "source_sha256": sha256(path),
    }


def _guard_trace(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {
            "present": False,
            "event_count": 0,
            "intervention_count": 0,
            "additional_agent_calls": 0,
            "sha256": None,
        }
    events = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    interventions = [
        event
        for event in events
        if event.get("event") == "proposal_evaluated"
        and event.get("allowed") is False
    ]
    additional_calls = sum(
        int(event.get("retry_index", 0)) > 0
        for event in events
        if event.get("event") == "proposal_evaluated"
    )
    return {
        "present": True,
        "event_count": len(events),
        "intervention_count": len(interventions),
        "additional_agent_calls": additional_calls,
        "sha256": sha256(path),
    }


def compare_arms(
    protocol: dict[str, Any],
    *,
    base_dir: str | Path,
    guarded_dir: str | Path,
) -> dict[str, Any]:
    base_root = Path(base_dir)
    guarded_root = Path(guarded_dir)
    rows: list[dict[str, Any]] = []
    for task_id in protocol["task_ids"]:
        base = _arm_row(base_root / f"task_{task_id}" / "returned_results.json")
        guarded = _arm_row(
            guarded_root / f"task_{task_id}" / "returned_results.json"
        )
        trace = _guard_trace(
            guarded_root / f"task_{task_id}" / "guard_trace.jsonl"
        )
        if base["task_id"] != str(task_id) or guarded["task_id"] != str(task_id):
            raise ValueError(f"Task identity mismatch for paired task {task_id}.")
        base_success = base["reward"] == 1
        guarded_success = guarded["reward"] == 1
        rows.append(
            {
                "task_id": str(task_id),
                "base": base,
                "guarded": guarded,
                "guard_trace": trace,
                "paired": {
                    "reward_delta": (
                        guarded["reward"] - base["reward"]
                        if isinstance(base["reward"], (int, float))
                        and isinstance(guarded["reward"], (int, float))
                        else None
                    ),
                    "recovered": not base_success and guarded_success,
                    "regressed": base_success and not guarded_success,
                    "both_success": base_success and guarded_success,
                    "both_failure": not base_success and not guarded_success,
                },
            }
        )
    base_successes = sum(row["base"]["reward"] == 1 for row in rows)
    guarded_successes = sum(row["guarded"]["reward"] == 1 for row in rows)
    return {
        "schema_version": "guard-online-ab-comparison-v1.0.0",
        "scope": "failure-selected paired subset; not a general Retail estimate",
        "task_count": len(rows),
        "summary": {
            "base_success_count": base_successes,
            "guarded_success_count": guarded_successes,
            "paired_business_success_delta": guarded_successes - base_successes,
            "guard_recovery_count": sum(
                row["paired"]["recovered"] for row in rows
            ),
            "guard_regression_count": sum(
                row["paired"]["regressed"] for row in rows
            ),
            "guard_intervention_count": sum(
                row["guard_trace"]["intervention_count"] for row in rows
            ),
            "additional_agent_calls": sum(
                row["guard_trace"]["additional_agent_calls"] for row in rows
            ),
            "guard_trace_complete": all(
                row["guard_trace"]["present"] for row in rows
            ),
            "all_pairs_present": len(rows) == len(protocol["task_ids"]),
            "v7_replay_required_before_final_claim": True,
        },
        "tasks": rows,
        "interpretation": protocol["interpretation"],
    }


def render_preflight(result: dict[str, Any]) -> str:
    lines = [
        "# Guard Online Paired A/B V1 Preflight",
        "",
        f"- Status: **{result['status']}**",
        f"- Tasks: {', '.join(result['task_ids'])}",
        "- Paid calls executed: no",
        "- API key value recorded: no",
        "",
        "## Checks",
        "",
        "| Check | Passed | Detail |",
        "|---|---|---|",
    ]
    for check in result["checks"]:
        lines.append(
            f"| {check['check_id']} | {'YES' if check['passed'] else 'NO'} | "
            f"{check['detail']} |"
        )
    lines.extend(
        [
            "",
            "## Boundary",
            "",
            "- This is a failure-selected three-task subset.",
            "- It cannot produce a general Retail success-rate claim.",
            "- Paid execution remains blocked until every preflight check passes.",
            "- Raw arm outputs must pass V7 replay before a final recovery claim.",
        ]
    )
    return "\n".join(lines) + "\n"


def write_preflight(
    protocol_path: str | Path,
    output_dir: str | Path,
    result: dict[str, Any],
) -> None:
    protocol_file = Path(protocol_path).resolve()
    destination = Path(output_dir).resolve()
    destination.mkdir(parents=True, exist_ok=True)
    snapshot = destination / "protocol_snapshot.json"
    snapshot.write_text(
        protocol_file.read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    preflight_path = destination / "preflight.json"
    preflight_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    report_path = destination / "analysis.md"
    report_path.write_text(render_preflight(result), encoding="utf-8")
    manifest = {
        "schema_version": "guard-online-ab-preflight-manifest-v1.0.0",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "project_commit": _git("rev-parse", "HEAD"),
        "project_git_dirty": bool(_git("status", "--porcelain")),
        "protocol": {
            "path": str(protocol_file.relative_to(PROJECT_ROOT)),
            "sha256": sha256(protocol_file),
        },
        "protocol_snapshot": {
            "path": str(snapshot.relative_to(PROJECT_ROOT)),
            "sha256": sha256(snapshot),
        },
        "preflight": {
            "path": str(preflight_path.relative_to(PROJECT_ROOT)),
            "sha256": sha256(preflight_path),
            "status": result["status"],
        },
        "report": {
            "path": str(report_path.relative_to(PROJECT_ROOT)),
            "sha256": sha256(report_path),
        },
        "paid_calls_executed": False,
    }
    (destination / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Preflight or compare a failure-selected Guard online A/B."
    )
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument(
        "--preflight-output",
        type=Path,
        default=DEFAULT_PREFLIGHT_OUTPUT,
    )
    parser.add_argument("--approve-paid-run", action="store_true")
    parser.add_argument("--base-dir", type=Path)
    parser.add_argument("--guarded-dir", type=Path)
    parser.add_argument("--comparison-output", type=Path)
    args = parser.parse_args()
    protocol = load_protocol(args.protocol)

    if args.base_dir or args.guarded_dir:
        if not args.base_dir or not args.guarded_dir:
            raise SystemExit("Both --base-dir and --guarded-dir are required.")
        result = compare_arms(
            protocol,
            base_dir=args.base_dir,
            guarded_dir=args.guarded_dir,
        )
        output = json.dumps(result, ensure_ascii=False, indent=2) + "\n"
        if args.comparison_output:
            args.comparison_output.write_text(output, encoding="utf-8")
        print(output)
        return

    result = preflight(protocol, paid_approval=args.approve_paid_run)
    write_preflight(args.protocol, args.preflight_output, result)
    print(render_preflight(result))
    if result["status"] != "READY":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
