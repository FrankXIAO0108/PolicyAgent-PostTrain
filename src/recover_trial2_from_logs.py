"""
Recover the completed four-task Trial-2 after Windows GBK console rendering
raised UnicodeEncodeError inside tau2's Rich progress display.

This script performs NO model calls and NO evaluator calls. It recovers only
facts already persisted in task.log, task.json, and llm_debug JSON files.
"""

from __future__ import annotations

import json
import re
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any

import run_retail_baseline20 as base


TASK_IDS = ["59", "98", "95", "107"]

RUN_DIR = (
    base.EXPERIMENTS_ROOT
    / "20260724_111341_retail_failure4_trial2_deepseek"
)

# Independently checked against each task's expected actions and the final
# environment state written at the end of task.log.
VERIFIED_DB_REWARD = {
    "59": 0.0,
    "98": 0.0,
    "95": 1.0,
    "107": 1.0,
}

DB_EVIDENCE = {
    "59": (
        "Final state cancelled #W2702727, while the task expected cancellation "
        "of #W8268610 and an address update for #W2702727."
    ),
    "98": (
        "Final exchange state used credit_card_3951670, while the expected "
        "exchange payment method was credit_card_8105988."
    ),
    "95": "Both expected laptop exchanges are present in the final state.",
    "107": "Both expected item exchanges are present in the final state.",
}

REWARD_RE = re.compile(
    r"Simulation complete:\s+domain=retail,\s+task=(\d+),\s+reward=([0-9.]+)"
)

LOG_TS_RE = re.compile(
    r"^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\.\d{3})",
    re.MULTILINE,
)

TOOL_CALL_RE = re.compile(
    r"ToolCall \(from assistant\)\r?\n"
    r"id:\s*(?P<id>[^\r\n]+)\r?\n"
    r"name:\s*(?P<name>[^\r\n]+)\r?\n"
    r"arguments:\r?\n"
    r"(?P<arguments>\{.*?\})\r?\n"
    r"(?=ToolCall \(from assistant\)|is_final_chunk:\s*True)",
    re.DOTALL,
)


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8-sig") as handle:
        return json.load(handle)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as handle:
        json.dump(
            value,
            handle,
            ensure_ascii=False,
            indent=2,
            default=str,
        )
        handle.write("\n")


def unique_file(
    root: Path,
    pattern: str,
    *,
    required: bool = True,
) -> Path | None:
    matches = sorted(root.rglob(pattern))

    if required and len(matches) != 1:
        raise RuntimeError(
            f"Expected exactly one {pattern!r} under {root}, "
            f"got {len(matches)}."
        )

    if not required and not matches:
        return None

    if len(matches) > 1:
        raise RuntimeError(
            f"Expected at most one {pattern!r} under {root}, "
            f"got {len(matches)}."
        )

    return matches[0]


def response_cost_lower_bound(
    task_dir: Path,
    pattern: str,
) -> float:
    total = 0.0

    for path in sorted(task_dir.rglob(pattern)):
        data = read_json(path)
        response = data.get("response") or {}
        value = response.get("cost")

        if isinstance(value, (int, float)):
            total += float(value)

    return total


def parse_tool_calls(
    log_text: str,
) -> list[dict[str, Any]]:
    calls: list[dict[str, Any]] = []

    for match in TOOL_CALL_RE.finditer(log_text):
        calls.append(
            {
                "id": match.group("id").strip(),
                "name": match.group("name").strip(),
                "arguments": json.loads(
                    match.group("arguments")
                ),
            }
        )

    if not calls:
        raise RuntimeError(
            "No assistant tool calls could be parsed from task.log."
        )

    return calls


def tool_type(name: str) -> str:
    if name.startswith(("get_", "find_")):
        return "read"

    return "write"


def reconstruct_action_checks(
    task: dict[str, Any],
    actual_calls: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    expected_actions = (
        task.get("evaluation_criteria") or {}
    ).get("actions") or []

    unused = list(range(len(actual_calls)))
    checks: list[dict[str, Any]] = []

    for expected in expected_actions:
        expected_name = expected["name"]
        expected_args = expected.get("arguments") or {}
        matched_index: int | None = None

        for index in unused:
            actual = actual_calls[index]

            if (
                actual["name"] == expected_name
                and actual["arguments"] == expected_args
            ):
                matched_index = index
                break

        if matched_index is not None:
            unused.remove(matched_index)
            actual_call = actual_calls[matched_index]
            action_reward = 1.0
        else:
            actual_call = None
            action_reward = 0.0

        checks.append(
            {
                "action_id": expected.get("action_id"),
                "tool_name": expected_name,
                "tool_type": tool_type(expected_name),
                "expected_arguments": expected_args,
                "matched_actual_call": actual_call,
                "action_reward": action_reward,
                "recovery_method": (
                    "EXACT_TOOL_NAME_AND_ARGUMENTS_MATCHED_FROM_TASK_LOG"
                ),
            }
        )

    return checks


def recover_nl_assertions(
    task_dir: Path,
    task: dict[str, Any],
) -> tuple[list[dict[str, Any]], float]:
    expected = (
        task.get("evaluation_criteria") or {}
    ).get("nl_assertions") or []

    judge_file = unique_file(
        task_dir,
        "*nl_assertions_eval*.json",
        required=False,
    )

    if not expected:
        if judge_file is not None:
            raise RuntimeError(
                f"{task_dir.name}: unexpected NL judge file for "
                "task with no NL assertions."
            )

        return [], 1.0

    if judge_file is None:
        raise RuntimeError(
            f"{task_dir.name}: NL assertions exist but judge "
            "output is missing."
        )

    debug_record = read_json(judge_file)
    response = debug_record.get("response") or {}
    content = response.get("content")

    if not isinstance(content, str):
        raise RuntimeError(
            f"{judge_file}: missing string response.content."
        )

    parsed = json.loads(content)
    results = parsed.get("results")

    if not isinstance(results, list):
        raise RuntimeError(
            f"{judge_file}: response has no results list."
        )

    observed_outcomes = [
        item.get("expectedOutcome")
        for item in results
    ]

    if observed_outcomes != expected:
        raise RuntimeError(
            f"{task_dir.name}: NL expectation order/content mismatch."
        )

    nl_reward = (
        1.0
        if results
        and all(
            item.get("metExpectation") is True
            for item in results
        )
        else 0.0
    )

    return results, nl_reward


def recover_task(
    task_id: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    task_dir = RUN_DIR / f"task_{task_id}"
    task_path = task_dir / "task.json"

    if not task_path.exists():
        raise RuntimeError(
            f"Missing task snapshot: {task_path}"
        )

    task = read_json(task_path)

    if str(task.get("id")) != task_id:
        raise RuntimeError(
            f"Task ID mismatch in {task_path}."
        )

    log_path = unique_file(task_dir, "task.log")
    assert log_path is not None

    log_text = log_path.read_text(
        encoding="utf-8-sig"
    )

    reward_matches = REWARD_RE.findall(log_text)

    if len(reward_matches) != 1:
        raise RuntimeError(
            f"Task {task_id}: expected one completed reward line, "
            f"got {reward_matches!r}."
        )

    logged_task_id, reward_text = reward_matches[0]

    if logged_task_id != task_id:
        raise RuntimeError(
            f"Task {task_id}: reward line has a different ID."
        )

    reward = float(reward_text)

    if "###STOP###" not in log_text:
        raise RuntimeError(
            f"Task {task_id}: user stop marker is missing."
        )

    timestamps = [
        datetime.strptime(
            value,
            "%Y-%m-%d %H:%M:%S.%f",
        )
        for value in LOG_TS_RE.findall(log_text)
    ]

    if len(timestamps) < 2:
        raise RuntimeError(
            f"Task {task_id}: insufficient timestamps."
        )

    duration = (
        timestamps[-1] - timestamps[0]
    ).total_seconds()

    actual_calls = parse_tool_calls(log_text)

    action_checks = reconstruct_action_checks(
        task,
        actual_calls,
    )

    nl_assertions, nl_reward = recover_nl_assertions(
        task_dir,
        task,
    )

    db_reward = VERIFIED_DB_REWARD[task_id]
    reconstructed_reward = min(
        db_reward,
        nl_reward,
    )

    if reconstructed_reward != reward:
        raise RuntimeError(
            f"Task {task_id}: logged reward {reward} conflicts "
            f"with recovered DB={db_reward}, NL={nl_reward}."
        )

    agent_cost_lb = response_cost_lower_bound(
        task_dir,
        "*agent_response*.json",
    )

    user_cost_lb = response_cost_lower_bound(
        task_dir,
        "*user_simulator_response*.json",
    )

    returned_results = {
        "timestamp": datetime.now().isoformat(),
        "info": {
            "recovery_status": (
                "RECOVERED_FROM_COMPLETED_TAU2_ARTIFACTS"
            ),
            "recovery_reason": (
                "tau2 completed evaluation, then Windows GBK "
                "rendering raised UnicodeEncodeError while "
                "printing the result."
            ),
            "no_model_or_evaluator_calls": True,
            "source_task_log": str(log_path),
            "limitations": [
                (
                    "The original in-memory Simulation object was "
                    "not persisted because the wrapper exception "
                    "occurred before return."
                ),
                (
                    "Action checks were reconstructed by exact "
                    "expected/actual tool name and JSON-argument "
                    "matching."
                ),
                (
                    "Agent and user costs are lower bounds from "
                    "available debug records; full costs are not "
                    "recoverable."
                ),
            ],
        },
        "tasks": [task],
        "simulations": [
            {
                "id": (
                    f"recovered_trial2_task_{task_id}"
                ),
                "task_id": task_id,
                "duration": duration,
                "termination_reason": "USER_STOP",
                "agent_cost": agent_cost_lb,
                "user_cost": user_cost_lb,
                "reward_info": {
                    "reward": reward,
                    "reward_breakdown": {
                        "DB": db_reward,
                        "NL_ASSERTION": nl_reward,
                    },
                    "action_checks": action_checks,
                    "nl_assertions": nl_assertions,
                    "recovery_evidence": {
                        "logged_terminal_reward": reward,
                        "db_evidence": (
                            DB_EVIDENCE[task_id]
                        ),
                    },
                },
            }
        ],
        "simulation_index": {
            task_id: [
                f"recovered_trial2_task_{task_id}"
            ]
        },
    }

    recovery_detail = {
        "task_id": task_id,
        "reward": reward,
        "db_reward": db_reward,
        "nl_assertion_reward": nl_reward,
        "action_passed": sum(
            item["action_reward"] == 1.0
            for item in action_checks
        ),
        "action_total": len(action_checks),
        "duration_seconds": duration,
        "agent_cost_usd_lower_bound": agent_cost_lb,
        "user_cost_usd_lower_bound": user_cost_lb,
        "task_log": str(log_path),
    }

    return returned_results, recovery_detail


def backup_wrapper_outputs() -> Path:
    backup_dir = (
        RUN_DIR / "wrapper_failure_snapshot"
    )

    backup_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    candidates = [
        RUN_DIR / "progress.json",
        RUN_DIR / "trial2_summary.json",
        RUN_DIR / "trial2_table.csv",
    ]

    for task_id in TASK_IDS:
        candidates.extend(
            [
                (
                    RUN_DIR
                    / f"task_{task_id}"
                    / "summary.json"
                ),
                (
                    RUN_DIR
                    / f"task_{task_id}"
                    / "error.json"
                ),
            ]
        )

    for source in candidates:
        if not source.exists():
            continue

        relative = source.relative_to(RUN_DIR)
        target = backup_dir / relative

        if target.exists():
            continue

        target.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        shutil.copy2(
            source,
            target,
        )

    return backup_dir


def main() -> None:
    if not RUN_DIR.is_dir():
        raise RuntimeError(
            "Trial-2 run directory does not exist: "
            f"{RUN_DIR}"
        )

    manifest = read_json(
        RUN_DIR / "run_manifest.json"
    )

    manifest_task_ids = [
        str(value)
        for value in manifest.get("task_ids", [])
    ]

    if manifest_task_ids != TASK_IDS:
        raise RuntimeError(
            "Run manifest does not match the four frozen tasks."
        )

    backup_dir = backup_wrapper_outputs()

    recovery_details: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []

    for task_id in TASK_IDS:
        task_dir = RUN_DIR / f"task_{task_id}"

        returned_path = (
            task_dir / "returned_results.json"
        )

        if returned_path.exists():
            existing = read_json(returned_path)

            recovery_status = (
                existing.get("info") or {}
            ).get("recovery_status")

            if (
                recovery_status
                != "RECOVERED_FROM_COMPLETED_TAU2_ARTIFACTS"
            ):
                raise RuntimeError(
                    "Refusing to overwrite non-recovery result: "
                    f"{returned_path}"
                )

            returned_results, detail = recover_task(
                task_id
            )

            existing_reward = (
                existing["simulations"][0]
                ["reward_info"]["reward"]
            )

            recovered_reward = (
                returned_results["simulations"][0]
                ["reward_info"]["reward"]
            )

            if existing_reward != recovered_reward:
                raise RuntimeError(
                    "Existing recovered reward changed: "
                    f"{task_id}"
                )
        else:
            returned_results, detail = recover_task(
                task_id
            )

            write_json(
                returned_path,
                returned_results,
            )

        summary = base.build_task_summary(
            task_id=task_id,
            task_dir=task_dir,
            execution_source=(
                "RECOVERED_FROM_COMPLETED_TAU2_ARTIFACTS"
            ),
            wall_clock_seconds=None,
            system_error=None,
        )

        summary["recovery_status"] = (
            "RECOVERED_FROM_COMPLETED_TAU2_ARTIFACTS"
        )

        summary["cost_accounting_status"] = (
            "PARTIAL_LOWER_BOUND_FROM_AVAILABLE_DEBUG_LOGS"
        )

        base.write_json(
            task_dir / "summary.json",
            summary,
        )

        summaries.append(summary)
        recovery_details.append(detail)

    run_config = base.load_json(
        base.RUN_CONFIG_PATH
    )

    run_id = RUN_DIR.name.removesuffix(
        "_retail_failure4_trial2_deepseek"
    )

    aggregate = base.build_aggregate(
        run_id=run_id,
        run_config=run_config,
        task_ids=TASK_IDS,
        summaries=summaries,
    )

    aggregate.update(
        {
            "experiment": (
                "retail_failure4_trial2_deepseek"
            ),
            "experiment_role": (
                "REPRESENTATIVE_TASK_SECOND_SAMPLE"
            ),
            "scope": (
                "Recovered Trial-2 second sample for tasks "
                "59, 98, 95, and 107; no model or evaluator "
                "calls were made during recovery."
            ),
            "parent_trial": (
                "retail_baseline20_trial1_deepseek"
            ),
            "task_ids": TASK_IDS,
            "recovery_status": (
                "RECOVERED_FROM_COMPLETED_TAU2_ARTIFACTS"
            ),
            "cost_accounting_status": (
                "PARTIAL_LOWER_BOUND_FROM_AVAILABLE_DEBUG_LOGS"
            ),
        }
    )

    base.write_json(
        RUN_DIR / "progress.json",
        aggregate,
    )

    base.write_json(
        RUN_DIR / "trial2_summary.json",
        aggregate,
    )

    base.write_baseline_csv(
        RUN_DIR / "trial2_table.csv",
        summaries,
    )

    recovery_report = {
        "recovered_at": datetime.now().isoformat(),
        "run_dir": str(RUN_DIR),
        "backup_dir": str(backup_dir),
        "no_model_or_evaluator_calls": True,
        "tasks": recovery_details,
        "aggregate": {
            "completed_count": (
                aggregate["completed_count"]
            ),
            "system_failed_count": (
                aggregate["system_failed_count"]
            ),
            "result_parse_failed_count": (
                aggregate["result_parse_failed_count"]
            ),
            "valid_reward_count": (
                aggregate["valid_reward_count"]
            ),
            "business_success_count": (
                aggregate["business_success_count"]
            ),
            "business_failure_count": (
                aggregate["business_failure_count"]
            ),
            "mean_reward": (
                aggregate["mean_reward"]
            ),
        },
    }

    base.write_json(
        RUN_DIR / "trial2_recovery_report.json",
        recovery_report,
    )

    print("=" * 88)
    print(
        "TRIAL-2 RECOVERY COMPLETE - "
        "NO MODEL OR EVALUATOR CALLS"
    )

    for detail in recovery_details:
        print(
            f"Task {detail['task_id']}: "
            f"reward={detail['reward']}, "
            f"DB={detail['db_reward']}, "
            f"NL={detail['nl_assertion_reward']}, "
            f"actions="
            f"{detail['action_passed']}/"
            f"{detail['action_total']}"
        )

    print(
        f"COMPLETED = "
        f"{aggregate['completed_count']}"
    )

    print(
        f"SYSTEM_FAILED = "
        f"{aggregate['system_failed_count']}"
    )

    print(
        f"RESULT_PARSE_FAILED = "
        f"{aggregate['result_parse_failed_count']}"
    )

    print(
        f"MEAN_REWARD = "
        f"{aggregate['mean_reward']}"
    )

    print(
        "REPORT = "
        f"{RUN_DIR / 'trial2_recovery_report.json'}"
    )

    recovery_ok = (
        aggregate["completed_count"] == 4
        and aggregate["system_failed_count"] == 0
        and aggregate["result_parse_failed_count"] == 0
        and aggregate["valid_reward_count"] == 4
    )

    if recovery_ok:
        print(
            "RETAIL_FAILURE4_TRIAL2_RECOVERY_OK"
        )
    else:
        print(
            "RETAIL_FAILURE4_TRIAL2_RECOVERY_INCOMPLETE"
        )
        raise SystemExit(1)

    print("=" * 88)


if __name__ == "__main__":
    main()