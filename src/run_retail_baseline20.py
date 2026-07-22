"""
Run or resume the frozen Retail 20-task Prompt Baseline Trial-1.

Key guarantees
--------------
1. Do not modify upstream tau2-bench source.
2. Task IDs come from the frozen baseline config.
3. Agent / User / Judge / runtime parameters come from the frozen run config.
4. Every task is executed independently.
5. Existing returned_results.json files are reused and NEVER re-run automatically.
6. Business failures (reward=0) are preserved exactly as observed.
7. System/API/runtime failures are distinguished from business failures.
8. Raw results are always parsed from persisted JSON, not live Pydantic objects.
9. NL Judge is explicitly forced to the configured model.
10. Per-task artifacts and aggregate tables are persisted continuously.

Important:
This is a frozen 20-task development Prompt Baseline Trial-1,
not an official tau2-bench leaderboard score.
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any

import tau2.config as tau2_config
import tau2.evaluator.evaluator_nl_assertions as nl_eval

from tau2.evaluator.evaluator import EvaluationType
from tau2.run import get_tasks, run_tasks


# =============================================================================
# Paths
# =============================================================================

PROJECT_ROOT = Path(
    r"D:\PolicyAgent-PostTrain"
)

UPSTREAM_ROOT = Path(
    r"D:\tau2-bench"
)

BASELINE_CONFIG_PATH = (
    PROJECT_ROOT
    / "configs"
    / "baseline_20_tasks.json"
)

RUN_CONFIG_PATH = (
    PROJECT_ROOT
    / "configs"
    / "baseline_trial1_run_config.json"
)

EXPERIMENTS_ROOT = (
    PROJECT_ROOT
    / "experiments"
)

RUN_SUFFIX = (
    "_retail_baseline20_trial1_deepseek"
)


# =============================================================================
# Generic helpers
# =============================================================================

def load_json(
    path: Path,
) -> Any:
    return json.loads(
        path.read_text(
            encoding="utf-8"
        )
    )


def write_json(
    path: Path,
    data: Any,
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    path.write_text(
        json.dumps(
            data,
            ensure_ascii=False,
            indent=2,
            default=str,
        ),
        encoding="utf-8",
    )


def append_jsonl(
    path: Path,
    data: dict[str, Any],
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with path.open(
        "a",
        encoding="utf-8",
    ) as file:
        file.write(
            json.dumps(
                data,
                ensure_ascii=False,
                default=str,
            )
            + "\n"
        )


def model_to_dict(
    obj: Any,
) -> Any:
    """
    Recursively convert Pydantic models and nested containers
    into JSON-compatible Python objects.

    This fixes structures such as:

        list[ActionCheck]
        list[NLAssertionCheck]

    where converting only the outer list is insufficient.
    """

    if obj is None:
        return None

    # Pydantic v2
    if hasattr(
        obj,
        "model_dump",
    ):
        return model_to_dict(
            obj.model_dump(
                mode="json"
            )
        )

    # Pydantic v1 / compatible
    if hasattr(
        obj,
        "dict",
    ):
        return model_to_dict(
            obj.dict()
        )

    if isinstance(
        obj,
        dict,
    ):
        return {
            key: model_to_dict(
                value
            )
            for key, value
            in obj.items()
        }

    if isinstance(
        obj,
        (list, tuple),
    ):
        return [
            model_to_dict(
                value
            )
            for value in obj
        ]

    return obj


def nested_get(
    obj: Any,
    *path: str,
    default: Any = None,
) -> Any:
    current = obj

    for key in path:

        if current is None:
            return default

        if isinstance(
            current,
            dict,
        ):
            current = (
                current.get(key)
            )

        else:
            current = getattr(
                current,
                key,
                None,
            )

    if current is None:
        return default

    return current


def to_float(
    value: Any,
    default: float = 0.0,
) -> float:
    if value is None:
        return default

    try:
        return float(value)

    except (
        TypeError,
        ValueError,
    ):
        return default


def enum_or_value(
    value: Any,
) -> Any:
    if value is None:
        return None

    if hasattr(
        value,
        "value",
    ):
        return value.value

    return str(value)


def get_upstream_commit() -> str:
    return subprocess.check_output(
        [
            "git",
            "-C",
            str(
                UPSTREAM_ROOT
            ),
            "rev-parse",
            "HEAD",
        ],
        text=True,
    ).strip()


# =============================================================================
# Run directory selection / resume
# =============================================================================

def find_existing_runs() -> list[Path]:
    if not EXPERIMENTS_ROOT.exists():
        return []

    runs = [
        path
        for path
        in EXPERIMENTS_ROOT.glob(
            f"*{RUN_SUFFIX}"
        )
        if path.is_dir()
    ]

    return sorted(
        runs,
        key=lambda path: path.name,
        reverse=True,
    )


def select_run_dir(
    *,
    force_new_run: bool,
) -> tuple[Path, bool]:
    """
    Returns:
        (run_dir, is_resume)

    Default behavior:
    - If an incomplete baseline run exists, resume the newest one.
    - If only completed runs exist, refuse to silently create another Trial-1.
    - If no run exists yet, create the first run.
    """

    existing_runs = (
        find_existing_runs()
    )

    if force_new_run:

        run_id = (
            datetime.now()
            .strftime(
                "%Y%m%d_%H%M%S"
            )
        )

        return (
            EXPERIMENTS_ROOT
            / f"{run_id}{RUN_SUFFIX}",
            False,
        )

    incomplete_runs = [
        path
        for path
        in existing_runs
        if not (
            path
            / "baseline_summary.json"
        ).exists()
    ]

    if incomplete_runs:

        return (
            incomplete_runs[0],
            True,
        )

    if existing_runs:

        raise RuntimeError(
            "\nA completed Baseline Trial-1 "
            "already exists.\n"
            "Refusing to silently create "
            "another Trial-1.\n\n"
            "If you intentionally want a "
            "new independent run, use:\n\n"
            "  --new-run\n"
        )

    run_id = (
        datetime.now()
        .strftime(
            "%Y%m%d_%H%M%S"
        )
    )

    return (
        EXPERIMENTS_ROOT
        / f"{run_id}{RUN_SUFFIX}",
        False,
    )


# =============================================================================
# Result extraction
# =============================================================================

def extract_simulations(
    results: Any,
) -> list[Any]:
    """
    Supports both raw dict loaded from returned_results.json
    and live tau2 Results objects.

    Verified persisted schema:

    {
        "timestamp": ...,
        "info": ...,
        "tasks": ...,
        "simulations": [...],
        "simulation_index": ...
    }
    """

    if isinstance(
        results,
        dict,
    ):

        simulations = (
            results.get(
                "simulations"
            )
        )

        if isinstance(
            simulations,
            list,
        ):
            return simulations

        return []

    simulations = getattr(
        results,
        "simulations",
        None,
    )

    if isinstance(
        simulations,
        list,
    ):
        return simulations

    dumped = model_to_dict(
        results
    )

    if isinstance(
        dumped,
        dict,
    ):

        simulations = (
            dumped.get(
                "simulations"
            )
        )

        if isinstance(
            simulations,
            list,
        ):
            return simulations

    return []


# =============================================================================
# Action evaluator extraction
# =============================================================================

def extract_action_stats(
    reward_info: dict[str, Any],
) -> dict[str, Any]:
    """
    Parse action_checks from persisted JSON.

    Important:
    Action matching is diagnostic.
    It is not assumed to be identical to final reward.
    """

    action_checks = (
        reward_info.get(
            "action_checks"
        )
        or []
    )

    if not isinstance(
        action_checks,
        list,
    ):
        action_checks = []

    normalized_checks = []

    for item in action_checks:

        if isinstance(
            item,
            dict,
        ):
            normalized_checks.append(
                item
            )

        else:
            converted = (
                model_to_dict(item)
            )

            if isinstance(
                converted,
                dict,
            ):
                normalized_checks.append(
                    converted
                )

    action_checks = (
        normalized_checks
    )

    def passed(
        item: dict[str, Any],
    ) -> bool:

        return (
            to_float(
                item.get(
                    "action_reward"
                ),
                default=0.0,
            )
            == 1.0
        )

    action_total = len(
        action_checks
    )

    action_passed = sum(
        1
        for item
        in action_checks
        if passed(item)
    )

    read_checks = [
        item
        for item
        in action_checks
        if item.get(
            "tool_type"
        ) == "read"
    ]

    write_checks = [
        item
        for item
        in action_checks
        if item.get(
            "tool_type"
        ) == "write"
    ]

    generic_checks = [
        item
        for item
        in action_checks
        if item.get(
            "tool_type"
        ) == "generic"
    ]

    return {
        "action_passed":
            action_passed,

        "action_total":
            action_total,

        "read_passed":
            sum(
                1
                for item
                in read_checks
                if passed(item)
            ),

        "read_total":
            len(
                read_checks
            ),

        "write_passed":
            sum(
                1
                for item
                in write_checks
                if passed(item)
            ),

        "write_total":
            len(
                write_checks
            ),

        "generic_passed":
            sum(
                1
                for item
                in generic_checks
                if passed(item)
            ),

        "generic_total":
            len(
                generic_checks
            ),

        "action_checks":
            action_checks,
    }


# =============================================================================
# NL Judge debug log extraction
# =============================================================================

def collect_judge_logs(
    task_dir: Path,
) -> dict[str, Any]:
    judge_files = sorted(
        task_dir.rglob(
            "*nl_assertions_eval*.json"
        )
    )

    calls: list[
        dict[str, Any]
    ] = []

    for path in judge_files:

        try:

            data = load_json(
                path
            )

            request = (
                data.get(
                    "request"
                )
                or {}
            )

            response = (
                data.get(
                    "response"
                )
                or {}
            )

            usage = (
                response.get(
                    "usage"
                )
                or {}
            )

            calls.append({
                "file":
                    str(path),

                "call_id":
                    data.get(
                        "call_id"
                    ),

                "call_name":
                    data.get(
                        "call_name"
                    ),

                "timestamp":
                    data.get(
                        "timestamp"
                    ),

                "model":
                    request.get(
                        "model"
                    ),

                "cost_usd":
                    to_float(
                        response.get(
                            "cost"
                        ),
                        default=0.0,
                    ),

                "prompt_tokens":
                    int(
                        usage.get(
                            "prompt_tokens",
                            0,
                        )
                        or 0
                    ),

                "completion_tokens":
                    int(
                        usage.get(
                            "completion_tokens",
                            0,
                        )
                        or 0
                    ),
            })

        except Exception as exc:

            calls.append({
                "file":
                    str(path),

                "parse_error_type":
                    type(
                        exc
                    ).__name__,

                "parse_error":
                    str(exc),
            })

    valid_calls = [
        call
        for call
        in calls
        if not call.get(
            "parse_error"
        )
    ]

    return {
        "judge_call_count":
            len(
                valid_calls
            ),

        "judge_log_file_count":
            len(
                judge_files
            ),

        "judge_cost_usd":
            sum(
                to_float(
                    call.get(
                        "cost_usd"
                    ),
                    default=0.0,
                )
                for call
                in valid_calls
            ),

        "judge_prompt_tokens":
            sum(
                int(
                    call.get(
                        "prompt_tokens",
                        0,
                    )
                    or 0
                )
                for call
                in valid_calls
            ),

        "judge_completion_tokens":
            sum(
                int(
                    call.get(
                        "completion_tokens",
                        0,
                    )
                    or 0
                )
                for call
                in valid_calls
            ),

        "judge_calls":
            calls,
    }


# =============================================================================
# Standardized per-task summary
# =============================================================================

def build_task_summary(
    *,
    task_id: str,
    task_dir: Path,
    execution_source: str,
    wall_clock_seconds: float | None,
    system_error: dict[str, Any] | None,
) -> dict[str, Any]:
    """
    Always parse from returned_results.json.

    This intentionally avoids parsing live Pydantic objects,
    so fresh runs and resumed runs share exactly the same parser.
    """

    returned_results_path = (
        task_dir
        / "returned_results.json"
    )

    summary: dict[
        str,
        Any,
    ] = {
        "task_id":
            str(
                task_id
            ),

        "status":
            (
                "COMPLETED"
                if returned_results_path.exists()
                else "SYSTEM_FAILED"
            ),

        "execution_source":
            execution_source,

        "reward":
            None,

        "db_reward":
            None,

        "nl_assertion_reward":
            None,

        "action_passed":
            0,

        "action_total":
            0,

        "read_passed":
            0,

        "read_total":
            0,

        "write_passed":
            0,

        "write_total":
            0,

        "generic_passed":
            0,

        "generic_total":
            0,

        "simulation_duration_seconds":
            None,

        "wall_clock_seconds":
            (
                round(
                    wall_clock_seconds,
                    3,
                )
                if wall_clock_seconds
                is not None
                else None
            ),

        "termination_reason":
            None,

        "agent_cost_usd":
            0.0,

        "user_cost_usd":
            0.0,

        "judge_cost_usd":
            0.0,

        "total_model_cost_usd":
            0.0,

        "judge_call_count":
            0,

        "judge_prompt_tokens":
            0,

        "judge_completion_tokens":
            0,

        "nl_assertion_count":
            0,

        "simulation_count":
            0,

        "error_type":
            (
                system_error.get(
                    "type"
                )
                if system_error
                else None
            ),

        "error_message":
            (
                system_error.get(
                    "message"
                )
                if system_error
                else None
            ),

        "raw_results_path":
            (
                str(
                    returned_results_path
                )
                if returned_results_path.exists()
                else None
            ),

        "task_snapshot_path":
            str(
                task_dir
                / "task.json"
            ),
    }

    if not (
        returned_results_path
        .exists()
    ):

        return summary

    raw_results = load_json(
        returned_results_path
    )

    simulations = (
        extract_simulations(
            raw_results
        )
    )

    summary[
        "simulation_count"
    ] = len(
        simulations
    )

    if not simulations:

        summary[
            "status"
        ] = "RESULT_PARSE_FAILED"

        summary[
            "error_type"
        ] = (
            "MissingSimulation"
        )

        summary[
            "error_message"
        ] = (
            "returned_results.json "
            "contains no simulations."
        )

        return summary

    if len(
        simulations
    ) != 1:

        summary[
            "result_warning"
        ] = (
            "Expected exactly 1 "
            f"simulation, got "
            f"{len(simulations)}."
        )

    simulation = (
        simulations[0]
    )

    if not isinstance(
        simulation,
        dict,
    ):

        simulation = (
            model_to_dict(
                simulation
            )
        )

    if not isinstance(
        simulation,
        dict,
    ):

        summary[
            "status"
        ] = "RESULT_PARSE_FAILED"

        summary[
            "error_type"
        ] = (
            "InvalidSimulationType"
        )

        return summary

    reward_info = (
        simulation.get(
            "reward_info"
        )
        or {}
    )

    if not isinstance(
        reward_info,
        dict,
    ):

        reward_info = (
            model_to_dict(
                reward_info
            )
            or {}
        )

    # -------------------------------------------------------------------------
    # Correct reward path:
    #
    # simulations[0].reward_info.reward
    # -------------------------------------------------------------------------

    summary[
        "reward"
    ] = reward_info.get(
        "reward"
    )

    reward_breakdown = (
        reward_info.get(
            "reward_breakdown"
        )
        or {}
    )

    if not isinstance(
        reward_breakdown,
        dict,
    ):

        reward_breakdown = (
            model_to_dict(
                reward_breakdown
            )
            or {}
        )

    summary[
        "reward_breakdown"
    ] = reward_breakdown

    summary[
        "db_reward"
    ] = reward_breakdown.get(
        "DB"
    )

    summary[
        "nl_assertion_reward"
    ] = reward_breakdown.get(
        "NL_ASSERTION"
    )

    summary[
        "simulation_duration_seconds"
    ] = simulation.get(
        "duration"
    )

    summary[
        "termination_reason"
    ] = enum_or_value(
        simulation.get(
            "termination_reason"
        )
    )

    summary[
        "agent_cost_usd"
    ] = to_float(
        simulation.get(
            "agent_cost"
        )
    )

    summary[
        "user_cost_usd"
    ] = to_float(
        simulation.get(
            "user_cost"
        )
    )

    # -------------------------------------------------------------------------
    # Action diagnostics
    # -------------------------------------------------------------------------

    action_stats = (
        extract_action_stats(
            reward_info
        )
    )

    for key in (
        "action_passed",
        "action_total",
        "read_passed",
        "read_total",
        "write_passed",
        "write_total",
        "generic_passed",
        "generic_total",
    ):

        summary[key] = (
            action_stats[key]
        )

    write_json(
        task_dir
        / "action_checks.json",

        action_stats[
            "action_checks"
        ],
    )

    # -------------------------------------------------------------------------
    # NL assertions
    # -------------------------------------------------------------------------

    nl_assertions = (
        reward_info.get(
            "nl_assertions"
        )
        or []
    )

    if not isinstance(
        nl_assertions,
        list,
    ):

        nl_assertions = (
            model_to_dict(
                nl_assertions
            )
            or []
        )

    if not isinstance(
        nl_assertions,
        list,
    ):
        nl_assertions = []

    summary[
        "nl_assertion_count"
    ] = len(
        nl_assertions
    )

    write_json(
        task_dir
        / "nl_assertions.json",

        nl_assertions,
    )

    # -------------------------------------------------------------------------
    # Judge costs / tokens from persisted debug logs
    # -------------------------------------------------------------------------

    judge_stats = (
        collect_judge_logs(
            task_dir
        )
    )

    summary[
        "judge_call_count"
    ] = judge_stats[
        "judge_call_count"
    ]

    summary[
        "judge_cost_usd"
    ] = judge_stats[
        "judge_cost_usd"
    ]

    summary[
        "judge_prompt_tokens"
    ] = judge_stats[
        "judge_prompt_tokens"
    ]

    summary[
        "judge_completion_tokens"
    ] = judge_stats[
        "judge_completion_tokens"
    ]

    summary[
        "total_model_cost_usd"
    ] = (
        summary[
            "agent_cost_usd"
        ]
        + summary[
            "user_cost_usd"
        ]
        + summary[
            "judge_cost_usd"
        ]
    )

    write_json(
        task_dir
        / "judge_stats.json",

        judge_stats,
    )

    return summary


# =============================================================================
# Aggregate
# =============================================================================

def build_aggregate(
    *,
    run_id: str,
    run_config: dict[str, Any],
    task_ids: list[str],
    summaries: list[dict[str, Any]],
) -> dict[str, Any]:

    completed = [
        row
        for row
        in summaries
        if row[
            "status"
        ] == "COMPLETED"
    ]

    system_failed = [
        row
        for row
        in summaries
        if row[
            "status"
        ] == "SYSTEM_FAILED"
    ]

    parse_failed = [
        row
        for row
        in summaries
        if row[
            "status"
        ] == "RESULT_PARSE_FAILED"
    ]

    valid_rewards = [
        float(
            row[
                "reward"
            ]
        )
        for row
        in completed
        if isinstance(
            row.get(
                "reward"
            ),
            (int, float),
        )
    ]

    success_count = sum(
        1
        for reward
        in valid_rewards
        if reward == 1.0
    )

    failure_count = sum(
        1
        for reward
        in valid_rewards
        if reward != 1.0
    )

    simulation_durations = [
        float(
            row[
                "simulation_duration_seconds"
            ]
        )
        for row
        in completed
        if isinstance(
            row.get(
                "simulation_duration_seconds"
            ),
            (int, float),
        )
    ]

    observed_wall_clocks = [
        float(
            row[
                "wall_clock_seconds"
            ]
        )
        for row
        in summaries
        if isinstance(
            row.get(
                "wall_clock_seconds"
            ),
            (int, float),
        )
    ]

    return {
        "run_id":
            run_id,

        "experiment":
            run_config[
                "name"
            ],

        "experiment_role":
            run_config[
                "experiment_role"
            ],

        "scope":
            (
                "Frozen 20-task Retail "
                "development Prompt Baseline Trial-1"
            ),

        "task_count":
            len(
                task_ids
            ),

        "completed_count":
            len(
                completed
            ),

        "system_failed_count":
            len(
                system_failed
            ),

        "result_parse_failed_count":
            len(
                parse_failed
            ),

        "valid_reward_count":
            len(
                valid_rewards
            ),

        "business_success_count":
            success_count,

        "business_failure_count":
            failure_count,

        "success_rate":
            (
                success_count
                / len(
                    valid_rewards
                )
                if valid_rewards
                else None
            ),

        "mean_reward":
            (
                sum(
                    valid_rewards
                )
                / len(
                    valid_rewards
                )
                if valid_rewards
                else None
            ),

        "total_agent_cost_usd":
            sum(
                to_float(
                    row.get(
                        "agent_cost_usd"
                    )
                )
                for row
                in summaries
            ),

        "total_user_cost_usd":
            sum(
                to_float(
                    row.get(
                        "user_cost_usd"
                    )
                )
                for row
                in summaries
            ),

        "total_judge_cost_usd":
            sum(
                to_float(
                    row.get(
                        "judge_cost_usd"
                    )
                )
                for row
                in summaries
            ),

        "total_model_cost_usd":
            sum(
                to_float(
                    row.get(
                        "total_model_cost_usd"
                    )
                )
                for row
                in summaries
            ),

        "total_judge_calls":
            sum(
                int(
                    row.get(
                        "judge_call_count",
                        0,
                    )
                    or 0
                )
                for row
                in summaries
            ),

        "total_judge_prompt_tokens":
            sum(
                int(
                    row.get(
                        "judge_prompt_tokens",
                        0,
                    )
                    or 0
                )
                for row
                in summaries
            ),

        "total_judge_completion_tokens":
            sum(
                int(
                    row.get(
                        "judge_completion_tokens",
                        0,
                    )
                    or 0
                )
                for row
                in summaries
            ),

        "total_simulation_duration_seconds":
            sum(
                simulation_durations
            ),

        "mean_simulation_duration_seconds":
            (
                sum(
                    simulation_durations
                )
                / len(
                    simulation_durations
                )
                if simulation_durations
                else None
            ),

        "observed_wall_clock_task_count":
            len(
                observed_wall_clocks
            ),

        "total_observed_wall_clock_seconds":
            sum(
                observed_wall_clocks
            ),

        "tasks":
            summaries,

        "interpretation": {
            "success_definition":
                (
                    "reward_info.reward == 1.0"
                ),

            "business_failure":
                (
                    "Simulation completed normally "
                    "but reward != 1.0."
                ),

            "system_failure":
                (
                    "Agent/API/runner execution "
                    "raised an exception before a "
                    "valid returned_results.json "
                    "was produced."
                ),

            "result_parse_failure":
                (
                    "A raw returned_results.json "
                    "exists, but local project "
                    "post-processing failed."
                ),

            "resume_policy":
                (
                    "Any task with an existing "
                    "returned_results.json is reused "
                    "offline and is not re-executed."
                ),

            "wall_clock_warning":
                (
                    "For tasks reused from a previous "
                    "partial run, outer Python wall-clock "
                    "duration may be unavailable. "
                    "Simulation duration remains preserved."
                ),

            "score_scope_warning":
                (
                    "This is a frozen 20-task "
                    "development baseline selected "
                    "from the train split, not an "
                    "official leaderboard score."
                ),
        },
    }


def write_baseline_csv(
    path: Path,
    rows: list[dict[str, Any]],
) -> None:

    fields = [
        "task_id",
        "status",
        "execution_source",
        "reward",
        "db_reward",
        "nl_assertion_reward",
        "action_passed",
        "action_total",
        "read_passed",
        "read_total",
        "write_passed",
        "write_total",
        "generic_passed",
        "generic_total",
        "simulation_duration_seconds",
        "wall_clock_seconds",
        "termination_reason",
        "agent_cost_usd",
        "user_cost_usd",
        "judge_cost_usd",
        "total_model_cost_usd",
        "judge_call_count",
        "judge_prompt_tokens",
        "judge_completion_tokens",
        "error_type",
        "error_message",
    ]

    with path.open(
        "w",
        encoding="utf-8-sig",
        newline="",
    ) as file:

        writer = (
            csv.DictWriter(
                file,
                fieldnames=fields,
            )
        )

        writer.writeheader()

        for row in rows:

            writer.writerow({
                field:
                    row.get(
                        field
                    )
                for field
                in fields
            })


# =============================================================================
# Main
# =============================================================================

def main() -> None:

    parser = (
        argparse.ArgumentParser()
    )

    parser.add_argument(
        "--new-run",
        action="store_true",
        help=(
            "Intentionally start a new "
            "independent baseline run. "
            "Do not use this for normal resume."
        ),
    )

    args = parser.parse_args()

    # -------------------------------------------------------------------------
    # Load frozen config
    # -------------------------------------------------------------------------

    baseline_config = load_json(
        BASELINE_CONFIG_PATH
    )

    run_config = load_json(
        RUN_CONFIG_PATH
    )

    assert (
        baseline_config[
            "status"
        ]
        == "FROZEN"
    )

    assert (
        run_config[
            "status"
        ]
        == "FROZEN"
    )

    task_ids = [
        str(
            task_id
        )
        for task_id
        in run_config[
            "task_ids"
        ]
    ]

    assert len(
        task_ids
    ) == 20

    assert len(
        set(
            task_ids
        )
    ) == 20

    assert (
        task_ids
        == [
            str(
                task_id
            )
            for task_id
            in baseline_config[
                "task_ids"
            ]
        ]
    )

    assert (
        run_config[
            "evaluation"
        ][
            "type"
        ]
        == "ALL_WITH_NL_ASSERTIONS"
    )

    assert (
        run_config[
            "runtime"
        ][
            "num_trials"
        ]
        == 1
    )

    # -------------------------------------------------------------------------
    # Upstream commit
    # -------------------------------------------------------------------------

    upstream_commit = (
        get_upstream_commit()
    )

    expected_commit = (
        run_config[
            "parent_baseline"
        ][
            "upstream_commit"
        ]
    )

    if (
        upstream_commit
        != expected_commit
    ):

        raise RuntimeError(
            "\nUpstream commit mismatch.\n"
            f"Expected: {expected_commit}\n"
            f"Actual:   {upstream_commit}\n"
        )

    # -------------------------------------------------------------------------
    # Select new or resumable run
    # -------------------------------------------------------------------------

    run_dir, is_resume = (
        select_run_dir(
            force_new_run=
                args.new_run
        )
    )

    run_id = (
        run_dir.name[
            : -len(
                RUN_SUFFIX
            )
        ]
    )

    if is_resume:

        print(
            "=" * 100
        )

        print(
            "RESUMING EXISTING "
            "BASELINE TRIAL-1"
        )

        print(
            "RUN_DIR =",
            run_dir,
        )

        print(
            "=" * 100
        )

    else:

        run_dir.mkdir(
            parents=True,
            exist_ok=False,
        )

    # -------------------------------------------------------------------------
    # Runtime config
    # -------------------------------------------------------------------------

    agent_cfg = (
        run_config[
            "agent"
        ]
    )

    user_cfg = (
        run_config[
            "user"
        ]
    )

    judge_cfg = (
        run_config[
            "nl_judge"
        ]
    )

    runtime_cfg = (
        run_config[
            "runtime"
        ]
    )

    judge_model = (
        judge_cfg[
            "model"
        ]
    )

    judge_args = {
        "temperature":
            judge_cfg[
                "temperature"
            ]
    }

    # -------------------------------------------------------------------------
    # Explicit NL Judge override
    # -------------------------------------------------------------------------

    tau2_config.DEFAULT_LLM_NL_ASSERTIONS = (
        judge_model
    )

    tau2_config.DEFAULT_LLM_NL_ASSERTIONS_ARGS = (
        judge_args
    )

    nl_eval.DEFAULT_LLM_NL_ASSERTIONS = (
        judge_model
    )

    nl_eval.DEFAULT_LLM_NL_ASSERTIONS_ARGS = (
        judge_args
    )

    # -------------------------------------------------------------------------
    # Run manifest
    # -------------------------------------------------------------------------

    manifest_path = (
        run_dir
        / "run_manifest.json"
    )

    if manifest_path.exists():

        manifest = load_json(
            manifest_path
        )

        manifest_task_ids = [
            str(
                task_id
            )
            for task_id
            in manifest.get(
                "task_ids",
                [],
            )
        ]

        if (
            manifest_task_ids
            != task_ids
        ):

            raise RuntimeError(
                "Existing run manifest "
                "task IDs do not match "
                "frozen configuration."
            )

        old_commit = (
            manifest.get(
                "upstream_commit"
            )
        )

        if (
            old_commit
            and old_commit
            != upstream_commit
        ):

            raise RuntimeError(
                "Existing run manifest "
                "upstream commit mismatch."
            )

    else:

        manifest = {
            "run_id":
                run_id,

            "created_at":
                datetime.now()
                .isoformat(),

            "experiment":
                run_config[
                    "name"
                ],

            "experiment_role":
                run_config[
                    "experiment_role"
                ],

            "scope":
                (
                    "Frozen 20-task Retail "
                    "development Prompt "
                    "Baseline Trial-1"
                ),

            "not_official_leaderboard_score":
                True,

            "upstream_commit":
                upstream_commit,

            "baseline_config_path":
                str(
                    BASELINE_CONFIG_PATH
                ),

            "run_config_path":
                str(
                    RUN_CONFIG_PATH
                ),

            "baseline_config_sha256":
                run_config[
                    "parent_baseline"
                ][
                    "baseline_config_sha256"
                ],

            "run_config_sha256":
                run_config[
                    "run_config_sha256"
                ],

            "task_ids":
                task_ids,

            "agent":
                agent_cfg,

            "user":
                user_cfg,

            "nl_judge":
                judge_cfg,

            "evaluation":
                run_config[
                    "evaluation"
                ],

            "runtime":
                runtime_cfg,

            "protocol":
                run_config[
                    "protocol"
                ],
        }

        write_json(
            manifest_path,
            manifest,
        )

    # -------------------------------------------------------------------------
    # NL Judge invocation trace
    # -------------------------------------------------------------------------

    judge_trace_path = (
        run_dir
        / "judge_call_trace.jsonl"
    )

    original_generate = (
        nl_eval.generate
    )

    def traced_generate(
        *call_args: Any,
        **call_kwargs: Any,
    ) -> Any:

        model = (
            call_kwargs.get(
                "model"
            )
        )

        if (
            model is None
            and call_args
        ):
            model = (
                call_args[0]
            )

        call_name = (
            call_kwargs.get(
                "call_name"
            )
        )

        append_jsonl(
            judge_trace_path,
            {
                "timestamp":
                    datetime.now()
                    .isoformat(),

                "event":
                    "NL_JUDGE_CALL_STARTED",

                "model":
                    model,

                "call_name":
                    call_name,
            },
        )

        try:

            result = (
                original_generate(
                    *call_args,
                    **call_kwargs,
                )
            )

            append_jsonl(
                judge_trace_path,
                {
                    "timestamp":
                        datetime.now()
                        .isoformat(),

                    "event":
                        "NL_JUDGE_CALL_SUCCEEDED",

                    "model":
                        model,

                    "call_name":
                        call_name,
                },
            )

            return result

        except Exception as exc:

            append_jsonl(
                judge_trace_path,
                {
                    "timestamp":
                        datetime.now()
                        .isoformat(),

                    "event":
                        "NL_JUDGE_CALL_FAILED",

                    "model":
                        model,

                    "call_name":
                        call_name,

                    "error_type":
                        type(
                            exc
                        ).__name__,

                    "error":
                        str(exc),
                },
            )

            raise

    nl_eval.generate = (
        traced_generate
    )

    # -------------------------------------------------------------------------
    # Start / resume
    # -------------------------------------------------------------------------

    print(
        "\n"
        + "=" * 100
    )

    print(
        "RETAIL BASELINE 20 "
        "TRIAL-1 START / RESUME"
    )

    print(
        "RUN_DIR =",
        run_dir,
    )

    print(
        "TASK_COUNT =",
        len(
            task_ids
        ),
    )

    print(
        "TASK_IDS =",
        task_ids,
    )

    print(
        "AGENT =",
        agent_cfg[
            "model"
        ],
    )

    print(
        "USER =",
        user_cfg[
            "model"
        ],
    )

    print(
        "NL_JUDGE =",
        judge_model,
    )

    print(
        "UPSTREAM =",
        upstream_commit,
    )

    print(
        "=" * 100
    )

    summaries: list[
        dict[str, Any]
    ] = []

    # =========================================================================
    # Each frozen task
    # =========================================================================

    for index, task_id in enumerate(
        task_ids,
        start=1,
    ):

        task_dir = (
            run_dir
            / f"task_{task_id}"
        )

        task_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        returned_results_path = (
            task_dir
            / "returned_results.json"
        )

        # ---------------------------------------------------------------------
        # Existing valid raw result:
        # NEVER call API again.
        # -------------------------------------------------------------------------

        if (
            returned_results_path
            .exists()
        ):

            print(
                "\n"
                + "#" * 100
            )

            print(
                f"BASELINE TASK "
                f"{index}/{len(task_ids)} "
                f"| TASK_ID={task_id}"
            )

            print(
                "REUSE EXISTING "
                "returned_results.json "
                "— NO API CALL"
            )

            print(
                "#" * 100
            )

            append_jsonl(
                task_dir
                / "attempt_history.jsonl",

                {
                    "timestamp":
                        datetime.now()
                        .isoformat(),

                    "event":
                        "REUSED_EXISTING_RESULT",

                    "task_id":
                        task_id,
                },
            )

            try:

                summary = (
                    build_task_summary(
                        task_id=
                            task_id,

                        task_dir=
                            task_dir,

                        execution_source=
                            "REUSED_EXISTING_RESULT",

                        wall_clock_seconds=
                            None,

                        system_error=
                            None,
                    )
                )

            except Exception as exc:

                summary = {
                    "task_id":
                        task_id,

                    "status":
                        "RESULT_PARSE_FAILED",

                    "execution_source":
                        "REUSED_EXISTING_RESULT",

                    "reward":
                        None,

                    "error_type":
                        type(
                            exc
                        ).__name__,

                    "error_message":
                        str(exc),

                    "traceback":
                        traceback
                        .format_exc(),
                }

                write_json(
                    task_dir
                    / "parser_error.json",

                    summary,
                )

            write_json(
                task_dir
                / "summary.json",

                summary,
            )

            summaries.append(
                summary
            )

            print(
                "STANDARDIZED SUMMARY:"
            )

            print(
                json.dumps(
                    summary,
                    ensure_ascii=False,
                    indent=2,
                    default=str,
                )
            )

            # Rebuild partial aggregate
            progress = (
                build_aggregate(
                    run_id=
                        run_id,

                    run_config=
                        run_config,

                    task_ids=
                        task_ids,

                    summaries=
                        summaries,
                )
            )

            write_json(
                run_dir
                / "progress.json",

                progress,
            )

            continue

        # ---------------------------------------------------------------------
        # Task has never produced a valid raw result.
        # Execute exactly once in this continuation.
        # -------------------------------------------------------------------------

        print(
            "\n"
            + "#" * 100
        )

        print(
            f"BASELINE TASK "
            f"{index}/{len(task_ids)} "
            f"| TASK_ID={task_id}"
        )

        print(
            "#" * 100
        )

        tasks = get_tasks(
            "retail",
            task_ids=[
                task_id
            ],
        )

        assert len(
            tasks
        ) == 1

        assert (
            str(
                tasks[0].id
            )
            == task_id
        )

        # Exact task snapshot
        write_json(
            task_dir
            / "task.json",

            model_to_dict(
                tasks[0]
            ),
        )

        append_jsonl(
            task_dir
            / "attempt_history.jsonl",

            {
                "timestamp":
                    datetime.now()
                    .isoformat(),

                "event":
                    "EXECUTION_STARTED",

                "task_id":
                    task_id,
            },
        )

        wall_start = (
            time.perf_counter()
        )

        system_error: (
            dict[str, Any]
            | None
        ) = None

        execution_succeeded = (
            False
        )

        try:

            results = run_tasks(
                domain="retail",

                tasks=tasks,

                agent=
                    agent_cfg[
                        "implementation"
                    ],

                user=
                    user_cfg[
                        "implementation"
                    ],

                llm_agent=
                    agent_cfg[
                        "model"
                    ],

                llm_args_agent={
                    "temperature":
                        agent_cfg[
                            "temperature"
                        ]
                },

                llm_user=
                    user_cfg[
                        "model"
                    ],

                llm_args_user={
                    "temperature":
                        user_cfg[
                            "temperature"
                        ]
                },

                num_trials=
                    runtime_cfg[
                        "num_trials"
                    ],

                max_steps=
                    runtime_cfg[
                        "max_steps"
                    ],

                max_errors=
                    runtime_cfg[
                        "max_errors"
                    ],

                save_dir=
                    task_dir
                    / "tau2_artifacts",

                console_display=True,

                evaluation_type=
                    EvaluationType
                    .ALL_WITH_NL_ASSERTIONS,

                max_concurrency=
                    runtime_cfg[
                        "max_concurrency"
                    ],

                seed=
                    runtime_cfg[
                        "seed"
                    ],

                log_level="INFO",

                verbose_logs=
                    runtime_cfg[
                        "verbose_logs"
                    ],

                max_retries=
                    runtime_cfg[
                        "max_retries"
                    ],

                auto_resume=False,

                auto_review=
                    runtime_cfg[
                        "auto_review"
                    ],
            )

            # IMPORTANT:
            # Persist raw result BEFORE any project-side parsing.
            write_json(
                returned_results_path,

                model_to_dict(
                    results
                ),
            )

            execution_succeeded = (
                True
            )

            append_jsonl(
                task_dir
                / "attempt_history.jsonl",

                {
                    "timestamp":
                        datetime.now()
                        .isoformat(),

                    "event":
                        "EXECUTION_SUCCEEDED",

                    "task_id":
                        task_id,
                },
            )

        except Exception as exc:

            system_error = {
                "type":
                    type(
                        exc
                    ).__name__,

                "message":
                    str(exc),

                "traceback":
                    traceback
                    .format_exc(),

                "timestamp":
                    datetime.now()
                    .isoformat(),
            }

            write_json(
                task_dir
                / "error.json",

                system_error,
            )

            append_jsonl(
                task_dir
                / "attempt_history.jsonl",

                {
                    "timestamp":
                        datetime.now()
                        .isoformat(),

                    "event":
                        "SYSTEM_FAILURE",

                    "task_id":
                        task_id,

                    "error_type":
                        system_error[
                            "type"
                        ],

                    "error_message":
                        system_error[
                            "message"
                        ],
                },
            )

        wall_clock_seconds = (
            time.perf_counter()
            - wall_start
        )

        # ---------------------------------------------------------------------
        # Parse raw result independently from execution.
        #
        # Even if parser fails, the next frozen task still runs.
        # -------------------------------------------------------------------------

        try:

            summary = (
                build_task_summary(
                    task_id=
                        task_id,

                    task_dir=
                        task_dir,

                    execution_source=
                        (
                            "EXECUTED_NOW"
                            if execution_succeeded
                            else "SYSTEM_FAILED"
                        ),

                    wall_clock_seconds=
                        wall_clock_seconds,

                    system_error=
                        system_error,
                )
            )

        except Exception as exc:

            summary = {
                "task_id":
                    task_id,

                "status":
                    (
                        "RESULT_PARSE_FAILED"
                        if returned_results_path.exists()
                        else "SYSTEM_FAILED"
                    ),

                "execution_source":
                    (
                        "EXECUTED_NOW"
                        if execution_succeeded
                        else "SYSTEM_FAILED"
                    ),

                "reward":
                    None,

                "wall_clock_seconds":
                    round(
                        wall_clock_seconds,
                        3,
                    ),

                "error_type":
                    type(
                        exc
                    ).__name__,

                "error_message":
                    str(exc),

                "traceback":
                    traceback
                    .format_exc(),
            }

            write_json(
                task_dir
                / "parser_error.json",

                summary,
            )

        write_json(
            task_dir
            / "summary.json",

            summary,
        )

        summaries.append(
            summary
        )

        print(
            "\nSTANDARDIZED TASK SUMMARY:"
        )

        print(
            json.dumps(
                summary,
                ensure_ascii=False,
                indent=2,
                default=str,
            )
        )

        # ---------------------------------------------------------------------
        # Persist progress after EVERY task.
        # -------------------------------------------------------------------------

        progress = (
            build_aggregate(
                run_id=
                    run_id,

                run_config=
                    run_config,

                task_ids=
                    task_ids,

                summaries=
                    summaries,
            )
        )

        write_json(
            run_dir
            / "progress.json",

            progress,
        )

    # =========================================================================
    # Final aggregate
    # =========================================================================

    aggregate = (
        build_aggregate(
            run_id=
                run_id,

            run_config=
                run_config,

            task_ids=
                task_ids,

            summaries=
                summaries,
        )
    )

    write_json(
        run_dir
        / "baseline_summary.json",

        aggregate,
    )

    write_baseline_csv(
        run_dir
        / "baseline_table.csv",

        summaries,
    )

    # -------------------------------------------------------------------------
    # Final terminal output
    # -------------------------------------------------------------------------

    print(
        "\n"
        + "=" * 100
    )

    print(
        "RETAIL BASELINE 20 "
        "TRIAL-1 FINISHED"
    )

    print(
        "RUN_DIR =",
        run_dir,
    )

    print(
        "TASK_COUNT =",
        aggregate[
            "task_count"
        ],
    )

    print(
        "COMPLETED =",
        aggregate[
            "completed_count"
        ],
    )

    print(
        "SYSTEM_FAILED =",
        aggregate[
            "system_failed_count"
        ],
    )

    print(
        "RESULT_PARSE_FAILED =",
        aggregate[
            "result_parse_failed_count"
        ],
    )

    print(
        "VALID_REWARDS =",
        aggregate[
            "valid_reward_count"
        ],
    )

    print(
        "BUSINESS_SUCCESS =",
        aggregate[
            "business_success_count"
        ],
    )

    print(
        "BUSINESS_FAILURE =",
        aggregate[
            "business_failure_count"
        ],
    )

    print(
        "SUCCESS_RATE =",
        aggregate[
            "success_rate"
        ],
    )

    print(
        "MEAN_REWARD =",
        aggregate[
            "mean_reward"
        ],
    )

    print(
        "TOTAL_AGENT_COST_USD =",
        round(
            aggregate[
                "total_agent_cost_usd"
            ],
            8,
        ),
    )

    print(
        "TOTAL_USER_COST_USD =",
        round(
            aggregate[
                "total_user_cost_usd"
            ],
            8,
        ),
    )

    print(
        "TOTAL_JUDGE_COST_USD =",
        round(
            aggregate[
                "total_judge_cost_usd"
            ],
            8,
        ),
    )

    print(
        "TOTAL_MODEL_COST_USD =",
        round(
            aggregate[
                "total_model_cost_usd"
            ],
            8,
        ),
    )

    print(
        "TOTAL_JUDGE_CALLS =",
        aggregate[
            "total_judge_calls"
        ],
    )

    print(
        "TOTAL_SIMULATION_DURATION_SECONDS =",
        round(
            aggregate[
                "total_simulation_duration_seconds"
            ],
            3,
        ),
    )

    print(
        "BASELINE_JSON =",
        run_dir
        / "baseline_summary.json",
    )

    print(
        "BASELINE_CSV =",
        run_dir
        / "baseline_table.csv",
    )

    if (
        aggregate[
            "completed_count"
        ] == 20
        and aggregate[
            "system_failed_count"
        ] == 0
        and aggregate[
            "result_parse_failed_count"
        ] == 0
        and aggregate[
            "valid_reward_count"
        ] == 20
    ):

        print(
            "BASELINE20_TRIAL1_RUN_OK"
        )

    else:

        print(
            "BASELINE20_TRIAL1_RUN_INCOMPLETE"
        )

    print(
        "=" * 100
    )


if __name__ == "__main__":
    main()