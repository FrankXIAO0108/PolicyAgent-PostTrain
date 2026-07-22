"""
Analyze a completed Retail 20-task baseline run.

Outputs:
1. Overall baseline metrics.
2. Failed task IDs.
3. Per-task diagnostic table.
4. Failure shortlist JSON for later manual badcase analysis.

This script performs offline analysis only.
It does not call any LLM API.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


RUN_DIR = Path(
    r"D:\PolicyAgent-PostTrain\experiments"
    r"\20260722_110504_retail_baseline20_trial1_deepseek"
)

SUMMARY_PATH = (
    RUN_DIR
    / "baseline_summary.json"
)

FAILURE_OUTPUT_PATH = (
    RUN_DIR
    / "baseline_failure_shortlist.json"
)


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
    path.write_text(
        json.dumps(
            data,
            ensure_ascii=False,
            indent=2,
            default=str,
        ),
        encoding="utf-8",
    )


def ratio(
    passed: Any,
    total: Any,
) -> str:

    try:
        passed = int(passed or 0)
        total = int(total or 0)

    except (
        TypeError,
        ValueError,
    ):
        return "-"

    if total == 0:
        return "0/0"

    return f"{passed}/{total}"


def main() -> None:

    baseline = load_json(
        SUMMARY_PATH
    )

    tasks = baseline.get(
        "tasks",
        [],
    )

    assert (
        baseline.get(
            "task_count"
        )
        == 20
    )

    assert len(tasks) == 20

    # -------------------------------------------------------------------------
    # Overall metrics
    # -------------------------------------------------------------------------

    print(
        "=== BASELINE OVERALL ==="
    )

    print(
        "TASK_COUNT =",
        baseline.get(
            "task_count"
        ),
    )

    print(
        "COMPLETED =",
        baseline.get(
            "completed_count"
        ),
    )

    print(
        "SYSTEM_FAILED =",
        baseline.get(
            "system_failed_count"
        ),
    )

    print(
        "RESULT_PARSE_FAILED =",
        baseline.get(
            "result_parse_failed_count"
        ),
    )

    print(
        "SUCCESS =",
        baseline.get(
            "business_success_count"
        ),
    )

    print(
        "FAILURE =",
        baseline.get(
            "business_failure_count"
        ),
    )

    print(
        "SUCCESS_RATE =",
        baseline.get(
            "success_rate"
        ),
    )

    print(
        "TOTAL_MODEL_COST_USD =",
        baseline.get(
            "total_model_cost_usd"
        ),
    )

    # -------------------------------------------------------------------------
    # Per-task concise table
    # -------------------------------------------------------------------------

    print(
        "\n=== PER-TASK RESULTS ==="
    )

    for task in tasks:

        print(
            "TASK "
            f"{str(task.get('task_id')):>3}"
            " | "
            f"reward={task.get('reward')}"
            " | "
            f"DB={task.get('db_reward')}"
            " | "
            f"NL={task.get('nl_assertion_reward')}"
            " | "
            "action="
            f"{ratio(task.get('action_passed'), task.get('action_total'))}"
            " | "
            "read="
            f"{ratio(task.get('read_passed'), task.get('read_total'))}"
            " | "
            "write="
            f"{ratio(task.get('write_passed'), task.get('write_total'))}"
            " | "
            "generic="
            f"{ratio(task.get('generic_passed'), task.get('generic_total'))}"
            " | "
            f"duration={task.get('simulation_duration_seconds')}"
            " | "
            f"cost=${float(task.get('total_model_cost_usd') or 0):.6f}"
        )

    # -------------------------------------------------------------------------
    # Failure shortlist
    # -------------------------------------------------------------------------

    failures = [
        task
        for task in tasks
        if task.get(
            "status"
        ) == "COMPLETED"
        and task.get(
            "reward"
        ) != 1.0
    ]

    print(
        "\n=== BUSINESS FAILURES ==="
    )

    print(
        "FAILURE_COUNT =",
        len(failures),
    )

    print(
        "FAILURE_TASK_IDS =",
        [
            str(
                task.get(
                    "task_id"
                )
            )
            for task
            in failures
        ],
    )

    failure_shortlist = []

    for task in failures:

        task_id = str(
            task.get(
                "task_id"
            )
        )

        task_dir = (
            RUN_DIR
            / f"task_{task_id}"
        )

        task_snapshot = {}

        task_path = (
            task_dir
            / "task.json"
        )

        if task_path.exists():

            task_snapshot = (
                load_json(
                    task_path
                )
            )

        item = {
            "task_id":
                task_id,

            "reward":
                task.get(
                    "reward"
                ),

            "db_reward":
                task.get(
                    "db_reward"
                ),

            "nl_assertion_reward":
                task.get(
                    "nl_assertion_reward"
                ),

            "action_passed":
                task.get(
                    "action_passed"
                ),

            "action_total":
                task.get(
                    "action_total"
                ),

            "read_passed":
                task.get(
                    "read_passed"
                ),

            "read_total":
                task.get(
                    "read_total"
                ),

            "write_passed":
                task.get(
                    "write_passed"
                ),

            "write_total":
                task.get(
                    "write_total"
                ),

            "generic_passed":
                task.get(
                    "generic_passed"
                ),

            "generic_total":
                task.get(
                    "generic_total"
                ),

            "termination_reason":
                task.get(
                    "termination_reason"
                ),

            "simulation_duration_seconds":
                task.get(
                    "simulation_duration_seconds"
                ),

            "total_model_cost_usd":
                task.get(
                    "total_model_cost_usd"
                ),

            "task_snapshot_path":
                str(
                    task_path
                ),

            "returned_results_path":
                str(
                    task_dir
                    / "returned_results.json"
                ),

            "task_snapshot":
                task_snapshot,

            "manual_review_status":
                "PENDING",

            "failure_type":
                None,

            "root_cause":
                None,

            "evaluator_suspect":
                None,

            "training_value":
                None,
        }

        failure_shortlist.append(
            item
        )

        print(
            "\n"
            f"TASK {task_id}"
        )

        print(
            "  Reward =",
            task.get(
                "reward"
            ),
        )

        print(
            "  DB =",
            task.get(
                "db_reward"
            ),
        )

        print(
            "  NL =",
            task.get(
                "nl_assertion_reward"
            ),
        )

        print(
            "  Action =",
            ratio(
                task.get(
                    "action_passed"
                ),
                task.get(
                    "action_total"
                ),
            ),
        )

        print(
            "  Read =",
            ratio(
                task.get(
                    "read_passed"
                ),
                task.get(
                    "read_total"
                ),
            ),
        )

        print(
            "  Write =",
            ratio(
                task.get(
                    "write_passed"
                ),
                task.get(
                    "write_total"
                ),
            ),
        )

        print(
            "  Generic =",
            ratio(
                task.get(
                    "generic_passed"
                ),
                task.get(
                    "generic_total"
                ),
            ),
        )

    write_json(
        FAILURE_OUTPUT_PATH,
        {
            "source_run":
                str(
                    RUN_DIR
                ),

            "baseline_success_rate":
                baseline.get(
                    "success_rate"
                ),

            "failure_count":
                len(
                    failure_shortlist
                ),

            "failure_task_ids":
                [
                    item[
                        "task_id"
                    ]
                    for item
                    in failure_shortlist
                ],

            "failures":
                failure_shortlist,
        },
    )

    print(
        "\nFAILURE_SHORTLIST_OUTPUT =",
        FAILURE_OUTPUT_PATH,
    )

    print(
        "\nBASELINE_ANALYSIS_OK"
    )


if __name__ == "__main__":
    main()