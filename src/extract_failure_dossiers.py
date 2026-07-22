"""
Extract detailed dossiers for failed Retail baseline tasks.

Purpose
-------
For each failed task, collect:
- frozen task definition
- evaluation criteria
- reward breakdown
- action checks
- message / trajectory history
- tool-related events
- NL assertion results
- basic evidence for manual root-cause analysis

Offline only.
No LLM API calls.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


RUN_DIR = Path(
    r"D:\PolicyAgent-PostTrain\experiments"
    r"\20260722_110504_retail_baseline20_trial1_deepseek"
)

FAILURE_IDS = [
    "59",
    "98",
    "95",
    "107",
]

OUTPUT_DIR = (
    RUN_DIR
    / "failure_dossiers"
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


def get_first_simulation(
    result: dict[str, Any],
) -> dict[str, Any]:

    simulations = (
        result.get(
            "simulations"
        )
        or []
    )

    if not simulations:

        raise ValueError(
            "No simulations found."
        )

    simulation = (
        simulations[0]
    )

    if not isinstance(
        simulation,
        dict,
    ):

        raise TypeError(
            "Simulation is not a dict."
        )

    return simulation


def compact_message(
    message: Any,
    index: int,
) -> dict[str, Any]:

    if not isinstance(
        message,
        dict,
    ):

        return {
            "index":
                index,

            "raw":
                message,
        }

    result = {
        "index":
            index,
    }

    # Keep common fields without assuming one exact schema.
    for key in (
        "role",
        "sender",
        "recipient",
        "content",
        "text",
        "tool_name",
        "tool_call",
        "tool_calls",
        "tool_result",
        "function",
        "name",
        "arguments",
        "cost",
    ):

        if key in message:

            result[key] = (
                message[key]
            )

    # Preserve full raw message for auditability.
    result["raw"] = message

    return result


def extract_task_dossier(
    task_id: str,
) -> dict[str, Any]:

    task_dir = (
        RUN_DIR
        / f"task_{task_id}"
    )

    task_path = (
        task_dir
        / "task.json"
    )

    result_path = (
        task_dir
        / "returned_results.json"
    )

    summary_path = (
        task_dir
        / "summary.json"
    )

    action_checks_path = (
        task_dir
        / "action_checks.json"
    )

    nl_assertions_path = (
        task_dir
        / "nl_assertions.json"
    )

    task = (
        load_json(
            task_path
        )
    )

    result = (
        load_json(
            result_path
        )
    )

    summary = (
        load_json(
            summary_path
        )
    )

    simulation = (
        get_first_simulation(
            result
        )
    )

    reward_info = (
        simulation.get(
            "reward_info"
        )
        or {}
    )

    messages = (
        simulation.get(
            "messages"
        )
        or []
    )

    compact_messages = [
        compact_message(
            message,
            index,
        )
        for index, message
        in enumerate(
            messages
        )
    ]

    action_checks = []

    if action_checks_path.exists():

        action_checks = (
            load_json(
                action_checks_path
            )
        )

    elif isinstance(
        reward_info,
        dict,
    ):

        action_checks = (
            reward_info.get(
                "action_checks"
            )
            or []
        )

    nl_assertions = []

    if nl_assertions_path.exists():

        nl_assertions = (
            load_json(
                nl_assertions_path
            )
        )

    elif isinstance(
        reward_info,
        dict,
    ):

        nl_assertions = (
            reward_info.get(
                "nl_assertions"
            )
            or []
        )

    dossier = {
        "task_id":
            task_id,

        "baseline_result": {
            "reward":
                summary.get(
                    "reward"
                ),

            "db_reward":
                summary.get(
                    "db_reward"
                ),

            "nl_assertion_reward":
                summary.get(
                    "nl_assertion_reward"
                ),

            "action":
                (
                    f"{summary.get('action_passed')}"
                    f"/{summary.get('action_total')}"
                ),

            "read":
                (
                    f"{summary.get('read_passed')}"
                    f"/{summary.get('read_total')}"
                ),

            "write":
                (
                    f"{summary.get('write_passed')}"
                    f"/{summary.get('write_total')}"
                ),

            "generic":
                (
                    f"{summary.get('generic_passed')}"
                    f"/{summary.get('generic_total')}"
                ),

            "termination_reason":
                summary.get(
                    "termination_reason"
                ),

            "duration_seconds":
                summary.get(
                    "simulation_duration_seconds"
                ),

            "total_model_cost_usd":
                summary.get(
                    "total_model_cost_usd"
                ),
        },

        "task_definition":
            task,

        "evaluation_criteria":
            task.get(
                "evaluation_criteria"
            )
            if isinstance(
                task,
                dict,
            )
            else None,

        "reward_info":
            reward_info,

        "action_checks":
            action_checks,

        "nl_assertions":
            nl_assertions,

        "trajectory_messages":
            compact_messages,

        "raw_paths": {
            "task":
                str(
                    task_path
                ),

            "result":
                str(
                    result_path
                ),

            "summary":
                str(
                    summary_path
                ),
        },

        "manual_analysis": {
            "user_final_intent":
                None,

            "expected_critical_actions":
                [],

            "actual_critical_actions":
                [],

            "first_divergence_point":
                None,

            "failure_type":
                None,

            "root_cause":
                None,

            "policy_issue":
                None,

            "tool_issue":
                None,

            "evaluator_issue":
                None,

            "claim_action_consistency":
                None,

            "training_value":
                None,

            "candidate_intervention":
                None,
        },
    }

    return dossier


def print_action_checks(
    dossier: dict[str, Any],
) -> None:

    print(
        "ACTION CHECKS:"
    )

    checks = (
        dossier.get(
            "action_checks"
        )
        or []
    )

    for index, check in enumerate(
        checks
    ):

        if not isinstance(
            check,
            dict,
        ):

            print(
                f"  [{index}]",
                check,
            )

            continue

        action = (
            check.get(
                "action"
            )
            or {}
        )

        if not isinstance(
            action,
            dict,
        ):

            action = {}

        print(
            f"  [{index}] "
            f"name={action.get('name')} "
            f"type={check.get('tool_type')} "
            f"reward={check.get('action_reward')} "
            f"match={check.get('action_match')}"
        )

        arguments = (
            action.get(
                "arguments"
            )
        )

        if arguments:

            print(
                "       args =",
                json.dumps(
                    arguments,
                    ensure_ascii=False,
                ),
            )


def main() -> None:

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    all_dossiers = []

    print(
        "=== FAILURE DOSSIER EXTRACTION ==="
    )

    for task_id in FAILURE_IDS:

        dossier = (
            extract_task_dossier(
                task_id
            )
        )

        all_dossiers.append(
            dossier
        )

        output_path = (
            OUTPUT_DIR
            / f"task_{task_id}_dossier.json"
        )

        write_json(
            output_path,
            dossier,
        )

        baseline = (
            dossier[
                "baseline_result"
            ]
        )

        print(
            "\n"
            + "=" * 80
        )

        print(
            "TASK",
            task_id,
        )

        print(
            "Reward =",
            baseline[
                "reward"
            ],
        )

        print(
            "DB =",
            baseline[
                "db_reward"
            ],
        )

        print(
            "NL =",
            baseline[
                "nl_assertion_reward"
            ],
        )

        print(
            "Action =",
            baseline[
                "action"
            ],
        )

        print(
            "Read =",
            baseline[
                "read"
            ],
        )

        print(
            "Write =",
            baseline[
                "write"
            ],
        )

        print_action_checks(
            dossier
        )

        print(
            "DOSSIER =",
            output_path,
        )

    write_json(
        OUTPUT_DIR
        / "all_failure_dossiers.json",

        {
            "source_run":
                str(
                    RUN_DIR
                ),

            "failure_task_ids":
                FAILURE_IDS,

            "dossiers":
                all_dossiers,
        },
    )

    print(
        "\n"
        + "=" * 80
    )

    print(
        "OUTPUT_DIR =",
        OUTPUT_DIR,
    )

    print(
        "FAILURE_DOSSIER_EXTRACTION_OK"
    )


if __name__ == "__main__":
    main()