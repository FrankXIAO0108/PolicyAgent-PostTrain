"""
Generate human-readable audit files for suspicious Reward=1 trajectories.

Purpose
-------
A benchmark reward of 1 does not automatically mean the trajectory is
high-quality enough for SFT Gold Data.

This script selects suspicious successful trajectories whose action-level
checks are not perfect, and exports one UTF-8 review file per task.

Priority tasks:
- 46: action 4/7
- 16: action 8/9
- 21: action 11/12
- 28: action 10/11

Offline only.
No API calls.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


# =============================================================================
# Paths
# =============================================================================

PROJECT_ROOT = Path(
    r"D:\PolicyAgent-PostTrain"
)

RUN_DIR = (
    PROJECT_ROOT
    / "experiments"
    / "20260722_110504_retail_baseline20_trial1_deepseek"
)

OUTPUT_DIR = (
    RUN_DIR
    / "success_audit_reviews"
)

BASELINE_SUMMARY_PATH = (
    RUN_DIR
    / "baseline_summary.json"
)


# =============================================================================
# Audit targets
# =============================================================================

AUDIT_TASK_IDS = [
    "46",
    "16",
    "21",
    "28",
]


# =============================================================================
# Helpers
# =============================================================================

def load_json(
    path: Path,
) -> Any:

    return json.loads(
        path.read_text(
            encoding="utf-8"
        )
    )


def shorten(
    value: Any,
    limit: int = 8000,
) -> str:

    if value is None:
        return ""

    if isinstance(
        value,
        (dict, list),
    ):

        text = json.dumps(
            value,
            ensure_ascii=False,
            indent=2,
            default=str,
        )

    else:

        text = str(
            value
        )

    if len(text) > limit:

        return (
            text[:limit]
            + "\n... [TRUNCATED]"
        )

    return text


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


def find_task_summary(
    baseline: dict[str, Any],
    task_id: str,
) -> dict[str, Any]:

    tasks = (
        baseline.get(
            "tasks"
        )
        or []
    )

    for task in tasks:

        if str(
            task.get(
                "task_id"
            )
        ) == task_id:

            return task

    raise KeyError(
        f"Task {task_id} not found "
        f"in baseline_summary.json"
    )


def detect_message_type(
    raw: dict[str, Any],
) -> str:

    role = str(
        raw.get(
            "role",
            ""
        )
    ).lower()

    sender = str(
        raw.get(
            "sender",
            ""
        )
    ).lower()

    combined = (
        role
        + " "
        + sender
    )

    if "user" in combined:
        return "USER"

    if (
        "assistant" in combined
        or "agent" in combined
    ):
        return "AGENT"

    if "tool" in combined:
        return "TOOL"

    return "MESSAGE"


def extract_primary_content(
    raw: dict[str, Any],
) -> Any:

    for key in (
        "content",
        "text",
        "message",
    ):

        value = raw.get(
            key
        )

        if value not in (
            None,
            "",
            [],
            {},
        ):

            return value

    return None


def extract_tool_calls(
    raw: dict[str, Any],
) -> list[Any]:

    calls: list[Any] = []

    for key in (
        "tool_calls",
        "tool_call",
        "function_call",
    ):

        value = raw.get(
            key
        )

        if not value:
            continue

        if isinstance(
            value,
            list,
        ):

            calls.extend(
                value
            )

        else:

            calls.append(
                value
            )

    return calls


def extract_tool_result(
    raw: dict[str, Any],
) -> Any:

    for key in (
        "tool_result",
        "result",
        "observation",
    ):

        value = raw.get(
            key
        )

        if value not in (
            None,
            "",
            [],
            {},
        ):

            return value

    return None


def extract_action_checks(
    simulation: dict[str, Any],
) -> list[Any]:

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

        return []

    return (
        reward_info.get(
            "action_checks"
        )
        or []
    )


def extract_nl_assertions(
    simulation: dict[str, Any],
) -> list[Any]:

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

        return []

    return (
        reward_info.get(
            "nl_assertions"
        )
        or []
    )


# =============================================================================
# Report builders
# =============================================================================

def build_baseline_section(
    summary: dict[str, Any],
) -> list[str]:

    return [
        "=== BASELINE RESULT ===",
        "",
        f"Reward = {summary.get('reward')}",
        f"DB = {summary.get('db_reward')}",
        f"NL = {summary.get('nl_assertion_reward')}",
        (
            "Action = "
            f"{summary.get('action_passed')}"
            f"/{summary.get('action_total')}"
        ),
        (
            "Read = "
            f"{summary.get('read_passed')}"
            f"/{summary.get('read_total')}"
        ),
        (
            "Write = "
            f"{summary.get('write_passed')}"
            f"/{summary.get('write_total')}"
        ),
        (
            "Generic = "
            f"{summary.get('generic_passed')}"
            f"/{summary.get('generic_total')}"
        ),
        (
            "Termination Reason = "
            f"{summary.get('termination_reason')}"
        ),
        "",
        "AUDIT QUESTION:",
        (
            "Why did this trajectory receive Reward=1 "
            "despite imperfect action-level checks?"
        ),
        "",
    ]


def build_task_definition_section(
    task: Any,
) -> list[str]:

    return [
        "=== TASK DEFINITION ===",
        "",
        shorten(
            task,
            limit=15000,
        ),
        "",
    ]


def build_action_checks_section(
    checks: list[Any],
) -> list[str]:

    lines = [
        "=== ACTION CHECKS ===",
        "",
    ]

    if not checks:

        lines.extend([
            "NO ACTION CHECKS",
            "",
        ])

        return lines

    for index, check in enumerate(
        checks
    ):

        if not isinstance(
            check,
            dict,
        ):

            lines.extend([
                f"[{index}]",
                shorten(
                    check
                ),
                "",
            ])

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

        lines.append(
            (
                f"[{index}] "
                f"name={action.get('name')} "
                f"| type={check.get('tool_type')} "
                f"| reward={check.get('action_reward')} "
                f"| match={check.get('action_match')}"
            )
        )

        lines.append(
            "ARGS:"
        )

        lines.append(
            shorten(
                action.get(
                    "arguments"
                ),
                limit=5000,
            )
        )

        lines.append("")

    return lines


def build_trajectory_section(
    simulation: dict[str, Any],
) -> list[str]:

    lines = [
        "=== ACTUAL TRAJECTORY ===",
        "",
    ]

    messages = (
        simulation.get(
            "messages"
        )
        or []
    )

    if not messages:

        lines.extend([
            "NO TRAJECTORY MESSAGES FOUND",
            "",
        ])

        return lines

    for index, raw in enumerate(
        messages
    ):

        lines.append(
            "-" * 90
        )

        if not isinstance(
            raw,
            dict,
        ):

            lines.extend([
                f"[{index}] MESSAGE",
                shorten(
                    raw
                ),
                "",
            ])

            continue

        message_type = (
            detect_message_type(
                raw
            )
        )

        lines.append(
            (
                f"[{index}] {message_type}"
                f" | role={raw.get('role')}"
                f" | sender={raw.get('sender')}"
                f" | recipient={raw.get('recipient')}"
            )
        )

        content = (
            extract_primary_content(
                raw
            )
        )

        tool_calls = (
            extract_tool_calls(
                raw
            )
        )

        tool_result = (
            extract_tool_result(
                raw
            )
        )

        if content is not None:

            lines.append(
                "CONTENT:"
            )

            lines.append(
                shorten(
                    content,
                    limit=10000,
                )
            )

        if tool_calls:

            lines.append(
                "TOOL CALL:"
            )

            for call in tool_calls:

                lines.append(
                    shorten(
                        call,
                        limit=8000,
                    )
                )

        if tool_result is not None:

            lines.append(
                "TOOL RESULT:"
            )

            lines.append(
                shorten(
                    tool_result,
                    limit=10000,
                )
            )

        if (
            content is None
            and not tool_calls
            and tool_result is None
        ):

            lines.append(
                "RAW:"
            )

            lines.append(
                shorten(
                    raw,
                    limit=12000,
                )
            )

        lines.append("")

    return lines


def build_nl_section(
    assertions: list[Any],
) -> list[str]:

    lines = [
        "=== NL ASSERTIONS ===",
        "",
    ]

    if not assertions:

        lines.extend([
            "NO NL ASSERTIONS",
            "",
        ])

        return lines

    for index, assertion in enumerate(
        assertions
    ):

        lines.extend([
            f"[{index}]",
            shorten(
                assertion,
                limit=8000,
            ),
            "",
        ])

    return lines


def build_manual_template(
    task_id: str,
) -> list[str]:

    return [
        "=== 成功轨迹人工质量审计 ===",
        "",
        f"任务 ID：{task_id}",
        "",
        "1. 用户最终意图：",
        "",
        "",
        "2. Reward=1 的直接依据：",
        "",
        "",
        "3. 未通过的 Action Check：",
        "",
        "",
        "4. 未通过检查的真实原因：",
        "",
        "",
        "5. 是否存在真实业务错误：",
        "",
        "",
        "6. 是否存在 Policy / Tool / Evaluator 覆盖缺口：",
        "",
        "",
        "7. Claim-Action Consistency：",
        "",
        "",
        "8. 是否存在冗余、错误或危险 Tool Call：",
        "",
        "",
        "9. 轨迹质量等级：",
        "",
        "",
        "10. 是否可进入 SFT Gold Pool：",
        "",
        "",
        "11. 是否需要修正后再进入训练集：",
        "",
        "",
        "12. 最终标签与候选用途：",
        "",
        "",
    ]


def build_report(
    task_id: str,
    summary: dict[str, Any],
    task: Any,
    simulation: dict[str, Any],
) -> str:

    lines: list[str] = [
        "=" * 100,
        f"TASK {task_id} SUCCESS TRAJECTORY QUALITY AUDIT",
        "=" * 100,
        "",
    ]

    lines.extend(
        build_baseline_section(
            summary
        )
    )

    lines.extend(
        build_task_definition_section(
            task
        )
    )

    lines.extend(
        build_action_checks_section(
            extract_action_checks(
                simulation
            )
        )
    )

    lines.extend(
        build_trajectory_section(
            simulation
        )
    )

    lines.extend(
        build_nl_section(
            extract_nl_assertions(
                simulation
            )
        )
    )

    lines.extend(
        build_manual_template(
            task_id
        )
    )

    return "\n".join(
        lines
    )


# =============================================================================
# Main
# =============================================================================

def main() -> None:

    baseline = (
        load_json(
            BASELINE_SUMMARY_PATH
        )
    )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    manifest_tasks = []

    for task_id in AUDIT_TASK_IDS:

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

        if not task_path.exists():

            raise FileNotFoundError(
                f"Missing task file: "
                f"{task_path}"
            )

        if not result_path.exists():

            raise FileNotFoundError(
                f"Missing result file: "
                f"{result_path}"
            )

        summary = (
            find_task_summary(
                baseline,
                task_id,
            )
        )

        if (
            summary.get(
                "reward"
            )
            != 1.0
        ):

            raise ValueError(
                f"Task {task_id} is not "
                f"a Reward=1 trajectory."
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

        simulation = (
            get_first_simulation(
                result
            )
        )

        report = (
            build_report(
                task_id=task_id,
                summary=summary,
                task=task,
                simulation=simulation,
            )
        )

        output_path = (
            OUTPUT_DIR
            / f"task_{task_id}_success_review.txt"
        )

        output_path.write_text(
            report,
            encoding="utf-8",
        )

        manifest_tasks.append({
            "task_id":
                task_id,

            "reward":
                summary.get(
                    "reward"
                ),

            "action_passed":
                summary.get(
                    "action_passed"
                ),

            "action_total":
                summary.get(
                    "action_total"
                ),

            "review_file":
                str(
                    output_path
                ),
        })

    manifest = {
        "source_run":
            str(
                RUN_DIR
            ),

        "audit_role":
            (
                "Reward=1 suspicious "
                "trajectory quality audit"
            ),

        "task_ids":
            AUDIT_TASK_IDS,

        "tasks":
            manifest_tasks,
    }

    (
        OUTPUT_DIR
        / "success_audit_manifest.json"
    ).write_text(
        json.dumps(
            manifest,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    # ASCII-only output for Windows console compatibility.
    print(
        "SUCCESS_AUDIT_REVIEWS_CREATED"
    )

    for item in manifest_tasks:

        print(
            "TASK",
            item[
                "task_id"
            ],
            "ACTION",
            (
                f"{item['action_passed']}"
                f"/{item['action_total']}"
            ),
        )

        print(
            "OUTPUT =",
            item[
                "review_file"
            ],
        )

    print(
        "SUCCESS_AUDIT_REVIEW_OK"
    )


if __name__ == "__main__":
    main()