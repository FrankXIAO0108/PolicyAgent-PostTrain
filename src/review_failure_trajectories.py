"""
Generate human-readable trajectory review files for failed baseline tasks.

Purpose
-------
For each failed task, generate a separate UTF-8 text report containing:

1. Baseline result
2. Expected / golden actions
3. Actual User-Agent-Tool trajectory
4. NL assertion evaluation

The reports are designed for manual First Divergence Point analysis.

Offline only.
No LLM API calls.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


# =============================================================================
# Paths
# =============================================================================

RUN_DIR = Path(
    r"D:\PolicyAgent-PostTrain\experiments"
    r"\20260722_110504_retail_baseline20_trial1_deepseek"
)

DOSSIER_DIR = (
    RUN_DIR
    / "failure_dossiers"
)

OUTPUT_DIR = (
    RUN_DIR
    / "failure_reviews"
)

FAILURE_IDS = [
    "59",
    "98",
    "95",
    "107",
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
    limit: int = 5000,
) -> str:
    """
    Convert arbitrary value to readable text.

    Keep enough information for manual trajectory audit,
    while preventing pathological single fields from producing
    extremely large reports.
    """

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

    class_name = str(
        raw.get(
            "__class__",
            ""
        )
    ).lower()

    combined = (
        role
        + " "
        + sender
        + " "
        + class_name
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

    values: list[Any] = []

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

            values.extend(
                value
            )

        else:

            values.append(
                value
            )

    return values


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


# =============================================================================
# Report construction
# =============================================================================

def build_baseline_section(
    dossier: dict[str, Any],
) -> list[str]:

    baseline = (
        dossier.get(
            "baseline_result"
        )
        or {}
    )

    return [
        "=== BASELINE RESULT ===",
        "",
        f"Reward = {baseline.get('reward')}",
        f"DB = {baseline.get('db_reward')}",
        f"NL = {baseline.get('nl_assertion_reward')}",
        f"Action = {baseline.get('action')}",
        f"Read = {baseline.get('read')}",
        f"Write = {baseline.get('write')}",
        f"Generic = {baseline.get('generic')}",
        (
            "Termination Reason = "
            f"{baseline.get('termination_reason')}"
        ),
        (
            "Duration Seconds = "
            f"{baseline.get('duration_seconds')}"
        ),
        (
            "Total Model Cost USD = "
            f"{baseline.get('total_model_cost_usd')}"
        ),
        "",
    ]


def build_task_definition_section(
    dossier: dict[str, Any],
) -> list[str]:

    task_definition = (
        dossier.get(
            "task_definition"
        )
    )

    return [
        "=== TASK DEFINITION ===",
        "",
        shorten(
            task_definition,
            limit=10000,
        ),
        "",
    ]


def build_expected_actions_section(
    dossier: dict[str, Any],
) -> list[str]:

    lines = [
        "=== EXPECTED / GOLDEN ACTIONS ===",
        "",
    ]

    checks = (
        dossier.get(
            "action_checks"
        )
        or []
    )

    if not checks:

        lines.append(
            "NO EXPECTED ACTION CHECKS"
        )

        lines.append("")

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
    dossier: dict[str, Any],
) -> list[str]:

    lines = [
        "=== ACTUAL TRAJECTORY ===",
        "",
    ]

    trajectory = (
        dossier.get(
            "trajectory_messages"
        )
        or []
    )

    if not trajectory:

        lines.extend([
            "NO TRAJECTORY MESSAGES FOUND",
            "",
        ])

        return lines

    for item in trajectory:

        if not isinstance(
            item,
            dict,
        ):

            lines.extend([
                shorten(
                    item
                ),
                "",
            ])

            continue

        index = (
            item.get(
                "index"
            )
        )

        raw = (
            item.get(
                "raw"
            )
            or {}
        )

        lines.append(
            "-" * 90
        )

        if not isinstance(
            raw,
            dict,
        ):

            lines.append(
                f"[{index}] MESSAGE"
            )

            lines.append(
                shorten(
                    raw
                )
            )

            lines.append("")

            continue

        message_type = (
            detect_message_type(
                raw
            )
        )

        role = (
            raw.get(
                "role"
            )
        )

        sender = (
            raw.get(
                "sender"
            )
        )

        recipient = (
            raw.get(
                "recipient"
            )
        )

        lines.append(
            (
                f"[{index}] {message_type}"
                f" | role={role}"
                f" | sender={sender}"
                f" | recipient={recipient}"
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
                    limit=8000,
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
                    limit=8000,
                )
            )

        # Fallback:
        # Preserve raw message if the schema does not expose
        # meaningful content using the common fields above.
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
                    limit=10000,
                )
            )

        lines.append("")

    return lines


def build_nl_assertion_section(
    dossier: dict[str, Any],
) -> list[str]:

    lines = [
        "=== NL ASSERTIONS ===",
        "",
    ]

    assertions = (
        dossier.get(
            "nl_assertions"
        )
        or []
    )

    if not assertions:

        lines.extend([
            "NO NL ASSERTIONS",
            "",
        ])

        return lines

    for index, assertion in enumerate(
        assertions
    ):

        lines.append(
            f"[{index}]"
        )

        lines.append(
            shorten(
                assertion,
                limit=8000,
            )
        )

        lines.append("")

    return lines


def build_manual_review_template(
    task_id: str,
) -> list[str]:

    return [
        "=== MANUAL ROOT-CAUSE ANALYSIS ===",
        "",
        f"Task ID: {task_id}",
        "",
        "1. User Final Intent:",
        "",
        "",
        "2. Expected Critical Actions:",
        "",
        "",
        "3. Actual Critical Actions:",
        "",
        "",
        "4. First Divergence Point:",
        "",
        "",
        "5. Failure Type:",
        "",
        "",
        "6. Root Cause:",
        "",
        "",
        "7. Policy Issue:",
        "",
        "",
        "8. Tool / Environment Issue:",
        "",
        "",
        "9. Evaluator Issue:",
        "",
        "",
        "10. Claim-Action Consistency:",
        "",
        "",
        "11. Training Value:",
        "",
        "",
        "12. Candidate Intervention:",
        "",
        "",
    ]


def build_task_report(
    dossier: dict[str, Any],
) -> str:

    task_id = str(
        dossier.get(
            "task_id"
        )
    )

    lines: list[str] = [
        "=" * 100,
        f"TASK {task_id} FAILURE TRAJECTORY REVIEW",
        "=" * 100,
        "",
    ]

    lines.extend(
        build_baseline_section(
            dossier
        )
    )

    lines.extend(
        build_task_definition_section(
            dossier
        )
    )

    lines.extend(
        build_expected_actions_section(
            dossier
        )
    )

    lines.extend(
        build_trajectory_section(
            dossier
        )
    )

    lines.extend(
        build_nl_assertion_section(
            dossier
        )
    )

    lines.extend(
        build_manual_review_template(
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

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    generated_files: list[
        Path
    ] = []

    for task_id in FAILURE_IDS:

        dossier_path = (
            DOSSIER_DIR
            / f"task_{task_id}_dossier.json"
        )

        if not dossier_path.exists():

            raise FileNotFoundError(
                f"Missing dossier: "
                f"{dossier_path}"
            )

        dossier = (
            load_json(
                dossier_path
            )
        )

        report = (
            build_task_report(
                dossier
            )
        )

        output_path = (
            OUTPUT_DIR
            / f"task_{task_id}_review.txt"
        )

        # Write UTF-8 directly.
        #
        # This avoids Windows console GBK encoding problems
        # caused by characters such as:
        #
        #   ✅
        #   ❌
        #
        output_path.write_text(
            report,
            encoding="utf-8",
        )

        generated_files.append(
            output_path
        )

    manifest = {
        "source_run":
            str(
                RUN_DIR
            ),

        "failure_task_ids":
            FAILURE_IDS,

        "generated_files":
            [
                str(path)
                for path
                in generated_files
            ],
    }

    (
        OUTPUT_DIR
        / "review_manifest.json"
    ).write_text(
        json.dumps(
            manifest,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    # Terminal output deliberately uses ASCII-only text
    # so Windows GBK console cannot crash the script.
    print(
        "FAILURE_REVIEW_FILES_CREATED"
    )

    for path in generated_files:

        print(
            "OUTPUT =",
            path,
        )

    print(
        "FAILURE_TRAJECTORY_REVIEW_OK"
    )


if __name__ == "__main__":
    main()