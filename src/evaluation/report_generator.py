from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def render_failure_analysis(report: dict[str, Any]) -> str:
    summary = report["summary"]
    lines = [
        "# Tau2-aligned Hybrid Evaluation v7",
        "",
        f"- Generated: `{report['generated_at']}`",
        f"- Experiment: `{report['experiment']}`",
        f"- Tasks: {summary['task_count']}",
        f"- Official successes: {summary['success_count']}",
        f"- Official failures: {summary['failure_count']}",
        f"- Replay inconsistencies: {summary['replay_inconsistency_count']}",
        "",
        "## Failure analysis",
        "",
    ]
    failures = [task for task in report["tasks"] if not task["task_success"]]
    for task in failures:
        signal = task["official_signal"]
        taxonomy = task["taxonomy"]
        lines.extend(
            [
                f"### Task {task['task_id']}",
                "",
                (
                    f"- Official signal: DB={signal['db_match']}, "
                    f"NL={signal['nl_match']}, reward={signal['recorded_reward']}"
                ),
                (
                    "- Layer 1 / official signal: "
                    + ", ".join(taxonomy["official_signal"])
                ),
                (
                    "- Layer 2 / root cause: "
                    + ", ".join(taxonomy["root_cause"])
                ),
                (
                    "  - Primary causal: "
                    + ", ".join(taxonomy["primary_causal_root_cause"])
                ),
                (
                    "  - Secondary findings: "
                    + (
                        ", ".join(taxonomy["secondary_findings"])
                        or "none"
                    )
                ),
                (
                    "- Layer 3 / business impact: "
                    + ", ".join(taxonomy["business_impact"])
                ),
                (
                    "- Quarantine recommended: "
                    f"{taxonomy['quarantine_recommended']}"
                ),
                f"- Agent hash: `{task['replay']['agent_hash']}`",
                f"- Gold hash: `{task['replay']['gold_hash']}`",
            ]
        )
        causal = [
            cause
            for cause in task["root_causes"]
            if cause["caused_official_failure"]
        ]
        secondary = [
            cause
            for cause in task["root_causes"]
            if not cause["caused_official_failure"]
        ]
        if causal:
            lines.append("- Causal evidence:")
            for cause in causal:
                lines.append(
                    f"  - `{cause['code']}`: {cause['explanation']}"
                )
        if secondary:
            lines.append("- Secondary findings:")
            for cause in secondary:
                lines.append(
                    f"  - `{cause['code']}`: {cause['explanation']}"
                )
        active_flags = [
            name for name, enabled in task["state_diff"]["flags"].items() if enabled
        ]
        lines.append(f"- DB diff flags: {', '.join(active_flags) or 'none'}")
        if task["improvement_suggestions"]:
            lines.append("- Improvement suggestions:")
            for suggestion in task["improvement_suggestions"]:
                lines.append(f"  - {suggestion}")
        lines.append("")

    lines.extend(
        [
            "## Interpretation boundary",
            "",
            "Layer 1 is reconstructed from Tau2 state replay and frozen official NL "
            "results. Layers 2 and 3 are downstream deterministic diagnoses; they "
            "are not Tau2-native labels. Cases with `benchmark_data_risk` should be "
            "quarantined instead of used as ordinary optimization negatives.",
            "",
        ]
    )
    return "\n".join(lines)


def write_evaluation_report(
    report: dict[str, Any],
    output_dir: str | Path,
) -> tuple[Path, Path]:
    output = Path(output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    json_path = output / "final_report.json"
    markdown_path = output / "failure_analysis.md"
    json_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    markdown_path.write_text(
        render_failure_analysis(report),
        encoding="utf-8",
    )
    return json_path, markdown_path
