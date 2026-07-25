"""
Compare Trial-1 and Trial-2 outcomes for the four Retail tasks that failed
in the frozen 20-task Trial-1 baseline.

This script makes NO model calls and NO evaluator calls.

Important:
- Trial-2 contains only the four Trial-1 failures.
- Its 50% success rate must not be compared directly with the 80% success
  rate of the full 20-task Trial-1 baseline.
- Repeated reward=0 does not automatically mean repeated agent failure.
  Human/verifier audit labels remain necessary.
"""

from __future__ import annotations

import csv
import json
from datetime import datetime
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
EXPERIMENTS_ROOT = PROJECT_ROOT / "experiments"

TRIAL1_RUN_DIR = (
    EXPERIMENTS_ROOT
    / "20260722_110504_retail_baseline20_trial1_deepseek"
)

TRIAL2_RUN_DIR = (
    EXPERIMENTS_ROOT
    / "20260724_111341_retail_failure4_trial2_deepseek"
)

OUTPUT_DIR = (
    EXPERIMENTS_ROOT
    / "20260724_failure4_trial1_trial2_stability"
)

TASK_IDS = ["59", "98", "95", "107"]

EXPECTED_TRIAL1_REWARDS = {
    "59": 0.0,
    "98": 0.0,
    "95": 0.0,
    "107": 0.0,
}

EXPECTED_TRIAL2_REWARDS = {
    "59": 0.0,
    "98": 0.0,
    "95": 1.0,
    "107": 1.0,
}

# Labels come from the existing human trajectory audit.
# They are not inferred from raw reward alone.
AUDIT_LABELS = {
    "59": {
        "audit_label": "BENCHMARK_ALIGNMENT_FAILURE",
        "trajectory_tier_trial1": "EXCLUDED",
        "interpretation": (
            "Repeated raw reward failure, but the existing audit found a "
            "dynamic user-intent versus static-gold mismatch. This must not "
            "be treated as a clean negative training example."
        ),
    },
    "98": {
        "audit_label": "MIXED_BADCASE",
        "trajectory_tier_trial1": "MIXED",
        "interpretation": (
            "Repeated raw reward failure. Existing audit found both benchmark "
            "alignment issues and genuine trajectory defects, including action "
            "scope and final-claim consistency problems."
        ),
    },
    "95": {
        "audit_label": "VALID_AGENT_FAILURE_IN_TRIAL1",
        "trajectory_tier_trial1": "VALID_NEGATIVE",
        "interpretation": (
            "Trial-1 failed but Trial-2 succeeded. The task is sampling-sensitive: "
            "the Trial-1 failure remains a useful audited negative trajectory, "
            "while the Trial-2 success requires its own quality audit before it "
            "can be used as positive training data."
        ),
    },
    "107": {
        "audit_label": "POLICY_GROUNDING_FAILURE_IN_TRIAL1",
        "trajectory_tier_trial1": "VALID_NEGATIVE",
        "interpretation": (
            "Trial-1 failed but Trial-2 succeeded. The observed policy-grounding "
            "failure is not deterministic. The Trial-2 success must still pass "
            "policy, authorization, state, and final-claim verification."
        ),
    },
}


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


def get_nested(
    value: Any,
    path: tuple[str | int, ...],
) -> Any:
    current = value

    for key in path:
        if isinstance(key, int):
            if not isinstance(current, list) or key >= len(current):
                return None
            current = current[key]
        else:
            if not isinstance(current, dict) or key not in current:
                return None
            current = current[key]

    return current


def extract_reward(summary: dict[str, Any]) -> float:
    candidate_paths = [
        ("reward",),
        ("reward_info", "reward"),
        ("simulation", "reward_info", "reward"),
        ("simulations", 0, "reward_info", "reward"),
        ("result", "reward"),
    ]

    for path in candidate_paths:
        value = get_nested(summary, path)

        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return float(value)

    raise RuntimeError(
        "Could not find a numeric task reward in summary.json. "
        f"Available top-level keys: {sorted(summary.keys())}"
    )


def load_task_reward(
    run_dir: Path,
    task_id: str,
) -> tuple[float, Path]:
    summary_path = run_dir / f"task_{task_id}" / "summary.json"

    if not summary_path.exists():
        raise RuntimeError(
            f"Missing task summary: {summary_path}"
        )

    summary = read_json(summary_path)

    if not isinstance(summary, dict):
        raise RuntimeError(
            f"Task summary is not a JSON object: {summary_path}"
        )

    status = summary.get("status")

    if status in {"SYSTEM_FAILED", "RESULT_PARSE_FAILED"}:
        raise RuntimeError(
            f"Task {task_id} has invalid status {status!r} in "
            f"{summary_path}"
        )

    return extract_reward(summary), summary_path


def transition_label(
    trial1_reward: float,
    trial2_reward: float,
) -> str:
    pair = (trial1_reward, trial2_reward)

    labels = {
        (0.0, 0.0): "REWARD_STABLE_0",
        (0.0, 1.0): "FAIL_TO_SUCCESS",
        (1.0, 0.0): "SUCCESS_TO_FAIL",
        (1.0, 1.0): "REWARD_STABLE_1",
    }

    if pair not in labels:
        raise RuntimeError(
            f"Unsupported binary reward transition: {pair}"
        )

    return labels[pair]


def validate_trial2_recovery() -> Path:
    report_path = (
        TRIAL2_RUN_DIR / "trial2_recovery_report.json"
    )

    if not report_path.exists():
        raise RuntimeError(
            "Missing Trial-2 recovery report: "
            f"{report_path}"
        )

    report = read_json(report_path)

    if report.get("no_model_or_evaluator_calls") is not True:
        raise RuntimeError(
            "Trial-2 recovery report does not confirm "
            "no_model_or_evaluator_calls=true."
        )

    aggregate = report.get("aggregate") or {}

    expected_counts = {
        "completed_count": 4,
        "system_failed_count": 0,
        "result_parse_failed_count": 0,
        "valid_reward_count": 4,
    }

    for key, expected in expected_counts.items():
        actual = aggregate.get(key)

        if actual != expected:
            raise RuntimeError(
                f"Unexpected Trial-2 recovery value: "
                f"{key}={actual!r}, expected {expected!r}."
            )

    return report_path


def build_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    for task_id in TASK_IDS:
        trial1_reward, trial1_summary = load_task_reward(
            TRIAL1_RUN_DIR,
            task_id,
        )

        trial2_reward, trial2_summary = load_task_reward(
            TRIAL2_RUN_DIR,
            task_id,
        )

        expected_trial1 = EXPECTED_TRIAL1_REWARDS[task_id]
        expected_trial2 = EXPECTED_TRIAL2_REWARDS[task_id]

        if trial1_reward != expected_trial1:
            raise RuntimeError(
                f"Task {task_id}: Trial-1 reward changed. "
                f"Actual={trial1_reward}, expected={expected_trial1}."
            )

        if trial2_reward != expected_trial2:
            raise RuntimeError(
                f"Task {task_id}: Trial-2 reward changed. "
                f"Actual={trial2_reward}, expected={expected_trial2}."
            )

        audit = AUDIT_LABELS[task_id]
        success_count = int(trial1_reward) + int(trial2_reward)

        rows.append(
            {
                "task_id": task_id,
                "trial1_reward": trial1_reward,
                "trial2_reward": trial2_reward,
                "successes_across_two_samples": success_count,
                "empirical_success_fraction": success_count / 2,
                "transition": transition_label(
                    trial1_reward,
                    trial2_reward,
                ),
                "audit_label": audit["audit_label"],
                "trajectory_tier_trial1": (
                    audit["trajectory_tier_trial1"]
                ),
                "interpretation": audit["interpretation"],
                "trial1_summary_path": str(trial1_summary),
                "trial2_summary_path": str(trial2_summary),
            }
        )

    return rows


def write_csv(
    path: Path,
    rows: list[dict[str, Any]],
) -> None:
    fieldnames = [
        "task_id",
        "trial1_reward",
        "trial2_reward",
        "successes_across_two_samples",
        "empirical_success_fraction",
        "transition",
        "audit_label",
        "trajectory_tier_trial1",
        "interpretation",
        "trial1_summary_path",
        "trial2_summary_path",
    ]

    with path.open(
        "w",
        encoding="utf-8-sig",
        newline="",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=fieldnames,
        )
        writer.writeheader()
        writer.writerows(rows)


def build_markdown(
    rows: list[dict[str, Any]],
    generated_at: str,
) -> str:
    stable_zero = sum(
        row["transition"] == "REWARD_STABLE_0"
        for row in rows
    )

    fail_to_success = sum(
        row["transition"] == "FAIL_TO_SUCCESS"
        for row in rows
    )

    total_successes = sum(
        row["successes_across_two_samples"]
        for row in rows
    )

    any_success_tasks = sum(
        row["successes_across_two_samples"] > 0
        for row in rows
    )

    lines = [
        "# Retail Failure-4 Trial-1 × Trial-2 Stability Report",
        "",
        f"Generated at: `{generated_at}`",
        "",
        "## Scope",
        "",
        (
            "This report compares the four tasks that received reward 0 "
            "in the frozen 20-task Retail train-split development Trial-1."
        ),
        "",
        (
            "Trial-2 is a second sample of this failure-selected subset. "
            "It is not an independent 20-task baseline and is not an "
            "official tau2-bench leaderboard result."
        ),
        "",
        "## Raw result comparison",
        "",
        (
            "| Task | Trial-1 | Trial-2 | Transition | "
            "Existing audit label |"
        ),
        "|---:|---:|---:|---|---|",
    ]

    for row in rows:
        lines.append(
            f"| {row['task_id']} "
            f"| {row['trial1_reward']:.1f} "
            f"| {row['trial2_reward']:.1f} "
            f"| {row['transition']} "
            f"| {row['audit_label']} |"
        )

    lines.extend(
        [
            "",
            "## Aggregate observations",
            "",
            f"- Repeated raw reward 0: `{stable_zero}/4` tasks.",
            f"- Trial-1 failure → Trial-2 success: `{fail_to_success}/4` tasks.",
            (
                f"- Successful samples across the eight task-runs: "
                f"`{total_successes}/8 = {total_successes / 8:.1%}`."
            ),
            (
                f"- Tasks with at least one success across two samples: "
                f"`{any_success_tasks}/4 = {any_success_tasks / 4:.1%}`."
            ),
            "",
            (
                "These numbers describe only the failure-selected subset. "
                "They must not be reported as the model's general Retail "
                "success rate."
            ),
            "",
            "## Interpretation by task",
            "",
        ]
    )

    for row in rows:
        lines.extend(
            [
                f"### Task {row['task_id']}",
                "",
                f"- Transition: `{row['transition']}`",
                f"- Audit label: `{row['audit_label']}`",
                (
                    "- Trial-1 trajectory tier: "
                    f"`{row['trajectory_tier_trial1']}`"
                ),
                f"- Interpretation: {row['interpretation']}",
                "",
            ]
        )

    lines.extend(
        [
            "## Main conclusions",
            "",
            (
                "1. Task 59 and Task 98 reproduced reward 0, but neither "
                "can be called a clean, stable agent failure from raw reward "
                "alone. Task 59 is benchmark-alignment-sensitive; Task 98 is "
                "a mixed badcase."
            ),
            "",
            (
                "2. Task 95 and Task 107 changed from reward 0 to reward 1. "
                "Their Trial-1 failures are therefore not deterministic."
            ),
            "",
            (
                "3. The successful Trial-2 trajectories for Task 95 and "
                "Task 107 are not automatically SFT Gold. They still require "
                "trajectory-quality verification."
            ),
            "",
            (
                "4. The result supports the project route: Raw Reward → "
                "Human/Verifier Audit → Trajectory Quality → Training Eligibility."
            ),
            "",
            "## Next verifier focus",
            "",
            (
                "- Task 59: Latest Explicit Authorized Intent and benchmark "
                "alignment."
            ),
            (
                "- Task 98: Authorization Scope and Claim–Action–State "
                "Consistency."
            ),
            (
                "- Task 95: Multi-goal Completeness and premature escalation."
            ),
            (
                "- Task 107: Policy Grounding and Policy–Tool Enforcement gap."
            ),
            "",
        ]
    )

    return "\n".join(lines)


def main() -> None:
    if not TRIAL1_RUN_DIR.is_dir():
        raise RuntimeError(
            f"Trial-1 directory does not exist: {TRIAL1_RUN_DIR}"
        )

    if not TRIAL2_RUN_DIR.is_dir():
        raise RuntimeError(
            f"Trial-2 directory does not exist: {TRIAL2_RUN_DIR}"
        )

    recovery_report_path = validate_trial2_recovery()
    rows = build_rows()

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    generated_at = datetime.now().isoformat()

    comparison = {
        "generated_at": generated_at,
        "analysis": "retail_failure4_trial1_trial2_stability",
        "not_official_leaderboard_score": True,
        "no_model_or_evaluator_calls": True,
        "trial1_run_dir": str(TRIAL1_RUN_DIR),
        "trial2_run_dir": str(TRIAL2_RUN_DIR),
        "trial2_recovery_report": str(recovery_report_path),
        "selection_warning": (
            "Trial-2 contains only the four tasks that failed Trial-1. "
            "Its aggregate reward must not be compared directly with "
            "the full 20-task Trial-1 baseline."
        ),
        "rows": rows,
        "aggregate": {
            "task_count": len(rows),
            "stable_reward_zero_count": sum(
                row["transition"] == "REWARD_STABLE_0"
                for row in rows
            ),
            "fail_to_success_count": sum(
                row["transition"] == "FAIL_TO_SUCCESS"
                for row in rows
            ),
            "successful_samples": sum(
                row["successes_across_two_samples"]
                for row in rows
            ),
            "total_samples": len(rows) * 2,
            "tasks_with_any_success": sum(
                row["successes_across_two_samples"] > 0
                for row in rows
            ),
        },
    }

    json_path = OUTPUT_DIR / "failure4_stability.json"
    csv_path = OUTPUT_DIR / "failure4_stability.csv"
    markdown_path = OUTPUT_DIR / "failure4_stability_report.md"

    write_json(json_path, comparison)
    write_csv(csv_path, rows)

    markdown_path.write_text(
        build_markdown(rows, generated_at),
        encoding="utf-8",
    )

    print("=" * 88)
    print("RETAIL FAILURE-4 TRIAL-1 x TRIAL-2 STABILITY")
    print("NO MODEL OR EVALUATOR CALLS")

    for row in rows:
        print(
            f"Task {row['task_id']}: "
            f"{row['trial1_reward']:.1f} -> "
            f"{row['trial2_reward']:.1f} | "
            f"{row['transition']} | "
            f"{row['audit_label']}"
        )

    print("STABLE_REWARD_ZERO = 2")
    print("FAIL_TO_SUCCESS = 2")
    print("SUCCESSFUL_SAMPLES = 2/8")
    print("TASKS_WITH_ANY_SUCCESS = 2/4")
    print(f"JSON = {json_path}")
    print(f"CSV = {csv_path}")
    print(f"REPORT = {markdown_path}")
    print("RETAIL_FAILURE4_STABILITY_ANALYSIS_OK")
    print("=" * 88)


if __name__ == "__main__":
    main()