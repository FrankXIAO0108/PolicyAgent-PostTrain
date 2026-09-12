from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


DEFAULT_CASE_IDS = ("95", "98", "107")
CASE_SUMMARIES = {
    "95": "把布尔型 availability 误解为库存数量，导致两个换货目标未完成。",
    "98": "写入了错误支付方式，并把商品级请求扩大成整单取消风险。",
    "107": "选择了错误变体，并提交了新旧商品相同的违规换货。",
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def _load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Missing frozen demo input: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def _active_flags(task: dict[str, Any]) -> list[str]:
    return [
        name
        for name, enabled in task["state_diff"]["flags"].items()
        if enabled
    ]


def build_project_summary(project_root: Path) -> dict[str, Any]:
    """Summarize frozen Baseline/Guard evidence, not current training status."""
    root = project_root.resolve()
    evaluation_path = root / "reports" / "evaluation" / "final_report.json"
    guard_path = (
        root
        / "experiments"
        / "20260726_pre_action_guard_v1"
        / "guard_audit.json"
    )
    comparison_path = (
        root
        / "experiments"
        / "20260726_v6_vs_v7_evaluation"
        / "comparison.json"
    )
    evaluation = _load_json(evaluation_path)
    guard = _load_json(guard_path)
    comparison = _load_json(comparison_path)

    evaluation_tasks = {
        str(task["task_id"]): task for task in evaluation["tasks"]
    }
    guard_tasks = {str(task["task_id"]): task for task in guard["tasks"]}
    missing = [
        task_id
        for task_id in DEFAULT_CASE_IDS
        if task_id not in evaluation_tasks or task_id not in guard_tasks
    ]
    if missing:
        raise ValueError(f"Frozen reports do not cover demo tasks: {missing}")

    cases: list[dict[str, Any]] = []
    for task_id in DEFAULT_CASE_IDS:
        task = evaluation_tasks[task_id]
        guard_task = guard_tasks[task_id]
        blocking_findings = [
            finding
            for finding in guard_task["runtime_guard"]["findings"]
            if finding["blocking"]
        ]
        cases.append(
            {
                "task_id": task_id,
                "business_summary": CASE_SUMMARIES[task_id],
                "official_signal": task["taxonomy"]["official_signal"],
                "primary_root_causes": task["taxonomy"][
                    "primary_causal_root_cause"
                ],
                "secondary_findings": task["taxonomy"]["secondary_findings"],
                "business_impact": task["taxonomy"]["business_impact"],
                "state_diff_flags": _active_flags(task),
                "runtime_guard_would_block": guard_task["runtime_guard"][
                    "would_block"
                ],
                "runtime_guard_rules": list(
                    dict.fromkeys(
                        finding["rule_id"] for finding in blocking_findings
                    )
                ),
            }
        )

    v6 = comparison["outcome_detection"]["v6_llm_pipeline"]
    v7 = comparison["outcome_detection"]["v7_replay_pipeline"]
    summary = evaluation["summary"]
    guard_summary = guard["summary"]
    return {
        "schema_version": "policy-agent-project-summary-v2.0",
        "project": "PolicyAgent-PostTrain",
        "positioning": (
            "构建于上游 tau2-bench Retail 环境之上的可复现 Tool Agent "
            "可靠性系统。"
        ),
        "thesis": (
            "最终 reward 正确，不足以证明轨迹安全且符合业务政策。"
        ),
        "demo_scope": {
            "kind": "frozen_baseline_guard_example",
            "description": (
                "只读取冻结 Baseline、V6/V7 对比与 Guard 产物，"
                "不重跑评测，也不判断当前后训练状态。"
            ),
            "current_training_status": "not_assessed",
            "current_training_status_reference": "TECHNICAL_REPORT.md",
        },
        "frozen_scope": {
            "task_count": summary["task_count"],
            "official_success_count": summary["success_count"],
            "official_failure_count": summary["failure_count"],
            "failure_task_ids": summary["failure_task_ids"],
            "system_failure_count": 0,
            "new_llm_calls_for_summary": 0,
        },
        "comparison": {
            "v6_llm_failure_recall": v6["recall"],
            "v6_false_negative_count": v6["confusion_matrix"]["fn"],
            "v7_replay_failure_recall": v7["recall"],
            "v7_false_negative_count": v7["confusion_matrix"]["fn"],
            "v7_replay_inconsistency_count": comparison[
                "v7_replay_inconsistency_count"
            ],
            "interpretation": (
                "该结果表示冻结开发集重放一致性，不表示 held-out 泛化能力。"
            ),
        },
        "guard": {
            "runtime_blocked_official_failures": guard_summary[
                "runtime_blocked_failure_count"
            ],
            "official_failure_count": guard_summary["official_failure_count"],
            "non_quarantined_failure_count": guard_summary[
                "non_quarantined_failure_count"
            ],
            "combined_non_quarantined_coverage": guard_summary[
                "combined_non_quarantined_failure_coverage"
            ],
            "interpretation": (
                "该结果是离线反事实拦截证据，不证明模型重新生成后一定成功。"
            ),
        },
        "cases": cases,
        "evidence": {
            "evaluation_report": {
                "path": evaluation_path.relative_to(root).as_posix(),
                "sha256": _sha256(evaluation_path),
            },
            "guard_audit": {
                "path": guard_path.relative_to(root).as_posix(),
                "sha256": _sha256(guard_path),
            },
            "comparison": {
                "path": comparison_path.relative_to(root).as_posix(),
                "sha256": _sha256(comparison_path),
            },
        },
        "boundaries": [
            "20 任务实验是冻结开发基线，不是排行榜成绩。",
            "V7 指标衡量冻结产物的重放一致性。",
            "Guard 拦截是离线证据，不是在线恢复率结论。",
            "本示例不评估当前标签审阅进度或后训练完成情况。",
            "本示例不提供 SFT、DPO 或 GRPO 业务提升证据。",
        ],
    }


def render_markdown(demo: dict[str, Any]) -> str:
    demo_scope = demo["demo_scope"]
    scope = demo["frozen_scope"]
    comparison = demo["comparison"]
    guard = demo["guard"]
    lines = [
        "# PolicyAgent-PostTrain — 冻结 Baseline/Guard 离线示例",
        "",
        f"> {demo['thesis']}",
        "",
        demo_scope["description"],
        "",
        "## 冻结实验",
        "",
        (
            f"- {scope['task_count']} 个 Retail 开发任务："
            f"{scope['official_success_count']} 成功，"
            f"{scope['official_failure_count']} 失败，0 系统失败"
        ),
        f"- 失败任务：{', '.join(scope['failure_task_ids'])}",
        "- 生成摘要新增 LLM 调用：0",
        "",
        "## 为什么不能只依赖轨迹 LLM Judge",
        "",
        "| 方法 | 失败召回率 | 漏报数 |",
        "|---|---:|---:|",
        (
            "| V6 轨迹 + LLM pipeline | "
            f"{comparison['v6_llm_failure_recall']:.0%} | "
            f"{comparison['v6_false_negative_count']} |"
        ),
        (
            "| V7 确定性状态重放 | "
            f"{comparison['v7_replay_failure_recall']:.0%} | "
            f"{comparison['v7_false_negative_count']} |"
        ),
        "",
        (
            "边界：上述 V7 数字是冻结开发集的重放一致性，"
            "不是未见任务泛化性能。"
        ),
        "",
        "## 三个代表性业务失败",
        "",
        "| Task | 业务问题 | 根因 | 业务影响 | Runtime Guard |",
        "|---|---|---|---|---|",
    ]
    for case in demo["cases"]:
        root_causes = ", ".join(case["primary_root_causes"]) or "none"
        impacts = ", ".join(case["business_impact"]) or "none"
        guard_result = (
            "BLOCK: " + ", ".join(case["runtime_guard_rules"])
            if case["runtime_guard_would_block"]
            else "ALLOW"
        )
        lines.append(
            f"| {case['task_id']} | {case['business_summary']} | "
            f"{root_causes} | {impacts} | {guard_result} |"
        )
    lines.extend(
        [
            "",
            "## Guard 离线结果",
            "",
            (
                "- Runtime-safe Guard 拦截官方失败："
                f"{guard['runtime_blocked_official_failures']}/"
                f"{guard['official_failure_count']}"
            ),
            (
                "- 排除数据冲突 Task 59 后，Runtime + reference diagnostic "
                f"覆盖：{guard['combined_non_quarantined_coverage']}/"
                f"{guard['non_quarantined_failure_count']}"
            ),
            "- 这是离线反事实拦截结果，不代表重新生成后一定成功。",
            "",
            "## 当前项目状态入口",
            "",
            (
                "当前后训练进展与证据边界见 "
                f"[技术报告]({demo_scope['current_training_status_reference']})"
                "（以报告核验日期为准）；本示例不据此生成训练状态结论。"
            ),
            "",
            "## 证据边界",
            "",
        ]
    )
    lines.extend(f"- {boundary}" for boundary in demo["boundaries"])
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="从本地冻结产物生成 Baseline/Guard 离线示例，不评估当前训练状态。"
    )
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
    )
    parser.add_argument(
        "--format",
        choices=("markdown", "json"),
        default="markdown",
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    demo = build_project_summary(args.project_root)
    rendered = (
        render_markdown(demo)
        if args.format == "markdown"
        else json.dumps(demo, ensure_ascii=False, indent=2) + "\n"
    )
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
        print(f"已保存：{args.output}")
    else:
        print(rendered, end="")


if __name__ == "__main__":
    main()
