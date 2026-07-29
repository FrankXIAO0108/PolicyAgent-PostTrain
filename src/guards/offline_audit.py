from __future__ import annotations

import argparse
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .retail_pre_action import (
    GuardContext,
    ToolProposal,
    evaluate_retail_actions,
    observe_tool_result,
)


DEFAULT_EXPERIMENT = Path(
    r"D:\PolicyAgent-PostTrain\experiments"
    r"\20260722_110504_retail_baseline20_trial1_deepseek"
)
DEFAULT_OUTPUT = Path(
    r"D:\PolicyAgent-PostTrain\experiments"
    r"\20260726_pre_action_guard_v1"
)


def _proposal(raw: dict[str, Any]) -> ToolProposal:
    return ToolProposal(
        id=str(raw.get("id", "")),
        name=str(raw.get("name", "")),
        arguments=dict(raw.get("arguments") or {}),
    )


def audit_artifact(path: str | Path) -> dict[str, Any]:
    artifact = Path(path)
    payload = json.loads(artifact.read_text(encoding="utf-8"))
    simulation = payload["simulations"][0]
    task = payload["tasks"][0]
    reference_actions = [
        _proposal(action)
        for action in task["evaluation_criteria"].get("actions") or []
    ]
    context = GuardContext(reference_actions=reference_actions)
    pending: dict[str, ToolProposal] = {}
    runtime_findings: list[dict[str, Any]] = []
    reference_findings: list[dict[str, Any]] = []
    runtime_decisions: list[dict[str, Any]] = []
    reference_decisions: list[dict[str, Any]] = []

    for message_index, message in enumerate(simulation.get("messages") or []):
        role = message.get("role")
        if role == "user":
            context.user_texts.append(str(message.get("content") or ""))
            continue
        if role == "assistant":
            proposals = [
                _proposal(call) for call in message.get("tool_calls") or []
            ]
            runtime = evaluate_retail_actions(
                proposals,
                context,
                assistant_content=str(message.get("content") or ""),
            )
            context.enforce_reference = True
            reference = evaluate_retail_actions(
                proposals,
                context,
                assistant_content=str(message.get("content") or ""),
            )
            context.enforce_reference = False
            if proposals and runtime.decision.value != "ALLOW":
                runtime_decisions.append(
                    {
                        "message_index": message_index,
                        "decision": runtime.decision.value,
                        "tool_names": [proposal.name for proposal in proposals],
                    }
                )
            if proposals and reference.decision.value != "ALLOW":
                reference_decisions.append(
                    {
                        "message_index": message_index,
                        "decision": reference.decision.value,
                        "tool_names": [proposal.name for proposal in proposals],
                    }
                )
            for finding in runtime.findings:
                runtime_findings.append(
                    {"message_index": message_index, **finding.to_dict()}
                )
            runtime_keys = {
                (
                    finding.rule_id,
                    json.dumps(finding.evidence, sort_keys=True),
                )
                for finding in runtime.findings
            }
            for finding in reference.findings:
                key = (
                    finding.rule_id,
                    json.dumps(finding.evidence, sort_keys=True),
                )
                if key not in runtime_keys:
                    reference_findings.append(
                        {"message_index": message_index, **finding.to_dict()}
                    )
            pending.update(
                {proposal.id: proposal for proposal in proposals if proposal.id}
            )
            continue
        if role == "tool":
            call_id = str(message.get("id") or message.get("tool_call_id") or "")
            proposal = pending.get(call_id)
            if proposal is not None:
                observe_tool_result(
                    context,
                    proposal,
                    message.get("content"),
                    error=bool(message.get("error", False)),
                )

    reward_info = simulation.get("reward_info") or {}
    reward = reward_info.get("reward")
    return {
        "task_id": str(task["id"]),
        "official_reward": reward,
        "official_failure": reward == 0,
        "runtime_guard": {
            "would_block": any(item["blocking"] for item in runtime_findings),
            "decisions": runtime_decisions,
            "findings": runtime_findings,
        },
        "reference_diagnostic": {
            "would_block": any(item["blocking"] for item in reference_findings),
            "decisions": reference_decisions,
            "findings": reference_findings,
        },
        "artifact": str(artifact.resolve()),
    }


def _categories(task: dict[str, Any], section: str) -> list[str]:
    return list(
        dict.fromkeys(
            finding["category"]
            for finding in task[section]["findings"]
            if finding["blocking"]
        )
    )


def _render(report: dict[str, Any]) -> str:
    summary = report["summary"]
    lines = [
        "# Deterministic pre-action guard V1 audit",
        "",
        "## Scope",
        "",
        "- Input: frozen 20-task Retail development trajectories.",
        "- Runtime-safe guard: policy/state/user-scope checks without gold actions.",
        "- Reference diagnostic: optional benchmark-only comparison to frozen gold.",
        "- No LLM calls and no trajectory regeneration.",
        "",
        "## Summary",
        "",
        f"- Tasks audited: {summary['task_count']}",
        (
            "- Runtime guard blocking coverage on official failures: "
            f"{summary['runtime_blocked_failure_count']}/"
            f"{summary['official_failure_count']}"
        ),
        (
            "- Runtime guard blocked successful trajectories: "
            f"{summary['runtime_blocked_success_count']}/"
            f"{summary['official_success_count']}"
        ),
        (
            "- Reference diagnostic coverage on non-quarantined failures: "
            f"{summary['combined_non_quarantined_failure_coverage']}/"
            f"{summary['non_quarantined_failure_count']}"
        ),
        "",
        "## Four failure cases",
        "",
        "| Task | Runtime blocking categories | Reference-only categories | Interpretation |",
        "|---|---|---|---|",
    ]
    interpretations = {
        "59": "Gold/user conflict remains evaluator quarantine; runtime policy findings are secondary.",
        "95": "Blocks premature transfer because boolean availability can satisfy both exchanges.",
        "98": "Blocks item-scoped whole-order cancellation and serializes multiple writes; reference mode exposes payment mismatch.",
        "107": "Blocks same-item exchange; reference mode also exposes wrong replacement variant.",
    }
    by_id = {task["task_id"]: task for task in report["tasks"]}
    for task_id in ("59", "95", "98", "107"):
        task = by_id[task_id]
        runtime = ", ".join(_categories(task, "runtime_guard")) or "none"
        reference = ", ".join(_categories(task, "reference_diagnostic")) or "none"
        lines.append(
            f"| {task_id} | {runtime} | {reference} | "
            f"{interpretations[task_id]} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation boundary",
            "",
            "A runtime guard cannot legitimately know Tau2 gold actions. Reference "
            "comparison is therefore isolated as a benchmark/training diagnostic and "
            "must not be described as a deployable safety rule. Blocking successful "
            "trajectories is not automatically a false positive: Tau2 reward does not "
            "score every policy requirement. Those cases require a live A/B rerun to "
            "measure utility impact.",
            "",
        ]
    )
    return "\n".join(lines)


def audit_experiment(
    experiment_dir: str | Path = DEFAULT_EXPERIMENT,
    output_dir: str | Path = DEFAULT_OUTPUT,
) -> dict[str, Any]:
    experiment = Path(experiment_dir).resolve()
    output = Path(output_dir).resolve()
    artifacts = sorted(
        experiment.glob("task_*/returned_results.json"),
        key=lambda path: int(path.parent.name.split("_")[-1]),
    )
    tasks = [audit_artifact(path) for path in artifacts]
    failures = [task for task in tasks if task["official_failure"]]
    successes = [task for task in tasks if not task["official_failure"]]
    quarantined = {"59"}
    non_quarantined = [
        task for task in failures if task["task_id"] not in quarantined
    ]
    combined_covered = [
        task
        for task in non_quarantined
        if task["runtime_guard"]["would_block"]
        or task["reference_diagnostic"]["would_block"]
    ]
    category_counts = Counter(
        finding["category"]
        for task in tasks
        for section in ("runtime_guard", "reference_diagnostic")
        for finding in task[section]["findings"]
        if finding["blocking"]
    )
    runtime_decision_counts = Counter(
        decision["decision"]
        for task in tasks
        for decision in task["runtime_guard"]["decisions"]
    )
    report = {
        "schema_version": "pre-action-guard-v1.0.0",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "experiment": str(experiment),
        "summary": {
            "task_count": len(tasks),
            "official_failure_count": len(failures),
            "official_success_count": len(successes),
            "runtime_blocked_failure_count": sum(
                task["runtime_guard"]["would_block"] for task in failures
            ),
            "runtime_blocked_success_count": sum(
                task["runtime_guard"]["would_block"] for task in successes
            ),
            "non_quarantined_failure_count": len(non_quarantined),
            "combined_non_quarantined_failure_coverage": len(combined_covered),
            "blocking_category_counts": dict(sorted(category_counts.items())),
            "runtime_decision_counts": dict(
                sorted(runtime_decision_counts.items())
            ),
            "new_llm_calls": 0,
        },
        "validity_notes": [
            "Task 59 is quarantined because latest user intent conflicts with static gold.",
            "Reference diagnostics use gold actions and are forbidden in deployment mode.",
            "Offline interception does not prove that a regenerated trajectory succeeds.",
        ],
        "tasks": tasks,
    }
    output.mkdir(parents=True, exist_ok=True)
    (output / "guard_audit.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (output / "analysis.md").write_text(_render(report), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit frozen trajectories with Guard V1.")
    parser.add_argument("--experiment", type=Path, default=DEFAULT_EXPERIMENT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    report = audit_experiment(args.experiment, args.output)
    print(json.dumps(report["summary"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
