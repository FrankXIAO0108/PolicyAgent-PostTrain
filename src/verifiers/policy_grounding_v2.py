from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Iterable

from src.guards.retail_pre_action import (
    GuardContext,
    ToolProposal,
    evaluate_retail_actions,
    observe_tool_result,
)

from .policy_grounding_v0 import _dimension_verdict
from .policy_grounding_v1 import verify_trajectory as verify_trajectory_v1
from .schemas import (
    ArtifactBundle,
    Dimension,
    Finding,
    MessageEvent,
    Severity,
    VerificationResult,
    Verdict,
)
from .trajectory_loader import load_task_artifacts


_V1_RULE_IDS = {
    "protocol.one_tool_call_per_turn",
    "protocol.no_content_with_tool_call",
}
_LATEST_INTENT_CATEGORIES = {
    "variant_error",
    "scope_error",
    "payment_error",
    "missing_action",
}


def _finding_code(rule_id: str) -> str:
    normalized = re.sub(r"[^A-Z0-9]+", "_", rule_id.upper()).strip("_")
    return f"PG_GUARD_{normalized}"


def _dimension(category: str) -> Dimension:
    if category in _LATEST_INTENT_CATEGORIES:
        return Dimension.LATEST_INTENT
    return Dimension.POLICY_COMPLIANCE


def _runtime_guard_findings(events: list[MessageEvent]) -> list[Finding]:
    """Replay runtime-safe Guard rules without reference actions or gold data."""

    context = GuardContext()
    pending: dict[str, ToolProposal] = {}
    findings: list[Finding] = []

    for event in events:
        if event.role == "user":
            context.user_texts.append(event.content)
            continue

        if event.role == "assistant":
            proposals = [
                ToolProposal(
                    id=call.id,
                    name=call.name,
                    arguments=dict(call.arguments),
                )
                for call in event.tool_calls
            ]
            result = evaluate_retail_actions(
                proposals,
                context,
                assistant_content=event.content,
            )
            for guard_finding in result.findings:
                # V1 already owns protocol cardinality and text/tool exclusivity.
                if guard_finding.rule_id in _V1_RULE_IDS:
                    continue
                findings.append(
                    Finding(
                        code=_finding_code(guard_finding.rule_id),
                        dimension=_dimension(guard_finding.category),
                        severity=(
                            Severity.MAJOR
                            if guard_finding.blocking
                            or guard_finding.severity.lower() == "major"
                            else Severity.MINOR
                        ),
                        message=guard_finding.message,
                        event_indices=[event.index],
                        evidence={
                            "guard_rule_id": guard_finding.rule_id,
                            "guard_category": guard_finding.category,
                            "blocking": guard_finding.blocking,
                            **guard_finding.evidence,
                        },
                    )
                )
            pending.update(
                {proposal.id: proposal for proposal in proposals if proposal.id}
            )
            continue

        if event.role == "tool":
            proposal = pending.get(str(event.tool_call_id or ""))
            if proposal is not None:
                observe_tool_result(
                    context,
                    proposal,
                    event.content,
                    error=event.tool_error,
                )

    return findings


def verify_trajectory(
    events: list[MessageEvent],
    *,
    task_id: str = "unknown",
    benchmark_verdict: Verdict = Verdict.NOT_EVALUATED,
) -> VerificationResult:
    """Run V1 intent grounding plus runtime-safe deterministic Guard rules."""

    result = verify_trajectory_v1(
        events,
        task_id=task_id,
        benchmark_verdict=benchmark_verdict,
    )
    guard_findings = _runtime_guard_findings(events)
    existing = {
        (finding.code, tuple(finding.event_indices))
        for finding in result.findings
    }
    result.findings.extend(
        finding
        for finding in guard_findings
        if (finding.code, tuple(finding.event_indices)) not in existing
    )

    for dimension in (
        Dimension.LATEST_INTENT,
        Dimension.EXPLICIT_CONFIRMATION,
        Dimension.POLICY_COMPLIANCE,
        Dimension.ACTION_RESULT_TRUTHFULNESS,
    ):
        result.dimensions[dimension] = _dimension_verdict(
            result.findings, dimension
        )

    strict_dimensions = (
        Dimension.LATEST_INTENT,
        Dimension.EXPLICIT_CONFIRMATION,
        Dimension.POLICY_COMPLIANCE,
        Dimension.ACTION_RESULT_TRUTHFULNESS,
    )
    result.verdict = (
        Verdict.FAIL
        if any(
            result.dimensions[dimension] == Verdict.FAIL
            for dimension in strict_dimensions
        )
        else Verdict.REVIEW
        if any(
            result.dimensions[dimension] == Verdict.REVIEW
            for dimension in strict_dimensions
        )
        else Verdict.PASS
    )
    result.metrics.update(
        {
            "verifier_version": "2.2",
            "runtime_guard_finding_count": len(guard_findings),
            "runtime_guard_major_finding_count": sum(
                finding.severity == Severity.MAJOR
                for finding in guard_findings
            ),
            "uses_reference_actions": False,
        }
    )
    result.notes = [
        "V2.2 composes V1.3 with the same runtime-safe deterministic rules used "
        "by the pre-action Guard.",
        "Cross-turn exchange/modify writes on the same order are blocked by the "
        "shared one-shot order-mutation rule.",
        "Runtime Guard replay is hydrated only from user messages and observed "
        "tool results. It does not read benchmark reference actions or gold state.",
        "Reference diagnostic rules remain outside this verifier.",
        "Major findings produce FAIL; minor-only findings produce REVIEW.",
        "Benchmark reward remains independent from policy-grounding dimensions.",
    ]
    return result


def verify_artifacts(bundle: ArtifactBundle) -> VerificationResult:
    benchmark_reward = bundle.summary.get("reward")
    benchmark_verdict = (
        Verdict.PASS
        if benchmark_reward == 1
        else Verdict.FAIL
        if benchmark_reward == 0
        else Verdict.NOT_EVALUATED
    )
    return verify_trajectory(
        bundle.events,
        task_id=bundle.task_id,
        benchmark_verdict=benchmark_verdict,
    )


def _iter_task_dirs(paths: Iterable[str]) -> Iterable[Path]:
    for raw_path in paths:
        path = Path(raw_path).expanduser().resolve()
        if (path / "task.json").exists():
            yield path
            continue
        yield from sorted(
            candidate.parent
            for candidate in path.rglob("task.json")
            if candidate.is_file()
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Verify Tau2 policy grounding V2 with runtime-safe Guard rules."
    )
    parser.add_argument("paths", nargs="+", help="Task or experiment directories")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    results = [
        verify_artifacts(load_task_artifacts(task_dir)).to_dict()
        for task_dir in _iter_task_dirs(args.paths)
    ]
    payload = json.dumps(results, ensure_ascii=False, indent=2) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload, encoding="utf-8")
    else:
        print(payload, end="")


if __name__ == "__main__":
    main()
