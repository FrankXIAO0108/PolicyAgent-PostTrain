from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path
from typing import Iterable

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


_WRITE_TOOL_PATTERNS = (
    "cancel_",
    "modify_",
    "return_",
    "exchange_",
)
_CONFIRM_RE = re.compile(
    r"\b(yes|confirm(?:ed)?|go ahead|proceed|do it|exactly as (?:listed|summarized))\b",
    re.IGNORECASE,
)
_SUCCESS_RE = re.compile(
    r"\b(successfully|completed|processed|cancelled|canceled|exchange requested|"
    r"return requested|has been changed|has been updated)\b",
    re.IGNORECASE,
)


def _is_write_tool(name: str) -> bool:
    lowered = name.lower()
    return lowered.startswith(_WRITE_TOOL_PATTERNS)


def _previous_user_event(
    events: list[MessageEvent], event_index: int
) -> MessageEvent | None:
    for event in reversed(events[:event_index]):
        if event.role == "user":
            return event
    return None


def _tool_result_map(events: Iterable[MessageEvent]) -> dict[str, MessageEvent]:
    return {
        event.tool_call_id: event
        for event in events
        if event.role == "tool" and event.tool_call_id
    }


def _benchmark_verdict(bundle: ArtifactBundle) -> Verdict:
    reward = bundle.summary.get("reward")
    if reward is None:
        return Verdict.NOT_EVALUATED
    try:
        return Verdict.PASS if float(reward) >= 1.0 else Verdict.FAIL
    except (TypeError, ValueError):
        return Verdict.NOT_EVALUATED


def verify_trajectory(
    events: list[MessageEvent],
    *,
    task_id: str = "unknown",
    benchmark_verdict: Verdict = Verdict.NOT_EVALUATED,
) -> VerificationResult:
    """Run deterministic policy-grounding checks.

    V0 intentionally separates strict system-policy compliance from semantic
    latest-intent alignment. It never treats a benchmark action list as the
    latest user intent because Task 59 demonstrates that such lists can become
    stale after the user changes their request.
    """
    findings: list[Finding] = []
    tool_results = _tool_result_map(events)
    write_call_count = 0
    failed_tool_call_ids: set[str] = set()

    for event in events:
        calls = event.tool_calls
        if len(calls) > 1:
            write_count = sum(_is_write_tool(call.name) for call in calls)
            severity = Severity.MAJOR if write_count else Severity.MINOR
            findings.append(
                Finding(
                    code="PG_TOOL_CALL_CARDINALITY",
                    dimension=Dimension.POLICY_COMPLIANCE,
                    severity=severity,
                    message=(
                        f"Assistant issued {len(calls)} tool calls in one turn; "
                        "the active retail policy permits at most one."
                    ),
                    event_indices=[event.index],
                    evidence={
                        "tool_names": [call.name for call in calls],
                        "write_call_count": write_count,
                    },
                )
            )

        if calls and event.content.strip():
            findings.append(
                Finding(
                    code="PG_MESSAGE_TOOL_EXCLUSIVITY",
                    dimension=Dimension.POLICY_COMPLIANCE,
                    severity=Severity.MINOR,
                    message=(
                        "Assistant combined user-facing message content and tool "
                        "calls in the same turn."
                    ),
                    event_indices=[event.index],
                    evidence={"tool_names": [call.name for call in calls]},
                )
            )

        for call in calls:
            if not _is_write_tool(call.name):
                continue
            write_call_count += 1
            previous_user = _previous_user_event(events, event.index)
            if previous_user is None or not _CONFIRM_RE.search(previous_user.content):
                findings.append(
                    Finding(
                        code="PG_WRITE_WITHOUT_EXPLICIT_CONFIRMATION",
                        dimension=Dimension.EXPLICIT_CONFIRMATION,
                        severity=Severity.MAJOR,
                        message=(
                            f"Write tool {call.name!r} was not immediately preceded "
                            "by an explicit user confirmation."
                        ),
                        event_indices=[
                            index
                            for index in (
                                previous_user.index if previous_user else None,
                                event.index,
                            )
                            if index is not None
                        ],
                        evidence={
                            "tool_name": call.name,
                            "arguments": call.arguments,
                            "previous_user_text": (
                                previous_user.content if previous_user else None
                            ),
                        },
                    )
                )

            result = tool_results.get(call.id)
            if result is not None and result.tool_error:
                failed_tool_call_ids.add(call.id)

    if failed_tool_call_ids:
        for event in events:
            if event.role != "assistant" or not _SUCCESS_RE.search(event.content):
                continue
            findings.append(
                Finding(
                    code="PG_FALSE_SUCCESS_CLAIM_AFTER_TOOL_ERROR",
                    dimension=Dimension.ACTION_RESULT_TRUTHFULNESS,
                    severity=Severity.MAJOR,
                    message=(
                        "Assistant claimed success although at least one write tool "
                        "reported an error."
                    ),
                    event_indices=[event.index],
                    evidence={"failed_tool_call_ids": sorted(failed_tool_call_ids)},
                )
            )
            break

    dimension_failures = Counter(finding.dimension for finding in findings)
    dimensions = {
        Dimension.BENCHMARK_REWARD: benchmark_verdict,
        Dimension.LATEST_INTENT: Verdict.REVIEW if write_call_count else Verdict.PASS,
        Dimension.EXPLICIT_CONFIRMATION: (
            Verdict.FAIL
            if dimension_failures[Dimension.EXPLICIT_CONFIRMATION]
            else Verdict.PASS
        ),
        Dimension.POLICY_COMPLIANCE: (
            Verdict.FAIL
            if dimension_failures[Dimension.POLICY_COMPLIANCE]
            else Verdict.PASS
        ),
        Dimension.ACTION_RESULT_TRUTHFULNESS: (
            Verdict.FAIL
            if dimension_failures[Dimension.ACTION_RESULT_TRUTHFULNESS]
            else Verdict.PASS
        ),
    }

    strict_dimensions = (
        Dimension.EXPLICIT_CONFIRMATION,
        Dimension.POLICY_COMPLIANCE,
        Dimension.ACTION_RESULT_TRUTHFULNESS,
    )
    verdict = (
        Verdict.FAIL
        if any(dimensions[dimension] == Verdict.FAIL for dimension in strict_dimensions)
        else Verdict.REVIEW
        if dimensions[Dimension.LATEST_INTENT] == Verdict.REVIEW
        else Verdict.PASS
    )

    return VerificationResult(
        task_id=task_id,
        verdict=verdict,
        dimensions=dimensions,
        findings=findings,
        metrics={
            "event_count": len(events),
            "tool_call_count": sum(len(event.tool_calls) for event in events),
            "write_call_count": write_call_count,
            "max_tool_calls_in_one_turn": max(
                (len(event.tool_calls) for event in events), default=0
            ),
        },
        notes=[
            "latest_intent=REVIEW means V0 found write actions but did not perform "
            "open-ended semantic comparison of evolving user constraints.",
            "Benchmark reward is reported independently and never used as a proxy "
            "for policy grounding.",
        ],
    )


def verify_artifacts(bundle: ArtifactBundle) -> VerificationResult:
    return verify_trajectory(
        bundle.events,
        task_id=bundle.task_id,
        benchmark_verdict=_benchmark_verdict(bundle),
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
    parser = argparse.ArgumentParser(description="Verify Tau2 policy grounding.")
    parser.add_argument("paths", nargs="+", help="Task or experiment directories")
    args = parser.parse_args()

    results = [
        verify_artifacts(load_task_artifacts(task_dir)).to_dict()
        for task_dir in _iter_task_dirs(args.paths)
    ]
    print(json.dumps(results, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
