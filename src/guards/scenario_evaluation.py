from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .retail_pre_action import (
    GuardContext,
    ToolProposal,
    evaluate_retail_actions,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SUITE = PROJECT_ROOT / "configs" / "guard_synthetic_diagnostic_v1.json"
DEFAULT_OUTPUT = (
    PROJECT_ROOT / "experiments" / "20260730_guard_synthetic_diagnostic_v1"
)
SCHEMA_VERSION = "guard-synthetic-diagnostic-result-v1.0.0"


def sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_value(*args: str) -> str | None:
    try:
        return subprocess.check_output(
            ["git", *args],
            cwd=PROJECT_ROOT,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _proposal(raw: dict[str, Any]) -> ToolProposal:
    return ToolProposal(
        id=str(raw.get("id", "")),
        name=str(raw["name"]),
        arguments=dict(raw.get("arguments") or {}),
    )


def _context(raw: dict[str, Any]) -> GuardContext:
    if raw.get("reference_actions") or raw.get("enforce_reference"):
        raise ValueError(
            "Synthetic runtime diagnostics must not contain reference actions."
        )
    return GuardContext(
        orders=dict(raw.get("orders") or {}),
        products=dict(raw.get("products") or {}),
        item_catalog=dict(raw.get("item_catalog") or {}),
        payment_method_ids={
            str(value) for value in raw.get("payment_method_ids") or []
        },
        user_texts=[str(value) for value in raw.get("user_texts") or []],
        completed_writes=[
            _proposal(value) for value in raw.get("completed_writes") or []
        ],
        reference_actions=[],
        enforce_reference=False,
    )


def load_suite(path: str | Path) -> dict[str, Any]:
    suite = json.loads(Path(path).read_text(encoding="utf-8"))
    cases = suite.get("cases")
    if not isinstance(cases, list) or not cases:
        raise ValueError("Scenario suite must contain a non-empty cases list.")
    case_ids = [str(case.get("case_id", "")) for case in cases]
    if any(not value for value in case_ids):
        raise ValueError("Every scenario must have a non-empty case_id.")
    if len(case_ids) != len(set(case_ids)):
        raise ValueError("Scenario case_id values must be unique.")
    if suite.get("official_metric") is not False:
        raise ValueError("Synthetic diagnostic suite must set official_metric=false.")
    return suite


def _safe_divide(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def _f1(precision: float | None, recall: float | None) -> float | None:
    if precision is None or recall is None or precision + recall == 0:
        return None
    return 2 * precision * recall / (precision + recall)


def evaluate_suite(suite: dict[str, Any]) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    category_counts: dict[str, dict[str, int]] = defaultdict(
        lambda: {"count": 0, "passed": 0}
    )

    for case in suite["cases"]:
        expected = dict(case["expected"])
        result = evaluate_retail_actions(
            [_proposal(value) for value in case.get("proposals") or []],
            _context(dict(case.get("context") or {})),
            assistant_content=str(case.get("assistant_content", "")),
        )
        all_rule_ids = {finding.rule_id for finding in result.findings}
        blocking_rule_ids = {
            finding.rule_id for finding in result.findings if finding.blocking
        }
        expected_blocking = set(expected.get("blocking_rule_ids") or [])
        expected_findings = set(expected.get("finding_rule_ids") or [])
        decision_match = result.decision.value == expected["decision"]
        blocking_rules_match = blocking_rule_ids == expected_blocking
        required_findings_present = expected_findings.issubset(all_rule_ids)
        passed = (
            decision_match
            and blocking_rules_match
            and required_findings_present
        )
        category = str(case.get("category", "uncategorized"))
        category_counts[category]["count"] += 1
        category_counts[category]["passed"] += int(passed)
        rows.append(
            {
                "case_id": str(case["case_id"]),
                "category": category,
                "description": str(case.get("description", "")),
                "expected": expected,
                "actual": result.to_dict(),
                "checks": {
                    "decision_match": decision_match,
                    "blocking_rules_match": blocking_rules_match,
                    "required_findings_present": required_findings_present,
                    "missing_blocking_rule_ids": sorted(
                        expected_blocking - blocking_rule_ids
                    ),
                    "unexpected_blocking_rule_ids": sorted(
                        blocking_rule_ids - expected_blocking
                    ),
                    "missing_finding_rule_ids": sorted(
                        expected_findings - all_rule_ids
                    ),
                },
                "passed": passed,
            }
        )

    tp = sum(
        row["expected"]["decision"] != "ALLOW"
        and row["actual"]["decision"] != "ALLOW"
        for row in rows
    )
    fp = sum(
        row["expected"]["decision"] == "ALLOW"
        and row["actual"]["decision"] != "ALLOW"
        for row in rows
    )
    fn = sum(
        row["expected"]["decision"] != "ALLOW"
        and row["actual"]["decision"] == "ALLOW"
        for row in rows
    )
    tn = sum(
        row["expected"]["decision"] == "ALLOW"
        and row["actual"]["decision"] == "ALLOW"
        for row in rows
    )
    precision = _safe_divide(tp, tp + fp)
    recall = _safe_divide(tp, tp + fn)
    passed_count = sum(row["passed"] for row in rows)
    return {
        "schema_version": SCHEMA_VERSION,
        "suite": {
            "schema_version": suite.get("schema_version"),
            "name": suite.get("name"),
            "scope": suite.get("scope"),
            "label_status": suite.get("label_status"),
            "official_metric": suite.get("official_metric"),
        },
        "summary": {
            "case_count": len(rows),
            "passed_count": passed_count,
            "failed_count": len(rows) - passed_count,
            "exact_case_accuracy": _safe_divide(passed_count, len(rows)),
            "decision_accuracy": _safe_divide(
                sum(row["checks"]["decision_match"] for row in rows),
                len(rows),
            ),
            "blocking_rule_exact_match": _safe_divide(
                sum(row["checks"]["blocking_rules_match"] for row in rows),
                len(rows),
            ),
            "blocking_detection": {
                "positive_class": "expected decision is not ALLOW",
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "tn": tn,
                "precision": precision,
                "recall": recall,
                "f1": _f1(precision, recall),
            },
            "category_results": dict(sorted(category_counts.items())),
            "new_llm_calls": 0,
            "uses_reference_actions": False,
        },
        "cases": rows,
        "validity_notes": list(suite.get("limitations") or []),
    }


def render_markdown(result: dict[str, Any]) -> str:
    summary = result["summary"]
    detection = summary["blocking_detection"]

    def metric(value: float | None) -> str:
        return "N/A" if value is None else f"{value:.2%}"

    lines = [
        "# Retail Guard Synthetic Generalization Diagnostic V1",
        "",
        "## Scope",
        "",
        f"- Cases: {summary['case_count']}",
        "- Labels: developer-authored deterministic expectations",
        "- Official metric: no",
        "- Reference actions / gold DB: not used",
        "- New LLM calls: 0",
        "",
        "## Result",
        "",
        f"- Exact case accuracy: {metric(summary['exact_case_accuracy'])}",
        f"- Decision accuracy: {metric(summary['decision_accuracy'])}",
        (
            "- Blocking-rule exact match: "
            f"{metric(summary['blocking_rule_exact_match'])}"
        ),
        (
            "- Blocking detection P/R/F1: "
            f"{metric(detection['precision'])} / "
            f"{metric(detection['recall'])} / {metric(detection['f1'])}"
        ),
        (
            f"- Confusion matrix: TP={detection['tp']}, FP={detection['fp']}, "
            f"FN={detection['fn']}, TN={detection['tn']}"
        ),
        "",
        "## Cases",
        "",
        "| Case | Category | Expected | Actual | Blocking rules | Pass |",
        "|---|---|---|---|---|---|",
    ]
    for row in result["cases"]:
        rules = ", ".join(
            finding["rule_id"]
            for finding in row["actual"]["findings"]
            if finding["blocking"]
        )
        lines.append(
            f"| {row['case_id']} | {row['category']} | "
            f"{row['expected']['decision']} | {row['actual']['decision']} | "
            f"{rules or '-'} | {'YES' if row['passed'] else 'NO'} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation boundary",
            "",
            *[f"- {note}" for note in result["validity_notes"]],
            "",
            "This suite is useful for deterministic regression and scenario transfer. "
            "It does not replace independent policy adjudication, frozen Tau2 held-out "
            "evaluation, or live Guard A/B measurement.",
        ]
    )
    return "\n".join(lines) + "\n"


def write_outputs(
    suite_path: str | Path,
    output_dir: str | Path,
    result: dict[str, Any],
) -> dict[str, Any]:
    suite_file = Path(suite_path).resolve()
    destination = Path(output_dir).resolve()
    destination.mkdir(parents=True, exist_ok=True)
    snapshot = destination / "suite_snapshot.json"
    snapshot.write_text(suite_file.read_text(encoding="utf-8"), encoding="utf-8")
    results_path = destination / "results.json"
    results_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    analysis_path = destination / "analysis.md"
    analysis_path.write_text(render_markdown(result), encoding="utf-8")

    status = _git_value("status", "--porcelain")
    manifest = {
        "schema_version": "guard-synthetic-diagnostic-manifest-v1.0.0",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "experiment": "retail_guard_synthetic_generalization_diagnostic_v1",
        "project_commit": _git_value("rev-parse", "HEAD"),
        "project_git_dirty": bool(status),
        "inputs": {
            "suite": {
                "path": str(suite_file.relative_to(PROJECT_ROOT)),
                "sha256": sha256(suite_file),
            },
            "suite_snapshot": {
                "path": str(snapshot.relative_to(PROJECT_ROOT)),
                "sha256": sha256(snapshot),
            },
            "guard_source": {
                "path": "src/guards/retail_pre_action.py",
                "sha256": sha256(
                    PROJECT_ROOT / "src" / "guards" / "retail_pre_action.py"
                ),
            },
            "runner_source": {
                "path": "src/guards/scenario_evaluation.py",
                "sha256": sha256(Path(__file__)),
            },
        },
        "outputs": {
            "results": {
                "path": str(results_path.relative_to(PROJECT_ROOT)),
                "sha256": sha256(results_path),
            },
            "analysis": {
                "path": str(analysis_path.relative_to(PROJECT_ROOT)),
                "sha256": sha256(analysis_path),
            },
        },
        "command": (
            "python -m src.guards.scenario_evaluation "
            f"--suite {suite_file.relative_to(PROJECT_ROOT)} "
            f"--output {destination.relative_to(PROJECT_ROOT)}"
        ),
        "new_llm_calls": 0,
        "uses_reference_actions": False,
        "official_metric": False,
        "limitations": result["validity_notes"],
    }
    manifest_path = destination / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate runtime-safe Retail Guard on a synthetic scenario suite."
    )
    parser.add_argument("--suite", type=Path, default=DEFAULT_SUITE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    suite = load_suite(args.suite)
    result = evaluate_suite(suite)
    write_outputs(args.suite, args.output, result)
    print(render_markdown(result))
    if result["summary"]["failed_count"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
