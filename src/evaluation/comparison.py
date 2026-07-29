from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


DEFAULT_V6 = Path(r"D:\PolicyAgent-PostTrain\reports\verifier\v6_verifier_result.jsonl")
DEFAULT_V7 = Path(r"D:\PolicyAgent-PostTrain\reports\evaluation\final_report.json")
DEFAULT_EXPECTED = Path(
    r"D:\PolicyAgent-PostTrain\experiments"
    r"\20260726_v6_vs_v7_evaluation\expected_failure_taxonomy.json"
)
DEFAULT_OUTPUT = Path(
    r"D:\PolicyAgent-PostTrain\experiments\20260726_v6_vs_v7_evaluation"
)


def _safe_div(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0


def _classification_metrics(
    gold_failure: dict[str, bool],
    predicted_failure: dict[str, bool],
) -> dict[str, Any]:
    common = sorted(set(gold_failure) & set(predicted_failure), key=int)
    tp = sum(gold_failure[key] and predicted_failure[key] for key in common)
    fp = sum(not gold_failure[key] and predicted_failure[key] for key in common)
    fn = sum(gold_failure[key] and not predicted_failure[key] for key in common)
    tn = sum(
        not gold_failure[key] and not predicted_failure[key] for key in common
    )
    return {
        "task_count": len(common),
        "confusion_matrix": {"tp": tp, "fp": fp, "fn": fn, "tn": tn},
        "accuracy": _safe_div(tp + tn, len(common)),
        "precision": _safe_div(tp, tp + fp),
        "recall": _safe_div(tp, tp + fn),
        "specificity": _safe_div(tn, tn + fp),
        "f1": _safe_div(2 * tp, 2 * tp + fp + fn),
        "false_positive_task_ids": [
            key
            for key in common
            if not gold_failure[key] and predicted_failure[key]
        ],
        "false_negative_task_ids": [
            key
            for key in common
            if gold_failure[key] and not predicted_failure[key]
        ],
    }


def _taxonomy_metrics(
    expected: dict[str, list[str]],
    actual: dict[str, list[str]],
) -> dict[str, Any]:
    expected_pairs = {
        (task_id, label)
        for task_id, labels in expected.items()
        for label in labels
    }
    actual_pairs = {
        (task_id, label)
        for task_id, labels in actual.items()
        for label in labels
        if task_id in expected
    }
    true_pairs = expected_pairs & actual_pairs
    return {
        "audit_task_count": len(expected),
        "exact_match_task_count": sum(
            set(expected[task_id]) == set(actual.get(task_id, []))
            for task_id in expected
        ),
        "micro_precision": _safe_div(len(true_pairs), len(actual_pairs)),
        "micro_recall": _safe_div(len(true_pairs), len(expected_pairs)),
        "expected_pairs": len(expected_pairs),
        "predicted_pairs": len(actual_pairs),
        "matched_pairs": len(true_pairs),
        "scope": "frozen_failure4_diagnostic_audit_not_held_out_generalization",
    }


def _render(report: dict[str, Any]) -> str:
    v6 = report["outcome_detection"]["v6_llm_pipeline"]
    v7 = report["outcome_detection"]["v7_replay_pipeline"]
    taxonomy = report["root_cause_audit"]["v7"]
    return "\n".join(
        [
            "# V6 vs V7 evaluation comparison",
            "",
            "## Method",
            "",
            "- Dataset: frozen 20-task Retail development baseline.",
            "- Gold outcome: recorded Tau2 overall reward.",
            "- V6 prediction: raw `prediction.has_failure` from the v6 JSONL.",
            "- V7 prediction: reward reconstructed from DB replay and frozen NL results.",
            "- Root-cause audit: four known failures; diagnostic set, not held-out.",
            "",
            "## Outcome detection",
            "",
            "| System | Accuracy | Failure recall | Precision | FP | FN | New LLM calls |",
            "|---|---:|---:|---:|---:|---:|---:|",
            (
                f"| V6 | {v6['accuracy']:.2%} | {v6['recall']:.2%} | "
                f"{v6['precision']:.2%} | {v6['confusion_matrix']['fp']} | "
                f"{v6['confusion_matrix']['fn']} | "
                f"{report['cost_profile']['v6_new_llm_calls']} |"
            ),
            (
                f"| V7 | {v7['accuracy']:.2%} | {v7['recall']:.2%} | "
                f"{v7['precision']:.2%} | {v7['confusion_matrix']['fp']} | "
                f"{v7['confusion_matrix']['fn']} | "
                f"{report['cost_profile']['v7_new_llm_calls']} |"
            ),
            "",
            f"- V6 false positive tasks: {v6['false_positive_task_ids']}",
            f"- V6 false negative tasks: {v6['false_negative_task_ids']}",
            f"- V7 replay inconsistencies: {report['v7_replay_inconsistency_count']}",
            (
                "- V7 measured replay time: "
                f"{report['cost_profile']['v7_latency_seconds']} seconds"
            ),
            "",
            "## Root-cause audit",
            "",
            (
                f"- Exact task-level taxonomy match: "
                f"{taxonomy['exact_match_task_count']}/{taxonomy['audit_task_count']}"
            ),
            f"- Micro precision: {taxonomy['micro_precision']:.2%}",
            f"- Micro recall: {taxonomy['micro_recall']:.2%}",
            "- This is an audited development slice, not an unbiased generalization score.",
            "",
            "## Conclusion",
            "",
            "V6 misses all four official failures because trajectory-only semantic "
            "judgment cannot observe the gold state transition. V7 reconstructs the "
            "official outcome deterministically, then diagnoses cause and business "
            "impact in separate layers. The 100% V7 outcome score measures replay "
            "fidelity on frozen artifacts, not performance on unseen tasks.",
            "",
        ]
    )


def compare(
    *,
    v6_path: str | Path = DEFAULT_V6,
    v7_path: str | Path = DEFAULT_V7,
    expected_path: str | Path = DEFAULT_EXPECTED,
    output_dir: str | Path = DEFAULT_OUTPUT,
) -> dict[str, Any]:
    v6_file = Path(v6_path).resolve()
    v7_file = Path(v7_path).resolve()
    expected_file = Path(expected_path).resolve()
    output = Path(output_dir).resolve()

    v6_rows = [
        json.loads(line)
        for line in v6_file.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    v7_report = json.loads(v7_file.read_text(encoding="utf-8"))
    expected = json.loads(expected_file.read_text(encoding="utf-8"))

    gold_failure = {
        str(row["task_id"]): float(row["gold_reward"]["overall"]) == 0.0
        for row in v6_rows
    }
    v6_prediction = {
        str(row["task_id"]): bool(row["prediction"]["has_failure"])
        for row in v6_rows
    }
    v7_prediction = {
        str(task["task_id"]): task["official_signal"]["reconstructed_reward"] == 0.0
        for task in v7_report["tasks"]
    }
    actual_taxonomy = {
        str(task["task_id"]): task["taxonomy"]["root_cause"]
        for task in v7_report["tasks"]
        if not task["task_success"]
    }

    report = {
        "schema_version": "v7-comparison-1.0.0",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "inputs": {
            "v6_raw_predictions": str(v6_file),
            "v7_report": str(v7_file),
            "expected_failure_taxonomy": str(expected_file),
        },
        "outcome_detection": {
            "v6_llm_pipeline": _classification_metrics(
                gold_failure, v6_prediction
            ),
            "v7_replay_pipeline": _classification_metrics(
                gold_failure, v7_prediction
            ),
        },
        "root_cause_audit": {
            "expected": expected,
            "actual": actual_taxonomy,
            "v7": _taxonomy_metrics(expected, actual_taxonomy),
        },
        "v7_replay_inconsistency_count": v7_report["summary"][
            "replay_inconsistency_count"
        ],
        "cost_profile": {
            "v6_new_llm_calls": 2 * len(v6_rows),
            "v6_call_basis": "one evidence extraction plus one classification call per task",
            "v7_new_llm_calls": 0,
            "v7_call_basis": "deterministic replay and reuse of frozen official NL results",
            "token_cost_usd": None,
            "v6_latency_seconds": None,
            "v7_latency_seconds": v7_report["summary"].get(
                "evaluation_elapsed_seconds"
            ),
            "missing_measurement_note": (
                "Historical v6 artifacts do not record token usage or wall-clock latency; "
                "V7 latency measures the local 20-task replay loop."
            ),
        },
        "validity_notes": [
            "The v6 metrics use raw predictions, not verifier_eval.json labels.",
            "V7 outcome accuracy measures exact replay fidelity on frozen artifacts.",
            "The four-task taxonomy audit was used during development and is not held out.",
        ],
    }

    output.mkdir(parents=True, exist_ok=True)
    (output / "comparison.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (output / "analysis.md").write_text(_render(report), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare raw V6 and replay-based V7.")
    parser.add_argument("--v6", type=Path, default=DEFAULT_V6)
    parser.add_argument("--v7", type=Path, default=DEFAULT_V7)
    parser.add_argument("--expected", type=Path, default=DEFAULT_EXPECTED)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    report = compare(
        v6_path=args.v6,
        v7_path=args.v7,
        expected_path=args.expected,
        output_dir=args.output,
    )
    print(json.dumps(report["outcome_detection"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
