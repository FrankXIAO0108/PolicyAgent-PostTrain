from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from src.training.teacher_evidence_pack import claim_state_consistency


VERDICTS = ("PASS", "FAIL", "REVIEW", "NOT_APPLICABLE")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def _order(value: dict[str, Any]) -> dict[str, Any]:
    row = {
        "order_id": value["order_id"],
        "status": value.get("status"),
    }
    if "created_at" in value:
        row["created_at"] = value["created_at"]
    if "payments" in value:
        row["payment_history"] = [
            {"transaction_type": "payment", "amount": amount}
            for amount in value["payments"]
        ]
    return row


def _checker_inputs(case: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    messages = [
        {
            "role": "tool",
            "content": json.dumps(_order(order)),
            "error": False,
        }
        for order in case.get("tool_orders", [])
    ]
    messages.append({"role": "assistant", "content": case["answer"]})
    final_orders = {
        order["order_id"]: _order(order) for order in case.get("orders", [])
    }
    return messages, {"agent": {"orders": final_orders}}


def _safe_div(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0


def evaluate(path: Path) -> dict[str, Any]:
    dataset = json.loads(path.read_text(encoding="utf-8"))
    if dataset.get("policy", {}).get("training_allowed") is not False:
        raise ValueError("holdout must be explicitly prohibited from training")

    rows: list[dict[str, Any]] = []
    confusion: dict[str, Counter[str]] = defaultdict(Counter)
    for case in dataset["cases"]:
        expected = case["expected_verdict"]
        if expected not in VERDICTS:
            raise ValueError(f"invalid expected verdict: {expected}")
        messages, final_state = _checker_inputs(case)
        result = claim_state_consistency(messages, final_state)
        predicted = result["verdict"]
        confusion[expected][predicted] += 1
        rows.append(
            {
                "case_id": case["case_id"],
                "category": case["category"],
                "expected": expected,
                "predicted": predicted,
                "exact_match": expected == predicted,
                "findings": result["findings"],
            }
        )

    total = len(rows)
    exact = sum(row["exact_match"] for row in rows)
    expected_fail = sum(row["expected"] == "FAIL" for row in rows)
    predicted_fail = sum(row["predicted"] == "FAIL" for row in rows)
    true_fail = sum(
        row["expected"] == row["predicted"] == "FAIL" for row in rows
    )
    fail_precision = _safe_div(true_fail, predicted_fail)
    fail_recall = _safe_div(true_fail, expected_fail)
    fail_f1 = _safe_div(2 * fail_precision * fail_recall, fail_precision + fail_recall)

    actionable = [row for row in rows if row["expected"] in {"PASS", "FAIL"}]
    determinate = [row for row in actionable if row["predicted"] in {"PASS", "FAIL"}]
    determinate_correct = sum(row["exact_match"] for row in determinate)
    expected_review = [row for row in rows if row["expected"] == "REVIEW"]
    false_fail = [row for row in rows if row["predicted"] == "FAIL" and row["expected"] != "FAIL"]
    review_auto_pass = [
        row for row in expected_review if row["predicted"] == "PASS"
    ]

    metrics = {
        "case_count": total,
        "exact_match_count": exact,
        "exact_match_rate": _safe_div(exact, total),
        "fail_precision": fail_precision,
        "fail_recall": fail_recall,
        "fail_f1": fail_f1,
        "actionable_coverage": _safe_div(len(determinate), len(actionable)),
        "selective_accuracy": _safe_div(determinate_correct, len(determinate)),
        "review_routing_recall": _safe_div(
            sum(row["predicted"] == "REVIEW" for row in expected_review),
            len(expected_review),
        ),
        "false_fail_count": len(false_fail),
        "review_auto_pass_count": len(review_auto_pass),
    }
    gates = {
        "fail_precision_at_least_0_95": fail_precision >= 0.95,
        "fail_recall_at_least_0_90": fail_recall >= 0.90,
        "review_routing_recall_at_least_0_90": metrics["review_routing_recall"] >= 0.90,
        "zero_false_fail": not false_fail,
        "zero_review_auto_pass": not review_auto_pass,
    }
    return {
        "schema_version": "retail-claim-state-holdout-evaluation-v1.0",
        "dataset": {
            "path": path.as_posix(),
            "sha256": _sha256(path),
            "training_allowed": False,
            "rule_tuning_allowed": False,
        },
        "metrics": metrics,
        "gates": gates,
        "ready_for_reward_penalty": all(gates.values()),
        "confusion": {
            expected: {predicted: confusion[expected][predicted] for predicted in VERDICTS}
            for expected in VERDICTS
        },
        "errors": [row for row in rows if not row["exact_match"]],
        "rows": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate frozen claim-state checker holdout.")
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = evaluate(args.dataset)
    rendered = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    else:
        print(rendered, end="")


if __name__ == "__main__":
    main()
