from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from src.training.retail_tool_sft_data import REPO_ROOT, sha256, write_jsonl


SOURCE = "developer_authored_synthetic_multistep_trajectory"
WRITE_TOOLS = {"modify_pending_order_items", "return_delivered_order_items"}


def _tool_call(call_id: str, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
    return {
        "role": "assistant",
        "content": None,
        "tool_calls": [{"id": call_id, "name": name, "arguments": arguments}],
    }


def _tool_result(call_id: str, content: Any) -> dict[str, Any]:
    if not isinstance(content, str):
        content = json.dumps(content, ensure_ascii=False, separators=(",", ":"))
    return {"role": "tool", "tool_call_id": call_id, "content": content}


def _trajectory(split: str, index: int, kind: str) -> dict[str, Any]:
    prefix = "TRN" if split == "train" else "HLD"
    number = (1000 if split == "train" else 8000) + index
    user_id = f"multistep_{split}_{number}"
    email = f"multistep-{split}-{number}@example.test"
    order_id = f"#W{number:07d}"
    product_id = str(5_000_000_000 + number)
    old_item = str(6_000_000_000 + number * 2)
    new_item = str(6_000_000_001 + number * 2)
    payment_id = f"credit_card_{number}"
    user = {
        "user_id": user_id,
        "name": {"first_name": f"Alex{number}", "last_name": f"Chen{number}"},
        "email": email,
        "payment_methods": {payment_id: {"source": "credit_card", "id": payment_id}},
        "orders": [order_id],
    }
    order_status = "delivered" if kind == "return_delivered" else "pending"
    order = {
        "order_id": order_id,
        "user_id": user_id,
        "status": order_status,
        "items": [{"item_id": old_item, "product_id": product_id, "price": 50.0}],
        "payment_history": [{"payment_method_id": payment_id, "amount": 50.0}],
    }
    messages: list[dict[str, Any]] = [
        {
            "role": "user",
            "content": (
                f"My account email is {email}. Please show me the status of order {order_id}."
                if kind == "lookup_order"
                else f"My account email is {email}. I want to {'return' if kind == 'return_delivered' else 'replace'} the item in order {order_id}."
            ),
        },
        _tool_call("call-01", "find_user_id_by_email", {"email": email}),
        _tool_result("call-01", user_id),
        _tool_call("call-02", "get_user_details", {"user_id": user_id}),
        _tool_result("call-02", user),
        _tool_call("call-03", "get_order_details", {"order_id": order_id}),
        _tool_result("call-03", order),
    ]
    if kind == "lookup_order":
        messages.append(
            _tool_call(
                "call-04",
                "respond_to_user",
                {"message": f"Order {order_id} is currently {order_status}."},
            )
        )
    elif kind == "modify_pending":
        product = {
            "product_id": product_id,
            "variants": {
                old_item: {"item_id": old_item, "available": True, "price": 50.0},
                new_item: {"item_id": new_item, "available": True, "price": 60.0},
            },
        }
        messages.extend(
            [
                _tool_call("call-04", "get_product_details", {"product_id": product_id}),
                _tool_result("call-04", product),
                _tool_call(
                    "call-05",
                    "respond_to_user",
                    {"message": f"Replacing {old_item} with {new_item} costs $10 more. Should I proceed using {payment_id}?"},
                ),
                _tool_result("call-05", "Yes, proceed with that replacement."),
                _tool_call(
                    "call-06",
                    "modify_pending_order_items",
                    {
                        "order_id": order_id,
                        "item_ids": [old_item],
                        "new_item_ids": [new_item],
                        "payment_method_id": payment_id,
                    },
                ),
                _tool_result("call-06", {**order, "status": "pending", "items": [{"item_id": new_item, "product_id": product_id, "price": 60.0}]}),
                _tool_call("call-07", "respond_to_user", {"message": f"Order {order_id} now contains {new_item}."}),
            ]
        )
    else:
        messages.extend(
            [
                _tool_call(
                    "call-04",
                    "respond_to_user",
                    {"message": f"I can return {old_item} from {order_id} to {payment_id}. Should I proceed?"},
                ),
                _tool_result("call-04", "Yes, please return it."),
                _tool_call(
                    "call-05",
                    "return_delivered_order_items",
                    {"order_id": order_id, "item_ids": [old_item], "payment_method_id": payment_id},
                ),
                _tool_result("call-05", {**order, "status": "return requested", "return_items": [old_item]}),
                _tool_call("call-06", "respond_to_user", {"message": f"The return for {old_item} has been requested."}),
            ]
        )
    return {
        "schema_version": "retail-multistep-tool-trajectory-v1",
        "trajectory_id": f"{split}-{kind}-{number}",
        "split": split,
        "category": kind,
        "messages": messages,
        "source": SOURCE,
        "contains_tau2_task_content": False,
        "human_adjudicated_business_gold": False,
        "business_metric_claim_allowed": False,
    }


def build_trajectories(split: str, count_per_category: int) -> list[dict[str, Any]]:
    if split not in {"train", "holdout"}:
        raise ValueError(split)
    return [
        _trajectory(split, index, kind)
        for index in range(count_per_category)
        for kind in ("lookup_order", "modify_pending", "return_delivered")
    ]


def decision_rows(trajectories: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for trajectory in trajectories:
        messages = trajectory["messages"]
        for index, target in enumerate(messages):
            if target.get("role") != "assistant" or not target.get("tool_calls"):
                continue
            call = target["tool_calls"][0]
            prior_tool_results = sum(message.get("role") == "tool" for message in messages[:index])
            rows.append(
                {
                    "schema_version": "retail-multistep-tool-sft-decision-v1",
                    "scenario_id": f"{trajectory['trajectory_id']}-decision-{index:02d}",
                    "trajectory_id": trajectory["trajectory_id"],
                    "split": trajectory["split"],
                    "category": trajectory["category"],
                    "decision_index": index,
                    "prior_tool_results": prior_tool_results,
                    "context_messages": messages[:index],
                    "expected_call": {"name": call["name"], "arguments": call["arguments"]},
                    "argument_policy": "exact",
                    "source": SOURCE,
                    "contains_tau2_task_content": False,
                    "human_adjudicated_business_gold": False,
                    "business_metric_claim_allowed": False,
                }
            )
    return rows


def validate_dataset(train_trajectories: list[dict[str, Any]], holdout_trajectories: list[dict[str, Any]]) -> dict[str, Any]:
    train_rows = decision_rows(train_trajectories)
    holdout_rows = decision_rows(holdout_trajectories)
    train_ids = {row["trajectory_id"] for row in train_rows}
    holdout_ids = {row["trajectory_id"] for row in holdout_rows}
    all_rows = [*train_rows, *holdout_rows]
    all_trajectories = [*train_trajectories, *holdout_trajectories]

    def call_result_pairs_are_bound(trajectory: dict[str, Any]) -> bool:
        messages = trajectory["messages"]
        for index, message in enumerate(messages):
            if message.get("role") != "assistant" or not message.get("tool_calls"):
                continue
            calls = message["tool_calls"]
            if len(calls) != 1:
                return False
            if index == len(messages) - 1:
                continue
            result = messages[index + 1]
            if result.get("role") != "tool" or result.get("tool_call_id") != calls[0]["id"]:
                return False
        return True

    def confirmed_writes_are_bound(row: dict[str, Any]) -> bool:
        if row["expected_call"]["name"] not in WRITE_TOOLS:
            return True
        context = row["context_messages"]
        if len(context) < 2 or context[-1].get("role") != "tool":
            return False
        confirmation_call = context[-2]
        if confirmation_call.get("role") != "assistant" or not confirmation_call.get("tool_calls"):
            return False
        return (
            confirmation_call["tool_calls"][0]["name"] == "respond_to_user"
            and context[-1].get("tool_call_id") == confirmation_call["tool_calls"][0]["id"]
            and str(context[-1].get("content", "")).strip().lower().startswith("yes")
        )

    checks = {
        "trajectory_ids_disjoint": not bool(train_ids & holdout_ids),
        "scenario_ids_unique": len({row["scenario_id"] for row in all_rows}) == len(all_rows),
        "tool_call_results_id_bound": all(call_result_pairs_are_bound(trajectory) for trajectory in all_trajectories),
        "all_rows_have_tool_result_context": all(row["prior_tool_results"] >= 1 for row in all_rows if row["decision_index"] >= 3),
        "post_tool_decisions_present": all(any(row["prior_tool_results"] >= 1 for row in rows) for rows in (train_rows, holdout_rows)),
        "confirmation_to_write_present": any(row["expected_call"]["name"] in WRITE_TOOLS for row in all_rows),
        "all_writes_have_bound_explicit_confirmation": all(confirmed_writes_are_bound(row) for row in all_rows),
        "no_business_claim": all(not row["human_adjudicated_business_gold"] and not row["business_metric_claim_allowed"] for row in all_rows),
    }
    checks["passed"] = all(checks.values())
    if not checks["passed"]:
        raise RuntimeError(f"Multi-step Tool SFT checks failed: {checks}")
    return checks


def build_dataset(output_dir: Path) -> dict[str, Any]:
    train_trajectories = build_trajectories("train", 8)
    holdout_trajectories = build_trajectories("holdout", 2)
    checks = validate_dataset(train_trajectories, holdout_trajectories)
    train_rows = decision_rows(train_trajectories)
    holdout_rows = decision_rows(holdout_trajectories)
    files = {
        "trajectories_train": ("trajectories_train.jsonl", train_trajectories),
        "trajectories_holdout": ("trajectories_holdout.jsonl", holdout_trajectories),
        "sft": ("sft.jsonl", train_rows),
        "holdout": ("holdout.jsonl", holdout_rows),
    }
    records = {}
    for key, (filename, rows) in files.items():
        path = output_dir / filename
        write_jsonl(path, rows)
        records[key] = {"path": filename, "rows": len(rows), "sha256": sha256(path)}
    manifest = {
        "schema_version": "retail-multistep-tool-sft-data-v1",
        "scope": "ISOLATED_MULTISTEP_TOOL_SFT_WARMUP",
        "source": SOURCE,
        "derivation": "complete synthetic trajectories sliced into next-assistant decision examples",
        "claims": {
            "formal_retail_gate_unchanged": True,
            "human_adjudicated_business_gold": False,
            "business_improvement_claim_allowed": False,
            "contains_tau2_frozen_tasks": False,
        },
        "files": records,
        "train_expected_tool_counts": dict(Counter(row["expected_call"]["name"] for row in train_rows)),
        "train_prior_tool_result_counts": dict(Counter(str(row["prior_tool_results"]) for row in train_rows)),
        "leakage_and_quality_checks": checks,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=REPO_ROOT / "data" / "retail_multistep_tool_sft_v1")
    args = parser.parse_args()
    print(json.dumps(build_dataset(args.output_dir.resolve()), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
