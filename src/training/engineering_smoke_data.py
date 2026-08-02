from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


SYSTEM_PROMPT = (
    "你是零售业务工具智能体。只能输出一个 JSON 对象，格式为 "
    '{"tool":"工具名","arguments":{...}}。涉及取消、退款、改地址等写操作时，'
    "若用户尚未明确确认，必须调用 request_confirmation，不能直接执行写操作。"
)


def _compact(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def _prompt(user_text: str) -> str:
    return f"系统：{SYSTEM_PROMPT}\n用户：{user_text}\n助手："


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def _scenario(index: int, split: str) -> dict[str, Any]:
    order_id = f"SYN-{split.upper()}-{index:03d}"
    variant = f"SKU-{index:03d}-BLUE-M"
    mode = index % 6
    if mode == 0:
        user = f"帮我查询订单 {order_id} 的物流状态。"
        target = {"tool": "get_order", "arguments": {"order_id": order_id}}
        rejected = {
            "tool": "cancel_order",
            "arguments": {"order_id": order_id},
        }
        category = "read_only_query"
    elif mode == 1:
        user = f"我在考虑取消订单 {order_id}，先告诉我下一步怎么做。"
        target = {
            "tool": "request_confirmation",
            "arguments": {"action": "cancel_order", "order_id": order_id},
        }
        rejected = {
            "tool": "cancel_order",
            "arguments": {"order_id": order_id},
        }
        category = "confirmation_before_write"
    elif mode == 2:
        user = f"把订单 {order_id} 的配送地址改掉，但我还没给新地址。"
        target = {
            "tool": "request_missing_information",
            "arguments": {"field": "new_address", "order_id": order_id},
        }
        rejected = {
            "tool": "modify_address",
            "arguments": {"order_id": order_id, "new_address": "unknown"},
        }
        category = "missing_required_argument"
    elif mode == 3:
        user = f"查询商品变体 {variant} 是否可用于订单 {order_id} 的换货。"
        target = {
            "tool": "check_variant_availability",
            "arguments": {"order_id": order_id, "variant_id": variant},
        }
        rejected = {
            "tool": "exchange_item",
            "arguments": {"order_id": order_id, "variant_id": variant},
        }
        category = "check_before_mutation"
    elif mode == 4:
        user = f"订单 {order_id} 的商品坏了，我想退款，但还没确认退款方式。"
        target = {
            "tool": "request_missing_information",
            "arguments": {"field": "refund_method", "order_id": order_id},
        }
        rejected = {
            "tool": "refund_order",
            "arguments": {"order_id": order_id, "refund_method": "default"},
        }
        category = "refund_method_grounding"
    else:
        user = f"订单 {order_id} 的目标换货规格不明确，请不要猜测。"
        target = {
            "tool": "transfer_to_human",
            "arguments": {"order_id": order_id, "reason": "missing_variant"},
        }
        rejected = {
            "tool": "exchange_item",
            "arguments": {"order_id": order_id, "variant_id": "guessed"},
        }
        category = "no_unsupported_guess"
    return {
        "scenario_id": f"{split}-{index:03d}",
        "split": split,
        "category": category,
        "prompt": _prompt(user),
        "expected_action": target,
        "rejected_action": rejected,
        "synthetic": True,
    }


def build_records() -> dict[str, list[dict[str, Any]]]:
    supervised = [_scenario(index, "train") for index in range(24)]
    rl = [_scenario(index, "rl") for index in range(8)]
    holdout = [_scenario(index, "holdout") for index in range(8)]
    return {
        "sft": [
            {
                "scenario_id": row["scenario_id"],
                "prompt": row["prompt"],
                "completion": _compact(row["expected_action"]),
                "category": row["category"],
            }
            for row in supervised
        ],
        "dpo": [
            {
                "scenario_id": row["scenario_id"],
                "prompt": row["prompt"],
                "chosen": _compact(row["expected_action"]),
                "rejected": _compact(row["rejected_action"]),
                "category": row["category"],
            }
            for row in supervised
        ],
        "grpo": [
            {
                "scenario_id": row["scenario_id"],
                "prompt": row["prompt"],
                "expected_action": _compact(row["expected_action"]),
                "category": row["category"],
            }
            for row in rl
        ],
        "holdout": [
            {
                "scenario_id": row["scenario_id"],
                "prompt": row["prompt"],
                "expected_action": _compact(row["expected_action"]),
                "category": row["category"],
            }
            for row in holdout
        ],
    }


def write_dataset(output_dir: Path) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    records = build_records()
    files: dict[str, dict[str, Any]] = {}
    for name, rows in records.items():
        path = output_dir / f"{name}.jsonl"
        text = "".join(
            json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows
        )
        path.write_text(text, encoding="utf-8")
        files[name] = {
            "path": path.name,
            "rows": len(rows),
            "sha256": _sha256(path),
        }

    train_ids = {row["scenario_id"] for row in records["sft"]}
    rl_ids = {row["scenario_id"] for row in records["grpo"]}
    holdout_ids = {row["scenario_id"] for row in records["holdout"]}
    manifest = {
        "schema_version": "posttrain-engineering-smoke-data-v1",
        "scope": "ISOLATED_ENGINEERING_SMOKE",
        "source": "developer-authored synthetic scenarios",
        "contains_tau2_frozen_tasks": False,
        "human_adjudicated_business_gold": False,
        "business_metric_claim_allowed": False,
        "files": files,
        "leakage_checks": {
            "train_vs_rl_overlap": sorted(train_ids & rl_ids),
            "train_vs_holdout_overlap": sorted(train_ids & holdout_ids),
            "rl_vs_holdout_overlap": sorted(rl_ids & holdout_ids),
            "passed": not (
                (train_ids & rl_ids)
                or (train_ids & holdout_ids)
                or (rl_ids & holdout_ids)
            ),
        },
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    manifest = write_dataset(args.output_dir)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
