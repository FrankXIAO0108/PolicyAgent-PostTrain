from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def _row(
    *,
    scenario_id: str,
    split: str,
    category: str,
    user_message: str,
    tool: str,
    arguments: dict[str, Any],
    argument_policy: str,
) -> dict[str, Any]:
    return {
        "schema_version": "retail-tool-protocol-sft-row-v1",
        "scenario_id": scenario_id,
        "split": split,
        "curriculum": "A_IDENTITY_AND_CUSTOMER_ROUTING",
        "category": category,
        "user_message": user_message,
        "expected_call": {"name": tool, "arguments": arguments},
        "argument_policy": argument_policy,
        "source": "developer_authored_synthetic_protocol_example",
        "contains_tau2_task_content": False,
        "human_adjudicated_business_gold": False,
        "business_metric_claim_allowed": False,
    }


def build_rows(split: str, count_per_category: int) -> list[dict[str, Any]]:
    if split not in {"train", "holdout"}:
        raise ValueError(split)
    prefix = "TRN" if split == "train" else "HLD"
    offset = 1000 if split == "train" else 8000
    rows: list[dict[str, Any]] = []
    for index in range(count_per_category):
        number = offset + index
        first = f"Alex{number}"
        last = f"Chen{number}"
        email = f"protocol-{prefix.lower()}-{number}@example.test"
        zip_code = f"{number:05d}"[-5:]
        variants = [
            (
                "exchange_missing_identity",
                f"I need to exchange two items from order {prefix}-EX-{number}, but I have not provided my account details yet.",
                "respond_to_user",
                {
                    "message": "Before I access the order, please provide either your account email or your full name and ZIP code so I can authenticate you."
                },
                "required_nonempty_keys",
            ),
            (
                "return_missing_identity",
                f"Please return everything in order {prefix}-RT-{number}. I have not told you who I am yet.",
                "respond_to_user",
                {
                    "message": "I need to authenticate you first. Please provide your account email, or your full name together with your ZIP code."
                },
                "required_nonempty_keys",
            ),
            (
                "modify_missing_identity",
                f"Change the shipping address for order {prefix}-MD-{number}. No identity information has been supplied.",
                "respond_to_user",
                {
                    "message": "Before changing or viewing the order, please provide your account email or your full name and ZIP code for authentication."
                },
                "required_nonempty_keys",
            ),
            (
                "authenticate_by_email",
                f"My account email is {email}. I need help with a recent order.",
                "find_user_id_by_email",
                {"email": email},
                "exact",
            ),
            (
                "authenticate_by_name_zip",
                f"My name is {first} {last}, and my billing ZIP code is {zip_code}. Please help with my order.",
                "find_user_id_by_name_zip",
                {"first_name": first, "last_name": last, "zip": zip_code},
                "exact",
            ),
        ]
        for category_index, (category, message, tool, arguments, policy) in enumerate(
            variants
        ):
            rows.append(
                _row(
                    scenario_id=f"{split}-{category_index:02d}-{number}",
                    split=split,
                    category=category,
                    user_message=message,
                    tool=tool,
                    arguments=arguments,
                    argument_policy=policy,
                )
            )
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def validate_rows(train: list[dict[str, Any]], holdout: list[dict[str, Any]]) -> dict[str, Any]:
    train_ids = {row["scenario_id"] for row in train}
    holdout_ids = {row["scenario_id"] for row in holdout}
    train_messages = {row["user_message"] for row in train}
    holdout_messages = {row["user_message"] for row in holdout}
    required_tools = {
        "respond_to_user",
        "find_user_id_by_email",
        "find_user_id_by_name_zip",
    }
    observed_tools = {
        row["expected_call"]["name"] for row in [*train, *holdout]
    }
    checks = {
        "scenario_ids_disjoint": not bool(train_ids & holdout_ids),
        "user_messages_disjoint": not bool(train_messages & holdout_messages),
        "all_rows_synthetic": all(
            row["contains_tau2_task_content"] is False for row in [*train, *holdout]
        ),
        "no_business_gold_claim": all(
            row["human_adjudicated_business_gold"] is False
            and row["business_metric_claim_allowed"] is False
            for row in [*train, *holdout]
        ),
        "required_tools_covered": required_tools <= observed_tools,
    }
    checks["passed"] = all(checks.values())
    if not checks["passed"]:
        raise RuntimeError(f"Tool SFT data checks failed: {checks}")
    return checks


def build_dataset(output_dir: Path) -> dict[str, Any]:
    train = build_rows("train", count_per_category=16)
    holdout = build_rows("holdout", count_per_category=4)
    checks = validate_rows(train, holdout)
    train_path = output_dir / "sft.jsonl"
    holdout_path = output_dir / "holdout.jsonl"
    write_jsonl(train_path, train)
    write_jsonl(holdout_path, holdout)
    manifest = {
        "schema_version": "retail-tool-protocol-sft-data-v1",
        "scope": "ISOLATED_TOOL_PROTOCOL_WARMUP",
        "curriculum": "A_IDENTITY_AND_CUSTOMER_ROUTING",
        "source": "developer-authored deterministic synthetic examples",
        "upstream_tool_contract": {
            "repository": "sierra-research/tau2-bench",
            "commit": "58e5e1ace69302e6982d27014569c03e0ffccdd2",
        },
        "claims": {
            "formal_retail_gate_unchanged": True,
            "human_adjudicated_business_gold": False,
            "business_improvement_claim_allowed": False,
            "contains_tau2_frozen_tasks": False,
        },
        "files": {
            "sft": {
                "path": "sft.jsonl",
                "rows": len(train),
                "sha256": sha256(train_path),
            },
            "holdout": {
                "path": "holdout.jsonl",
                "rows": len(holdout),
                "sha256": sha256(holdout_path),
            },
        },
        "category_counts": {
            category: sum(row["category"] == category for row in train)
            for category in sorted({row["category"] for row in train})
        },
        "leakage_checks": checks,
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "data" / "retail_tool_protocol_sft_v1",
    )
    args = parser.parse_args()
    print(
        json.dumps(
            build_dataset(args.output_dir.resolve()), ensure_ascii=False, indent=2
        )
    )


if __name__ == "__main__":
    main()
