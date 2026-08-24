"""Build a protocol-aligned SFT view of owner-reviewed tau2 trajectories.

The native tau2 teacher transcripts represent customer-facing assistant text as
plain assistant messages followed by user messages.  The TRL Agentic environment
instead exposes customer interaction as a ``respond_to_user`` tool whose result is
the next customer utterance.  This builder changes only that representation:

* the initial greeting is removed and the initial customer request remains a
  normal user message, matching the RL prompt;
* later assistant-to-customer turns become ``respond_to_user`` tool calls;
* the observed next customer utterance becomes the corresponding tool result;
* existing Retail tool calls and results are preserved byte-for-byte as values;
* a fixed non-tool completion is appended after the observed STOP/TRANSFER marker
  so the model is explicitly supervised to terminate the TRL tool loop.

The output remains owner-reviewed development data.  It is not independent gold
and does not open a formal Retail or business-improvement gate.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from copy import deepcopy
from pathlib import Path
from typing import Any

from src.training.run_retail_agentic_grpo import (
    REPO_ROOT,
    load_json,
    load_jsonl,
    sha256,
    wrap_retail_policy_for_agentic_protocol,
)


SCOPE = "AGENTIC_PROTOCOL_BRIDGE_SFT"
EXPECTED_GREETING = "Hi! How can I help you today?"
STOP_MARKER = "###STOP###"
TRANSFER_MARKER = "###TRANSFER###"
TERMINAL_COMPLETION = "Interaction complete."


def _message(
    role: str, content: str = "", tool_calls: list[dict[str, Any]] | None = None
) -> dict[str, Any]:
    return {
        "role": role,
        "content": content,
        "tool_calls": deepcopy(tool_calls or []),
        "loss_mask": 1 if role == "assistant" else 0,
    }


def _bridge_call_id(candidate_id: str, ordinal: int) -> str:
    digest = hashlib.sha256(candidate_id.encode("utf-8")).hexdigest()[:12]
    return f"bridge-{digest}-{ordinal:03d}"


def transform_messages(
    messages: list[dict[str, Any]], candidate_id: str
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Convert one native teacher transcript to the Agentic tool protocol."""

    if len(messages) < 4:
        raise ValueError(f"{candidate_id}: transcript is too short")
    greeting, opening = messages[0], messages[1]
    if (
        greeting.get("role") != "assistant"
        or (greeting.get("tool_calls") or [])
        or str(greeting.get("content") or "").strip() != EXPECTED_GREETING
    ):
        raise ValueError(f"{candidate_id}: frozen initial greeting mismatch")
    if opening.get("role") != "user" or not str(opening.get("content") or "").strip():
        raise ValueError(f"{candidate_id}: initial customer request is missing")

    output = [_message("user", str(opening.get("content") or ""))]
    existing_tool_calls = 0
    bridge_calls = 0
    index = 2
    while index < len(messages):
        current = messages[index]
        role = str(current.get("role") or "")
        calls = current.get("tool_calls") or []
        if role != "assistant":
            raise ValueError(
                f"{candidate_id}: expected assistant at source index {index}, got {role!r}"
            )

        if calls:
            if len(calls) != 1 or str(current.get("content") or "").strip():
                raise ValueError(
                    f"{candidate_id}: source tool call at index {index} is not atomic"
                )
            if index + 1 >= len(messages) or messages[index + 1].get("role") != "tool":
                raise ValueError(
                    f"{candidate_id}: source tool call at index {index} has no result"
                )
            output.append(_message("assistant", "", calls))
            output.append(
                _message("tool", str(messages[index + 1].get("content") or ""))
            )
            existing_tool_calls += 1
            index += 2
            continue

        assistant_text = str(current.get("content") or "").strip()
        if not assistant_text:
            raise ValueError(f"{candidate_id}: empty assistant text at index {index}")
        if index + 1 >= len(messages) or messages[index + 1].get("role") != "user":
            raise ValueError(
                f"{candidate_id}: customer-facing text at index {index} has no user reply"
            )
        user_text = str(messages[index + 1].get("content") or "")
        call_id = _bridge_call_id(candidate_id, bridge_calls + 1)
        output.append(
            _message(
                "assistant",
                "",
                [
                    {
                        "id": call_id,
                        "name": "respond_to_user",
                        "arguments": {"message": assistant_text},
                    }
                ],
            )
        )
        output.append(_message("tool", user_text))
        bridge_calls += 1
        index += 2

    if output[-1]["role"] != "tool":
        raise ValueError(f"{candidate_id}: converted transcript does not end in a tool result")
    terminal_observation = str(output[-1]["content"])
    if STOP_MARKER not in terminal_observation and TRANSFER_MARKER not in terminal_observation:
        raise ValueError(f"{candidate_id}: terminal STOP/TRANSFER marker is missing")
    output.append(_message("assistant", TERMINAL_COMPLETION))

    call_ids = [
        str(call["id"])
        for message in output
        for call in (message.get("tool_calls") or [])
    ]
    if len(call_ids) != len(set(call_ids)):
        raise ValueError(f"{candidate_id}: tool-call ids are not unique")
    return output, {
        "removed_initial_greetings": 1,
        "existing_tool_calls_preserved": existing_tool_calls,
        "respond_to_user_calls_added": bridge_calls,
        "terminal_completions_added": 1,
    }


def transform_row(row: dict[str, Any]) -> tuple[dict[str, Any], dict[str, int]]:
    candidate_id = str(row.get("candidate_id") or "")
    if not candidate_id:
        raise ValueError("source row is missing candidate_id")
    messages, counts = transform_messages(list(row.get("messages") or []), candidate_id)
    transformed = deepcopy(row)
    transformed["schema_version"] = "agentic-protocol-bridge-sft-row-v1"
    transformed["source_system_policy"] = str(row.get("system_policy") or "")
    transformed["system_policy"] = wrap_retail_policy_for_agentic_protocol(
        transformed["source_system_policy"]
    )
    transformed["messages"] = messages
    transformed["protocol_bridge"] = {
        "source_representation": "tau2_native_user_messages",
        "target_representation": "trl_environment_respond_to_user_tool",
        "terminal_completion": TERMINAL_COMPLETION,
        **counts,
    }
    return transformed, counts


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def build_dataset(source_dir: Path, output_dir: Path) -> dict[str, Any]:
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty output: {output_dir}")
    source_manifest_path = source_dir / "manifest.json"
    source_manifest = load_json(source_manifest_path)
    source_record = source_manifest["files"]["sft_dataset"]
    source_dataset = source_dir / source_record["path"]
    if sha256(source_dataset) != source_record["sha256"]:
        raise ValueError("source SFT dataset hash mismatch")
    source_rows = load_jsonl(source_dataset)
    if len(source_rows) != int(source_record["rows"]):
        raise ValueError("source SFT row count mismatch")
    if (source_manifest.get("leakage_and_quality_checks") or {}).get("passed") is not True:
        raise ValueError("source SFT quality/leakage gate is not open")

    transformed_rows: list[dict[str, Any]] = []
    aggregate: Counter[str] = Counter()
    by_split: dict[str, Counter[str]] = {}
    for row in source_rows:
        transformed, counts = transform_row(row)
        transformed_rows.append(transformed)
        split = str(row.get("split") or "")
        by_split.setdefault(split, Counter()).update(counts)
        aggregate.update(counts)

    task_splits: dict[str, set[str]] = {}
    for row in transformed_rows:
        task_splits.setdefault(str(row["task_id"]), set()).add(str(row["split"]))
    task_leakage = sorted(task for task, splits in task_splits.items() if len(splits) > 1)
    candidate_ids = [str(row["candidate_id"]) for row in transformed_rows]
    checks = {
        "source_quality_gate_passed": True,
        "task_leakage_across_splits": task_leakage,
        "duplicate_candidate_ids": len(candidate_ids) - len(set(candidate_ids)),
        "all_rows_start_with_user": all(row["messages"][0]["role"] == "user" for row in transformed_rows),
        "all_rows_end_with_terminal_assistant": all(
            row["messages"][-1]
            == _message("assistant", TERMINAL_COMPLETION)
            for row in transformed_rows
        ),
        "plain_assistant_before_terminal": sum(
            1
            for row in transformed_rows
            for message in row["messages"][:-1]
            if message["role"] == "assistant" and not message["tool_calls"]
        ),
        "pii_scan_inherited_from_bound_source": (
            source_manifest.get("leakage_and_quality_checks", {}).get("pii_hits") == 0
            and source_manifest.get("leakage_and_quality_checks", {}).get("pii_errors") == 0
        ),
    }
    checks["passed"] = (
        not checks["task_leakage_across_splits"]
        and checks["duplicate_candidate_ids"] == 0
        and checks["all_rows_start_with_user"]
        and checks["all_rows_end_with_terminal_assistant"]
        and checks["plain_assistant_before_terminal"] == 0
        and checks["pii_scan_inherited_from_bound_source"]
    )
    if not checks["passed"]:
        raise RuntimeError(f"protocol bridge checks failed: {checks}")

    dataset_path = output_dir / "sft_dataset.jsonl"
    _write_jsonl(dataset_path, transformed_rows)
    manifest = {
        "schema_version": "agentic-protocol-bridge-sft-manifest-v1",
        "scope": SCOPE,
        "claims": {
            "formal_retail_gate_unchanged": True,
            "human_adjudicated_business_gold": False,
            "business_improvement_claim_allowed": False,
            "owner_reviewed_development_data": True,
            "protocol_alignment_only": True,
        },
        "source": {
            "manifest_path": str(source_manifest_path.relative_to(REPO_ROOT)),
            "manifest_sha256": sha256(source_manifest_path),
            "dataset_path": str(source_dataset.relative_to(REPO_ROOT)),
            "dataset_sha256": sha256(source_dataset),
            "rows": len(source_rows),
        },
        "files": {
            "sft_dataset": {
                "path": dataset_path.name,
                "sha256": sha256(dataset_path),
                "rows": len(transformed_rows),
            }
        },
        "counts": {
            "rows": len(transformed_rows),
            "tasks": len(task_splits),
            "splits": dict(Counter(str(row["split"]) for row in transformed_rows)),
            "aggregate": dict(aggregate),
            "by_split": {key: dict(value) for key, value in sorted(by_split.items())},
        },
        "leakage_and_quality_checks": checks,
    }
    _write_json(output_dir / "manifest.json", manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    source_dir = (REPO_ROOT / args.source_dir).resolve()
    output_dir = (REPO_ROOT / args.output_dir).resolve()
    manifest = build_dataset(source_dir, output_dir)
    print(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
