from __future__ import annotations

import argparse
import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.verifiers.intent_state import is_write_tool


SCHEMA_VERSION = "sft-residual-adjudication-v1.0.0"
POSITIVE_DECISIONS = {
    "ACCEPT_AS_PREFERRED_POSITIVE",
    "ACCEPT_AS_POSITIVE_ALTERNATIVE",
}
VALID_NEGATIVE_DECISION = "ACCEPT_AS_VALID_NEGATIVE"
AFFIRMATIVE = re.compile(
    r"\b(?:yes|confirm(?:ed)?|proceed|go ahead|sure|do it)\b|确认|同意|继续",
    re.I,
)
CONFIRM_QUESTION = re.compile(
    r"\b(?:confirm|confirmation|yes\s*/\s*no|proceed|go ahead|would you like)\b|确认|是否继续",
    re.I,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _simulation(row: dict[str, Any]) -> dict[str, Any]:
    path = Path(row["source"]["returned_results"]).resolve()
    if _sha256(path) != str(row["source"]["returned_results_sha256"]).upper():
        raise ValueError(f"Returned-results hash mismatch: {path}")
    simulations = _load(path).get("simulations") or []
    index = int(row["trial_index"])
    if index >= len(simulations):
        raise IndexError(f"Trial {index} not found in {path}")
    return simulations[index]


def _call_records(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    results = {
        str(message.get("id") or ""): message
        for message in messages
        if message.get("role") == "tool"
    }
    rows: list[dict[str, Any]] = []
    for message_index, message in enumerate(messages):
        if message.get("role") != "assistant":
            continue
        for call in message.get("tool_calls") or []:
            result = results.get(str(call.get("id") or ""), {})
            rows.append(
                {
                    "message_index": message_index,
                    "name": str(call.get("name") or ""),
                    "arguments": dict(call.get("arguments") or {}),
                    "result_error": bool(result.get("error", False)),
                    "result_content": str(result.get("content") or ""),
                }
            )
    return rows


def _max_no_progress_run(calls: list[dict[str, Any]]) -> int:
    maximum = 0
    current = 0
    previous: str | None = None
    for call in calls:
        signature = json.dumps(
            {
                "name": call["name"],
                "arguments": call["arguments"],
                "result_error": call["result_error"],
                "result_content": call["result_content"],
            },
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        current = current + 1 if signature == previous else 1
        maximum = max(maximum, current)
        previous = signature
    return maximum


def _missing_confirmation_count(messages: list[dict[str, Any]]) -> int:
    missing = 0
    last_write_index = -1
    for index, message in enumerate(messages):
        if message.get("role") != "assistant":
            continue
        writes = [
            call
            for call in message.get("tool_calls") or []
            if is_write_tool(str(call.get("name") or ""))
        ]
        if not writes:
            continue
        user_indices = [
            prior
            for prior in range(last_write_index + 1, index)
            if messages[prior].get("role") == "user"
            and AFFIRMATIVE.search(str(messages[prior].get("content") or ""))
        ]
        confirmed = False
        for user_index in user_indices:
            confirmed = any(
                messages[prior].get("role") == "assistant"
                and CONFIRM_QUESTION.search(str(messages[prior].get("content") or ""))
                for prior in range(last_write_index + 1, user_index)
            )
            if confirmed:
                break
        if not confirmed:
            missing += len(writes)
        last_write_index = index
    return missing


def _signals(row: dict[str, Any]) -> dict[str, Any]:
    simulation = _simulation(row)
    messages = list(simulation.get("messages") or [])
    calls = _call_records(messages)
    max_no_progress = _max_no_progress_run(calls)
    max_steps = str(simulation.get("termination_reason") or "") == "max_steps"
    dialogue_loop = bool(
        (row.get("trajectory_digest") or {})
        .get("evaluation_card", {})
        .get("dialogue_use", {})
        .get("dialogue_repeat_candidate", False)
    )
    missing_confirmation = _missing_confirmation_count(messages)
    return {
        "dialogue_loop": dialogue_loop,
        "no_progress_tool_loop": max_steps and max_no_progress >= 3,
        "unconfirmed_write": missing_confirmation > 0,
        "evidence": {
            "termination_reason": simulation.get("termination_reason"),
            "max_consecutive_same_call_and_result": max_no_progress,
            "missing_confirmation_write_count": missing_confirmation,
        },
    }


def build_adjudication(
    *, review_packet: str | Path, owner_reviews_dir: str | Path, output_dir: str | Path
) -> dict[str, Any]:
    packet_path = Path(review_packet).resolve()
    reviews_dir = Path(owner_reviews_dir).resolve()
    output = Path(output_dir).resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty output: {output}")
    output.mkdir(parents=True, exist_ok=True)

    packet_hash = _sha256(packet_path)
    packet = _load(packet_path)
    packet_rows = {str(row["review_id"]): row for row in packet.get("rows") or []}
    if len(packet_rows) != len(packet.get("rows") or []):
        raise ValueError("Duplicate review_id in review packet")

    decisions: dict[str, dict[str, Any]] = {}
    review_sources: list[dict[str, Any]] = []
    for path in sorted(reviews_dir.glob("*_owner_review.json")):
        review = _load(path)
        bound = review.get("source_review_packet") or {}
        if str(bound.get("sha256") or "").upper() != packet_hash:
            raise ValueError(f"Owner review packet hash mismatch: {path}")
        if Path(str(bound.get("path"))).resolve() != packet_path:
            raise ValueError(f"Owner review packet path mismatch: {path}")
        review_sources.append({"path": str(path.resolve()), "sha256": _sha256(path)})
        for decision in review.get("decisions") or []:
            review_id = str(decision["review_id"])
            if review_id in decisions:
                raise ValueError(f"Duplicate owner decision: {review_id}")
            decisions[review_id] = decision

    if set(decisions) != set(packet_rows):
        raise ValueError(
            "Owner-decision coverage mismatch; "
            f"missing={sorted(set(packet_rows) - set(decisions))}, "
            f"extra={sorted(set(decisions) - set(packet_rows))}"
        )

    adjudicated_rows: list[dict[str, Any]] = []
    for review_id, row in packet_rows.items():
        decision = decisions[review_id]
        adjudicated_rows.append(
            {
                "review_id": review_id,
                "task_id": str(row["task_id"]),
                "trial_index": int(row["trial_index"]),
                "benchmark_reward": row["post_run_benchmark_evidence"]["outcome"][
                    "reward"
                ],
                "owner_decision": decision["decision"],
                "owner_label": decision["label"],
                "owner_rationale": decision["rationale"],
                "signals": _signals(row),
                "source": row["source"],
            }
        )

    by_task: dict[str, list[dict[str, Any]]] = {}
    for row in adjudicated_rows:
        by_task.setdefault(row["task_id"], []).append(row)
    pair_candidates: list[dict[str, Any]] = []
    for task_id, task_rows in sorted(by_task.items()):
        chosen = [
            row
            for row in task_rows
            if row["owner_decision"] == "ACCEPT_AS_PREFERRED_POSITIVE"
        ]
        if len(chosen) != 1:
            raise ValueError(f"Task {task_id} must have exactly one preferred positive")
        rejected = [
            row for row in task_rows if row["owner_decision"] == VALID_NEGATIVE_DECISION
        ]
        for negative in rejected:
            pair_candidates.append(
                {
                    "pair_id": f"{task_id}:chosen-{chosen[0]['trial_index']}:rejected-{negative['trial_index']}",
                    "task_id": task_id,
                    "chosen_review_id": chosen[0]["review_id"],
                    "rejected_review_id": negative["review_id"],
                    "slice_required": True,
                    "training_release_allowed": False,
                    "release_blockers": [
                        "The task was used for residual-failure discovery and reward design.",
                        "The critical decision turn has not been sliced and independently validated.",
                        "An entity-disjoint training split and untouched final holdout are not bound.",
                    ],
                }
            )

    positive_rows = [
        row for row in adjudicated_rows if row["owner_decision"] in POSITIVE_DECISIONS
    ]
    positive_false_positives = [
        {
            "review_id": row["review_id"],
            "active_signals": [
                name
                for name in (
                    "dialogue_loop",
                    "no_progress_tool_loop",
                    "unconfirmed_write",
                )
                if row["signals"][name]
            ],
        }
        for row in positive_rows
        if any(
            row["signals"][name]
            for name in ("dialogue_loop", "no_progress_tool_loop", "unconfirmed_write")
        )
    ]
    expected_signal_by_label = {
        "DIALOGUE_LOOP_AND_FALSE_CAPABILITY": "dialogue_loop",
        "NO_PROGRESS_TOOL_LOOP": "no_progress_tool_loop",
        "UNCONFIRMED_STATE_MUTATION": "unconfirmed_write",
    }
    targeted = [
        {
            "review_id": row["review_id"],
            "owner_label": row["owner_label"],
            "expected_signal": expected_signal_by_label[row["owner_label"]],
            "detected": row["signals"][expected_signal_by_label[row["owner_label"]]],
        }
        for row in adjudicated_rows
        if row["owner_label"] in expected_signal_by_label
    ]

    result_path = output / "adjudication.json"
    result = {
        "schema_version": SCHEMA_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source_review_packet": {"path": str(packet_path), "sha256": packet_hash},
        "owner_reviews": review_sources,
        "summary": {
            "row_count": len(adjudicated_rows),
            "owner_review_count": len(review_sources),
            "benchmark_positive_but_not_owner_positive_count": sum(
                row["benchmark_reward"] == 1
                and row["owner_decision"] not in POSITIVE_DECISIONS
                for row in adjudicated_rows
            ),
            "preference_pair_candidate_count": len(pair_candidates),
            "positive_signal_false_positive_count": len(positive_false_positives),
            "targeted_negative_count": len(targeted),
            "targeted_negative_detected_count": sum(
                row["detected"] for row in targeted
            ),
        },
        "reward_signal_validation": {
            "signals": ["dialogue_loop", "no_progress_tool_loop", "unconfirmed_write"],
            "positive_rows": [row["review_id"] for row in positive_rows],
            "positive_false_positives": positive_false_positives,
            "targeted_negatives": targeted,
            "development_checks_passed": not positive_false_positives
            and all(row["detected"] for row in targeted),
            "independent_holdout_passed": False,
            "eligible_for_grpo_reward": False,
        },
        "preference_pair_candidates": pair_candidates,
        "rows": adjudicated_rows,
        "release_gate": {
            "fully_owner_adjudicated": True,
            "dpo_training_data_released": False,
            "grpo_reward_released": False,
            "reason": "Development discovery evidence is complete, but slicing, leakage control, and an untouched holdout remain open.",
        },
    }
    result_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "adjudication": {"path": str(result_path), "sha256": _sha256(result_path)},
        "summary": result["summary"],
        "release_gate": result["release_gate"],
        "contains_private_trajectory_references": True,
        "safe_to_publish": False,
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Aggregate owner-reviewed residual SFT evidence."
    )
    parser.add_argument("--review-packet", type=Path, required=True)
    parser.add_argument("--owner-reviews", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(
        json.dumps(
            build_adjudication(
                review_packet=args.review_packet,
                owner_reviews_dir=args.owner_reviews,
                output_dir=args.output,
            ),
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
