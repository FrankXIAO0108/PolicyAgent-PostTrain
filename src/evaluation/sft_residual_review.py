from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.verifiers.intent_state import is_write_tool


SCHEMA_VERSION = "sft-residual-review-v1.0.0"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _cards(
    report: dict[str, Any], run_name: str
) -> dict[tuple[str, int], dict[str, Any]]:
    rows = report.get(run_name)
    if not isinstance(rows, list):
        raise ValueError(f"Evaluation report has no {run_name!r} card list")
    return {(str(row["task_id"]), int(row.get("trial_index", 0))): row for row in rows}


def _simulation(card: dict[str, Any]) -> tuple[Path, dict[str, Any]]:
    path = Path(card["artifact"]["returned_results"]).resolve()
    actual_hash = _sha256(path)
    expected_hash = str(card["artifact"]["sha256"]).upper()
    if actual_hash != expected_hash:
        raise ValueError(f"Artifact hash mismatch: {path}")
    simulations = _load(path).get("simulations") or []
    index = int(card.get("trial_index", 0))
    if index >= len(simulations):
        raise IndexError(f"Trial {index} not found in {path}")
    return path, simulations[index]


def _calls(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    results = {
        str(message.get("id")): message
        for message in messages
        if message.get("role") == "tool"
    }
    rows: list[dict[str, Any]] = []
    for event_index, message in enumerate(messages):
        if message.get("role") != "assistant":
            continue
        for call in message.get("tool_calls") or []:
            result = results.get(str(call.get("id")), {})
            rows.append(
                {
                    "event_index": event_index,
                    "call_id": str(call.get("id") or ""),
                    "name": str(call.get("name") or ""),
                    "arguments": dict(call.get("arguments") or {}),
                    "is_write": is_write_tool(str(call.get("name") or "")),
                    "result_error": bool(result.get("error", False)),
                    "result_excerpt": str(result.get("content") or "")[:500],
                }
            )
    return rows


def _final_answer(messages: list[dict[str, Any]]) -> str:
    for message in reversed(messages):
        content = str(message.get("content") or "").strip()
        if message.get("role") == "assistant" and content:
            return content
    return ""


def _group_calls(calls: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: list[dict[str, Any]] = []
    for call in calls:
        if not groups or groups[-1]["name"] != call["name"]:
            groups.append(
                {
                    "name": call["name"],
                    "count": 0,
                    "error_count": 0,
                    "first_arguments": call["arguments"],
                    "last_arguments": call["arguments"],
                }
            )
        groups[-1]["count"] += 1
        groups[-1]["error_count"] += int(call["result_error"])
        groups[-1]["last_arguments"] = call["arguments"]
    return groups


def _canonical_action(action: dict[str, Any]) -> str:
    return json.dumps(
        {"name": action["name"], "arguments": action.get("arguments") or {}},
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _task_snapshot(
    result_path: Path, expected_hash: str
) -> tuple[Path, dict[str, Any]]:
    path = result_path.with_name("task_snapshot.json")
    if _sha256(path) != expected_hash.upper():
        raise ValueError(f"Task snapshot hash mismatch: {path}")
    return path, _load(path)


def verify_residual_tasks(
    *, evaluation_cards: str | Path, spec_path: str | Path, output_dir: str | Path
) -> dict[str, Any]:
    report_path = Path(evaluation_cards).resolve()
    specs_path = Path(spec_path).resolve()
    output = Path(output_dir).resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty output: {output}")
    output.mkdir(parents=True, exist_ok=True)

    report = _load(report_path)
    specs = _load(specs_path)
    if _sha256(report_path) != str(specs["source_evaluation_cards_sha256"]).upper():
        raise ValueError("Evaluation-card report hash does not match verifier spec")
    cards = _cards(report, str(specs.get("run_name") or "candidate"))
    rows: list[dict[str, Any]] = []
    seen_task_ids: set[str] = set()

    for task_spec in specs.get("tasks") or []:
        task_id = str(task_spec["task_id"])
        if task_id in seen_task_ids:
            raise ValueError(f"Duplicate task verifier spec: {task_id}")
        seen_task_ids.add(task_id)
        trial_indexes = sorted(index for tid, index in cards if tid == task_id)
        if not trial_indexes:
            raise KeyError(f"Verifier task not found in evaluation cards: {task_id}")
        for trial_index in trial_indexes:
            card = cards[(task_id, trial_index)]
            result_path, simulation = _simulation(card)
            snapshot_path, _ = _task_snapshot(
                result_path, str(task_spec["task_snapshot_sha256"])
            )
            messages = list(simulation.get("messages") or [])
            calls = _calls(messages)
            successful_calls = [call for call in calls if not call["result_error"]]
            observed_writes = Counter(
                _canonical_action(call) for call in successful_calls if call["is_write"]
            )
            required_writes = Counter(
                _canonical_action(action)
                for action in task_spec.get("required_writes") or []
            )
            missing = list((required_writes - observed_writes).elements())
            unexpected = list((observed_writes - required_writes).elements())
            checks = [
                {
                    "name": "required_write_coverage",
                    "passed": not missing,
                    "evidence": {
                        "missing_actions": [json.loads(value) for value in missing]
                    },
                },
                {
                    "name": "no_unexpected_successful_writes",
                    "passed": not unexpected,
                    "evidence": {
                        "unexpected_actions": [
                            json.loads(value) for value in unexpected
                        ]
                    },
                },
                {
                    "name": "no_tool_error",
                    "passed": not any(call["result_error"] for call in calls),
                    "evidence": {
                        "error_call_ids": [
                            call["call_id"] for call in calls if call["result_error"]
                        ]
                    },
                },
            ]
            answer = _final_answer(messages)
            for claim in task_spec.get("required_final_claims") or []:
                value = str(claim["value"])
                supporting_results = [
                    call["call_id"]
                    for call in successful_calls
                    if value in call["result_excerpt"]
                ]
                checks.append(
                    {
                        "name": str(claim["name"]),
                        "passed": value in answer
                        and (
                            bool(supporting_results)
                            if claim.get("require_tool_evidence")
                            else True
                        ),
                        "evidence": {
                            "value": value,
                            "present_in_final_answer": value in answer,
                            "supporting_tool_call_ids": supporting_results,
                        },
                    }
                )
            passed = all(check["passed"] for check in checks)
            rows.append(
                {
                    "task_id": task_id,
                    "trial_index": trial_index,
                    "source": {
                        "path": str(result_path),
                        "sha256": _sha256(result_path),
                    },
                    "task_snapshot": {
                        "path": str(snapshot_path),
                        "sha256": _sha256(snapshot_path),
                    },
                    "benchmark_outcome": card["outcome"],
                    "checks": checks,
                    "verdict": "PASS" if passed else "FAIL",
                    "scope": "DEVELOPMENT_TASK_BOUND_DIAGNOSTIC",
                }
            )

    result_path = output / "verifier_results.json"
    result = {
        "schema_version": SCHEMA_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source_evaluation_cards": {
            "path": str(report_path),
            "sha256": _sha256(report_path),
        },
        "source_spec": {"path": str(specs_path), "sha256": _sha256(specs_path)},
        "scope": {
            "task_bound": True,
            "uses_post_run_task_specification": True,
            "eligible_as_general_reward": False,
            "eligible_for_frozen_metric_claim": False,
        },
        "rows": rows,
    }
    result_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "result": {
            "path": str(result_path),
            "sha256": _sha256(result_path),
            "row_count": len(rows),
        },
        "pass_count": sum(row["verdict"] == "PASS" for row in rows),
        "fail_count": sum(row["verdict"] == "FAIL" for row in rows),
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return manifest


def build_preference_review_packet(
    *, evaluation_cards: str | Path, selection_path: str | Path, output_dir: str | Path
) -> dict[str, Any]:
    report_path = Path(evaluation_cards).resolve()
    selected_path = Path(selection_path).resolve()
    output = Path(output_dir).resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty output: {output}")
    output.mkdir(parents=True, exist_ok=True)

    report = _load(report_path)
    selections = _load(selected_path)
    if (
        _sha256(report_path)
        != str(selections["source_evaluation_cards_sha256"]).upper()
    ):
        raise ValueError("Evaluation-card report hash does not match selection spec")
    cards = _cards(report, str(selections.get("run_name") or "candidate"))
    rows: list[dict[str, Any]] = []
    seen_keys: set[tuple[str, int]] = set()
    for selection in selections.get("selections") or []:
        key = (str(selection["task_id"]), int(selection["trial_index"]))
        if key in seen_keys:
            raise ValueError(f"Duplicate selected task/trial: {key}")
        seen_keys.add(key)
        if key not in cards:
            raise KeyError(f"Selected task/trial not found: {key}")
        card = cards[key]
        result_path, simulation = _simulation(card)
        messages = list(simulation.get("messages") or [])
        calls = _calls(messages)
        snapshot_path = result_path.with_name("task_snapshot.json")
        snapshot = _load(snapshot_path)
        rows.append(
            {
                "review_id": f"candidate:{key[0]}:{key[1]}",
                "task_id": key[0],
                "trial_index": key[1],
                "candidate_role": selection["candidate_role"],
                "codex_proposal": {
                    "status": "PROPOSED_BY_CODEX",
                    "rationale": selection["rationale"],
                    "is_human_gold": False,
                },
                "task_goal": (snapshot.get("user_scenario") or {}).get("instructions"),
                "post_run_benchmark_evidence": {
                    "visibility": "REVIEW_ONLY_NOT_MODEL_INPUT",
                    "evaluation_criteria": snapshot.get("evaluation_criteria") or {},
                    "outcome": card["outcome"],
                },
                "trajectory_digest": {
                    "user_turns": [
                        {
                            "event_index": index,
                            "content": str(message.get("content") or "")[:1200],
                        }
                        for index, message in enumerate(messages)
                        if message.get("role") == "user"
                    ],
                    "consecutive_tool_groups": _group_calls(calls),
                    "write_calls": [call for call in calls if call["is_write"]],
                    "tool_errors": [call for call in calls if call["result_error"]],
                    "final_answer": _final_answer(messages),
                    "evaluation_card": {
                        "infrastructure": card["infrastructure"],
                        "outcome": card["outcome"],
                        "tool_use": card["tool_use"],
                        "dialogue_use": card.get("dialogue_use") or {},
                        "policy_diagnostic": card["policy_diagnostic"],
                    },
                },
                "source": {
                    "returned_results": str(result_path),
                    "returned_results_sha256": _sha256(result_path),
                    "task_snapshot": str(snapshot_path),
                    "task_snapshot_sha256": _sha256(snapshot_path),
                },
                "human_review": {
                    "status": "PENDING",
                    "decision": "",
                    "reviewer_id": "",
                    "rationale": "",
                },
                "training_release_allowed": False,
            }
        )

    packet_path = output / "review_packet.json"
    packet = {
        "schema_version": SCHEMA_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source_evaluation_cards": {
            "path": str(report_path),
            "sha256": _sha256(report_path),
        },
        "source_selection": {
            "path": str(selected_path),
            "sha256": _sha256(selected_path),
        },
        "review_scope": {
            "row_count": len(rows),
            "human_acceptance_required": True,
            "preference_or_rl_release_allowed": False,
        },
        "rows": rows,
    }
    packet_path.write_text(
        json.dumps(packet, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "review_packet": {
            "path": str(packet_path),
            "sha256": _sha256(packet_path),
            "row_count": len(rows),
        },
        "contains_private_trajectory_text": True,
        "safe_to_publish": False,
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Audit frozen SFT residuals without running models."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    for name in ("verify", "review-packet"):
        child = subparsers.add_parser(name)
        child.add_argument("--evaluation-cards", type=Path, required=True)
        child.add_argument("--spec", type=Path, required=True)
        child.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "verify":
        result = verify_residual_tasks(
            evaluation_cards=args.evaluation_cards,
            spec_path=args.spec,
            output_dir=args.output,
        )
    else:
        result = build_preference_review_packet(
            evaluation_cards=args.evaluation_cards,
            selection_path=args.spec,
            output_dir=args.output,
        )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
