from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from dataclasses import asdict
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Mapping, Sequence

from src.evaluation.retail_predicates import (
    RetailPredicateContext,
    evaluate_retail_predicate,
)
from src.evaluation.strict_task_evaluator import aggregate_strict_task_evaluation
from src.evaluation.task_rubric import load_task_rubric


TARGET_TASK_IDS = ("21", "24", "50", "59", "107")
STAGES = ("base", "sft", "rl")
SCHEMA_VERSION = "retail-strict-shadow-replay-v1"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def _jsonable(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return payload


def _normalise_messages(raw_messages: Any) -> tuple[dict[str, Any], ...]:
    if not isinstance(raw_messages, list):
        raise ValueError("simulation messages must be an array")
    messages: list[dict[str, Any]] = []
    for index, raw in enumerate(raw_messages):
        if not isinstance(raw, Mapping):
            raise ValueError(f"messages[{index}] must be an object")
        role = raw.get("role")
        if not isinstance(role, str):
            raise ValueError(f"messages[{index}].role must be a string")
        message: dict[str, Any] = {"role": role, "content": raw.get("content")}
        if role == "assistant":
            raw_calls = raw.get("tool_calls") or []
            if not isinstance(raw_calls, list):
                raise ValueError(f"messages[{index}].tool_calls must be an array")
            calls: list[dict[str, Any]] = []
            for call_index, raw_call in enumerate(raw_calls):
                if not isinstance(raw_call, Mapping):
                    raise ValueError(
                        f"messages[{index}].tool_calls[{call_index}] must be an object"
                    )
                calls.append(
                    {
                        "id": raw_call.get("id"),
                        "name": raw_call.get("name"),
                        "arguments": raw_call.get("arguments", {}),
                    }
                )
            message["tool_calls"] = calls
        elif role == "tool":
            message["tool_call_id"] = raw.get("tool_call_id", raw.get("id"))
            error = raw.get("error")
            if isinstance(error, bool):
                message["success"] = not error
        messages.append(message)
    return tuple(messages)


def _last_assistant_answer(messages: Sequence[Mapping[str, Any]]) -> str | None:
    for message in reversed(messages):
        content = message.get("content")
        if message.get("role") == "assistant" and isinstance(content, str) and content.strip():
            return content
    return None


def _extract_single_simulation(payload: Mapping[str, Any], task_id: str) -> Mapping[str, Any]:
    simulations = payload.get("simulations")
    if not isinstance(simulations, list) or len(simulations) != 1:
        raise ValueError("returned_results.json must contain exactly one simulation")
    simulation = simulations[0]
    if not isinstance(simulation, Mapping):
        raise ValueError("simulation must be an object")
    if str(simulation.get("task_id")) != task_id:
        raise ValueError(
            f"simulation task_id mismatch: expected {task_id}, observed {simulation.get('task_id')}"
        )
    return simulation


def _predicate_payload(result: Any) -> dict[str, Any]:
    return _jsonable(asdict(result))


def _evaluation_payload(evaluation: Any) -> dict[str, Any]:
    return _jsonable(asdict(evaluation))


def evaluate_source_artifact(
    *,
    task_id: str,
    stage: str,
    source_path: Path,
    rubric_path: Path,
) -> dict[str, Any]:
    before_hash = sha256_file(source_path)
    rubric_hash = sha256_file(rubric_path)
    payload = _load_json(source_path)
    simulation = _extract_single_simulation(payload, task_id)
    reward_info = simulation.get("reward_info")
    common = {
        "task_id": task_id,
        "stage": stage,
        "source_path": str(source_path.resolve()),
        "source_sha256": before_hash,
        "rubric_path": str(rubric_path.resolve()),
        "rubric_sha256": rubric_hash,
        "termination_reason": simulation.get("termination_reason"),
    }
    if reward_info is None:
        after_hash = sha256_file(source_path)
        if after_hash != before_hash:
            raise RuntimeError(f"source artifact changed during read: {source_path}")
        return {
            **common,
            "status": "INFRASTRUCTURE_FAILURE",
            "source_unchanged": True,
            "tau2_result": None,
            "strict_evaluation": None,
            "reason": "reward_info is absent; this row is not counted as a model failure",
        }

    messages = _normalise_messages(simulation.get("messages"))
    final_answer = _last_assistant_answer(messages)
    rubric = load_task_rubric(rubric_path)
    context = RetailPredicateContext(
        messages=messages,
        final_answer=final_answer,
        stopped=simulation.get("termination_reason") is not None,
        metadata={
            "source_path": str(source_path.resolve()),
            "source_sha256": before_hash,
            "missing_semantic_sidecars": [
                "initial_state",
                "final_state",
                "confirmation_message_index_by_call_id",
                "latest_intent_revision_message_index_by_call_id",
                "claim_checks",
            ],
        },
    )
    predicate_results = tuple(
        evaluate_retail_predicate(spec, context) for spec in rubric.predicates
    )
    evaluation = aggregate_strict_task_evaluation(
        rubric,
        predicate_results,
        tau2_result=reward_info,
    )
    after_hash = sha256_file(source_path)
    if after_hash != before_hash:
        raise RuntimeError(f"source artifact changed during read: {source_path}")
    return {
        **common,
        "status": "EVALUATED" if evaluation.evaluation_valid else "EVALUATED_EVIDENCE_INCOMPLETE",
        "source_unchanged": True,
        "tau2_result": reward_info,
        "strict_evaluation": _evaluation_payload(evaluation),
        "predicate_results": [_predicate_payload(result) for result in predicate_results],
        "evidence_contract": {
            "raw_messages_available": True,
            "semantic_sidecars_available": False,
            "missing_semantic_sidecars": list(context.metadata["missing_semantic_sidecars"]),
            "rule": "Missing evidence produces ERROR and never becomes model FAIL or strict PASS.",
        },
    }


def default_source_matrix(root: Path) -> dict[str, dict[str, Path | None]]:
    base = root / "_local_private_runs" / "teacher_eval_base_v1" / "private_evaluation"
    sft = root / "_local_private_runs" / "teacher_eval_sft_v1" / "private_evaluation"
    rerun_107 = (
        root
        / "_local_private_runs"
        / "teacher_eval_sft_task107_rerun_20260820_v1"
        / "private_evaluation"
        / "task_107"
        / "returned_results.json"
    )
    return {
        "base": {
            task_id: None
            if task_id == "21"
            else base / f"task_{task_id}" / "returned_results.json"
            for task_id in TARGET_TASK_IDS
        },
        "sft": {
            task_id: (
                None
                if task_id == "21"
                else rerun_107
                if task_id == "107"
                else sft / f"task_{task_id}" / "returned_results.json"
            )
            for task_id in TARGET_TASK_IDS
        },
        "rl": {task_id: None for task_id in TARGET_TASK_IDS},
    }


def build_shadow_report(root: Path, *, command: str | None = None) -> dict[str, Any]:
    matrix = default_source_matrix(root)
    rows: list[dict[str, Any]] = []
    for task_id in TARGET_TASK_IDS:
        rubric_path = root / "configs" / "evaluation" / "retail_strict_v1" / f"task_{task_id}.json"
        if not rubric_path.is_file():
            raise FileNotFoundError(f"rubric is missing: {rubric_path}")
        for stage in STAGES:
            source_path = matrix[stage][task_id]
            if source_path is None:
                if stage == "rl":
                    status = "NOT_EVALUATED_NO_TASK_OVERLAP"
                    reason = (
                        "Existing RL artifacts cover identity-authentication tasks "
                        "7/10/11/13/15/20/22 under a truncated protocol; they are not "
                        "comparable to this full-task rubric."
                    )
                else:
                    status = "NOT_EVALUATED_SOURCE_MISSING"
                    reason = "Task 21 is absent from the frozen Qwen Base/SFT task set."
                rows.append(
                    {
                        "task_id": task_id,
                        "stage": stage,
                        "status": status,
                        "source_path": None,
                        "source_sha256": None,
                        "rubric_path": str(rubric_path.resolve()),
                        "rubric_sha256": sha256_file(rubric_path),
                        "reason": reason,
                    }
                )
                continue
            if not source_path.is_file():
                raise FileNotFoundError(f"bound source artifact is missing: {source_path}")
            rows.append(
                evaluate_source_artifact(
                    task_id=task_id,
                    stage=stage,
                    source_path=source_path,
                    rubric_path=rubric_path,
                )
            )

    status_counts: dict[str, int] = {}
    for row in rows:
        status = str(row["status"])
        status_counts[status] = status_counts.get(status, 0) + 1
    plan_path = root / "configs" / "execution" / "strict_eval_p0_plan_v1.json"
    return {
        "schema_version": SCHEMA_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "read_only": True,
        "execution": {
            "command": command,
            "plan_path": str(plan_path.resolve()),
            "plan_sha256": sha256_file(plan_path),
        },
        "claim_scope": {
            "comparable_strict_rows": "Base/SFT tasks 24, 50, 59, and 107 only",
            "task_21": "source missing for Base and SFT",
            "rl": "no strict-task overlap and protocol-incompatible",
            "metric_improvement_claim_allowed": False,
            "three_way_comparison_claim_allowed": False,
        },
        "target_task_ids": list(TARGET_TASK_IDS),
        "stages": list(STAGES),
        "status_counts": status_counts,
        "rows": rows,
    }


def write_shadow_report(
    root: Path,
    output_dir: Path,
    *,
    command: str | None = None,
) -> Path:
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite existing output: {output_dir}")
    report = build_shadow_report(root, command=command)
    output_dir.mkdir(parents=True, exist_ok=False)
    output_path = output_dir / "strict_shadow_report.json"
    output_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return output_path


def main() -> int:
    parser = argparse.ArgumentParser(description="Read-only strict shadow evaluation")
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("output/strict_eval_p0/shadow_replay_v1"),
    )
    args = parser.parse_args()
    root = args.root.resolve()
    output_dir = args.output_dir.resolve()
    command = subprocess.list2cmdline(
        [
            "python",
            "-m",
            "src.evaluation.run_strict_shadow_eval",
            "--root",
            str(root),
            "--output-dir",
            str(output_dir),
        ]
    )
    output_path = write_shadow_report(root, output_dir, command=command)
    print(output_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
