from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.evaluation.teacher_eval_cards import analyze_task
from src.evaluation.db_diff import analyze_db_diff
from src.evaluation.replay_evaluator import replay_results_artifact
from src.verifiers.intent_state import is_write_tool


SCHEMA_VERSION = "teacher-paired-human-review-v1.0.0"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _result_path(run_dir: Path, task_id: str) -> Path:
    matches = sorted(
        run_dir.glob(f"private_evaluation/task_{task_id}/returned_results.json")
    )
    if len(matches) != 1:
        raise FileNotFoundError(
            f"Expected one returned_results.json for task {task_id} under {run_dir}"
        )
    return matches[0].resolve()


def _snapshot_path(run_dir: Path, task_id: str) -> Path:
    path = run_dir / "private_evaluation" / f"task_{task_id}" / "task_snapshot.json"
    if not path.is_file():
        raise FileNotFoundError(path)
    return path.resolve()


def _simulation(path: Path) -> dict[str, Any]:
    simulations = _load(path).get("simulations") or []
    if len(simulations) != 1:
        raise ValueError(f"Expected one simulation in {path}")
    return simulations[0]


def _truncate(value: Any, limit: int = 600) -> str:
    text = str(value or "")
    return text if len(text) <= limit else text[:limit] + "...[TRUNCATED]"


def _call_rows(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    tool_results = {
        str(message.get("id")): message
        for message in messages
        if message.get("role") == "tool"
    }
    rows: list[dict[str, Any]] = []
    for event_index, message in enumerate(messages):
        if message.get("role") != "assistant":
            continue
        for call in message.get("tool_calls") or []:
            result = tool_results.get(str(call.get("id")), {})
            rows.append(
                {
                    "event_index": event_index,
                    "call_id": str(call.get("id") or ""),
                    "name": str(call.get("name") or ""),
                    "arguments": dict(call.get("arguments") or {}),
                    "is_write": is_write_tool(str(call.get("name") or "")),
                    "result_error": bool(result.get("error", False)),
                    "result_excerpt": _truncate(result.get("content"), 300),
                }
            )
    return rows


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
                    "event_range": [call["event_index"], call["event_index"]],
                }
            )
        group = groups[-1]
        group["count"] += 1
        group["error_count"] += int(call["result_error"])
        group["last_arguments"] = call["arguments"]
        group["event_range"][1] = call["event_index"]
    return groups


def _final_answer(messages: list[dict[str, Any]]) -> str:
    for message in reversed(messages):
        if message.get("role") == "assistant" and str(
            message.get("content") or ""
        ).strip():
            return str(message["content"]).strip()
    return ""


def _candidate_evidence(
    result_path: Path,
    *,
    run_name: str,
    is_replacement: bool,
    tau2_root: Path | None,
) -> dict[str, Any]:
    simulation = _simulation(result_path)
    messages = list(simulation.get("messages") or [])
    calls = _call_rows(messages)
    card = analyze_task(
        result_path,
        run_name=run_name,
        is_replacement=is_replacement,
    )
    replay_evidence: dict[str, Any] | None = None
    if tau2_root is not None:
        replay = replay_results_artifact(result_path, tau2_root=tau2_root)
        diff = analyze_db_diff(
            replay.initial_state, replay.agent_state, replay.gold_state
        )
        replay_evidence = {
            "replay": replay.to_dict(),
            "db_diff": diff.to_dict(),
        }
    return {
        "run_name": run_name,
        "source": {
            "path": str(result_path),
            "sha256": _sha256(result_path),
            "is_replacement": is_replacement,
        },
        "user_turns": [
            {
                "event_index": index,
                "content": _truncate(message.get("content"), 1000),
            }
            for index, message in enumerate(messages)
            if message.get("role") == "user"
        ],
        "tool_call_sequence": [call["name"] for call in calls],
        "consecutive_tool_groups": _group_calls(calls),
        "write_calls": [call for call in calls if call["is_write"]],
        "tool_errors": [call for call in calls if call["result_error"]],
        "final_answer": _final_answer(messages),
        "evaluation_card": {
            "infrastructure": card["infrastructure"],
            "outcome": card["outcome"],
            "tool_use": card["tool_use"],
            "policy_diagnostic": card["policy_diagnostic"],
            "dimension_card": card["dimension_card"],
        },
        "deterministic_state_replay": replay_evidence,
    }


def build_task_pack(
    *,
    task_id: str,
    base_dir: Path,
    candidate_dir: Path,
    replacement_dir: Path | None = None,
    tau2_root: Path | None = None,
) -> dict[str, Any]:
    base_result = _result_path(base_dir, task_id)
    candidate_root = replacement_dir or candidate_dir
    candidate_result = _result_path(candidate_root, task_id)
    snapshot = _snapshot_path(candidate_root, task_id)
    task = _load(snapshot)
    criteria = task.get("evaluation_criteria") or {}
    return {
        "schema_version": SCHEMA_VERSION,
        "task_id": task_id,
        "review_status": "PENDING_HUMAN_ADJUDICATION",
        "task_snapshot": {
            "path": str(snapshot),
            "sha256": _sha256(snapshot),
        },
        "task_goal": (task.get("user_scenario") or {}).get("instructions"),
        "post_run_benchmark_evidence": {
            "visibility": "REVIEW_ONLY_NOT_MODEL_INPUT",
            "reference_actions": criteria.get("actions") or [],
            "communicate_info": criteria.get("communicate_info") or [],
            "nl_assertions": criteria.get("nl_assertions") or [],
            "reward_basis": criteria.get("reward_basis") or [],
        },
        "base": _candidate_evidence(
            base_result,
            run_name="base",
            is_replacement=False,
            tau2_root=tau2_root,
        ),
        "sft": _candidate_evidence(
            candidate_result,
            run_name="sft",
            is_replacement=replacement_dir is not None,
            tau2_root=tau2_root,
        ),
        "review_questions": [
            "What is the final user intent after all revisions and confirmations?",
            "Which tools are required, allowed, or forbidden for that final intent?",
            "Which reads are necessary for authentication, state lookup, and decision support?",
            "Where is the earliest safe and truthful stopping point?",
            "Did Base or SFT perform redundant exploration, an unsafe write, or a false success claim?",
        ],
    }


def _review_template(task_id: str, pack_path: Path) -> dict[str, Any]:
    return {
        "task_id": task_id,
        "evidence_pack": {"path": str(pack_path), "sha256": _sha256(pack_path)},
        "codex_proposal_status": "PENDING",
        "human_decision": "",
        "human_reviewer_id": "",
        "reviewed_at": "",
        "human_rationale": "",
        "human_corrections": {},
        "allowed_human_decisions": [
            "ACCEPT",
            "CORRECTION_REQUIRED",
            "REJECT",
        ],
    }


def build_batch(
    *,
    base_dir: str | Path,
    candidate_dir: str | Path,
    task_ids: list[str],
    output_dir: str | Path,
    replacements: dict[str, str | Path] | None = None,
    tau2_root: str | Path | None = None,
) -> dict[str, Any]:
    base = Path(base_dir).resolve()
    candidate = Path(candidate_dir).resolve()
    output = Path(output_dir).resolve()
    resolved_tau2 = Path(tau2_root).resolve() if tau2_root is not None else None
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty output: {output}")
    packs_dir = output / "evidence_packs"
    packs_dir.mkdir(parents=True, exist_ok=True)
    template_rows: list[dict[str, Any]] = []
    pack_sources: list[dict[str, Any]] = []
    for task_id in task_ids:
        replacement = (replacements or {}).get(str(task_id))
        pack = build_task_pack(
            task_id=str(task_id),
            base_dir=base,
            candidate_dir=candidate,
            replacement_dir=(Path(replacement).resolve() if replacement else None),
            tau2_root=resolved_tau2,
        )
        pack_path = packs_dir / f"task_{task_id}.json"
        pack_path.write_text(
            json.dumps(pack, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        template_rows.append(_review_template(str(task_id), pack_path))
        pack_sources.append(
            {"task_id": str(task_id), "path": str(pack_path), "sha256": _sha256(pack_path)}
        )
    template_path = output / "human_review_template.jsonl"
    template_path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in template_rows),
        encoding="utf-8",
    )
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "task_ids": [str(task_id) for task_id in task_ids],
        "inputs": {
            "base_dir": str(base),
            "candidate_dir": str(candidate),
            "replacements": {
                str(task_id): str(Path(path).resolve())
                for task_id, path in (replacements or {}).items()
            },
            "tau2_root": str(resolved_tau2) if resolved_tau2 else None,
        },
        "evidence_packs": pack_sources,
        "human_review_template": {
            "path": str(template_path),
            "sha256": _sha256(template_path),
        },
        "label_identity": {
            "codex_proposals_are_human_gold": False,
            "human_acceptance_required": True,
        },
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return manifest


def _replacement(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("replacement must be TASK_ID=RUN_DIR")
    task_id, path = value.split("=", 1)
    return task_id, Path(path)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build compact paired Base/SFT evidence packs for human review."
    )
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--task-id", action="append", required=True)
    parser.add_argument("--replacement", action="append", type=_replacement, default=[])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tau2-root", type=Path)
    args = parser.parse_args()
    manifest = build_batch(
        base_dir=args.base,
        candidate_dir=args.candidate,
        task_ids=args.task_id,
        output_dir=args.output,
        replacements=dict(args.replacement),
        tau2_root=args.tau2_root,
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
