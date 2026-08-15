"""Build ENVIRONMENT_REPLAY corrections for tau2 teacher trajectory candidates.

A correction applies three deterministic operations to a frozen candidate:

- ``tool_call_serialization``: split assistant turns containing multiple
  parallel tool calls into serial turns, each immediately followed by its
  matched frozen tool observation (matched by tool-call id);
- ``assistant_text_cleanup``: remove narration text from every assistant
  turn that carries tool calls (no text/tool mixing);
- ``pii_masking``: replace optional exposed internal identifiers in every
  assistant text message (for example a payment method id). Tool-call
  arguments and frozen tool observations are never rewritten.

Corrected tool observations are never hand-written: they are the frozen
observations from the source simulation. The corrected message history is
replayed through the pinned tau2 environment and the resulting database hash
must match the original replay, which is recorded in the replay manifest.

Tau2 imports are deferred so message-level unit tests run without the
upstream checkout.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

AUTHOR_DEFAULT = "tau2_teacher_correction_pipeline_v1"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=str) + "\n",
        encoding="utf-8",
    )


def _call_id(call: dict[str, Any]) -> str:
    return str(call.get("id") or "")


def _tool_results_by_id(messages: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {
        str(message.get("id") or message.get("tool_call_id") or ""): message
        for message in messages
        if str(message.get("role", "")) == "tool"
    }


def serialize_and_clean(messages: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Split parallel tool-call turns and strip narration on tool-call turns."""
    tools_by_id = _tool_results_by_id(messages)
    corrected: list[dict[str, Any]] = []
    change_log: list[dict[str, Any]] = []
    for index, message in enumerate(messages):
        role = str(message.get("role", ""))
        calls = message.get("tool_calls") or []
        if role == "tool":
            continue
        if role != "assistant" or not calls:
            corrected.append(message)
            continue
        if str(message.get("content") or "").strip():
            change_log.append(
                {
                    "category": "assistant_text_cleanup",
                    "reason": (
                        f"assistant turn {index} narration removed; "
                        "text/tool mixing is forbidden in corrected targets"
                    ),
                }
            )
        if len(calls) > 1:
            change_log.append(
                {
                    "category": "tool_call_serialization",
                    "reason": (
                        f"assistant turn {index} split {len(calls)} parallel "
                        "tool calls into serial turns with matched tool results"
                    ),
                }
            )
        for call in calls:
            turn = dict(message)
            turn["content"] = ""
            turn["tool_calls"] = [call]
            corrected.append(turn)
            result = tools_by_id.get(_call_id(call))
            if result is None:
                raise ValueError(
                    f"assistant turn {index}: no frozen tool result for call {_call_id(call)!r}"
                )
            corrected.append(result)
    return corrected, change_log


def apply_assistant_replacements(
    messages: list[dict[str, Any]], replacements: dict[str, str]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Apply PII masking to every non-empty assistant text message.

    Tool-call arguments and tool observations are left untouched so the
    ENVIRONMENT_REPLAY state-preservation binding stays intact.
    """
    if not replacements:
        return messages, []
    change_log: list[dict[str, Any]] = []
    masked = list(messages)
    found_target = False
    for index, message in enumerate(messages):
        if str(message.get("role", "")) != "assistant":
            continue
        content = str(message.get("content") or "")
        if not content.strip():
            continue
        found_target = True
        updated = content
        for old, new in replacements.items():
            if old in updated:
                updated = updated.replace(old, new)
                change_log.append(
                    {
                        "category": "pii_masking",
                        "reason": (
                            f"assistant turn {index} exposed internal identifier "
                            f"{old!r}; replaced with {new!r}"
                        ),
                    }
                )
        if updated != content:
            masked[index] = {**message, "content": updated}
    if not found_target:
        raise ValueError("no assistant text message to mask")
    return masked, change_log


def validate_structure(messages: list[dict[str, Any]]) -> None:
    """Mirror correction_validation message-structure rules."""
    if not messages:
        raise ValueError("corrected messages must not be empty")
    roles = [str(message.get("role", "")) for message in messages]
    if "user" not in roles or "assistant" not in roles:
        raise ValueError("correction requires user and assistant messages")
    pending_tool_ids: set[str] = set()
    for index, message in enumerate(messages):
        role = str(message.get("role", ""))
        if role not in {"system", "user", "assistant", "tool"}:
            raise ValueError(f"message {index}: unsupported role {role!r}")
        calls = message.get("tool_calls") or []
        content = str(message.get("content") or "")
        if role == "assistant" and calls:
            if len(calls) != 1:
                raise ValueError(f"message {index}: exactly one tool call is allowed")
            if content.strip():
                raise ValueError(f"message {index}: text/tool mixing is forbidden")
            call = calls[0]
            if not call.get("id") or not call.get("name"):
                raise ValueError(f"message {index}: tool call id and name are required")
            if not isinstance(call.get("arguments"), dict):
                raise ValueError(f"message {index}: tool arguments must be an object")
            pending_tool_ids.add(str(call["id"]))
        if role == "tool":
            tool_id = str(message.get("id") or message.get("tool_call_id") or "")
            if tool_id not in pending_tool_ids:
                raise ValueError(f"message {index}: tool result has no matching call")
            pending_tool_ids.remove(tool_id)
    if pending_tool_ids:
        raise ValueError(f"tool calls without results: {sorted(pending_tool_ids)}")


def load_source(run_dir: Path, candidate_id: str) -> tuple[Path, dict[str, Any]]:
    """Return the task results path and the candidate simulation dict."""
    for path in sorted((run_dir / "private_evaluation").glob("task_*/returned_results.json")):
        payload = json.loads(path.read_text(encoding="utf-8-sig"))
        for simulation in payload.get("simulations") or []:
            if str(simulation.get("id") or "") == candidate_id:
                return path, simulation
    raise ValueError(f"candidate {candidate_id!r} not found under {run_dir}")


def replay_corrected(
    tau2_root: Path,
    task_results_path: Path,
    candidate_id: str,
    corrected_messages: list[dict[str, Any]],
    upstream_commit: str,
) -> dict[str, Any]:
    """Replay corrected messages and verify the final database hash is preserved."""
    from tau2.data_model.simulation import SimulationRun

    from src.evaluation.replay_evaluator import Tau2Runtime, replay_task_simulation

    runtime = Tau2Runtime(tau2_root)
    results, raw = runtime.load_results(task_results_path)
    simulation = next(
        item for item in results.simulations if str(item.id) == candidate_id
    )
    task = next(
        item for item in results.tasks if str(item.id) == str(simulation.task_id)
    )
    domain = results.info.environment_info.domain_name
    constructor = runtime.registry.get_env_constructor(domain)
    original = replay_task_simulation(
        task, simulation, constructor=constructor, domain=domain, raw_results=raw
    )
    simulation_dict = simulation.model_dump(mode="json")
    simulation_dict["messages"] = corrected_messages
    corrected_simulation = SimulationRun.model_validate(simulation_dict)
    corrected = replay_task_simulation(
        task,
        corrected_simulation,
        constructor=constructor,
        domain=domain,
        raw_results=raw,
    )
    return {
        "schema_version": "tau2-teacher-correction-replay-v1",
        "candidate_id": candidate_id,
        "upstream_commit": upstream_commit,
        "original_replay": {
            "agent_db_hash": original.agent_hash,
            "gold_db_hash": original.gold_hash,
            "db_match": original.db_match,
            "replay_errors": original.replay_errors,
        },
        "corrected_replay": {
            "agent_db_hash": corrected.agent_hash,
            "gold_db_hash": corrected.gold_hash,
            "db_match": corrected.db_match,
            "replay_errors": corrected.replay_errors,
        },
        "state_preserved": bool(
            original.agent_hash and original.agent_hash == corrected.agent_hash
        ),
    }


def build_correction(
    *,
    run_dir: Path,
    candidate_id: str,
    output_dir: Path,
    tau2_root: Path,
    policy_path: Path,
    upstream_commit: str,
    author_id: str,
    assistant_replacements: dict[str, str],
) -> dict[str, Any]:
    task_results_path, simulation = load_source(run_dir, candidate_id)
    messages = simulation.get("messages") or []
    corrected, change_log = serialize_and_clean(messages)
    corrected, pii_log = apply_assistant_replacements(corrected, assistant_replacements)
    change_log.extend(pii_log)
    validate_structure(corrected)

    replay = replay_corrected(
        tau2_root, task_results_path, candidate_id, corrected, upstream_commit
    )
    if not replay["state_preserved"]:
        raise ValueError(
            f"candidate {candidate_id}: corrected replay did not preserve final state"
        )

    task_id = str(simulation.get("task_id") or "")
    source_path = output_dir / f"source_{candidate_id}.json"
    replay_path = output_dir / f"replay_manifest_{candidate_id}.json"
    correction_path = output_dir / f"corrected_{candidate_id}.json"

    write_json(source_path, {"simulations": [simulation]})
    write_json(replay_path, replay)
    correction = {
        "schema_version": "tau2-teacher-correction-v1",
        "task_id": task_id,
        "candidate_id": candidate_id,
        "author_id": author_id,
        "authored_at": datetime.now(timezone.utc).isoformat(),
        "generation_mode": "ENVIRONMENT_REPLAY",
        "source": {
            "path": str(source_path),
            "sha256": sha256(source_path),
        },
        "source_provenance": {
            "task_results_path": str(task_results_path),
            "task_results_sha256": sha256(task_results_path),
        },
        "policy": {
            "path": str(policy_path),
            "sha256": sha256(policy_path),
        },
        "system_policy": policy_path.read_text(encoding="utf-8-sig"),
        "change_log": change_log,
        "messages": corrected,
        "replay_manifest": {
            "path": str(replay_path),
            "sha256": sha256(replay_path),
        },
    }
    write_json(correction_path, correction)
    return {
        "candidate_id": candidate_id,
        "task_id": task_id,
        "correction": {"path": str(correction_path), "sha256": sha256(correction_path)},
        "replay_manifest": {"path": str(replay_path), "sha256": sha256(replay_path)},
        "source": {"path": str(source_path), "sha256": sha256(source_path)},
        "change_log": change_log,
        "state_preserved": replay["state_preserved"],
        "db_match_gold": replay["corrected_replay"]["db_match"],
    }


def load_plan(plan_path: Path | None) -> dict[str, dict[str, str]]:
    if plan_path is None:
        return {}
    payload = json.loads(plan_path.read_text(encoding="utf-8-sig"))
    plans: dict[str, dict[str, str]] = {}
    for row in payload.get("candidates") or []:
        replacements = dict(row.get("assistant_replacements") or {})
        if not replacements:
            replacements = dict(row.get("final_answer_replacements") or {})
        plans[str(row["candidate_id"])] = replacements
    return plans


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--candidate-id", type=str, action="append", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--tau2-root", type=Path, default=Path(r"D:\tau2-bench"))
    parser.add_argument("--policy", type=Path)
    parser.add_argument("--plan", type=Path)
    parser.add_argument("--author-id", default=AUTHOR_DEFAULT)
    parser.add_argument("--upstream-commit", default="58e5e1ace69302e6982d27014569c03e0ffccdd2")
    args = parser.parse_args()

    output_dir = args.output_dir.resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite existing output: {output_dir}")
    policy_path = (args.policy or args.tau2_root / "data" / "tau2" / "domains" / "retail" / "policy.md").resolve()
    plan = load_plan(args.plan)
    results = []
    for candidate_id in args.candidate_id:
        results.append(
            build_correction(
                run_dir=args.run_dir.resolve(),
                candidate_id=candidate_id,
                output_dir=output_dir,
                tau2_root=args.tau2_root.resolve(),
                policy_path=policy_path,
                upstream_commit=args.upstream_commit,
                author_id=args.author_id,
                assistant_replacements=plan.get(candidate_id, {}),
            )
        )
    summary = {
        "schema_version": "tau2-teacher-correction-run-v1",
        "author_id": args.author_id,
        "policy": {"path": str(policy_path), "sha256": sha256(policy_path)},
        "corrections": results,
    }
    write_json(output_dir / "correction_summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
