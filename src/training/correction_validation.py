from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from src.verifiers.gold_validation import load_jsonl


MODES = {"ASSISTANT_TEXT_EDIT", "ENVIRONMENT_REPLAY"}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def _timestamp(value: str, field: str) -> None:
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as error:
        raise ValueError(f"{field} must be an ISO-8601 timestamp") from error
    if parsed.tzinfo is None:
        raise ValueError(f"{field} must include a timezone")


def _verify_file(path: Path, expected: str, field: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"{field} does not exist: {path}")
    actual = sha256(path)
    if actual != expected.upper():
        raise ValueError(
            f"{field} hash mismatch: expected={expected.upper()}, actual={actual}"
        )


def _raw_messages(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    simulations = payload.get("simulations", [])
    if len(simulations) != 1:
        raise ValueError("Source must contain exactly one frozen simulation")
    return simulations[0].get("messages", [])


def _canonical_observed(
    messages: list[dict[str, Any]], roles: set[str]
) -> list[dict[str, Any]]:
    return [
        {
            "role": str(message.get("role", "")),
            "content": str(message.get("content") or ""),
            "id": message.get("id"),
            "error": message.get("error"),
        }
        for message in messages
        if str(message.get("role", "")) in roles
    ]


def _assistant_calls(messages: list[dict[str, Any]]) -> list[list[dict[str, Any]]]:
    return [
        [
            {
                "id": call.get("id"),
                "name": call.get("name"),
                "arguments": call.get("arguments"),
            }
            for call in (message.get("tool_calls") or [])
        ]
        for message in messages
        if str(message.get("role", "")) == "assistant"
    ]


def _validate_messages(messages: list[dict[str, Any]]) -> None:
    if not messages:
        raise ValueError("Correction messages must not be empty")
    roles = [str(message.get("role", "")) for message in messages]
    if "user" not in roles or "assistant" not in roles:
        raise ValueError("Correction requires user and assistant messages")
    pending_tool_ids: set[str] = set()
    for index, message in enumerate(messages):
        role = str(message.get("role", ""))
        if role not in {"system", "user", "assistant", "tool"}:
            raise ValueError(f"Message {index}: unsupported role {role!r}")
        calls = message.get("tool_calls") or []
        content = str(message.get("content") or "")
        if role == "assistant" and calls:
            if len(calls) != 1:
                raise ValueError(f"Message {index}: exactly one tool call is allowed")
            if content.strip():
                raise ValueError(f"Message {index}: text/tool mixing is forbidden")
            call = calls[0]
            if not call.get("id") or not call.get("name"):
                raise ValueError(f"Message {index}: tool call id and name are required")
            if not isinstance(call.get("arguments"), dict):
                raise ValueError(f"Message {index}: tool arguments must be an object")
            pending_tool_ids.add(str(call["id"]))
        if role == "tool":
            tool_id = str(message.get("id") or message.get("tool_call_id") or "")
            if tool_id not in pending_tool_ids:
                raise ValueError(
                    f"Message {index}: tool result has no matching earlier call"
                )
            pending_tool_ids.remove(tool_id)
    if pending_tool_ids:
        raise ValueError(
            f"Correction has tool calls without results: {sorted(pending_tool_ids)}"
        )


@dataclass(frozen=True, slots=True)
class Approval:
    task_id: str
    correction_sha256: str
    reviewer_id: str
    verdict: str
    reviewed_at: str
    rationale: str

    @classmethod
    def from_dict(cls, row: dict[str, Any]) -> Approval:
        approval = cls(
            task_id=str(row["task_id"]),
            correction_sha256=str(row.get("correction_sha256", "")).upper(),
            reviewer_id=str(row.get("reviewer_id", "")).strip(),
            verdict=str(row.get("verdict", "")).upper(),
            reviewed_at=str(row.get("reviewed_at", "")).strip(),
            rationale=str(row.get("rationale", "")).strip(),
        )
        if not approval.correction_sha256:
            raise ValueError("Approval correction_sha256 is required")
        if not approval.reviewer_id:
            raise ValueError("Approval reviewer_id is required")
        if approval.verdict not in {"APPROVE", "REJECT"}:
            raise ValueError("Approval verdict must be APPROVE or REJECT")
        _timestamp(approval.reviewed_at, "reviewed_at")
        if not approval.rationale:
            raise ValueError("Approval rationale is required")
        evidence = row.get("evidence_files", [])
        if not isinstance(evidence, list) or not evidence:
            raise ValueError("Approval evidence_files is required")
        return approval


def validate_correction(
    correction_path: Path | None,
    approvals_path: Path | None,
) -> dict[str, Any]:
    if correction_path is None:
        return {
            "ready": False,
            "reasons": ["No corrected trajectory supplied."],
            "task_id": None,
            "correction_sha256": None,
            "inputs": {"correction": None, "approvals": None},
        }
    correction_hash = sha256(correction_path)
    payload = json.loads(correction_path.read_text(encoding="utf-8-sig"))
    task_id = str(payload["task_id"])
    author_id = str(payload.get("author_id", "")).strip()
    authored_at = str(payload.get("authored_at", "")).strip()
    mode = str(payload.get("generation_mode", "")).upper()
    if not author_id:
        raise ValueError("author_id is required")
    _timestamp(authored_at, "authored_at")
    if mode not in MODES:
        raise ValueError(f"generation_mode must be one of {sorted(MODES)}")

    source = payload.get("source", {})
    source_path = Path(str(source["path"]))
    source_hash = str(source.get("sha256", "")).upper()
    _verify_file(source_path, source_hash, "source")
    policy = payload.get("policy", {})
    policy_path = Path(str(policy["path"]))
    policy_hash = str(policy.get("sha256", "")).upper()
    _verify_file(policy_path, policy_hash, "policy")
    if str(payload.get("system_policy", "")) != policy_path.read_text(
        encoding="utf-8-sig"
    ):
        raise ValueError("system_policy must exactly match the frozen policy file")

    change_log = payload.get("change_log")
    if not isinstance(change_log, list) or not change_log:
        raise ValueError("change_log is required")
    for index, change in enumerate(change_log):
        if not change.get("reason") or not change.get("category"):
            raise ValueError(f"change_log[{index}] requires category and reason")

    messages = payload.get("messages")
    if not isinstance(messages, list):
        raise ValueError("messages must be a list")
    _validate_messages(messages)
    source_messages = _raw_messages(source_path)
    if mode == "ASSISTANT_TEXT_EDIT":
        if _canonical_observed(messages, {"user", "tool"}) != _canonical_observed(
            source_messages, {"user", "tool"}
        ):
            raise ValueError(
                "ASSISTANT_TEXT_EDIT must preserve all user/tool observations"
            )
        if _assistant_calls(messages) != _assistant_calls(source_messages):
            raise ValueError(
                "ASSISTANT_TEXT_EDIT must preserve assistant tool calls"
            )
    else:
        replay = payload.get("replay_manifest", {})
        replay_path = Path(str(replay.get("path", "")))
        replay_hash = str(replay.get("sha256", "")).upper()
        if not replay_hash:
            raise ValueError("ENVIRONMENT_REPLAY requires replay manifest and hash")
        _verify_file(replay_path, replay_hash, "replay_manifest")

    reasons: list[str] = []
    approvals: list[Approval] = []
    matching: list[Approval] = []
    if approvals_path is None:
        reasons.append("No independent correction approvals supplied.")
    else:
        approvals = [
            Approval.from_dict(row) for row in load_jsonl(approvals_path)
        ]
        matching = [
            approval
            for approval in approvals
            if approval.task_id == task_id
            and approval.correction_sha256 == correction_hash
        ]
        reviewer_ids = {approval.reviewer_id for approval in matching}
        if author_id in reviewer_ids:
            reasons.append("Correction author cannot approve their own artifact.")
        if len(reviewer_ids) < 2:
            reasons.append("Two independent reviewer identities are required.")
        if any(approval.verdict == "REJECT" for approval in matching):
            reasons.append("At least one matching reviewer rejected the correction.")
        if len([row for row in matching if row.verdict == "APPROVE"]) < 2:
            reasons.append("Two matching APPROVE decisions are required.")

    return {
        "ready": not reasons,
        "reasons": reasons,
        "task_id": task_id,
        "generation_mode": mode,
        "source_sha256": source_hash,
        "policy_sha256": policy_hash,
        "correction_sha256": correction_hash,
        "approval_count": len(matching),
        "inputs": {
            "correction": {
                "path": str(correction_path),
                "sha256": correction_hash,
            },
            "approvals": (
                {
                    "path": str(approvals_path),
                    "sha256": sha256(approvals_path),
                }
                if approvals_path is not None
                else None
            ),
        },
    }


def write_report(result: dict[str, Any], output_dir: Path) -> None:
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty output: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "correction_validation.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate corrected trajectories and independent approvals."
    )
    parser.add_argument("--correction", type=Path)
    parser.add_argument("--approvals", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = validate_correction(args.correction, args.approvals)
    write_report(result, args.output)
    print(json.dumps(result, ensure_ascii=False))
    if not result["ready"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
