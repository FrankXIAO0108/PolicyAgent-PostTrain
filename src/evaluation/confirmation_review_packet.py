from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


SCHEMA_VERSION = "retail-confirmation-review-packet-v1.0.0"
ALLOWED_PROPOSAL_LABELS = {"ACCEPTABLE", "POLICY_VIOLATION", "UNCERTAIN"}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _simulation(path: Path) -> dict[str, Any]:
    simulations = _load(path).get("simulations") or []
    if len(simulations) != 1:
        raise ValueError(f"Expected one simulation in {path}")
    return simulations[0]


def _truncate(value: Any, limit: int = 1600) -> str:
    text = str(value or "")
    return text if len(text) <= limit else text[:limit] + "...[TRUNCATED]"


def _review_excerpt(value: Any, limit: int = 6000) -> str:
    """Keep the action-detail tail of long proposal messages reviewable."""

    text = str(value or "")
    if len(text) <= limit:
        return text
    head_limit = min(1200, limit // 3)
    tail_limit = limit - head_limit
    return (
        text[:head_limit]
        + "\n...[MIDDLE TRUNCATED; SOURCE HASH PRESERVED]...\n"
        + text[-tail_limit:]
    )


def _find_write(
    messages: list[dict[str, Any]], tool_call_id: str
) -> tuple[int, dict[str, Any]]:
    matches = []
    for index, message in enumerate(messages):
        if message.get("role") != "assistant":
            continue
        for call in message.get("tool_calls") or []:
            if str(call.get("id") or "") == tool_call_id:
                matches.append((index, call))
    if len(matches) != 1:
        raise ValueError(
            f"Expected one tool call {tool_call_id!r}, found {len(matches)}"
        )
    return matches[0]


def _conversation_window(
    messages: list[dict[str, Any]], write_index: int, limit: int = 6
) -> list[dict[str, Any]]:
    candidates = [
        {
            "event_index": index,
            "role": str(message.get("role") or ""),
            "content": _review_excerpt(message.get("content")),
        }
        for index, message in enumerate(messages[:write_index])
        if message.get("role") in {"assistant", "user"}
        and str(message.get("content") or "").strip()
    ]
    return candidates[-limit:]


def _tool_result(
    messages: list[dict[str, Any]], write_index: int, tool_call_id: str
) -> dict[str, Any] | None:
    for message in messages[write_index + 1 :]:
        if message.get("role") != "tool":
            continue
        if str(message.get("id") or "") == tool_call_id:
            return {
                "error": bool(message.get("error", False)),
                "content_excerpt": _truncate(message.get("content"), 500),
            }
    return None


def _proposal_index(
    proposals_path: Path | None, source_audit_sha256: str
) -> tuple[dict[str, dict[str, Any]], dict[str, Any] | None]:
    if proposals_path is None:
        return {}, None
    proposals_path = proposals_path.resolve()
    payload = _load(proposals_path)
    if payload.get("source_audit_sha256") != source_audit_sha256:
        raise ValueError("Proposal source audit hash does not match the review source")
    index: dict[str, dict[str, Any]] = {}
    for proposal in payload.get("proposals") or []:
        review_id = str(proposal.get("review_id") or "")
        if not review_id or review_id in index:
            raise ValueError(f"Missing or duplicate proposal review_id: {review_id!r}")
        if proposal.get("status") != "PROPOSED_BY_CODEX":
            raise ValueError(f"Invalid proposal status for {review_id}")
        if proposal.get("label") not in ALLOWED_PROPOSAL_LABELS:
            raise ValueError(f"Invalid proposal label for {review_id}")
        if not str(proposal.get("rationale") or "").strip():
            raise ValueError(f"Missing proposal rationale for {review_id}")
        index[review_id] = proposal
    return index, {
        "path": str(proposals_path),
        "sha256": _sha256(proposals_path),
        "schema_version": payload.get("schema_version"),
    }


def build_review_packet(
    audit_path: Path,
    output_dir: Path,
    proposals_path: Path | None = None,
) -> dict[str, Any]:
    audit_path = audit_path.resolve()
    output_dir = output_dir.resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty output: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    audit = _load(audit_path)
    source_audit_sha256 = _sha256(audit_path)
    proposals, proposal_source = _proposal_index(
        proposals_path, source_audit_sha256
    )
    rows = []
    source_cache: dict[Path, list[dict[str, Any]]] = {}
    for pair in audit.get("pairs") or []:
        for run_name in ("run_a", "run_b"):
            row = pair[run_name]
            artifact = Path(row["artifact"]["path"]).resolve()
            if _sha256(artifact) != row["artifact"]["sha256"]:
                raise ValueError(f"Source artifact hash mismatch: {artifact}")
            if artifact not in source_cache:
                source_cache[artifact] = list(_simulation(artifact).get("messages") or [])
            messages = source_cache[artifact]
            for check in row["confirmation_diagnostics"]["checks"]:
                binding = check["parameter_binding"]
                if not check["confirmed"] or binding["verdict"] != "REVIEW":
                    continue
                call_id = str(check["tool_call_id"])
                write_index, call = _find_write(messages, call_id)
                review_id = f"{row['task_id']}:{run_name}:{call_id}"
                proposal = proposals.get(review_id)
                rows.append(
                    {
                        "review_id": review_id,
                        "task_id": row["task_id"],
                        "run_name": run_name,
                        "source": {
                            "path": str(artifact),
                            "sha256": row["artifact"]["sha256"],
                        },
                        "benchmark": row["benchmark"],
                        "write": {
                            "event_index": write_index,
                            "tool_call_id": call_id,
                            "tool": str(call.get("name") or ""),
                            "arguments": dict(call.get("arguments") or {}),
                            "result": _tool_result(messages, write_index, call_id),
                        },
                        "confirmation_context": _conversation_window(
                            messages, write_index
                        ),
                        "automatic_diagnostic": binding,
                        "codex_proposal": {
                            "status": (
                                proposal["status"] if proposal else "PENDING"
                            ),
                            "label": proposal["label"] if proposal else "",
                            "rationale": (
                                proposal["rationale"] if proposal else ""
                            ),
                            "allowed_labels": sorted(ALLOWED_PROPOSAL_LABELS),
                        },
                        "human_review": {
                            "status": "PENDING",
                            "decision": "",
                            "reviewer_id": "",
                            "rationale": "",
                        },
                    }
                )

    selected_ids = {row["review_id"] for row in rows}
    if proposal_source is not None and set(proposals) != selected_ids:
        missing = sorted(selected_ids - set(proposals))
        extra = sorted(set(proposals) - selected_ids)
        raise ValueError(
            f"Proposal coverage mismatch; missing={missing}, extra={extra}"
        )

    packet_path = output_dir / "review_packet.json"
    packet = {
        "schema_version": SCHEMA_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source_audit": {
            "path": str(audit_path),
            "sha256": source_audit_sha256,
            "schema_version": audit.get("schema_version"),
        },
        "review_scope": {
            "selection": "confirmed writes with parameter_binding=REVIEW",
            "row_count": len(rows),
            "codex_proposals_are_human_gold": False,
            "human_acceptance_required": True,
            "reward_integration_allowed": False,
            "proposal_source": proposal_source,
        },
        "rows": rows,
    }
    packet_path.write_text(
        json.dumps(packet, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "source_audit": packet["source_audit"],
        "proposal_source": proposal_source,
        "review_packet": {
            "path": str(packet_path),
            "sha256": _sha256(packet_path),
            "row_count": len(rows),
        },
        "contains_private_trajectory_text": True,
        "safe_to_publish": False,
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build a private evidence packet for confirmation REVIEW rows."
    )
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--proposals", type=Path)
    args = parser.parse_args()
    print(
        json.dumps(
            build_review_packet(args.audit, args.output, args.proposals), indent=2
        )
    )


if __name__ == "__main__":
    main()
