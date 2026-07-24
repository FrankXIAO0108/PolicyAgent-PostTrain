from __future__ import annotations

import argparse
import json
import re
from json import JSONDecoder
from pathlib import Path
from typing import Any, Iterable

from .schemas import ArtifactBundle, MessageEvent, ToolCall


_STEP_RE = re.compile(
    r"^\S.*?orchestrator:step:\d+ - Step (?P<step>\d+)\.\s*$",
    re.MULTILINE,
)
_ROLE_RE = re.compile(r"^From role: Role\.(?P<role>[A-Z_]+)\s*$", re.MULTILINE)
_TOOL_CALL_RE = re.compile(
    r"ToolCall \(from assistant\)\s*"
    r"id:\s*(?P<id>[^\r\n]+)\s*"
    r"name:\s*(?P<name>[^\r\n]+)\s*"
    r"arguments:\s*",
    re.MULTILINE,
)


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8-sig") as handle:
        value = json.load(handle)
    return value if isinstance(value, dict) else {}


def _content_to_text(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(str(part) for part in content)
    return json.dumps(content, ensure_ascii=False, sort_keys=True)


def _arguments_from_debug(call: dict[str, Any]) -> dict[str, Any]:
    arguments: Any = call.get("arguments")
    function = call.get("function")
    if arguments is None and isinstance(function, dict):
        arguments = function.get("arguments")
    if isinstance(arguments, dict):
        return arguments
    if isinstance(arguments, str):
        try:
            parsed = json.loads(arguments)
        except json.JSONDecodeError:
            return {"_raw": arguments}
        return parsed if isinstance(parsed, dict) else {"_value": parsed}
    return {}


def _events_from_debug(path: Path) -> list[MessageEvent]:
    payload = _read_json(path)
    request = payload.get("request", {})
    messages = request.get("messages", []) if isinstance(request, dict) else []
    events: list[MessageEvent] = []

    for raw in messages:
        if not isinstance(raw, dict):
            continue
        calls = [
            ToolCall(
                id=str(call.get("id", "")),
                name=str(
                    call.get("name")
                    or (call.get("function") or {}).get("name")
                    or ""
                ),
                arguments=_arguments_from_debug(call),
            )
            for call in (raw.get("tool_calls") or [])
            if isinstance(call, dict)
        ]
        events.append(
            MessageEvent(
                index=len(events),
                role=str(raw.get("role", "")).lower(),
                content=_content_to_text(raw.get("content")),
                tool_calls=calls,
                tool_call_id=raw.get("tool_call_id"),
                tool_error=bool(raw.get("error", False)),
                source=str(path),
            )
        )

    # An agent-response debug file stores the newly generated assistant message
    # under ``response`` rather than in ``request.messages``.  Without this
    # append, the loader silently drops the final assistant turn and cannot
    # verify post-action claims.
    response = payload.get("response")
    if isinstance(response, dict):
        response_calls = [
            ToolCall(
                id=str(call.get("id", "")),
                name=str(
                    call.get("name")
                    or (call.get("function") or {}).get("name")
                    or ""
                ),
                arguments=_arguments_from_debug(call),
            )
            for call in (response.get("tool_calls") or [])
            if isinstance(call, dict)
        ]
        response_content = _content_to_text(response.get("content"))
        if response_content or response_calls:
            events.append(
                MessageEvent(
                    index=len(events),
                    role="assistant",
                    content=response_content,
                    tool_calls=response_calls,
                    tool_error=bool(response.get("error", False)),
                    source=str(path),
                )
            )
    return events


def _extract_json_at(text: str, start: int) -> tuple[dict[str, Any], int]:
    while start < len(text) and text[start].isspace():
        start += 1
    try:
        value, length = JSONDecoder().raw_decode(text[start:])
    except json.JSONDecodeError:
        line = text[start:].splitlines()[0] if text[start:] else ""
        return {"_raw": line}, start + len(line)
    if not isinstance(value, dict):
        value = {"_value": value}
    return value, start + length


def _tool_calls_from_log_block(block: str) -> list[ToolCall]:
    calls: list[ToolCall] = []
    for match in _TOOL_CALL_RE.finditer(block):
        arguments, _ = _extract_json_at(block, match.end())
        calls.append(
            ToolCall(
                id=match.group("id").strip(),
                name=match.group("name").strip(),
                arguments=arguments,
            )
        )
    return calls


def _content_from_log_block(block: str) -> str:
    marker = "\ncontent:"
    start = block.find(marker)
    if start < 0:
        return ""
    start += len(marker)
    end_candidates = [
        position
        for token in ("\nToolCalls:", "\nis_final_chunk:", "\n202")
        if (position := block.find(token, start)) >= 0
    ]
    end = min(end_candidates) if end_candidates else len(block)
    return block[start:end].strip()


def _events_from_task_log(path: Path) -> list[MessageEvent]:
    text = path.read_text(encoding="utf-8-sig", errors="replace")
    matches = list(_STEP_RE.finditer(text))
    events: list[MessageEvent] = []

    for offset, match in enumerate(matches):
        end = matches[offset + 1].start() if offset + 1 < len(matches) else len(text)
        block = text[match.end() : end]
        role_match = _ROLE_RE.search(block)
        if role_match is None:
            continue
        role = role_match.group("role").lower()
        events.append(
            MessageEvent(
                index=len(events),
                role={"agent": "assistant", "env": "tool"}.get(role, role),
                content=_content_from_log_block(block),
                tool_calls=_tool_calls_from_log_block(block),
                tool_error="error=True" in block,
                source=str(path),
            )
        )
    return events


def _latest_agent_debug(task_dir: Path) -> Path | None:
    candidates = sorted(
        task_dir.rglob("*agent_response*.json"),
        key=lambda path: (path.stat().st_mtime_ns, str(path)),
    )
    return candidates[-1] if candidates else None


def _first_task_log(task_dir: Path) -> Path | None:
    candidates = sorted(task_dir.rglob("task.log"))
    return candidates[0] if candidates else None


def load_trajectory(task_dir: str | Path) -> tuple[list[MessageEvent], Path]:
    """Load a Tau2 trajectory, preferring structured LLM debug messages."""
    root = Path(task_dir).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Task directory does not exist: {root}")

    debug_path = _latest_agent_debug(root)
    if debug_path is not None:
        events = _events_from_debug(debug_path)
        if events:
            return events, debug_path

    log_path = _first_task_log(root)
    if log_path is None:
        raise FileNotFoundError(f"No agent debug JSON or task.log under: {root}")
    events = _events_from_task_log(log_path)
    if not events:
        raise ValueError(f"No trajectory events parsed from: {log_path}")
    return events, log_path


def load_task_artifacts(task_dir: str | Path) -> ArtifactBundle:
    root = Path(task_dir).expanduser().resolve()
    events, source_path = load_trajectory(root)
    return ArtifactBundle(
        task_dir=root,
        events=events,
        task=_read_json(root / "task.json"),
        summary=_read_json(root / "summary.json"),
        source_path=source_path,
    )


def _iter_task_dirs(paths: Iterable[str]) -> Iterable[Path]:
    for raw_path in paths:
        path = Path(raw_path).expanduser().resolve()
        if (path / "task.json").exists():
            yield path
            continue
        yield from sorted(
            candidate.parent
            for candidate in path.rglob("task.json")
            if candidate.is_file()
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Parse Tau2 task trajectories.")
    parser.add_argument("paths", nargs="+", help="Task or experiment directories")
    args = parser.parse_args()

    output = []
    for task_dir in _iter_task_dirs(args.paths):
        bundle = load_task_artifacts(task_dir)
        output.append(
            {
                "task_id": bundle.task_id,
                "task_dir": str(bundle.task_dir),
                "source_path": str(bundle.source_path),
                "events": [event.to_dict() for event in bundle.events],
            }
        )
    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
