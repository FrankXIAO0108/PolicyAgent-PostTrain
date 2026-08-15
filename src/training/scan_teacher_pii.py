"""Scan teacher trajectories for internal identifiers in assistant text.

Layer-1 pilot quality gate 2 (see
``docs/04_数据治理与后训练/2026-08-15_分层教师Pilot设计.md``): every assistant
text message must be free of internal identifiers before a candidate enters
the correction/masking plan. This scanner implements the audit lesson from
2026-08-15 (a payment-method id leaked in intermediate and final assistant
messages and was only caught during the continuation audit) as a
deterministic, fail-closed check.

Scope decisions:

- Only ``role == "assistant"`` text content is scanned. Tool-call arguments
  and frozen tool observations are the legitimate home of internal
  identifiers and are never flagged.
- Internal identifiers are collected from the frozen payload itself (tool
  results, tool-call arguments, state snapshots) and then looked up verbatim
  in assistant text, so detection is data-driven instead of pattern guessing.
- Distinctive formats (payment-method ids, emails, UUIDs) are also matched
  directly in assistant text as a safety net for identifiers that are
  echoed but absent from the frozen payload.
- Customer-facing business data (order numbers like ``#W2378156`` and
  product/item ids) is not an internal identifier and is never flagged.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


SCHEMA_VERSION = "teacher-pii-scan-v0.1"

PAYMENT_RE = re.compile(r"(?i)\b(?:credit_card|gift_card)_\d+\b")
EMAIL_RE = re.compile(r"\b[\w.+-]+@[\w-]+\.[A-Za-z0-9.-]+\b")
UUID_RE = re.compile(
    r"\b[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}\b",
    re.IGNORECASE,
)
USER_ID_RE = re.compile(r"(?i)\b[a-z]+_[a-z]+_\d{2,4}\b")

# Keys whose string values are internal identifiers when found in payload data.
_ID_KEYS = {"user_id", "payment_method_id", "email"}
_PAYMENT_METHODS_KEY = "payment_methods"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def _payload_messages(payload: Any) -> list[dict[str, Any]]:
    """Extract the message list from supported frozen trajectory formats."""
    if isinstance(payload, dict):
        if isinstance(payload.get("messages"), list):
            return payload["messages"]
        simulations = payload.get("simulations")
        if isinstance(simulations, list) and simulations:
            messages = simulations[0].get("messages")
            if isinstance(messages, list):
                return messages
        raise ValueError(
            "payload has no usable 'messages' (expected a message list, a "
            "{'messages': [...]} object, or a {'simulations': [...]} envelope)"
        )
    if isinstance(payload, list):
        return payload
    raise ValueError(
        f"unsupported trajectory payload type {type(payload).__name__}"
    )


def _classify(key: str | None, value: str, collected: dict[str, set[str]]) -> None:
    """Add internal identifiers found in one string value."""
    if key == "user_id" and USER_ID_RE.fullmatch(value):
        collected["user_id"].add(value)
    elif key == "payment_method_id" and PAYMENT_RE.fullmatch(value):
        collected["payment_method_id"].add(value)
    elif key == "email" and EMAIL_RE.fullmatch(value):
        collected["email"].add(value)
    for match in PAYMENT_RE.findall(value):
        collected["payment_method_id"].add(match)
    for match in USER_ID_RE.findall(value):
        collected["user_id"].add(match)
    for match in EMAIL_RE.findall(value):
        collected["email"].add(match)


def _walk(obj: Any, key: str | None, collected: dict[str, set[str]]) -> None:
    if isinstance(obj, dict):
        if key == _PAYMENT_METHODS_KEY:
            for method_id in obj:
                if PAYMENT_RE.fullmatch(str(method_id)):
                    collected["payment_method_id"].add(str(method_id))
        role = obj.get("role")
        for nested_key, nested in obj.items():
            if role == "assistant" and nested_key == "content":
                # assistant prose is the scan target, never a source of ids
                continue
            if isinstance(nested, str):
                _classify(nested_key, nested, collected)
            else:
                _walk(nested, nested_key, collected)
    elif isinstance(obj, list):
        for item in obj:
            _walk(item, None, collected)


def collect_internal_identifiers(payload: Any) -> dict[str, set[str]]:
    """Collect internal identifiers from the frozen payload.

    Returns a mapping from category to the set of concrete identifier strings.
    Identifiers come from known keys (``user_id``, ``payment_method_id``,
    ``email``, keys under ``payment_methods``) plus distinctive formats found
    in any string value. Product/item ids and order numbers are not internal
    identifiers and are intentionally not collected.
    """
    collected: dict[str, set[str]] = {
        "payment_method_id": set(),
        "user_id": set(),
        "email": set(),
    }
    _walk(payload, None, collected)
    return collected


def _static_content_hits(content: str) -> list[tuple[str, str]]:
    """Safety-net matches for distinctive formats in free assistant text.

    The user-id format is intentionally excluded here: in free text it is
    indistinguishable from ordinary prose, so user ids are only flagged when
    the concrete value exists in the frozen payload.
    """
    hits: list[tuple[str, str]] = []
    for category, pattern in (
        ("payment_method_id", PAYMENT_RE),
        ("email", EMAIL_RE),
        ("uuid", UUID_RE),
    ):
        for match in pattern.findall(content):
            hits.append((category, match))
    return hits


@dataclass(frozen=True, slots=True)
class PiiHit:
    message_index: int
    category: str
    identifier: str
    context: str
    on_tool_turn: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "message_index": self.message_index,
            "category": self.category,
            "identifier": self.identifier,
            "context": self.context,
            "on_tool_turn": self.on_tool_turn,
        }


def scan_messages(
    messages: list[dict[str, Any]],
    identifiers: dict[str, set[str]] | None = None,
) -> list[PiiHit]:
    """Return every internal-identifier hit found in assistant text."""
    if identifiers is None:
        identifiers = collect_internal_identifiers({"messages": messages})
    known: list[tuple[str, str]] = [
        (category, identifier)
        for category, values in identifiers.items()
        for identifier in sorted(values)
    ]
    hits: list[PiiHit] = []
    for index, message in enumerate(messages):
        if str(message.get("role", "")) != "assistant":
            continue
        content = str(message.get("content") or "")
        if not content.strip():
            continue
        on_tool_turn = bool(message.get("tool_calls"))
        found: dict[str, str] = {}
        for category, identifier in known:
            if identifier and identifier in content:
                found[identifier] = category
        for category, identifier in _static_content_hits(content):
            found.setdefault(identifier, category)
        for identifier, category in sorted(found.items()):
            position = content.find(identifier)
            hits.append(
                PiiHit(
                    message_index=index,
                    category=category,
                    identifier=identifier,
                    context=content[max(0, position - 40) : position + 40],
                    on_tool_turn=on_tool_turn,
                )
            )
    return hits


def _candidate_meta(payload: Any, path: Path) -> tuple[str, str]:
    candidate_id = str(path.stem)
    for prefix in ("source_", "corrected_", "sim_"):
        if candidate_id.startswith(prefix):
            candidate_id = candidate_id[len(prefix) :]
            break
    task_id = ""
    if isinstance(payload, dict):
        if payload.get("candidate_id"):
            candidate_id = str(payload["candidate_id"])
        if payload.get("task_id") is not None:
            task_id = str(payload["task_id"])
        simulations = payload.get("simulations")
        if isinstance(simulations, list) and simulations:
            sim = simulations[0]
            if sim.get("candidate_id"):
                candidate_id = str(sim["candidate_id"])
            if not task_id and sim.get("task_id") is not None:
                task_id = str(sim["task_id"])
    if not task_id:
        task_id = candidate_id
    return candidate_id, task_id


def scan_trajectory(payload: Any, path: Path) -> dict[str, Any]:
    """Scan one trajectory payload and return the per-candidate report row."""
    messages = _payload_messages(payload)
    if not any(
        str(message.get("role", "")) == "assistant"
        and str(message.get("content") or "").strip()
        for message in messages
    ):
        raise ValueError(f"{path}: no assistant text message to scan")
    identifiers = collect_internal_identifiers(payload)
    hits = scan_messages(messages, identifiers)
    candidate_id, task_id = _candidate_meta(payload, path)
    return {
        "candidate_id": candidate_id,
        "task_id": task_id,
        "source": str(path),
        "source_sha256": sha256(path),
        "status": "HITS" if hits else "CLEAN",
        "identifiers": {
            category: sorted(values)
            for category, values in identifiers.items()
            if values
        },
        "hits": [hit.to_dict() for hit in hits],
    }


def load_trajectory(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _error_row(path: Path, error: Exception, candidate_id: str | None = None) -> dict[str, Any]:
    return {
        "candidate_id": candidate_id or str(path.stem),
        "task_id": "",
        "source": str(path),
        "source_sha256": sha256(path),
        "status": "ERROR",
        "error": str(error),
        "identifiers": {},
        "hits": [],
    }


def scan_paths(paths: Iterable[Path]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in paths:
        if path.suffix.lower() == ".jsonl":
            rows.extend(scan_jsonl(path))
            continue
        try:
            rows.append(scan_trajectory(load_trajectory(path), path))
        except (ValueError, json.JSONDecodeError) as error:
            rows.append(_error_row(path, error))
    return rows


def scan_jsonl(path: Path) -> list[dict[str, Any]]:
    """Scan a JSONL file where each line is one trajectory payload."""
    rows: list[dict[str, Any]] = []
    for index, line in enumerate(path.read_text(encoding="utf-8-sig").splitlines()):
        if not line.strip():
            continue
        label = f"{path.stem}#{index}"
        try:
            payload = json.loads(line)
            row = scan_trajectory(payload, path)
            if not row["candidate_id"] or row["candidate_id"] == str(path.stem):
                row["candidate_id"] = label
            row["source_line"] = index
            rows.append(row)
        except (ValueError, json.JSONDecodeError) as error:
            row = _error_row(path, error, candidate_id=label)
            row["source_line"] = index
            rows.append(row)
    return rows


def scan_directory(directory: Path) -> list[dict[str, Any]]:
    files = sorted(
        candidate
        for candidate in directory.rglob("*")
        if candidate.is_file() and candidate.suffix.lower() in {".json", ".jsonl"}
    )
    return scan_paths(files)


def summarize(rows: list[dict[str, Any]]) -> dict[str, int]:
    return {
        "candidates": len(rows),
        "clean": sum(1 for row in rows if row["status"] == "CLEAN"),
        "hits": sum(1 for row in rows if row["status"] == "HITS"),
        "errors": sum(1 for row in rows if row["status"] == "ERROR"),
    }


def write_outputs(rows: list[dict[str, Any]], output_dir: Path) -> None:
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(
            f"Refusing to overwrite non-empty output: {output_dir}"
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    report = {
        "schema_version": SCHEMA_VERSION,
        "summary": summarize(rows),
        "candidates": rows,
    }
    (output_dir / "pii_scan_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    with (output_dir / "pii_hits.jsonl").open("w", encoding="utf-8") as handle:
        for row in rows:
            for hit in row["hits"]:
                handle.write(
                    json.dumps(
                        {
                            "candidate_id": row["candidate_id"],
                            "task_id": row["task_id"],
                            "source": row["source"],
                            "source_sha256": row["source_sha256"],
                            **hit,
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Scan teacher trajectories for internal identifiers in assistant text."
        )
    )
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    if args.input.is_dir():
        rows = scan_directory(args.input)
    elif args.input.is_file():
        rows = scan_paths([args.input])
    else:
        parser.error(f"input path does not exist: {args.input}")

    if args.output is not None:
        write_outputs(rows, args.output)

    summary = summarize(rows)
    print(json.dumps(summary))
    for row in rows:
        if row["status"] == "ERROR":
            print(f"{row['candidate_id']} ERROR: {row['error']}")
        for hit in row["hits"]:
            print(
                f"{row['candidate_id']} message {hit['message_index']} "
                f"{hit['category']} {hit['identifier']!r}"
            )
    if summary["hits"] or summary["errors"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
