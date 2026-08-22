"""Assemble protocol correction artifacts from completed scripted replays."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.training.correction_validation import sha256, validate_correction
from src.training.run_scripted_replay import (
    REPO_ROOT,
    UPSTREAM_COMMIT,
    git_value,
    load_json,
    remap_path,
    sha256_lf,
    validate_spec_dir,
    write_json,
)


def validate_replay_evidence(
    replay: dict[str, Any],
    corrected_path: Path,
    *,
    expected_task_id: str,
    expected_spec_sha256_lf: str,
) -> None:
    errors: list[str] = []
    if replay.get("status") != "COMPLETED":
        errors.append("replay status is not COMPLETED")
    if str(replay.get("task_id")) != expected_task_id:
        errors.append("replay task_id mismatch")
    if str((replay.get("spec") or {}).get("sha256_lf", "")).upper() != (
        expected_spec_sha256_lf.upper()
    ):
        errors.append("replay spec LF hash mismatch")
    if not bool((replay.get("result") or {}).get("replay_seed_matches_spec")):
        errors.append("replay seed does not match spec")
    if int(replay.get("tool_result_mismatches") or 0) != 0:
        errors.append("replay has tool-result mismatches")
    protocol = replay.get("protocol") or {}
    if protocol.get("tool_result_pairs_ok") is not True:
        errors.append("tool call/result pairing failed")
    if int(protocol.get("mixed_messages") or 0) != 0:
        errors.append("replay contains mixed assistant text/tool messages")
    expected_messages_hash = str(
        (replay.get("corrected_messages") or {}).get("sha256") or ""
    ).upper()
    if not corrected_path.is_file():
        errors.append(f"corrected messages missing: {corrected_path}")
    elif sha256(corrected_path) != expected_messages_hash:
        errors.append("corrected messages hash mismatch")
    if errors:
        raise ValueError("; ".join(errors))


def build_correction_payload(
    *,
    spec: dict[str, Any],
    source_path: Path,
    policy_path: Path,
    messages: list[dict[str, Any]],
    replay_path: Path,
    replay: dict[str, Any],
    run_manifest_path: Path,
    run_manifest_sha256: str,
) -> dict[str, Any]:
    return {
        "schema_version": "retail-scripted-replay-correction-v1",
        "task_id": str(spec["task_id"]),
        "author_id": str(spec["author_id"]),
        "authored_at": datetime.now(timezone.utc).isoformat(),
        "generation_mode": "ENVIRONMENT_REPLAY",
        "source": {
            "path": str(source_path),
            "sha256": str((spec.get("source") or {})["sha256"]).upper(),
        },
        "policy": {
            "path": str(policy_path),
            "sha256": str((spec.get("policy") or {})["sha256"]).upper(),
        },
        "system_policy": policy_path.read_text(encoding="utf-8-sig"),
        "change_log": spec["change_log"],
        "messages": messages,
        "replay_manifest": {
            "path": str(replay_path),
            "sha256": sha256(replay_path),
        },
        "replay_evidence": {
            "run_manifest": {
                "path": str(run_manifest_path),
                "sha256": run_manifest_sha256,
            },
            "result": replay.get("result"),
            "branch": replay.get("branch"),
            "protocol": replay.get("protocol"),
            "state": replay.get("state"),
            "prefix_user_mismatches": replay.get("prefix_user_mismatches"),
            "tool_result_mismatches": replay.get("tool_result_mismatches"),
        },
    }


def assemble(
    *,
    spec_dir: Path,
    replay_output_dir: Path,
    output_dir: Path,
    remaps: list[tuple[str, str]],
    seed_source: int,
    upstream_commit: str,
    command: list[str] | None = None,
) -> dict[str, Any]:
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite existing output: {output_dir}")
    validated = validate_spec_dir(
        spec_dir, remaps, seed_source, upstream_commit
    )
    errors = list(validated["errors"])
    for record in validated["specs"]:
        errors.extend(record["errors"])
    if errors:
        raise ValueError(f"spec validation failed: {errors}")

    run_manifest_path = replay_output_dir / "run_manifest.json"
    run_manifest = load_json(run_manifest_path)
    if run_manifest.get("status") != "COMPLETED":
        raise ValueError("replay run is not COMPLETED")
    if (
        str((run_manifest.get("bindings") or {}).get("manifest_sha256_lf", "")).upper()
        != validated["manifest_sha256_lf"]
    ):
        raise ValueError("run manifest is bound to a different spec manifest")
    expected_tasks = {str(row["task_id"]) for row in validated["specs"]}
    if {str(task) for task in run_manifest.get("task_ids") or []} != expected_tasks:
        raise ValueError("run manifest task set differs from validated specs")

    output_dir.mkdir(parents=True, exist_ok=False)
    rows: list[dict[str, Any]] = []
    for record in validated["specs"]:
        task_id = str(record["task_id"])
        spec_path = spec_dir / record["name"]
        spec = load_json(spec_path)
        task_dir = replay_output_dir / f"task_{task_id}"
        replay_path = task_dir / "replay_manifest.json"
        corrected_path = task_dir / "corrected_messages.json"
        replay = load_json(replay_path)
        validate_replay_evidence(
            replay,
            corrected_path,
            expected_task_id=task_id,
            expected_spec_sha256_lf=sha256_lf(spec_path),
        )
        corrected = load_json(corrected_path)
        if str(corrected.get("task_id")) != task_id:
            raise ValueError(f"task {task_id}: corrected messages task_id mismatch")
        messages = corrected.get("messages")
        if not isinstance(messages, list):
            raise ValueError(f"task {task_id}: corrected messages must be a list")
        source_path = Path(remap_path(str(spec["source"]["path"]), remaps))
        policy_path = Path(remap_path(str(spec["policy"]["path"]), remaps))
        payload = build_correction_payload(
            spec=spec,
            source_path=source_path,
            policy_path=policy_path,
            messages=messages,
            replay_path=replay_path.resolve(),
            replay=replay,
            run_manifest_path=run_manifest_path.resolve(),
            run_manifest_sha256=sha256(run_manifest_path),
        )
        correction_path = output_dir / f"correction_task_{task_id}.json"
        write_json(correction_path, payload)
        precheck = validate_correction(correction_path, None)
        expected_reason = ["No independent correction approvals supplied."]
        if precheck["ready"] or precheck["reasons"] != expected_reason:
            raise ValueError(
                f"task {task_id}: unexpected correction precheck {precheck}"
            )
        rows.append(
            {
                "task_id": task_id,
                "correction": {
                    "path": str(correction_path.resolve()),
                    "sha256": sha256(correction_path),
                },
                "replay_manifest_sha256": sha256(replay_path),
                "precheck": precheck,
            }
        )

    summary = {
        "schema_version": "retail-scripted-replay-correction-assembly-v1",
        "status": "ASSEMBLED_AWAITING_APPROVALS",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "project": {
            "commit": git_value(REPO_ROOT, "rev-parse", "HEAD"),
            "branch": git_value(REPO_ROOT, "branch", "--show-current"),
            "dirty_at_start": bool(git_value(REPO_ROOT, "status", "--porcelain")),
        },
        "command": command,
        "spec_manifest_sha256_lf": validated["manifest_sha256_lf"],
        "run_manifest": {
            "path": str(run_manifest_path.resolve()),
            "sha256": sha256(run_manifest_path),
        },
        "corrections": rows,
    }
    write_json(output_dir / "assembly_summary.json", summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Assemble correction artifacts from scripted replay output."
    )
    parser.add_argument("--spec-dir", type=Path, required=True)
    parser.add_argument("--replay-output-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed-source", type=int, default=20260818)
    parser.add_argument("--upstream-commit", default=UPSTREAM_COMMIT)
    parser.add_argument("--path-remap", action="append", default=[])
    args = parser.parse_args()
    remaps = []
    for item in args.path_remap:
        if "=" not in item:
            raise SystemExit(f"--path-remap must be OLD=NEW, got {item!r}")
        remaps.append(tuple(item.split("=", 1)))
    result = assemble(
        spec_dir=args.spec_dir.resolve(),
        replay_output_dir=args.replay_output_dir.resolve(),
        output_dir=args.output_dir.resolve(),
        remaps=remaps,
        seed_source=args.seed_source,
        upstream_commit=args.upstream_commit,
        command=[
            sys.executable,
            "-m",
            "src.training.assemble_scripted_replay_corrections",
            *sys.argv[1:],
        ],
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
