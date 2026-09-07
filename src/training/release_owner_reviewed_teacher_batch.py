"""Release a candidate-level teacher shard under the owner-review development gate.

This explicit path implements the 2026-08-21 single-owner development policy. It
does not create independent gold. Task, user and order identities are hard split
keys; shared Retail catalog product IDs are reported but are not hard split keys.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from src.training.scan_teacher_pii import scan_trajectory
from src.training.sft_release import _normalize_messages, sha256


HARD_ENTITY_KEYS = {"user_id", "order_id"}
REPORTED_ENTITY_KEYS = {"product_id"}


def _load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _walk_entities(value: Any, keys: set[str], result: set[str]) -> None:
    if isinstance(value, dict):
        for key, nested in value.items():
            if key in keys and isinstance(nested, (str, int)):
                result.add(f"{key}:{nested}")
            _walk_entities(nested, keys, result)
    elif isinstance(value, list):
        for nested in value:
            _walk_entities(nested, keys, result)
    elif isinstance(value, str) and value.lstrip().startswith(("{", "[")):
        try:
            _walk_entities(json.loads(value), keys, result)
        except (TypeError, ValueError):
            pass


def entity_groups(payload: Any, keys: set[str]) -> set[str]:
    result: set[str] = set()
    _walk_entities(payload, keys, result)
    return result


def _fingerprint(messages: list[dict[str, Any]]) -> str:
    cleaned = []
    for message in messages:
        calls = []
        for call in message.get("tool_calls") or []:
            arguments = call.get("arguments", {})
            if isinstance(arguments, str):
                try:
                    arguments = json.loads(arguments)
                except ValueError:
                    pass
            calls.append({"name": call.get("name"), "arguments": arguments})
        content: Any = message.get("content") or ""
        if message.get("role") == "tool" and isinstance(content, str):
            try:
                content = json.loads(content)
            except ValueError:
                pass
        cleaned.append({"role": message.get("role"), "content": content, "tool_calls": calls})
    canonical = json.dumps(cleaned, ensure_ascii=False, sort_keys=True).encode()
    return hashlib.sha256(canonical).hexdigest().upper()


def assess_batch(approval_path: Path, existing_dataset_path: Path, final_holdout_config_path: Path) -> dict[str, Any]:
    approval = _load(approval_path)
    if approval.get("verdict") != "APPROVE" or approval.get("training_release_authorized") is not True:
        raise ValueError("owner approval must explicitly approve development release")
    if approval.get("review_mode") != "OWNER_REVIEWED_DEVELOPMENT":
        raise ValueError("owner approval review_mode mismatch")
    policy = approval.get("split_policy") or {}
    if set(policy.get("hard_keys") or []) != {"task_id", "user_id", "order_id"}:
        raise ValueError("approved hard split policy mismatch")
    if set(policy.get("reported_not_hard") or []) != {"product_id"}:
        raise ValueError("approved product reporting policy mismatch")

    existing = [json.loads(line) for line in existing_dataset_path.read_text(encoding="utf-8-sig").splitlines() if line.strip()]
    existing_ids = {str(row["candidate_id"]) for row in existing}
    existing_fingerprints = {_fingerprint(row["messages"]) for row in existing}
    existing_task_splits: dict[str, set[str]] = {}
    validation_hard: set[str] = set()
    validation_products: set[str] = set()
    for row in existing:
        task = str(row["task_id"])
        existing_task_splits.setdefault(task, set()).add(str(row["split"]).upper())
        if str(row["split"]).upper() == "VALIDATION":
            validation_hard |= entity_groups(row["messages"], HARD_ENTITY_KEYS)
            validation_hard |= {str(g) for g in row.get("group_ids", []) if str(g).split(":", 1)[0] in HARD_ENTITY_KEYS}
            validation_products |= entity_groups(row["messages"], REPORTED_ENTITY_KEYS)
            validation_products |= {str(g) for g in row.get("group_ids", []) if str(g).startswith("product_id:")}

    holdout_tasks = set(map(str, _load(final_holdout_config_path)["task_ids"]))
    seen_candidates: set[str] = set()
    seen_fingerprints: set[str] = set()
    records = []
    audit_rows = []
    for binding in approval.get("corrections") or []:
        path = Path(str(binding["correction_path"]))
        expected = str(binding["correction_sha256"]).upper()
        if sha256(path) != expected:
            raise ValueError(f"correction hash mismatch: {path}")
        correction = _load(path)
        candidate_id = str(correction["candidate_id"])
        task_id = str(correction["task_id"])
        if candidate_id != str(binding["candidate_id"]) or task_id != str(binding["task_id"]):
            raise ValueError(f"approval binding mismatch: {candidate_id}")
        if candidate_id in seen_candidates or candidate_id in existing_ids:
            raise ValueError(f"duplicate candidate_id: {candidate_id}")
        if task_id in holdout_tasks:
            raise ValueError(f"final holdout task cannot enter training: {task_id}")
        if existing_task_splits.get(task_id) != {"TRAIN"}:
            raise ValueError(f"task {task_id} is not bound exclusively to TRAIN")
        if str(correction.get("author_id")) == str(approval.get("reviewer_id")):
            raise ValueError(f"correction author cannot owner-approve: {candidate_id}")
        source = correction.get("source") or {}
        source_path = Path(str(source["path"]))
        if sha256(source_path) != str(source["sha256"]).upper():
            raise ValueError(f"source hash mismatch: {candidate_id}")
        supersedes = correction.get("supersedes") or {}
        if sha256(Path(str(supersedes["path"]))) != str(supersedes["sha256"]).upper():
            raise ValueError(f"superseded correction hash mismatch: {candidate_id}")
        replay_binding = correction.get("replay_manifest") or {}
        replay_path = Path(str(replay_binding["path"]))
        if sha256(replay_path) != str(replay_binding["sha256"]).upper():
            raise ValueError(f"replay hash mismatch: {candidate_id}")
        replay = _load(replay_path)
        replay_ok = replay.get("state_preserved") is True and all(
            replay[key].get("db_match") is True and not replay[key].get("replay_errors")
            for key in ("original_replay", "corrected_replay")
        )
        if not replay_ok:
            raise ValueError(f"replay/state gate failed: {candidate_id}")
        if scan_trajectory(correction, path)["hits"]:
            raise ValueError(f"assistant prose PII gate failed: {candidate_id}")
        messages = _normalize_messages(correction["messages"], candidate_id)
        fingerprint = _fingerprint(correction["messages"])
        if fingerprint in seen_fingerprints or fingerprint in existing_fingerprints:
            raise ValueError(f"exact duplicate trajectory: {candidate_id}")
        source_payload = _load(source_path)
        hard_groups = entity_groups(source_payload, HARD_ENTITY_KEYS)
        products = entity_groups(source_payload, REPORTED_ENTITY_KEYS)
        hard_overlap = sorted(hard_groups & validation_hard)
        if hard_overlap:
            raise ValueError(f"hard entity leakage for {candidate_id}: {hard_overlap}")
        product_overlap = sorted(products & validation_products)
        seen_candidates.add(candidate_id)
        seen_fingerprints.add(fingerprint)
        records.append({"candidate_id": candidate_id, "task_id": task_id, "split": "TRAIN",
                        "disposition": "CORRECTED_POSITIVE", "corrected": True,
                        "source_path": str(path), "source_sha256": expected,
                        "group_ids": sorted(hard_groups), "reported_product_ids": sorted(products),
                        "system_policy": str(correction.get("system_policy") or ""), "messages": messages})
        audit_rows.append({"candidate_id": candidate_id, "task_id": task_id,
                           "hard_entity_groups": sorted(hard_groups),
                           "product_overlap_with_validation": product_overlap})
    return {"ready": True, "reasons": [], "records": records,
            "review_mode": "OWNER_REVIEWED_DEVELOPMENT",
            "counts": {"released": len(records), "train": len(records), "validation": 0,
                       "tasks": len({row["task_id"] for row in records})},
            "split_policy": policy, "audit_rows": audit_rows,
            "checks": {"hash_chain": True, "state_replay": True, "assistant_pii_hits": 0,
                       "exact_duplicates": 0, "hard_entity_leakage": 0,
                       "final_holdout_task_overlap": 0,
                       "product_overlap_candidates_reported": sum(bool(row["product_overlap_with_validation"]) for row in audit_rows)},
            "inputs": {"approval": {"path": str(approval_path), "sha256": sha256(approval_path)},
                       "existing_dataset": {"path": str(existing_dataset_path), "sha256": sha256(existing_dataset_path)},
                       "final_holdout_config": {"path": str(final_holdout_config_path), "sha256": sha256(final_holdout_config_path)}}}


def write_release(result: dict[str, Any], output_dir: Path) -> None:
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty output: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    dataset = output_dir / "sft_dataset.jsonl"
    dataset.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in result["records"]), encoding="utf-8", newline="\n")
    report = {key: value for key, value in result.items() if key != "records"}
    (output_dir / "release_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n")
    manifest = {"schema_version": "owner-reviewed-teacher-shard-v1", "review_mode": result["review_mode"],
                "dataset_sha256": sha256(dataset), "counts": result["counts"],
                "split_policy": result["split_policy"], "checks": result["checks"], "inputs": result["inputs"]}
    (output_dir / "dataset_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n")
