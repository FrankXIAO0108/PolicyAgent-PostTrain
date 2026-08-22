from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.verifiers.gold_validation import GoldAnnotation, load_annotations, load_jsonl


DISPOSITIONS = {"RAW_POSITIVE", "CORRECTED_POSITIVE", "HOLDOUT"}
SPLITS = {"TRAIN", "VALIDATION"}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


@dataclass(frozen=True, slots=True)
class QualityDecision:
    task_id: str
    status: str
    disposition: str
    split: str | None
    source_split: str
    source_path: Path
    source_sha256: str
    correction_path: Path | None
    correction_sha256: str | None
    correction_validation_path: Path | None
    correction_validation_sha256: str | None
    group_ids: tuple[str, ...]
    rationale: str

    @classmethod
    def from_dict(
        cls,
        row: dict[str, Any],
        *,
        allow_owner_reviewed_development: bool = False,
    ) -> QualityDecision:
        task_id = str(row["task_id"])
        status = str(row.get("status", "")).upper()
        disposition = str(row.get("disposition", "")).upper()
        split_value = row.get("split")
        split = str(split_value).upper() if split_value else None
        source_split = str(row.get("source_split", "")).upper()
        source_path = Path(str(row["source_path"]))
        source_hash = str(row.get("source_sha256", "")).upper()
        correction_value = row.get("correction_path")
        correction_path = Path(str(correction_value)) if correction_value else None
        correction_hash_value = row.get("correction_sha256")
        correction_hash = (
            str(correction_hash_value).upper() if correction_hash_value else None
        )
        validation_value = row.get("correction_validation_path")
        validation_path = Path(str(validation_value)) if validation_value else None
        validation_hash_value = row.get("correction_validation_sha256")
        validation_hash = (
            str(validation_hash_value).upper() if validation_hash_value else None
        )
        group_ids = tuple(
            sorted(
                {
                    str(value).strip()
                    for value in row.get("group_ids", [])
                    if str(value).strip()
                }
            )
        )
        rationale = str(row.get("rationale", "")).strip()

        accepted_statuses = {"ADJUDICATED"}
        if allow_owner_reviewed_development:
            accepted_statuses.add("HUMAN_ADJUDICATED")
        if status not in accepted_statuses:
            raise ValueError(
                f"Task {task_id}: quality status {status!r} is not permitted "
                "in the selected review mode"
            )
        if disposition not in DISPOSITIONS:
            raise ValueError(
                f"Task {task_id}: disposition must be one of {sorted(DISPOSITIONS)}"
            )
        if disposition == "HOLDOUT":
            if split is not None:
                raise ValueError(f"Task {task_id}: HOLDOUT must not have a split")
        elif split not in SPLITS:
            raise ValueError(
                f"Task {task_id}: released rows require split TRAIN or VALIDATION"
            )
        if source_split not in {"TRAIN", "TEST"}:
            raise ValueError(f"Task {task_id}: source_split must be TRAIN or TEST")
        if disposition != "HOLDOUT" and source_split != "TRAIN":
            raise ValueError(
                f"Task {task_id}: official TEST tasks cannot enter SFT data"
            )
        if not source_hash:
            raise ValueError(f"Task {task_id}: source_sha256 is required")
        if disposition == "CORRECTED_POSITIVE":
            if (
                correction_path is None
                or not correction_hash
                or validation_path is None
                or not validation_hash
            ):
                raise ValueError(
                    f"Task {task_id}: corrected rows require correction and "
                    "validation paths and hashes"
                )
        elif any(
            value is not None
            for value in (
                correction_path,
                correction_hash,
                validation_path,
                validation_hash,
            )
        ):
            raise ValueError(
                f"Task {task_id}: only CORRECTED_POSITIVE may specify correction data"
            )
        if disposition != "HOLDOUT" and not group_ids:
            raise ValueError(f"Task {task_id}: released rows require group_ids")
        if not rationale:
            raise ValueError(f"Task {task_id}: rationale is required")
        return cls(
            task_id=task_id,
            status=status,
            disposition=disposition,
            split=split,
            source_split=source_split,
            source_path=source_path,
            source_sha256=source_hash,
            correction_path=correction_path,
            correction_sha256=correction_hash,
            correction_validation_path=validation_path,
            correction_validation_sha256=validation_hash,
            group_ids=group_ids,
            rationale=rationale,
        )


def load_decisions(
    path: Path,
    *,
    allow_owner_reviewed_development: bool = False,
) -> dict[str, QualityDecision]:
    decisions: dict[str, QualityDecision] = {}
    for row in load_jsonl(path):
        decision = QualityDecision.from_dict(
            row,
            allow_owner_reviewed_development=allow_owner_reviewed_development,
        )
        if decision.task_id in decisions:
            raise ValueError(f"Duplicate quality decision task ID: {decision.task_id}")
        decisions[decision.task_id] = decision
    return decisions


def _verify_file(path: Path, expected_hash: str, task_id: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"Task {task_id}: missing source file {path}")
    actual = sha256(path)
    if actual != expected_hash:
        raise ValueError(
            f"Task {task_id}: hash mismatch for {path}; "
            f"expected={expected_hash}, actual={actual}"
        )


def _raw_messages(path: Path, task_id: str) -> tuple[list[dict[str, Any]], str]:
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    simulations = payload.get("simulations", [])
    if len(simulations) != 1:
        raise ValueError(
            f"Task {task_id}: expected exactly one frozen simulation, "
            f"got {len(simulations)}"
        )
    messages = simulations[0].get("messages", [])
    policy = payload.get("info", {}).get("environment_info", {}).get("policy", "")
    return messages, str(policy)


def entity_groups(path: Path) -> set[str]:
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    groups: set[str] = set()
    keys = {"user_id", "order_id", "product_id"}

    def visit(value: Any) -> None:
        if isinstance(value, dict):
            for key, nested in value.items():
                if key in keys and isinstance(nested, (str, int)):
                    groups.add(f"{key}:{nested}")
                visit(nested)
        elif isinstance(value, list):
            for nested in value:
                visit(nested)

    visit(payload)
    return groups


def _corrected_messages(path: Path, task_id: str) -> tuple[list[dict[str, Any]], str]:
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    messages = payload.get("messages")
    if not isinstance(messages, list):
        raise ValueError(f"Task {task_id}: correction must contain messages")
    return messages, str(payload.get("system_policy", ""))


def _normalize_messages(
    messages: list[dict[str, Any]], task_id: str
) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    for index, message in enumerate(messages):
        role = str(message.get("role", ""))
        if role not in {"assistant", "user", "tool", "system"}:
            raise ValueError(f"Task {task_id}: unsupported message role {role!r}")
        calls = message.get("tool_calls") or []
        content = str(message.get("content") or "")
        if role == "assistant" and calls:
            if len(calls) > 1:
                raise ValueError(
                    f"Task {task_id}: assistant message {index} has multiple tool calls"
                )
            if content.strip():
                raise ValueError(
                    f"Task {task_id}: assistant message {index} mixes text and tool call"
                )
        normalized.append(
            {
                "role": role,
                "content": content,
                "tool_calls": calls,
                "loss_mask": 1 if role == "assistant" else 0,
            }
        )
    if not any(row["loss_mask"] for row in normalized):
        raise ValueError(f"Task {task_id}: no assistant targets found")
    return normalized


def assess_release(
    annotations: list[GoldAnnotation],
    decisions: dict[str, QualityDecision] | None,
    *,
    allow_owner_reviewed_development: bool = False,
) -> dict[str, Any]:
    reasons: list[str] = []
    accepted_statuses = {"ADJUDICATED"}
    if allow_owner_reviewed_development:
        accepted_statuses.add("HUMAN_ADJUDICATED")
    non_adjudicated = sorted(
        row.task_id for row in annotations if row.status not in accepted_statuses
    )
    if non_adjudicated:
        reasons.append(
            "Policy annotations are not fully ADJUDICATED: "
            + ", ".join(non_adjudicated)
        )
    if decisions is None:
        reasons.append("No adjudicated trajectory-quality decision file supplied.")
        return {
            "ready": False,
            "reasons": reasons,
            "records": [],
            "counts": {"annotations": len(annotations), "released": 0},
        }

    annotation_by_task = {row.task_id: row for row in annotations}
    missing = sorted(annotation_by_task.keys() - decisions.keys())
    unknown = sorted(decisions.keys() - annotation_by_task.keys())
    if missing or unknown:
        reasons.append(f"Decision coverage mismatch: missing={missing}, unknown={unknown}")

    records: list[dict[str, Any]] = []
    group_splits: dict[str, set[str]] = defaultdict(set)
    if not reasons:
        for task_id, annotation in annotation_by_task.items():
            decision = decisions[task_id]
            _verify_file(decision.source_path, decision.source_sha256, task_id)
            if decision.disposition == "HOLDOUT":
                continue
            if (
                decision.disposition == "RAW_POSITIVE"
                and annotation.label != "PASS"
            ):
                raise ValueError(
                    f"Task {task_id}: RAW_POSITIVE requires adjudicated policy PASS"
                )
            selected_path = decision.source_path
            selected_hash = decision.source_sha256
            corrected = decision.disposition == "CORRECTED_POSITIVE"
            record_groups = set(decision.group_ids) | entity_groups(
                decision.source_path
            )
            if corrected:
                assert decision.correction_path is not None
                assert decision.correction_sha256 is not None
                assert decision.correction_validation_path is not None
                assert decision.correction_validation_sha256 is not None
                _verify_file(
                    decision.correction_path,
                    decision.correction_sha256,
                    task_id,
                )
                _verify_file(
                    decision.correction_validation_path,
                    decision.correction_validation_sha256,
                    task_id,
                )
                validation = json.loads(
                    decision.correction_validation_path.read_text(
                        encoding="utf-8-sig"
                    )
                )
                if not validation.get("ready"):
                    raise ValueError(
                        f"Task {task_id}: correction validation is not ready"
                    )
                if (
                    str(validation.get("task_id")) != task_id
                    or str(validation.get("correction_sha256", "")).upper()
                    != decision.correction_sha256
                ):
                    raise ValueError(
                        f"Task {task_id}: correction validation binding mismatch"
                    )
                selected_path = decision.correction_path
                selected_hash = decision.correction_sha256
                messages, policy = _corrected_messages(selected_path, task_id)
                if not policy:
                    _, policy = _raw_messages(decision.source_path, task_id)
            else:
                messages, policy = _raw_messages(selected_path, task_id)
            normalized = _normalize_messages(messages, task_id)
            for group_id in record_groups:
                group_splits[group_id].add(str(decision.split))
            records.append(
                {
                    "candidate_id": (
                        f"policy-agent-task-{task_id}-"
                        f"{selected_hash[:12].lower()}"
                    ),
                    "task_id": task_id,
                    "split": decision.split,
                    "disposition": decision.disposition,
                    "corrected": corrected,
                    "source_path": str(selected_path),
                    "source_sha256": selected_hash,
                    "group_ids": sorted(record_groups),
                    "system_policy": policy,
                    "messages": normalized,
                }
            )

    leakage = sorted(
        group_id for group_id, splits in group_splits.items() if len(splits) > 1
    )
    if leakage:
        reasons.append("Group leakage across splits: " + ", ".join(leakage))
        records = []
    return {
        "ready": not reasons,
        "reasons": reasons,
        "review_mode": (
            "OWNER_REVIEWED_DEVELOPMENT"
            if allow_owner_reviewed_development
            else "INDEPENDENT_ADJUDICATION"
        ),
        "records": records,
        "counts": {
            "annotations": len(annotations),
            "released": len(records),
            "train": sum(row["split"] == "TRAIN" for row in records),
            "validation": sum(row["split"] == "VALIDATION" for row in records),
        },
    }


def write_release(result: dict[str, Any], output_dir: Path) -> None:
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty output: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    report = {key: value for key, value in result.items() if key != "records"}
    (output_dir / "release_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    if result["ready"]:
        dataset = output_dir / "sft_dataset.jsonl"
        with dataset.open("w", encoding="utf-8") as handle:
            for row in result["records"]:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        (output_dir / "dataset_manifest.json").write_text(
            json.dumps(
                {
                    "schema_version": "policy-agent-sft-v0.1",
                    "review_mode": result["review_mode"],
                    "dataset_sha256": sha256(dataset),
                    "counts": result["counts"],
                },
                ensure_ascii=False,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Fail-closed SFT data release gate.")
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--decisions", type=Path)
    parser.add_argument(
        "--allow-owner-reviewed-development",
        action="store_true",
        help=(
            "Permit HUMAN_ADJUDICATED owner-review evidence for development "
            "training only; does not create independent gold."
        ),
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    annotations = load_annotations(args.annotations)
    decisions = (
        load_decisions(
            args.decisions,
            allow_owner_reviewed_development=args.allow_owner_reviewed_development,
        )
        if args.decisions
        else None
    )
    result = assess_release(
        annotations,
        decisions,
        allow_owner_reviewed_development=args.allow_owner_reviewed_development,
    )
    write_release(result, args.output)
    print(json.dumps(result["counts"], ensure_ascii=False))
    if not result["ready"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
