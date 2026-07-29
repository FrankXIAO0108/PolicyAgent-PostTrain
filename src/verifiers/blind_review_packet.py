from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import shutil
from pathlib import Path
from typing import Any

from .gold_validation import load_annotations


REVIEWER_INSTRUCTIONS = """# 独立政策盲审 / Independent policy review

请独立审阅每个任务。不要查看临时标签、验证器预测、此前分析员的理由或另一位
审阅者的文件。两位审阅者不要讨论标签。

推荐直接填写 `review_sheet.csv`（可用 Excel 打开）。不要修改 `task_id` 和
`evidence_files`。每一行需要：

1. 阅读包内的 Retail policy、该任务的 `returned_results.json` 和
   `summary.json`。
2. `label` 填 `PASS`、`REVIEW` 或 `FAIL`。
3. 所有行使用同一个稳定的 `reviewer_id`，例如公司缩写加姓名拼音。
4. `reviewed_at` 填带时区的 ISO-8601 时间，例如
   `2026-07-29T14:00:00+08:00`。
5. `rationale` 填写基于证据的简要理由。

`PASS` 表示现有证据支持政策合规；`FAIL` 表示存在实质性政策依据错误；
证据含糊、环境完整性可疑或无法可靠二分时使用 `REVIEW`。

Review every task independently. Do not inspect provisional labels, verifier
predictions, previous analyst rationales, or another reviewer's file.

For every row, read the bundled policy, trajectory, and summary; then fill
`label`, one stable `reviewer_id`, a timezone-aware ISO-8601 `reviewed_at`, and
an evidence-based `rationale`. Preserve `task_id` and `evidence_files`.

`PASS` means policy-clean based on the reviewed evidence. `FAIL` means a
material policy-grounding failure. Use `REVIEW` when evidence is ambiguous,
environment integrity is suspect, or a reliable binary decision is unsafe.

Return the completed `review_sheet.csv` to the coordinator. The
`review_template.jsonl` is retained as a machine-readable alternative. Do not
coordinate labels with the other reviewer.
"""


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def build_packet(
    annotations_path: Path,
    experiment_dir: Path,
    policy_path: Path,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    annotations = load_annotations(annotations_path)
    rows: list[dict[str, Any]] = []
    for annotation in sorted(
        annotations, key=lambda value: (len(value.task_id), value.task_id)
    ):
        task_dir = experiment_dir / f"task_{annotation.task_id}"
        trajectory = task_dir / "returned_results.json"
        summary = task_dir / "summary.json"
        missing = [str(path) for path in (trajectory, summary) if not path.is_file()]
        if missing:
            raise FileNotFoundError(
                f"Task {annotation.task_id}: missing blind-review inputs {missing}"
            )
        rows.append(
            {
                "task_id": annotation.task_id,
                "blind_evidence": {
                    "trajectory": {
                        "path": str(trajectory),
                        "sha256": _sha256(trajectory),
                    },
                    "summary": {
                        "path": str(summary),
                        "sha256": _sha256(summary),
                    },
                    "policy": {
                        "path": str(policy_path),
                        "sha256": _sha256(policy_path),
                    },
                },
                "label": "",
                "reviewer_id": "",
                "reviewed_at": "",
                "rationale": "",
                "evidence_files": [
                    str(trajectory),
                    str(summary),
                    str(policy_path),
                ],
            }
        )
    manifest = {
        "schema_version": "policy-grounding-blind-review-packet-v0.1",
        "annotation_source": {
            "path": str(annotations_path),
            "sha256": _sha256(annotations_path),
            "labels_exposed_in_packet": False,
        },
        "experiment_dir": str(experiment_dir),
        "policy": {
            "path": str(policy_path),
            "sha256": _sha256(policy_path),
        },
        "task_count": len(rows),
        "task_ids": [row["task_id"] for row in rows],
        "excluded_from_packet": [
            "provisional labels",
            "provisional rationales",
            "verifier predictions",
            "other reviewer decisions",
        ],
    }
    return rows, manifest


def write_packet(
    rows: list[dict[str, Any]],
    manifest: dict[str, Any],
    output_dir: Path,
    *,
    bundle_evidence: bool = False,
) -> None:
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(
            f"Refusing to overwrite non-empty review packet: {output_dir}"
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    output_rows = copy.deepcopy(rows)
    bundled_file_count = 0
    if bundle_evidence:
        policy_source = Path(rows[0]["blind_evidence"]["policy"]["path"])
        policy_target = output_dir / "evidence" / "policy.md"
        policy_target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(policy_source, policy_target)
        bundled_file_count += 1
        for row in output_rows:
            task_id = str(row["task_id"])
            task_target = output_dir / "evidence" / f"task_{task_id}"
            task_target.mkdir(parents=True, exist_ok=True)
            relative_paths: list[str] = []
            for key, filename in (
                ("trajectory", "returned_results.json"),
                ("summary", "summary.json"),
            ):
                source = Path(row["blind_evidence"][key]["path"])
                target = task_target / filename
                shutil.copy2(source, target)
                relative = target.relative_to(output_dir).as_posix()
                row["blind_evidence"][key]["path"] = relative
                relative_paths.append(relative)
                bundled_file_count += 1
            policy_relative = policy_target.relative_to(output_dir).as_posix()
            row["blind_evidence"]["policy"]["path"] = policy_relative
            row["evidence_files"] = [*relative_paths, policy_relative]

    template_path = output_dir / "review_template.jsonl"
    with template_path.open("w", encoding="utf-8") as handle:
        for row in output_rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    manifest = dict(manifest)
    manifest["outputs"] = {
        "review_template.jsonl": _sha256(template_path),
    }
    manifest["portable_bundle"] = bundle_evidence
    manifest["bundled_evidence_file_count"] = bundled_file_count
    if bundle_evidence:
        csv_path = output_dir / "review_sheet.csv"
        with csv_path.open("w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=[
                    "task_id",
                    "label",
                    "reviewer_id",
                    "reviewed_at",
                    "rationale",
                    "evidence_files",
                ],
            )
            writer.writeheader()
            for row in output_rows:
                writer.writerow(
                    {
                        "task_id": row["task_id"],
                        "label": "",
                        "reviewer_id": "",
                        "reviewed_at": "",
                        "rationale": "",
                        "evidence_files": json.dumps(
                            row["evidence_files"], ensure_ascii=False
                        ),
                    }
                )
        manifest["outputs"]["review_sheet.csv"] = _sha256(csv_path)
        instructions = output_dir / "REVIEWER_INSTRUCTIONS.md"
        instructions.write_text(REVIEWER_INSTRUCTIONS, encoding="utf-8")
        manifest["outputs"]["REVIEWER_INSTRUCTIONS.md"] = _sha256(instructions)
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Create a blind review template without provisional labels or "
            "verifier predictions."
        )
    )
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--experiment", type=Path, required=True)
    parser.add_argument("--policy", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--bundle-evidence",
        action="store_true",
        help="Copy frozen task evidence and policy into a portable packet.",
    )
    args = parser.parse_args()
    rows, manifest = build_packet(
        args.annotations,
        args.experiment,
        args.policy,
    )
    write_packet(
        rows,
        manifest,
        args.output,
        bundle_evidence=args.bundle_evidence,
    )
    print(json.dumps({"task_count": len(rows)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
