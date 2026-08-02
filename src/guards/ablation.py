from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .retail_pre_action import GuardFinding, resolve_guard_decision
from .scenario_evaluation import evaluate_suite, load_suite


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PROTOCOL = PROJECT_ROOT / "configs" / "guard_ablation_v1.json"
DEFAULT_OUTPUT = PROJECT_ROOT / "experiments" / "20260802_guard_ablation_v1"


def sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_value(*args: str) -> str | None:
    try:
        return subprocess.check_output(
            ["git", *args],
            cwd=PROJECT_ROOT,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def load_protocol(path: str | Path) -> dict[str, Any]:
    protocol = json.loads(Path(path).read_text(encoding="utf-8"))
    if protocol.get("official_metric") is not False:
        raise ValueError("Guard ablation must set official_metric=false.")
    variants = protocol.get("variants")
    if not isinstance(variants, list) or not variants:
        raise ValueError("Guard ablation requires a non-empty variants list.")
    variant_ids = [str(variant.get("variant_id", "")) for variant in variants]
    if any(not value for value in variant_ids):
        raise ValueError("Every ablation variant requires a variant_id.")
    if len(variant_ids) != len(set(variant_ids)):
        raise ValueError("Ablation variant_id values must be unique.")
    if variant_ids.count("full_guard") != 1 or variant_ids.count("no_guard") != 1:
        raise ValueError("Ablation requires exactly one full_guard and no_guard.")
    for variant in variants:
        prefixes = variant.get("disabled_rule_prefixes") or []
        if any(not str(prefix) for prefix in prefixes):
            raise ValueError("Disabled rule prefixes must be non-empty.")
    return protocol


def _disabled(rule_id: str, variant: dict[str, Any]) -> bool:
    if variant.get("disable_all_blocking_rules") is True:
        return True
    exact = {str(value) for value in variant.get("disabled_rule_ids") or []}
    prefixes = [
        str(value) for value in variant.get("disabled_rule_prefixes") or []
    ]
    return rule_id in exact or any(rule_id.startswith(prefix) for prefix in prefixes)


def _safe_divide(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def _f1(precision: float | None, recall: float | None) -> float | None:
    if precision is None or recall is None or precision + recall == 0:
        return None
    return 2 * precision * recall / (precision + recall)


def evaluate_ablation(
    suite: dict[str, Any],
    protocol: dict[str, Any],
) -> dict[str, Any]:
    full = evaluate_suite(suite)
    variants: list[dict[str, Any]] = []

    for variant in protocol["variants"]:
        rows: list[dict[str, Any]] = []
        for source in full["cases"]:
            retained: list[GuardFinding] = []
            removed_blocking_rule_ids: list[str] = []
            for raw in source["actual"]["findings"]:
                finding = GuardFinding(**raw)
                if finding.blocking and _disabled(finding.rule_id, variant):
                    removed_blocking_rule_ids.append(finding.rule_id)
                else:
                    retained.append(finding)
            decision = resolve_guard_decision(retained).value
            expected_block = source["expected"]["decision"] != "ALLOW"
            actual_block = decision != "ALLOW"
            rows.append(
                {
                    "case_id": source["case_id"],
                    "category": source["category"],
                    "expected_decision": source["expected"]["decision"],
                    "full_guard_decision": source["actual"]["decision"],
                    "ablated_decision": decision,
                    "expected_block": expected_block,
                    "actual_block": actual_block,
                    "decision_match": decision == source["expected"]["decision"],
                    "changed_from_full_guard": (
                        decision != source["actual"]["decision"]
                    ),
                    "removed_blocking_rule_ids": sorted(
                        removed_blocking_rule_ids
                    ),
                }
            )

        tp = sum(row["expected_block"] and row["actual_block"] for row in rows)
        fp = sum(not row["expected_block"] and row["actual_block"] for row in rows)
        fn = sum(row["expected_block"] and not row["actual_block"] for row in rows)
        tn = sum(
            not row["expected_block"] and not row["actual_block"] for row in rows
        )
        precision = _safe_divide(tp, tp + fp)
        recall = _safe_divide(tp, tp + fn)
        variants.append(
            {
                "variant_id": variant["variant_id"],
                "description": variant.get("description", ""),
                "disabled_rule_ids": list(
                    variant.get("disabled_rule_ids") or []
                ),
                "disabled_rule_prefixes": list(
                    variant.get("disabled_rule_prefixes") or []
                ),
                "disable_all_blocking_rules": bool(
                    variant.get("disable_all_blocking_rules")
                ),
                "summary": {
                    "tp": tp,
                    "fp": fp,
                    "fn": fn,
                    "tn": tn,
                    "precision": precision,
                    "recall": recall,
                    "f1": _f1(precision, recall),
                    "decision_accuracy": _safe_divide(
                        sum(row["decision_match"] for row in rows),
                        len(rows),
                    ),
                    "missed_risky_case_ids": sorted(
                        row["case_id"]
                        for row in rows
                        if row["expected_block"] and not row["actual_block"]
                    ),
                    "safe_control_regression_ids": sorted(
                        row["case_id"]
                        for row in rows
                        if not row["expected_block"] and row["actual_block"]
                    ),
                    "changed_case_ids": sorted(
                        row["case_id"]
                        for row in rows
                        if row["changed_from_full_guard"]
                    ),
                },
                "cases": rows,
            }
        )

    full_guard = next(
        item for item in variants if item["variant_id"] == "full_guard"
    )
    full_tp = int(full_guard["summary"]["tp"])
    for variant in variants:
        variant["summary"]["lost_risky_detections_vs_full"] = (
            full_tp - int(variant["summary"]["tp"])
        )

    return {
        "schema_version": "guard-ablation-result-v1.0.0",
        "suite": full["suite"],
        "protocol": {
            "schema_version": protocol.get("schema_version"),
            "name": protocol.get("name"),
            "official_metric": protocol.get("official_metric"),
        },
        "source_suite_summary": full["summary"],
        "variants": variants,
        "validity_notes": list(protocol.get("limitations") or []),
        "new_llm_calls": 0,
        "uses_reference_actions": False,
    }


def render_markdown(result: dict[str, Any]) -> str:
    def metric(value: float | None) -> str:
        return "N/A" if value is None else f"{value:.2%}"

    lines = [
        "# Retail Guard 规则族消融实验 V1",
        "",
        "## 实验边界",
        "",
        "- 数据：15 个开发者构造的合成场景",
        "- 官方指标：否",
        "- 新增 LLM 调用：0",
        "- Reference action / gold DB：未使用",
        "- 方法：移除指定 blocking finding 后重新计算 Guard decision",
        "",
        "## 结果",
        "",
        "| Variant | TP | FP | FN | TN | Recall | Decision accuracy | 漏掉的风险场景 |",
        "|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for variant in result["variants"]:
        summary = variant["summary"]
        missed = ", ".join(summary["missed_risky_case_ids"]) or "-"
        lines.append(
            f"| {variant['variant_id']} | {summary['tp']} | {summary['fp']} | "
            f"{summary['fn']} | {summary['tn']} | "
            f"{metric(summary['recall'])} | "
            f"{metric(summary['decision_accuracy'])} | {missed} |"
        )
    lines.extend(
        [
            "",
            "## 如何解释",
            "",
            "- `no_guard` 是合成场景上的无防护基线。",
            "- `full_guard` 是完整规则集合。",
            "- 其他行每次只移除一个规则族，用于定位该规则族覆盖的风险 case。",
            "- 若安全对照回归数为 0，只说明本套合成负对照未被误拦截。",
            "",
            "## 限制",
            "",
            *[f"- {note}" for note in result["validity_notes"]],
        ]
    )
    return "\n".join(lines) + "\n"


def write_outputs(
    protocol_path: str | Path,
    suite_path: str | Path,
    output_dir: str | Path,
    result: dict[str, Any],
) -> dict[str, Any]:
    protocol_file = Path(protocol_path).resolve()
    suite_file = Path(suite_path).resolve()
    destination = Path(output_dir).resolve()
    if destination.exists() and any(destination.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty {destination}")
    destination.mkdir(parents=True, exist_ok=True)

    protocol_snapshot = destination / "protocol_snapshot.json"
    protocol_snapshot.write_text(
        protocol_file.read_text(encoding="utf-8"), encoding="utf-8"
    )
    results_path = destination / "results.json"
    results_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    analysis_path = destination / "analysis.md"
    analysis_path.write_text(render_markdown(result), encoding="utf-8")

    manifest = {
        "schema_version": "guard-ablation-manifest-v1.0.0",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "experiment": "retail_guard_rule_family_ablation_v1",
        "project_commit": _git_value("rev-parse", "HEAD"),
        "project_git_dirty": bool(_git_value("status", "--porcelain")),
        "inputs": {
            "protocol": {
                "path": str(protocol_file.relative_to(PROJECT_ROOT)),
                "sha256": sha256(protocol_file),
            },
            "protocol_snapshot": {
                "path": str(protocol_snapshot.relative_to(PROJECT_ROOT)),
                "sha256": sha256(protocol_snapshot),
            },
            "suite": {
                "path": str(suite_file.relative_to(PROJECT_ROOT)),
                "sha256": sha256(suite_file),
            },
            "guard_source": {
                "path": "src/guards/retail_pre_action.py",
                "sha256": sha256(
                    PROJECT_ROOT / "src" / "guards" / "retail_pre_action.py"
                ),
            },
            "runner_source": {
                "path": "src/guards/ablation.py",
                "sha256": sha256(Path(__file__)),
            },
        },
        "outputs": {
            "results": {
                "path": str(results_path.relative_to(PROJECT_ROOT)),
                "sha256": sha256(results_path),
            },
            "analysis": {
                "path": str(analysis_path.relative_to(PROJECT_ROOT)),
                "sha256": sha256(analysis_path),
            },
        },
        "command": (
            "python -m src.guards.ablation "
            f"--protocol {protocol_file.relative_to(PROJECT_ROOT)} "
            f"--output {destination.relative_to(PROJECT_ROOT)}"
        ),
        "new_llm_calls": 0,
        "uses_reference_actions": False,
        "official_metric": False,
        "limitations": result["validity_notes"],
    }
    (destination / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run finding-level ablation on the Retail Guard suite."
    )
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    protocol = load_protocol(args.protocol)
    suite_path = PROJECT_ROOT / protocol["suite"]
    result = evaluate_ablation(load_suite(suite_path), protocol)
    write_outputs(args.protocol, suite_path, args.output, result)
    print(render_markdown(result))


if __name__ == "__main__":
    main()
