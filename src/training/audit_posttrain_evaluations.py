from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


STAGES = ("base", "sft", "dpo", "grpo")


def is_strict_json_object(text: str) -> bool:
    """Return True only when the entire completion is one JSON object."""
    try:
        value = json.loads(text.strip())
    except (json.JSONDecodeError, TypeError):
        return False
    return isinstance(value, dict)


def audit_evaluation(payload: dict[str, Any]) -> dict[str, Any]:
    rows = payload.get("rows", [])
    if not rows:
        raise ValueError("Evaluation contains no rows")

    audited_rows: list[dict[str, Any]] = []
    for row in rows:
        strict = is_strict_json_object(row.get("completion", ""))
        extractable = bool(row.get("valid_json"))
        audited_rows.append(
            {
                "scenario_id": row.get("scenario_id"),
                "extractable_json": extractable,
                "strict_json_object": strict,
                "format_gap": extractable and not strict,
            }
        )

    count = len(audited_rows)
    return {
        "rows": count,
        "extractable_json_rate": sum(row["extractable_json"] for row in audited_rows)
        / count,
        "strict_json_object_rate": sum(
            row["strict_json_object"] for row in audited_rows
        )
        / count,
        "format_gap_count": sum(row["format_gap"] for row in audited_rows),
        "format_gap_scenarios": [
            row["scenario_id"] for row in audited_rows if row["format_gap"]
        ],
        "audited_rows": audited_rows,
    }


def audit_run(run_dir: Path) -> dict[str, Any]:
    stages: dict[str, Any] = {}
    for stage in STAGES:
        path = run_dir / f"evaluation_{stage}.json"
        if not path.is_file():
            raise FileNotFoundError(path)
        payload = json.loads(path.read_text(encoding="utf-8-sig"))
        stages[stage] = audit_evaluation(payload)

    return {
        "schema_version": "posttrain-evaluation-format-audit-v1",
        "source_run_dir": str(run_dir.resolve()),
        "metric_definitions": {
            "extractable_json_rate": "Completion contains a decodable JSON object.",
            "strict_json_object_rate": "The entire stripped completion is exactly one JSON object.",
        },
        "stages": stages,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    report = audit_run(args.run_dir.resolve())
    output = args.output or args.run_dir / "format_audit.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
