from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

from src.rl.retail_agentic_env import confirmation_parameter_binding


SCHEMA_VERSION = "retail-confirmation-parameter-holdout-report-v1.0.0"
ALLOWED_EXPECTED = {"PASS", "REVIEW", "NOT_EVALUABLE"}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def evaluate_holdout(config_path: Path) -> dict[str, Any]:
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    cases = payload.get("cases")
    if not isinstance(cases, list) or not cases:
        raise ValueError("Holdout config must contain a non-empty cases list.")

    seen: set[str] = set()
    rows = []
    for case in cases:
        case_id = str(case.get("case_id") or "")
        expected = str(case.get("expected_verdict") or "")
        if not case_id or case_id in seen:
            raise ValueError(f"Missing or duplicate case_id: {case_id!r}")
        if expected not in ALLOWED_EXPECTED:
            raise ValueError(f"Invalid expected_verdict for {case_id}: {expected!r}")
        seen.add(case_id)
        result = confirmation_parameter_binding(
            str(case.get("tool") or ""),
            dict(case.get("arguments") or {}),
            str(case.get("confirmation_text") or ""),
            value_aliases={
                str(key): [str(alias) for alias in aliases]
                for key, aliases in dict(case.get("value_aliases") or {}).items()
            },
        )
        rows.append(
            {
                "case_id": case_id,
                "category": str(case.get("category") or ""),
                "expected_verdict": expected,
                "observed_verdict": result["verdict"],
                "matched": result["verdict"] == expected,
                "diagnostic": result,
            }
        )

    expected_counts = Counter(row["expected_verdict"] for row in rows)
    observed_counts = Counter(row["observed_verdict"] for row in rows)
    matched_count = sum(row["matched"] for row in rows)
    return {
        "schema_version": SCHEMA_VERSION,
        "scope": "SYNTHETIC_CONTRACT_HOLDOUT",
        "source_config": str(config_path.as_posix()),
        "source_config_sha256": _sha256(config_path),
        "summary": {
            "case_count": len(rows),
            "matched_count": matched_count,
            "mismatch_count": len(rows) - matched_count,
            "exact_match_rate": matched_count / len(rows),
            "expected_verdict_counts": dict(sorted(expected_counts.items())),
            "observed_verdict_counts": dict(sorted(observed_counts.items())),
            "reward_integration_allowed": False,
        },
        "cases": rows,
        "validity_notes": [
            "This hand-authored synthetic contract holdout tests deterministic semantics, not real-trajectory generalization.",
            "REVIEW means evidence is missing or not literally resolvable; it is not proof of a policy violation.",
            "No result from this holdout may be used as a scalar reward without an independently frozen real-trajectory validation set.",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate literal confirmation-parameter binding on a frozen synthetic contract holdout."
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = evaluate_holdout(args.config)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report["summary"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
