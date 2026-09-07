"""Offline replay of reward scoring only; never calls a model or environment."""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import sys
from pathlib import Path

from src.evaluation.staged_reward_shadow import (
    score_rollout,
    _confirmation_diagnostic_from_serialized_messages,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--old-config", type=Path, required=True)
    parser.add_argument("--new-config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise SystemExit("Refusing to overwrite an existing audit directory")
    rawpath = args.run_dir / "raw_rollouts.jsonl"
    evpath = args.run_dir / "rollout_evidence.jsonl"
    raw = [
        json.loads(s)
        for s in rawpath.read_text(encoding="utf-8").splitlines()
        if s.strip()
    ]
    evidence = [
        json.loads(s)
        for s in evpath.read_text(encoding="utf-8").splitlines()
        if s.strip()
    ]
    if len(raw) != len(evidence) or len(raw) % 4:
        raise ValueError("Unpaired rollout evidence or incomplete n=4 groups")
    configs = [
        json.loads(p.read_text(encoding="utf-8"))
        for p in [args.old_config, args.new_config]
    ]
    specs = [c["reward"]["staged_reward_spec"] for c in configs]
    rows = []
    for i, (r, e) in enumerate(zip(raw, evidence), 1):
        legacy = _confirmation_diagnostic_from_serialized_messages(r["messages"])
        old = score_rollout(r, e, specs[0], legacy)
        new = score_rollout(r, e, specs[1], legacy)
        override = (r.get("reward") or {}).get("reward_override")
        for score in (old, new):
            score["effective_training_reward"] = score["staged_reward"]
            if override:
                if override.get("reason") != "tool_iteration_limit_reached":
                    raise ValueError("Unknown runtime reward override")
                score["effective_training_reward"] = 0.0
                score["runtime_override"] = override
        rows.append(
            {
                "row": i,
                "raw_evidence_binding": r["evidence_sha256"],
                "old": old,
                "new": new,
            }
        )

    def mixed(version):
        return sum(
            statistics.pstdev(
                [r[version]["effective_training_reward"] for r in rows[j : j + 4]]
            )
            > 1e-12
            for j in range(0, len(rows), 4)
        )

    old_low = [
        r
        for r in rows
        if r["old"]["terminal_success"] and r["old"]["staged_reward"] <= 0.15
    ]
    summary = {
        "scope": "OFFLINE_RESCORE_NOT_MODEL_IMPROVEMENT",
        "external_api_called": False,
        "rows": len(rows),
        "groups": len(rows) // 4,
        "old_mixed_groups": mixed("old"),
        "new_mixed_groups": mixed("new"),
        "previous_low_success_rows": [
            {
                "row": r["row"],
                "old": r["old"]["staged_reward"],
                "new": r["new"]["staged_reward"],
                "confirmation": r["new"]["components"]["confirmation_binding"][
                    "verdict"
                ],
            }
            for r in old_low
        ],
    }
    paths = [
        rawpath,
        evpath,
        args.old_config,
        args.new_config,
        Path("src/evaluation/staged_reward_shadow.py"),
        Path("src/evaluation/task44_reward_evidence.py"),
        Path("src/rl/retail_agentic_env.py"),
        Path(__file__),
    ]
    manifest = {
        "command": [sys.executable, *sys.argv],
        "source_sha256": {
            str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths
        },
        "configs": configs,
    }
    args.output_dir.mkdir(parents=True)
    for name, value in [
        ("result.json", rows),
        ("summary.json", summary),
        ("run_manifest.json", manifest),
    ]:
        (args.output_dir / name).write_text(
            json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
    print(json.dumps(summary, ensure_ascii=True))


if __name__ == "__main__":
    main()
