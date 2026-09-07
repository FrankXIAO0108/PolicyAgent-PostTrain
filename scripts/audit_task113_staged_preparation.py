"""Re-score saved development trajectories; never runs a model or network call."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.evaluation.staged_reward_shadow import (  # noqa: E402
    score_rollout, _confirmation_diagnostic_from_serialized_messages,
)
from src.training.run_retail_agentic_grpo import (  # noqa: E402
    load_json, load_jsonl, sha256, validate_config_and_split,
    validate_optimization_contract, save_json,
)


def audit(run_dir: Path, spec: dict) -> dict:
    raw = load_jsonl(run_dir / "raw_rollouts.jsonl")
    evidence = load_jsonl(run_dir / "rollout_evidence.jsonl")
    if len(raw) != 4 or len(evidence) != 4:
        raise ValueError("Expected exactly one complete n=4 development group")
    scores = []
    old_spec = load_json(ROOT / "configs/retail_agentic_qwen3_4b_task113_staged_claim_v7_grpo_10step_n4_v2.json")["reward"]["staged_reward_spec"]
    for i, (r, e) in enumerate(zip(raw, evidence, strict=True)):
        confirmation = _confirmation_diagnostic_from_serialized_messages(r["messages"])
        s = score_rollout(r, e, spec, confirmation)
        old = score_rollout(r, e, old_spec, confirmation)
        scores.append({
            "row": i, "terminal_reward": s["terminal_reward"],
            "old_v7": old["staged_reward"], "v7_1": s["staged_reward"],
            "components": s["additive_components"], "penalties": s["penalties"],
            "authorization": s["components"]["confirmation_binding"],
            "claim_check": s["components"]["claim_evidence_consistency"],
            "stop_reason": e["completion"]["stop_reason"],
            "evidence_sha256": r["evidence_sha256"],
        })
    rewards = [r["v7_1"] for r in scores]
    mean, std = statistics.mean(rewards), statistics.stdev(rewards)
    for row in scores:
        row["standardized_advantage"] = (row["v7_1"] - mean) / (std + 1e-4)
    return {
        "run_dir": str(run_dir), "raw_sha256": sha256(run_dir / "raw_rollouts.jsonl"),
        "evidence_sha256": sha256(run_dir / "rollout_evidence.jsonl"),
        "rows": scores, "sample_std_ddof1": std, "mean": mean,
        "terminal_failure_positive_advantage_count": sum(
            r["terminal_reward"] == 0 and r["standardized_advantage"] > 0 for r in scores
        ),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError("Use a new output directory; do not overwrite an audit")
    validated = validate_config_and_split(args.config)
    config = validated["config"]
    spec = config["reward"]["staged_reward_spec"]
    batches = {
        "new_sft30": audit(ROOT / "_local_private_runs/tr0905/s100/probe_v2/results_v1/task113_n4", spec),
        "historical_sft": audit(ROOT / "_local_private_runs/task113_prescreen_20260902/passk-n4-v1", spec),
    }
    assert [r["v7_1"] for r in batches["new_sft30"]["rows"]] == [0.0, 1.0, 0.2, 1.0]
    assert [r["v7_1"] for r in batches["historical_sft"]["rows"]] == [0.0, 0.75, 1.0, 0.0]
    report = {
        "status": "OFFLINE_AUDIT_PASSED_CUDA_UPDATE_NOT_TESTED",
        "external_api_called": False, "gpu_used": False, "parameter_update_performed": False,
        "config_sha256": sha256(args.config), "config": config,
        "command": [sys.executable, *sys.argv],
        "optimization_contract": validate_optimization_contract(config),
        "sft_manifest_binding": validated["sft_manifest_binding"], "batches": batches,
        "interpretation": "Development evidence, not independent verifier precision or recall.",
    }
    save_json(args.output_dir / "report.json", report)
    print(json.dumps({"status": report["status"], "output": str(args.output_dir),
                      "reward_vectors": {k: [r["v7_1"] for r in v["rows"]] for k,v in batches.items()}}))


if __name__ == "__main__":
    main()
