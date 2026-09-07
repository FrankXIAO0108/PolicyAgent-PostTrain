"""One authorized four-request batch on recovered traces; no GPU or rollout."""

import argparse
import hashlib
import json
import statistics
import subprocess
import sys
from pathlib import Path

from src.evaluation.task44_hybrid_reward import (
    INTERPRETER_REVISION,
    PROMPT,
    candidate_packet,
    hybrid_score,
    validate_settings,
)
from src.evaluation.semantic_shadow_judge import digest, validate_endpoint


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def prepare(config):
    if (
        config["mode"] != "FROZEN_RECOVERED_TASK44_SEMANTIC_RESCORE"
        or config["max_requests"] != 4
        or config["training_authorized"] is not False
        or config["gpu_authorized"] is not False
    ):
        raise ValueError("Scope must be four semantic requests only")
    if (
        config["prompt_sha256"] != hashlib.sha256(PROMPT.encode()).hexdigest()
        or config["interpreter_revision"] != INTERPRETER_REVISION
    ):
        raise ValueError("Frozen interpreter/prompt mismatch")
    for path, expected in config["source_sha256"].items():
        if sha(path) != expected:
            raise ValueError(f"Frozen source mismatch: {path}")
    policy = Path(config["policy_path"]).read_text(encoding="utf-8")
    rows = []
    for i in range(4):
        row = load(Path(config["recovery_dir"]) / f"row_{i}.json")
        if (
            row["row_index_zero_based"] != i
            or not row["initial_state_matches"]
            or not row["final_state_matches"]
        ):
            raise ValueError("Recovery identity/state mismatch")
        data = row["scoring_input"]
        settings = data["spec"]["semantic_assistance"]
        validate_settings(settings)
        if hashlib.sha256(policy.encode()).hexdigest() != settings["policy_sha256"]:
            raise ValueError("Policy content mismatch")
        if rows and data["spec"] != rows[0]["scoring_input"]["spec"]:
            raise ValueError("Group reward spec mismatch")
        if str(data["raw"]["task_id"]) != "44":
            raise ValueError("Only frozen Task44 permitted")
        rows.append(row)
    return rows, policy


def execute(rows, policy, output, *, scorer=hybrid_score):
    results = []
    for i, row in enumerate(rows):
        data = row["scoring_input"]
        record = {
            "row_index_zero_based": i,
            "rule_only_reward": row["rule_only_reward"],
            "hybrid_reward": None,
            "training_eligible": False,
        }
        try:
            result = scorer(
                data["base"], data["raw"], data["spec"], policy=policy, directory=output
            )
            result.update(
                used_as_training_reward=False, scope="FROZEN_RECOVERED_SEMANTIC_RESCORE"
            )
            record.update(
                status="READY", hybrid_reward=result["offline_reward"], details=result
            )
        except Exception as exc:
            # A batch may inspect remaining frozen rows, never replace/retry a row.
            record.update(status="SCORING_ERROR", error_type=type(exc).__name__)
        with (output / f"result_{i}.json").open("x", encoding="utf-8") as handle:
            json.dump(record, handle, ensure_ascii=False, indent=2, allow_nan=False)
        results.append(record)
        print(
            json.dumps({k: v for k, v in record.items() if k != "details"}), flush=True
        )
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--execute-api", action="store_true")
    args = parser.parse_args()
    config = load(args.config)
    rows, policy = prepare(config)
    if not args.execute_api:
        print(
            json.dumps(
                {"status": "VALIDATED", "rows": len(rows), "external_api_called": False}
            )
        )
        return
    if args.output_dir is None or args.output_dir.exists():
        raise FileExistsError("A new output directory is required")
    validate_endpoint(
        {"judge": rows[0]["scoring_input"]["spec"]["semantic_assistance"]}
    )
    args.output_dir.mkdir(parents=True)

    def save(name, data):
        with (args.output_dir / name).open("x", encoding="utf-8") as out:
            json.dump(data, out, ensure_ascii=False, indent=2, allow_nan=False)

    save(
        "launch_manifest.json",
        {
            "command": [sys.executable, *sys.argv],
            "config": config,
            "config_sha256": sha(args.config),
            "runner_sha256": sha(__file__),
            "project_head": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], text=True
            ).strip(),
            "max_requests": 4,
            "retries": 0,
            "gpu_used": False,
            "training_authorized": False,
            "old_responses_reused": False,
        },
    )
    save(
        "frozen_packets.json",
        [candidate_packet(r["scoring_input"]["raw"], policy) for r in rows],
    )
    save(
        "frozen_prompt.json",
        {"prompt": PROMPT, "sha256": hashlib.sha256(PROMPT.encode()).hexdigest()},
    )
    results = execute(rows, policy, args.output_dir)
    # Any mid-run drift invalidates the batch; frozen raw API receipts remain intact.
    check_rows, check_policy = prepare(config)
    if digest(check_rows) != digest(rows) or check_policy != policy:
        raise RuntimeError("Input changed during requests")
    ready = all(r["status"] == "READY" for r in results)
    rewards = [r["hybrid_reward"] for r in results] if ready else None
    summary = {
        "status": "ALL_SCORED" if ready else "SCORING_INCOMPLETE",
        "results": results,
        "group_rewards": rewards,
        "reward_std_population": statistics.pstdev(rewards) if ready else None,
        "reward_std_sample": statistics.stdev(rewards) if ready else None,
        "all_same_reward": len(set(rewards)) == 1 if ready else None,
        "request_receipts": len(
            list(args.output_dir.glob("semantic_cache/*/request.json"))
        ),
        "response_receipts": len(
            list(args.output_dir.glob("semantic_cache/*/response.json"))
        ),
        "gpu_used": False,
        "training_started": False,
    }
    save("summary.json", summary)
    print(json.dumps({k: v for k, v in summary.items() if k != "results"}), flush=True)


if __name__ == "__main__":
    main()
