"""Re-score a recovered group against saved responses; network is prohibited."""
import argparse
import hashlib
import json
from pathlib import Path
import statistics
import subprocess
import sys


def load(path):
    return json.loads(path.read_text(encoding="utf-8"))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--recovery-dir", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("Never overwrite a frozen audit")

    def no_network(event, _):
        if event in {"socket.connect", "socket.getaddrinfo", "socket.sendto"}:
            raise RuntimeError("Network disabled for offline audit")

    sys.addaudithook(no_network)
    from src.evaluation.task44_hybrid_reward import (
        INTERPRETER_REVISION, PROMPT, score_candidate_response,
    )
    from src.evaluation.semantic_shadow_judge import build_packet

    responses = []
    paths = [Path(__file__), args.recovery_dir / "recovery_manifest.json"]
    for root in args.cache_root:
        for p in sorted(root.glob("*/response.json")):
            request_path = p.with_name("request.json")
            responses.append((load(request_path), load(p), p))
            paths.extend([request_path, p])
    paths.extend(Path("src/evaluation") / f for f in (
        "task44_hybrid_reward.py", "task44_claim_roles.py",
        "task44_partial_semantics.py", "task44_reward_evidence.py",
        "semantic_shadow_judge.py",
    ))
    rows = []
    for index in range(4):
        path = args.recovery_dir / f"row_{index}.json"
        paths.append(path)
        row = load(path)
        assert row["row_index_zero_based"] == index
        assert row["initial_state_matches"] and row["final_state_matches"]
        data = row["scoring_input"]
        assert str(data["raw"]["task_id"]) == "44"
        matching = [(q, a, p) for q, a, p in responses
                    if build_packet(data["raw"], q["packet"]["policy"], row=0)["messages"]
                    == q["packet"]["messages"]]
        if len(matching) != 1:
            raise ValueError("Require exactly one saved response per frozen row")
        request, response, response_path = matching[0]
        assert request["prompt_sha256"] == hashlib.sha256(PROMPT.encode()).hexdigest()
        assert response["model"] == "deepseek-v4-flash"
        assert response["choices"][0]["finish_reason"] == "stop"
        assert request["scoring_input"]["base"] == data["base"]
        assert request["scoring_input"]["spec"] == data["spec"]
        record = {"row": index, "reward": None, "status": "BLOCKED",
                  "response_path": str(response_path)}
        try:
            result = score_candidate_response(data["base"], data["raw"], data["spec"],
                       request["packet"], response["choices"][0]["message"]["content"])
            result.update(used_as_training_reward=False, scope="OFFLINE_SAVED_GROUP_REINTERPRETATION")
            record.update(status="READY", reward=result["offline_reward"], details=result)
        except (ValueError, RuntimeError) as exc:
            record.update(error_type=type(exc).__name__, error=str(exc))
        rows.append(record)
    ready = all(r["status"] == "READY" for r in rows)
    rewards = [r["reward"] for r in rows] if ready else None
    report = {
        "interpreter_revision": INTERPRETER_REVISION,
        "command": [sys.executable, *sys.argv],
        "project_head": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "source_sha256": {str(p): sha(p) for p in paths},
        "external_api_called": False, "gpu_used": False, "training_started": False,
        "new_responses": 0, "all_scores_ready": ready, "rows": rows,
        "group_rewards": rewards,
        "reward_std_population": statistics.pstdev(rewards) if ready else None,
        "reward_std_sample": statistics.stdev(rewards) if ready else None,
        "scope": "FOUR_KNOWN_DEVELOPMENT_CASES_NOT_INDEPENDENT_GOLD",
    }
    with args.output.open("x", encoding="utf-8") as out:
        json.dump(report, out, ensure_ascii=False, indent=2, allow_nan=False)
    print(json.dumps({k: v for k, v in report.items() if k not in {"rows", "source_sha256", "command"}}))


if __name__ == "__main__":
    main()
