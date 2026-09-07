"""Audit cached extraction only; no API, model loading, or fabricated reward."""

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

from src.evaluation.task44_hybrid_reward import (
    INTERPRETER_REVISION,
    validate_candidate_extraction,
)
from src.evaluation.semantic_shadow_judge import build_packet
from src.evaluation.task44_partial_semantics import check_authorization, check_claim
from src.evaluation.task44_reward_evidence import confirmation_evidence


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("Refusing to overwrite audit")

    def load(path):
        return json.loads(path.read_text(encoding="utf-8"))

    def sha(path):
        return hashlib.sha256(path.read_bytes()).hexdigest()

    raw_path = args.run_dir / "raw_rollouts.jsonl"
    raw = [
        json.loads(line) for line in raw_path.read_text(encoding="utf-8").splitlines()
    ]
    failure_path = args.run_dir / "failure_manifest.json"
    expected = load(failure_path)["partial_artifacts"]["raw_rollouts.jsonl"]["sha256"]
    if sha(raw_path) != expected.lower():
        raise ValueError("Frozen raw hash mismatch")
    paths = [
        raw_path,
        failure_path,
        args.run_dir / "config.json",
        Path(__file__),
        Path("src/evaluation/task44_hybrid_reward.py"),
        Path("src/evaluation/task44_partial_semantics.py"),
        Path("src/evaluation/task44_reward_evidence.py"),
        Path("src/evaluation/semantic_shadow_judge.py"),
    ]
    results = []
    for cached in sorted((args.run_dir / "semantic_cache").glob("*/response.json")):
        request_path = cached.with_name("request.json")
        request = load(request_path)
        packet = request["packet"]
        matches = [
            i
            for i, row in enumerate(raw)
            if build_packet(row, packet["policy"], row=0)["messages"]
            == packet["messages"]
        ]
        if len(matches) != 1:
            raise ValueError("Ambiguous/missing cached trajectory binding")
        index = matches[0]
        choice = load(cached)["choices"][0]
        if choice["finish_reason"] != "stop":
            raise ValueError("Incomplete cached response")
        extraction, uncertain = validate_candidate_extraction(
            choice["message"]["content"], packet
        )
        rule_auth = {
            c["write_message_index"]: c
            for c in confirmation_evidence(raw[index]["messages"])["checks"]
        }
        results.append(
            {
                "row_index_zero_based": index,
                "status": "CACHED_EXTRACTION_REINTERPRETED_NOT_NEW_API_RESULT",
                "original_prompt_sha256": request["prompt_sha256"],
                "context_decisions": extraction["context_decisions"],
                "unresolved_candidate_ids": uncertain,
                "claim_checks": [
                    check_claim(raw[index]["messages"], c) for c in extraction["claims"]
                ],
                "authorization_checks": [
                    check_authorization(
                        raw[index]["messages"], a, rule_auth[a["write_message_index"]]
                    )
                    for a in extraction["authorizations"]
                ],
                "reward": None,
                "reward_not_computed_reason": "Frozen failure artifact lacks complete pre-semantic scoring input",
            }
        )
        paths.extend([request_path, cached])
    reviewed = {r["row_index_zero_based"] for r in results}
    report = {
        "scope": "OFFLINE_EXTRACTION_DIAGNOSTIC_ONLY",
        "command": [sys.executable, *sys.argv],
        "project_head": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip(),
        "interpreter_revision": INTERPRETER_REVISION,
        "source_sha256": {str(p): sha(p) for p in paths},
        "external_api_called": False,
        "gpu_used": False,
        "training_eligible": False,
        "online_prescreen_passed": False,
        "raw_count": len(raw),
        "cached_response_count": len(results),
        "unreviewed_row_indices_zero_based": [
            i for i in range(len(raw)) if i not in reviewed
        ],
        "group_rewards": None,
        "group_reward_std": None,
        "results": results,
    }
    with args.output.open("x", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2, allow_nan=False)
    print(
        json.dumps(
            {
                "cached_responses": len(results),
                "raw_count": len(raw),
                "external_api_called": False,
                "output": str(args.output),
            }
        )
    )


if __name__ == "__main__":
    main()
