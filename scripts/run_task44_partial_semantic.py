"""Prepare/extract local semantics and rescore frozen Task44 traces offline."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from src.evaluation.semantic_shadow_judge import build_packet, validate_endpoint
from src.evaluation.staged_reward_shadow import score_rollout
from src.evaluation.task44_partial_semantics import PROMPT, validate_extraction, rescore


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--execute-api", action="store_true")
    parser.add_argument("--responses-dir", type=Path)
    a = parser.parse_args()
    if a.execute_api and a.responses_dir:
        raise ValueError("Choose API or offline cached extraction, not both")
    if a.output_dir.exists():
        raise FileExistsError("Refusing overwrite")

    def load(p):
        return json.loads(Path(p).read_text(encoding="utf-8"))

    def sha(p):
        return hashlib.sha256(Path(p).read_bytes()).hexdigest()

    cfg = load(a.config)
    sourcepath = Path(cfg["source_config"])
    source = load(sourcepath)
    if (
        cfg["mode"] != "OFFLINE_PARTIAL_SEMANTIC_CANDIDATE"
        or cfg["online_promotion_allowed"] is not False
    ):
        raise ValueError("Offline-only config required")
    checks = [
        (source["raw_rollouts"], source["raw_sha256"]),
        (source["policy_path"], source["policy_sha256"]),
        (cfg["baseline_config"], cfg["baseline_config_sha256"]),
        (cfg["evidence_path"], cfg["evidence_sha256"]),
    ]
    for path, expected in checks:
        if sha(path) != expected.lower():
            raise ValueError("Source hash mismatch: " + path)
    raw = [
        json.loads(s)
        for s in Path(source["raw_rollouts"]).read_text(encoding="utf-8").splitlines()
        if s.strip()
    ]
    ev = [
        json.loads(s)
        for s in Path(cfg["evidence_path"]).read_text(encoding="utf-8").splitlines()
        if s.strip()
    ]
    if len(raw) != len(ev):
        raise ValueError("Unpaired evidence")
    spec = load(cfg["baseline_config"])["reward"]["staged_reward_spec"]
    rows = source["rows"]
    if (
        len(set(rows)) != len(rows)
        or len(rows) > source["max_requests"]
        or any(type(r) is not int or r < 1 or r > len(raw) for r in rows)
    ):
        raise ValueError("Invalid row selection")
    policy = Path(source["policy_path"]).read_text(encoding="utf-8")
    packets = [build_packet(raw[r - 1], policy, row=r) for r in rows]
    requests = [
        [
            {"role": "system", "content": PROMPT + "\nRetail policy:\n" + policy},
            {
                "role": "user",
                "content": json.dumps(
                    {k: v for k, v in p.items() if k != "policy"}, ensure_ascii=False
                ),
            },
        ]
        for p in packets
    ]
    if any(
        len(json.dumps(r, ensure_ascii=False)) > source["max_request_chars"]
        for r in requests
    ):
        raise ValueError("Input too long; no truncation")
    endpoint = validate_endpoint(source) if a.execute_api else None
    a.output_dir.mkdir(parents=True)

    def save(name, value):
        (a.output_dir / name).write_text(
            json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )

    paths = [
        a.config,
        sourcepath,
        *[Path(p) for p, _ in checks],
        Path(__file__),
        Path("src/evaluation/task44_partial_semantics.py"),
        Path("src/evaluation/staged_reward_shadow.py"),
        Path("src/evaluation/task44_reward_evidence.py"),
        Path("src/evaluation/semantic_shadow_judge.py"),
    ]
    manifest = {
        "status": "RUNNING",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "command": [sys.executable, *sys.argv],
        "project_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip(),
        "upstream_commit": cfg["upstream_commit"],
        "config": cfg,
        "source_config": source,
        "source_sha256": {str(p): sha(p) for p in paths},
        "rows": rows,
        "external_api_called": False,
        "used_as_training_reward": False,
        "prompt_sha256": hashlib.sha256(PROMPT.encode()).hexdigest(),
        "sampling": {
            "model": source["judge"]["model"],
            "temperature": 0,
            "thinking": "disabled",
            "max_output_tokens": cfg["max_output_tokens"],
            "seed": None,
            "max_retries": 0,
        },
        "requests_attempted": 0,
    }
    save("run_manifest.json", manifest)
    save("packets.json", packets)
    save("requests.json", requests)
    client = None
    if a.execute_api:
        from openai import OpenAI

        client = OpenAI(
            api_key=os.environ[source["judge"]["api_key_env"]],
            base_url=endpoint,
            timeout=source["timeout_seconds"],
            max_retries=0,
        )
    results = []
    stop = False
    for row, packet, request in zip(rows, packets, requests):
        extraction = None
        status = "NOT_REQUESTED"
        error = None
        response = None
        if client:
            manifest["external_api_called"] = True
            manifest["requests_attempted"] += 1
            save("run_manifest.json", manifest)
            try:
                response = client.chat.completions.create(
                    model=source["judge"]["model"],
                    messages=request,
                    temperature=0,
                    max_tokens=cfg["max_output_tokens"],
                    response_format={"type": "json_object"},
                    extra_body={"thinking": {"type": "disabled"}},
                ).model_dump(mode="json")
                save(f"row_{row}_raw_response.json", response)
            except Exception as exc:
                status = "API_ERROR"
                error = type(exc).__name__
                stop = True
        elif a.responses_dir:
            path = a.responses_dir / f"row_{row}_raw_response.json"
            if path.exists():
                # Cached results are copied into this run for an immutable audit trail.
                response = load(path)
                save(f"row_{row}_raw_response.json", response)
            else:
                status = "MISSING_RESPONSE"
        if response is not None:
            try:
                c = response["choices"][0]
                if c["finish_reason"] != "stop":
                    raise ValueError("Incomplete extraction response")
                extraction = validate_extraction(c["message"]["content"], packet)
                status = "VALID_EXTRACTION_PROVISIONAL"
                save(f"row_{row}_extraction.json", extraction)
            except Exception as exc:
                status = "EXTRACTION_ERROR"
                error = type(exc).__name__ + ": " + str(exc)[:160]
        baseline = score_rollout(raw[row - 1], ev[row - 1], spec)
        baseline_effective = (
            0.0
            if (raw[row - 1].get("reward") or {}).get("reward_override")
            else baseline["staged_reward"]
        )
        # No semantic evidence must reproduce the original rule score exactly.
        fallback = rescore(baseline, raw[row - 1], None, spec)
        if abs(fallback["offline_reward"] - baseline_effective) > 1e-10:
            raise ValueError("Fallback score drift")
        candidate = rescore(baseline, raw[row - 1], extraction, spec)
        references = (
            (
                [
                    ref
                    for item in extraction["authorizations"]
                    for ref in item["evidence"]
                ]
                + [
                    {"message_index": c["message_index"], "quote": c["quote"]}
                    for c in extraction["claims"]
                ]
            )
            if extraction
            else []
        )
        warnings = [
            {
                "kind": "EXACT_QUOTE_EXCEEDS_RECOMMENDED_LENGTH",
                "message_index": ref["message_index"],
                "length": len(ref["quote"]),
            }
            for ref in references
            if len(ref["quote"]) > 240
        ]
        results.append(
            {
                "row": row,
                "extraction_status": status,
                "error": error,
                "baseline_rule_reward": baseline_effective,
                "baseline": baseline,
                "candidate": candidate,
                "delta": round(candidate["offline_reward"] - baseline_effective, 12),
                "warnings": warnings,
            }
        )
        save("results.json", results)
        print(
            json.dumps(
                {
                    "row": row,
                    "status": status,
                    "old": baseline_effective,
                    "new": candidate["offline_reward"],
                }
            ),
            flush=True,
        )
        if stop:
            break
    summary = {
        "scope": "OFFLINE_RESCORE_NOT_MODEL_IMPROVEMENT",
        "rows": len(results),
        "extraction_statuses": dict(Counter(r["extraction_status"] for r in results)),
        "changed_rows": [
            {
                "row": r["row"],
                "old": r["baseline_rule_reward"],
                "new": r["candidate"]["offline_reward"],
            }
            for r in results
            if r["delta"]
        ],
        "external_api_called": manifest["external_api_called"],
        "used_as_training_reward": False,
        "group_variance_not_reported": "Selected diagnostic rows do not form complete n=4 groups",
    }
    save("summary.json", summary)
    manifest["status"] = "API_ERROR_STOPPED" if stop else "COMPLETED_OFFLINE_RESCORE"
    save("run_manifest.json", manifest)
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
