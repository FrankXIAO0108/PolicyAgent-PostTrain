from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any

from src.evaluation.replay_evaluator import Tau2Runtime, replay_task_simulation
from src.training.teacher_evidence_pack import build_evidence_pack
from src.training.teacher_trajectory_quality import audit_simulation, summarize_audits


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def load_candidates(run_dir: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    candidates: list[dict[str, Any]] = []
    sources: list[dict[str, Any]] = []
    for path in sorted(
        (run_dir / "private_evaluation").glob("task_*/returned_results.json")
    ):
        payload = json.loads(path.read_text(encoding="utf-8"))
        simulations = list(payload.get("simulations") or [])
        candidates.extend(
            {"simulation": simulation, "source_path": path}
            for simulation in simulations
        )
        sources.append(
            {
                "path": str(path.relative_to(run_dir)),
                "sha256": sha256(path),
                "simulation_count": len(simulations),
            }
        )
    return candidates, sources


def _prompt_audit_index(run_dir: Path) -> dict[tuple[str, int], dict[str, Any]]:
    path = run_dir / "private_evaluation" / "teacher_prompt_audit.jsonl"
    if not path.is_file():
        return {}
    grouped: dict[tuple[str, int], list[dict[str, Any]]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        key = (str(row["task_id"]), int(row["seed"]))
        grouped.setdefault(key, []).append(row)
    return {
        key: {
            "private_source_path": str(path.relative_to(run_dir)),
            "private_source_sha256": sha256(path),
            "request_count": len(rows),
            "all_gold_visibility_checks_passed": all(
                row.get("gold_visibility_check_passed") is True for row in rows
            ),
            "request_sha256": [row["request_sha256"] for row in rows],
            "requested_models": sorted(
                {str(row["response"].get("requested_model")) for row in rows}
            ),
            "reported_models": sorted(
                {
                    str(row["response"].get("reported_model"))
                    for row in rows
                    if row["response"].get("reported_model")
                }
            ),
            "system_fingerprints": sorted(
                {
                    str(row["response"].get("system_fingerprint"))
                    for row in rows
                    if row["response"].get("system_fingerprint")
                }
            ),
        }
        for key, rows in grouped.items()
    }


def audit_run(run_dir: Path) -> dict[str, Any]:
    candidates, sources = load_candidates(run_dir)
    if not candidates:
        raise ValueError(f"No completed teacher candidates found under {run_dir}")
    tau2_root = os.environ.get("POLICYAGENT_TAU2_ROOT")
    if not tau2_root:
        raise RuntimeError("POLICYAGENT_TAU2_ROOT is required for state replay")
    runtime = Tau2Runtime(tau2_root)
    prompt_index = _prompt_audit_index(run_dir)
    rows: list[dict[str, Any]] = []
    packs_dir = run_dir / "review_packets"
    packs_dir.mkdir(parents=True, exist_ok=True)
    for candidate in candidates:
        source_path = candidate["source_path"]
        results, raw_results = runtime.load_results(source_path)
        simulation_id = str(candidate["simulation"].get("id") or "")
        simulation = next(
            item for item in results.simulations if str(item.id) == simulation_id
        )
        task = next(item for item in results.tasks if str(item.id) == str(simulation.task_id))
        constructor = runtime.registry.get_env_constructor(
            results.info.environment_info.domain_name
        )
        replay = replay_task_simulation(
            task,
            simulation,
            constructor=constructor,
            domain=results.info.environment_info.domain_name,
            raw_results=raw_results,
        )
        raw_simulation = simulation.model_dump(mode="json")
        row = audit_simulation(raw_simulation)
        prompt_key = (str(simulation.task_id), int(simulation.seed))
        prompt_audit = prompt_index.get(prompt_key)
        if prompt_audit is None or not prompt_audit["all_gold_visibility_checks_passed"]:
            row["review_reasons"].append("teacher_gold_visibility_not_proven")
            if row["automatic_label"] == "AUTO_PASS_CANDIDATE":
                row["automatic_label"] = "REVIEW_REQUIRED"
        pack = build_evidence_pack(
            simulation=raw_simulation,
            task=task.model_dump(mode="json"),
            initial_state=replay.initial_state,
            final_state=replay.agent_state,
            automatic_audit=row,
            prompt_audit=prompt_audit,
        )
        pack["state_replay"] = {
            **replay.to_dict(),
            "replay_errors": replay.replay_errors,
        }
        claim_verdict = pack["claim_state_consistency"]["verdict"]
        if claim_verdict == "FAIL":
            row["hard_rejection_reasons"].append("final_answer_contradicts_final_state")
            row["automatic_label"] = "REJECTED"
        elif claim_verdict == "REVIEW":
            row["review_reasons"].append("final_answer_claim_requires_review")
            if row["automatic_label"] == "AUTO_PASS_CANDIDATE":
                row["automatic_label"] = "REVIEW_REQUIRED"
        pack["automatic_verification"] = row
        pack["recommended_label"] = row["automatic_label"]
        pack_path = packs_dir / f"candidate_{simulation_id}.json"
        pack_path.write_text(
            json.dumps(pack, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        row["evidence_pack"] = {
            "path": str(pack_path.relative_to(run_dir)),
            "sha256": sha256(pack_path),
        }
        row["state_replay"] = pack["state_replay"]
        rows.append(row)
    output = run_dir / "candidate_audit.jsonl"
    output.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )
    report = {
        "schema_version": "retail-tau2-teacher-candidate-audit-v1",
        "scope": "TAU2_GROUNDED_TEACHER_DATA_ENGINEERING_SMOKE",
        "sources": sources,
        "summary": summarize_audits(rows),
        "candidate_audit": {
            "path": output.name,
            "sha256": sha256(output),
        },
        "release_gate": {
            "automatic_gold_labels_allowed": False,
            "human_adjudication_required": True,
            "sft_data_released": False,
        },
    }
    report_path = run_dir / "audit_summary.json"
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(audit_run(args.run_dir.resolve()), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
