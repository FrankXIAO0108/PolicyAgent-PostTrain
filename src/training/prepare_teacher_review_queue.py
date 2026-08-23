from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def _canonical_tool_trace(simulation: dict[str, Any]) -> str:
    calls: list[dict[str, Any]] = []
    for message in simulation.get("messages") or []:
        if message.get("role") != "assistant":
            continue
        for call in message.get("tool_calls") or []:
            calls.append(
                {
                    "name": call.get("name"),
                    "arguments": call.get("arguments") or {},
                }
            )
    return json.dumps(calls, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def _load_simulations(run_dir: Path) -> dict[str, dict[str, Any]]:
    simulations: dict[str, dict[str, Any]] = {}
    for path in sorted((run_dir / "public_candidates").glob("task_*/candidate_trajectories.jsonl")):
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            simulation = json.loads(line)
            simulations[str(simulation["id"])] = simulation
    return simulations


def _load_audits(run_dir: Path) -> list[dict[str, Any]]:
    path = run_dir / "candidate_audit.jsonl"
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _load_temperatures(run_dir: Path) -> tuple[dict[str, float], list[dict[str, str]]]:
    temperatures: dict[str, float] = {}
    sources: list[dict[str, str]] = []
    for path in sorted((run_dir / "private_evaluation").glob("task_*/temperature_map.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        for binding in payload.values():
            temperatures[str(binding["simulation_id"])] = float(binding["temperature"])
        sources.append({"path": str(path), "sha256": sha256(path)})
    return temperatures, sources


def _representative_rank(row: dict[str, Any]) -> tuple[Any, ...]:
    metrics = row.get("metrics") or {}
    return (
        metrics.get("db_match") is not True,
        float(metrics.get("tau2_reward") or 0.0) != 1.0,
        int(metrics.get("tool_error_count") or 0),
        int(metrics.get("unexpected_write_count") or 0),
        len(row.get("review_reasons") or []),
        float(row["teacher_temperature"]),
        str(row["candidate_id"]),
    )


def build_review_queue(
    run_dir: Path,
    *,
    benchmark_anomaly_tasks: set[str] | None = None,
) -> dict[str, Any]:
    benchmark_anomaly_tasks = benchmark_anomaly_tasks or set()
    audits = _load_audits(run_dir)
    simulations = _load_simulations(run_dir)
    temperatures, temperature_sources = _load_temperatures(run_dir)
    if len(audits) != len(simulations):
        raise ValueError(
            f"audit/simulation count mismatch: {len(audits)} != {len(simulations)}"
        )

    by_task: dict[str, list[dict[str, Any]]] = {}
    for row in audits:
        candidate_id = str(row["candidate_id"])
        simulation = simulations.get(candidate_id)
        if simulation is None:
            raise ValueError(f"missing public candidate: {candidate_id}")
        if candidate_id not in temperatures:
            raise ValueError(f"missing teacher temperature binding: {candidate_id}")
        enriched = dict(row)
        enriched["teacher_temperature"] = temperatures[candidate_id]
        enriched["tool_trace_fingerprint"] = hashlib.sha256(
            _canonical_tool_trace(simulation).encode("utf-8")
        ).hexdigest().upper()
        by_task.setdefault(str(row["task_id"]), []).append(enriched)

    rejected_only_tasks = {
        task_id
        for task_id, rows in by_task.items()
        if all(row["automatic_label"] == "REJECTED" for row in rows)
    }
    reward_flip_tasks = {
        task_id
        for task_id, rows in by_task.items()
        if len({float((row.get("metrics") or {}).get("tau2_reward") or 0.0) for row in rows}) > 1
    }

    queue: list[dict[str, Any]] = []
    duplicate_holdup: list[dict[str, Any]] = []
    for task_id in sorted(by_task, key=lambda value: int(value)):
        rows = by_task[task_id]
        review_groups: dict[str, list[dict[str, Any]]] = {}
        for row in rows:
            if row["automatic_label"] == "REVIEW_REQUIRED":
                review_groups.setdefault(row["tool_trace_fingerprint"], []).append(row)
            else:
                queue.append(_queue_row(row, task_id, rejected_only_tasks, reward_flip_tasks, benchmark_anomaly_tasks))

        for fingerprint, group in sorted(review_groups.items()):
            ordered = sorted(group, key=_representative_rank)
            representative = ordered[0]
            queue.append(
                _queue_row(
                    representative,
                    task_id,
                    rejected_only_tasks,
                    reward_flip_tasks,
                    benchmark_anomaly_tasks,
                )
            )
            for duplicate in ordered[1:]:
                duplicate_holdup.append(
                    {
                        "task_id": task_id,
                        "candidate_id": duplicate["candidate_id"],
                        "representative_candidate_id": representative["candidate_id"],
                        "tool_trace_fingerprint": fingerprint,
                        "disposition": "EXACT_TOOL_TRACE_DUPLICATE_HOLDUP",
                        "note": "Exact tool sequence duplicate only; semantic equivalence not asserted.",
                    }
                )

    priority_order = {"P0": 0, "P1": 1, "P2": 2}
    queue.sort(key=lambda row: (priority_order[row["priority"]], int(row["task_id"]), int(row["trial"])))
    return {
        "schema_version": "teacher-review-queue-v1",
        "source": {
            "candidate_audit": {
                "path": str(run_dir / "candidate_audit.jsonl"),
                "sha256": sha256(run_dir / "candidate_audit.jsonl"),
            },
            "temperature_maps": temperature_sources,
        },
        "interpretation_boundary": [
            "Queue priority is routing, not a quality label.",
            "Exact tool-trace deduplication does not assert semantic equivalence.",
            "No queue row is training-released without correction, replay, and owner review.",
        ],
        "counts": {
            "source_candidates": len(audits),
            "review_queue": len(queue),
            "duplicate_holdup": len(duplicate_holdup),
            "priority": {
                priority: sum(row["priority"] == priority for row in queue)
                for priority in ("P0", "P1", "P2")
            },
        },
        "rejected_only_tasks": sorted(rejected_only_tasks, key=int),
        "reward_flip_tasks": sorted(reward_flip_tasks, key=int),
        "benchmark_anomaly_tasks": sorted(benchmark_anomaly_tasks, key=int),
        "queue": queue,
        "duplicate_holdup": duplicate_holdup,
    }


def _queue_row(
    row: dict[str, Any],
    task_id: str,
    rejected_only_tasks: set[str],
    reward_flip_tasks: set[str],
    benchmark_anomaly_tasks: set[str],
) -> dict[str, Any]:
    metrics = row.get("metrics") or {}
    reasons: list[str] = []
    if task_id in rejected_only_tasks:
        reasons.append("rejected_only_task")
    if metrics.get("db_match") is not True:
        reasons.append("final_database_state_mismatch")
    if task_id in reward_flip_tasks:
        reasons.append("temperature_reward_flip")
    if task_id in benchmark_anomaly_tasks:
        reasons.append("benchmark_or_golden_action_anomaly")
    if row["automatic_label"] == "REJECTED":
        reasons.append("automatic_rejected")
    priority = "P0" if reasons else "P2"
    if priority != "P0" and (
        float(metrics.get("tau2_reward") or 0.0) != 1.0
        or row.get("hard_rejection_reasons")
    ):
        priority = "P1"
    return {
        "priority": priority,
        "priority_reasons": reasons,
        "task_id": task_id,
        "candidate_id": row["candidate_id"],
        "trial": int(row.get("trial") or 0),
        "teacher_temperature": float(row["teacher_temperature"]),
        "automatic_label": row["automatic_label"],
        "tau2_reward": float(metrics.get("tau2_reward") or 0.0),
        "db_match": metrics.get("db_match"),
        "tool_error_count": int(metrics.get("tool_error_count") or 0),
        "hard_rejection_reasons": row.get("hard_rejection_reasons") or [],
        "review_reasons": row.get("review_reasons") or [],
        "tool_trace_fingerprint": row["tool_trace_fingerprint"],
        "evidence_pack": row["evidence_pack"],
        "proposed_disposition": "OWNER_REVIEW_REQUIRED",
    }


def write_outputs(payload: dict[str, Any], output_dir: Path) -> None:
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty output: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    queue_path = output_dir / "review_queue.json"
    queue_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    manifest = {
        "schema_version": "teacher-review-queue-manifest-v1",
        "command": [sys.executable, *sys.argv],
        "inputs": payload["source"],
        "outputs": {"review_queue.json": sha256(queue_path)},
        "counts": payload["counts"],
        "external_api_called": False,
        "training_data_released": False,
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare a deduplicated teacher candidate owner-review queue.")
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--benchmark-anomaly-task", action="append", default=[])
    args = parser.parse_args()
    payload = build_review_queue(
        args.run_dir.resolve(),
        benchmark_anomaly_tasks={str(value) for value in args.benchmark_anomaly_task},
    )
    write_outputs(payload, args.output_dir.resolve())
    print(json.dumps(payload["counts"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
