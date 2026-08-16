"""Run the Layer-1 teacher pilot on the pinned tau2 mirror.

Layer 1 (``docs/04_数据治理与后训练/2026-08-15_分层教师Pilot设计.md``) expands
the teacher data pool to 8 unused ``rl_train`` tasks x 4 candidates using a
per-candidate teacher temperature ladder ``[0.2, 0.4, 0.6, 0.8]`` to mitigate
near-duplicates. The audited blind teacher agent and the user simulator are
the same implementations as the smoke so results stay comparable.

Per-candidate provenance: every candidate gets a distinct
``seed = base_seed + trial_index`` so the downstream prompt audit (indexed by
``(task_id, seed)``) stays one-to-one per candidate, and each trial records its
temperature and simulation id in ``temperature_map.json``.

The output layout mirrors the smoke run so the downstream audit,
correction, and decision pipelines consume it unchanged:

- ``private_evaluation/task_<id>/returned_results.json`` (aggregated trials)
- ``private_evaluation/task_<id>/temperature_map.json`` (trial provenance)
- ``public_candidates/task_<id>/candidate_trajectories.jsonl``
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.training.run_retail_agentic_grpo import validate_upstream_checkout
from src.training.run_tau2_teacher_trajectory_smoke import (
    REPO_ROOT,
    git_value,
    sha256,
    write_json,
)

SCOPE = "TAU2_GROUNDED_TEACHER_DATA_ENGINEERING_PILOT_LAYER1"
DEFAULT_CONFIG = REPO_ROOT / "configs" / "retail_tau2_teacher_pilot_layer1_v1.json"


def validate_config(path: Path) -> dict[str, Any]:
    config = json.loads(path.read_text(encoding="utf-8-sig"))
    if config.get("status") != "FROZEN":
        raise ValueError("Layer-1 config must be FROZEN")
    if config.get("scope") != SCOPE:
        raise ValueError("Layer-1 scope mismatch")
    visibility = config["teacher_visibility"]
    forbidden = (
        "evaluation_criteria",
        "expected_actions",
        "gold_database_state",
        "hidden_user_scenario",
    )
    if any(visibility.get(key) is not False for key in forbidden):
        raise ValueError("Teacher visibility exposes hidden or post-generation gold")
    generation = config["generation"]
    if generation["agent"]["implementation"] != "audited_teacher_llm_agent":
        raise ValueError("Layer-1 requires the audited outbound-request agent")
    ladder = list(generation["agent"]["temperature_ladder"])
    candidates_per_task = int(generation["candidates_per_task"])
    if len(ladder) != candidates_per_task:
        raise ValueError(
            "temperature_ladder length must equal candidates_per_task "
            f"({len(ladder)} != {candidates_per_task})"
        )
    if not all(isinstance(value, (int, float)) and 0.0 <= float(value) <= 1.0 for value in ladder):
        raise ValueError("temperature_ladder entries must be within [0, 1]")
    if float(generation["user"]["temperature"]) != 0.0:
        raise ValueError("Layer-1 user simulator temperature must be 0.0")
    if candidates_per_task < 1:
        raise ValueError("candidates_per_task must be at least 1")
    split_path = (REPO_ROOT / config["task_split"]).resolve()
    split = json.loads(split_path.read_text(encoding="utf-8"))
    task_ids = [str(row["task_id"]) for row in config["tasks"]]
    if len(task_ids) != len(set(task_ids)):
        raise ValueError("Layer-1 tasks must be unique")
    allowed = set(split["splits"][config["task_subset"]])
    if not set(task_ids).issubset(allowed):
        raise ValueError("Layer-1 contains a task outside rl_train")
    if set(task_ids) & set(split["splits"]["rl_validation"]):
        raise ValueError("Layer-1 overlaps rl_validation")
    if set(task_ids) & set(split["splits"]["development_audit"]):
        raise ValueError("Layer-1 overlaps development_audit")
    if config["offline_evaluation"]["llm_judge_used"] is not False:
        raise ValueError("Layer-1 must not use an LLM judge")
    if config["offline_evaluation"]["maximum_automatic_label"] != "AUTO_PASS_CANDIDATE":
        raise ValueError("Automatic GOLD labels are forbidden")
    return {
        "config": config,
        "config_path": path,
        "config_sha256": sha256(path),
        "split_path": split_path,
        "split_sha256": sha256(split_path),
        "task_ids": task_ids,
        "temperature_ladder": ladder,
        "candidates_per_task": candidates_per_task,
        "base_seed": int(generation["seed"]),
    }


def trial_specs(validated: dict[str, Any]) -> list[dict[str, Any]]:
    """Return one spec per candidate: (trial_index, temperature, seed).

    Every candidate gets a distinct ``seed = base_seed + trial_index`` so the
    downstream prompt audit (indexed by ``(task_id, seed)``) stays one-to-one.
    """
    return [
        {
            "trial_index": index,
            "temperature": temperature,
            "seed": validated["base_seed"] + index,
        }
        for index, temperature in enumerate(validated["temperature_ladder"])
    ]


def run(validated: dict[str, Any], output_dir: Path, *, allow_dirty: bool) -> dict[str, Any]:
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite existing output: {output_dir}")
    dirty = bool(git_value(REPO_ROOT, "status", "--porcelain"))
    if dirty and not allow_dirty:
        raise RuntimeError("Commit the frozen Layer-1 inputs or pass --allow-dirty")
    output_dir.mkdir(parents=True, exist_ok=False)
    config = validated["config"]
    upstream = validate_upstream_checkout(
        config["upstream"]["commit"],
        config["upstream"].get("source_package_sha256"),
        config["upstream"].get("required_files"),
    )

    from src.agents.audited_teacher_agent import (
        PROMPT_AUDIT_LOG_ENV,
        register_audited_teacher_agent,
    )

    register_audited_teacher_agent()
    private_dir = output_dir / "private_evaluation"
    private_dir.mkdir(parents=True, exist_ok=True)
    os.environ[PROMPT_AUDIT_LOG_ENV] = str(
        private_dir / "teacher_prompt_audit.jsonl"
    )

    generation = config["generation"]
    manifest = {
        "schema_version": "retail-tau2-teacher-pilot-layer1-run-v1",
        "status": "STARTED",
        "started_at": datetime.now(timezone.utc).isoformat(),
        "scope": config["scope"],
        "project": {
            "commit": git_value(REPO_ROOT, "rev-parse", "HEAD"),
            "branch": git_value(REPO_ROOT, "branch", "--show-current"),
            "dirty_at_start": dirty,
        },
        "bindings": {
            "config_path": str(validated["config_path"].relative_to(REPO_ROOT)),
            "config_sha256": validated["config_sha256"],
            "task_split_path": str(validated["split_path"].relative_to(REPO_ROOT)),
            "task_split_sha256": validated["split_sha256"],
            "upstream": upstream,
        },
        "task_ids": validated["task_ids"],
        "teacher_visibility": config["teacher_visibility"],
        "generation": generation,
        "claims": config["claims"],
    }
    write_json(output_dir / "run_manifest.json", manifest)

    from tau2.evaluator.evaluator import EvaluationType
    from tau2.run import get_tasks, run_tasks

    failures: list[dict[str, Any]] = []
    completed = 0
    for task_id in validated["task_ids"]:
        task_dir = private_dir / f"task_{task_id}"
        task_dir.mkdir(parents=True, exist_ok=False)
        public_dir = output_dir / "public_candidates" / f"task_{task_id}"
        public_dir.mkdir(parents=True, exist_ok=False)
        tasks = get_tasks("retail", task_ids=[task_id])
        if len(tasks) != 1 or str(tasks[0].id) != task_id:
            raise RuntimeError(f"Unable to resolve exactly one Retail task {task_id}")
        write_json(task_dir / "task_snapshot.json", tasks[0].model_dump(mode="json"))
        started = time.perf_counter()
        try:
            simulations: list[dict[str, Any]] = []
            temperature_map: dict[str, dict[str, Any]] = {}
            for spec in trial_specs(validated):
                trial_index = spec["trial_index"]
                temperature = spec["temperature"]
                trial_seed = spec["seed"]
                results = run_tasks(
                    domain="retail",
                    tasks=tasks,
                    agent=generation["agent"]["implementation"],
                    user=generation["user"]["implementation"],
                    llm_agent=generation["agent"]["model"],
                    llm_args_agent={"temperature": temperature},
                    llm_user=generation["user"]["model"],
                    llm_args_user={"temperature": generation["user"]["temperature"]},
                    num_trials=1,
                    max_steps=int(generation["max_steps"]),
                    max_errors=int(generation["max_errors"]),
                    save_dir=task_dir / "tau2_artifacts" / f"trial_{trial_index}",
                    console_display=True,
                    evaluation_type=EvaluationType.ALL_IGNORE_BASIS,
                    max_concurrency=int(generation["max_concurrency"]),
                    seed=trial_seed,
                    log_level="INFO",
                    verbose_logs=True,
                    max_retries=int(generation["max_retries"]),
                    auto_resume=False,
                    auto_review=False,
                )
                if len(results.simulations) != 1:
                    raise RuntimeError(
                        f"task {task_id} trial {trial_index}: expected 1 simulation"
                    )
                simulation = results.simulations[0]
                simulation_dict = simulation.model_dump(mode="json")
                simulations.append(simulation_dict)
                temperature_map[str(trial_index)] = {
                    "temperature": temperature,
                    "seed": trial_seed,
                    "simulation_id": str(
                        simulation_dict.get("id") or simulation.id
                    ),
                }
            write_json(task_dir / "temperature_map.json", temperature_map)
            write_json(
                task_dir / "returned_results.json",
                {"simulations": simulations},
            )
            public_path = public_dir / "candidate_trajectories.jsonl"
            public_path.write_text(
                "".join(
                    json.dumps(simulation, ensure_ascii=False, default=str) + "\n"
                    for simulation in simulations
                ),
                encoding="utf-8",
            )
            completed += len(simulations)
        except Exception as exc:
            failure = {
                "task_id": task_id,
                "exception_type": type(exc).__name__,
                "message": str(exc),
                "traceback": traceback.format_exc(),
                "elapsed_seconds": time.perf_counter() - started,
            }
            failures.append(failure)
            write_json(task_dir / "system_failure.json", failure)

    from src.training.audit_tau2_teacher_trajectories import audit_run

    audit = audit_run(output_dir) if completed else None
    manifest.update(
        {
            "status": "COMPLETED_WITH_FAILURES" if failures else "COMPLETED",
            "completed_at": datetime.now(timezone.utc).isoformat(),
            "candidate_count": completed,
            "expected_candidate_count": len(validated["task_ids"])
            * validated["candidates_per_task"],
            "system_failures": failures,
            "audit_summary_sha256": (
                sha256(output_dir / "audit_summary.json") if audit else None
            ),
            "training_data_released": False,
            "teacher_qualified": False,
            "directory_boundaries": {
                "private_evaluation": "Contains hidden task definitions, expected actions, raw tau2 Results, per-candidate temperature maps, and exact outbound teacher requests.",
                "public_candidates": "Contains generated SimulationRun trajectories only; still not training-released.",
                "review_packets": "Contains reviewer evidence and may include hidden evaluation context.",
                "released_training_data": "Not created by this pilot run.",
            },
        }
    )
    write_json(output_dir / "run_manifest.json", manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the Layer-1 teacher pilot on the pinned tau2 mirror."
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--allow-dirty", action="store_true")
    args = parser.parse_args()
    validated = validate_config(args.config.resolve())
    if args.validate_only:
        print(
            json.dumps(
                {
                    "status": "VALIDATED",
                    "task_ids": validated["task_ids"],
                    "candidates_per_task": validated["candidates_per_task"],
                    "temperature_ladder": validated["temperature_ladder"],
                    "config_sha256": validated["config_sha256"],
                    "split_sha256": validated["split_sha256"],
                    "external_api_called": False,
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return
    if args.output_dir is None:
        parser.error("--output-dir is required unless --validate-only")
    manifest = run(
        validated,
        args.output_dir.resolve(),
        allow_dirty=args.allow_dirty,
    )
    print(
        json.dumps(
            {
                "status": manifest["status"],
                "candidate_count": manifest["candidate_count"],
                "expected_candidate_count": manifest["expected_candidate_count"],
            }
        )
    )
    if manifest["status"] != "COMPLETED":
        for failure in manifest["system_failures"]:
            print(f"task {failure['task_id']}: {failure['exception_type']}")
        raise SystemExit(2)


if __name__ == "__main__":
    main()
