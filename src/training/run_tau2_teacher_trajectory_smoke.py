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


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = REPO_ROOT / "configs" / "retail_tau2_teacher_trajectory_smoke_v1.json"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=str) + "\n",
        encoding="utf-8",
    )


def git_value(cwd: Path, *args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=cwd, check=True, capture_output=True, text=True
    ).stdout.strip()


def validate_config(path: Path) -> dict[str, Any]:
    config = json.loads(path.read_text(encoding="utf-8-sig"))
    if config.get("status") != "FROZEN":
        raise ValueError("Teacher smoke config must be FROZEN")
    if config.get("scope") != "TAU2_GROUNDED_TEACHER_DATA_ENGINEERING_SMOKE":
        raise ValueError("Teacher smoke scope mismatch")
    visibility = config["teacher_visibility"]
    forbidden = (
        "evaluation_criteria",
        "expected_actions",
        "gold_database_state",
        "hidden_user_scenario",
    )
    if any(visibility.get(key) is not False for key in forbidden):
        raise ValueError("Teacher visibility exposes hidden or post-generation gold")
    if config["generation"]["agent"]["implementation"] != "audited_teacher_llm_agent":
        raise ValueError("Teacher smoke requires the audited outbound-request agent")
    split_path = (REPO_ROOT / config["task_split"]).resolve()
    split = json.loads(split_path.read_text(encoding="utf-8"))
    task_ids = [str(row["task_id"]) for row in config["tasks"]]
    if len(task_ids) != 3 or len(task_ids) != len(set(task_ids)):
        raise ValueError("Teacher smoke requires exactly three unique tasks")
    allowed = set(split["splits"][config["task_subset"]])
    if not set(task_ids).issubset(allowed):
        raise ValueError("Teacher smoke contains a task outside rl_train")
    if set(task_ids) & set(split["splits"]["rl_validation"]):
        raise ValueError("Teacher smoke overlaps rl_validation")
    if set(task_ids) & set(split["splits"]["development_audit"]):
        raise ValueError("Teacher smoke overlaps development_audit")
    if config["offline_evaluation"]["llm_judge_used"] is not False:
        raise ValueError("Teacher smoke must not use an LLM judge")
    if config["offline_evaluation"]["maximum_automatic_label"] != "AUTO_PASS_CANDIDATE":
        raise ValueError("Automatic GOLD labels are forbidden")
    return {
        "config": config,
        "config_path": path,
        "config_sha256": sha256(path),
        "split_path": split_path,
        "split_sha256": sha256(split_path),
        "task_ids": task_ids,
    }


def run(validated: dict[str, Any], output_dir: Path, *, allow_dirty: bool) -> dict[str, Any]:
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite existing output: {output_dir}")
    dirty = bool(git_value(REPO_ROOT, "status", "--porcelain"))
    if dirty and not allow_dirty:
        raise RuntimeError("Commit the frozen smoke inputs or pass --allow-dirty")
    output_dir.mkdir(parents=True, exist_ok=False)
    config = validated["config"]
    from src.training.run_retail_agentic_grpo import validate_upstream_checkout

    upstream = validate_upstream_checkout(
        config["upstream"]["commit"],
        config["upstream"].get("source_package_sha256"),
        config["upstream"].get("required_files"),
    )

    # Gold is intentionally unavailable to the teacher agent. Tau2's normal
    # generation behavior is retained while the audited wrapper records the
    # exact outbound request in a private artifact.
    from tau2.evaluator.evaluator import EvaluationType
    from tau2.run import get_tasks, run_tasks

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
        "schema_version": "retail-tau2-teacher-trajectory-smoke-run-v1",
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
        # Keep hidden task/evaluation data outside public trajectory artifacts.
        write_json(task_dir / "task_snapshot.json", tasks[0].model_dump(mode="json"))
        started = time.perf_counter()
        try:
            results = run_tasks(
                domain="retail",
                tasks=tasks,
                agent=generation["agent"]["implementation"],
                user=generation["user"]["implementation"],
                llm_agent=generation["agent"]["model"],
                llm_args_agent={"temperature": generation["agent"]["temperature"]},
                llm_user=generation["user"]["model"],
                llm_args_user={"temperature": generation["user"]["temperature"]},
                num_trials=int(generation["candidates_per_task"]),
                max_steps=int(generation["max_steps"]),
                max_errors=int(generation["max_errors"]),
                save_dir=task_dir / "tau2_artifacts",
                console_display=True,
                evaluation_type=EvaluationType.ALL_IGNORE_BASIS,
                max_concurrency=int(generation["max_concurrency"]),
                seed=int(generation["seed"]),
                log_level="INFO",
                verbose_logs=True,
                max_retries=int(generation["max_retries"]),
                auto_resume=False,
                auto_review=False,
            )
            write_json(task_dir / "returned_results.json", results.model_dump(mode="json"))
            public_path = public_dir / "candidate_trajectories.jsonl"
            public_path.write_text(
                "".join(
                    json.dumps(
                        simulation.model_dump(mode="json"),
                        ensure_ascii=False,
                        default=str,
                    )
                    + "\n"
                    for simulation in results.simulations
                ),
                encoding="utf-8",
            )
            completed += len(results.simulations)
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
            * int(generation["candidates_per_task"]),
            "system_failures": failures,
            "audit_summary_sha256": (
                sha256(output_dir / "audit_summary.json") if audit else None
            ),
            "training_data_released": False,
            "teacher_qualified": False,
            "directory_boundaries": {
                "private_evaluation": "Contains hidden task definitions, expected actions, raw tau2 Results, and exact outbound teacher requests.",
                "public_candidates": "Contains generated SimulationRun trajectories only; still not training-released.",
                "review_packets": "Contains reviewer evidence and may include hidden evaluation context.",
                "released_training_data": "Not created by this smoke.",
            },
        }
    )
    write_json(output_dir / "run_manifest.json", manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
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
        parser.error("--output-dir is required unless --validate-only is used")
    print(json.dumps(run(validated, args.output_dir.resolve(), allow_dirty=args.allow_dirty), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
