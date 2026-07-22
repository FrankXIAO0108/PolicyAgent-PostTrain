"""
Run the frozen Retail 5-task engineering smoke subset.

This script is intentionally separate from upstream tau2-bench:
- upstream source is not modified;
- task IDs and run config come from PolicyAgent-PostTrain/configs;
- NL Judge is explicitly patched to DeepSeek;
- each task is run separately for failure isolation;
- raw Results plus an aggregate summary are persisted.
"""

from __future__ import annotations

import json
import subprocess
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any

import tau2.config as tau2_config
import tau2.evaluator.evaluator_nl_assertions as nl_eval

from tau2.evaluator.evaluator import EvaluationType
from tau2.run import get_tasks, run_tasks


PROJECT_ROOT = Path(r"D:\PolicyAgent-PostTrain")
UPSTREAM_ROOT = Path(r"D:\tau2-bench")

SMOKE_CONFIG_PATH = PROJECT_ROOT / "configs" / "smoke_5_tasks.json"

RUN_ID = datetime.now().strftime("%Y%m%d_%H%M%S")
RUN_DIR = (
    PROJECT_ROOT
    / "experiments"
    / f"{RUN_ID}_retail_smoke5_deepseek"
)

JUDGE_MODEL = "deepseek/deepseek-chat"
JUDGE_ARGS = {"temperature": 0.0}


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(data, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )


def get_upstream_commit() -> str:
    return subprocess.check_output(
        ["git", "-C", str(UPSTREAM_ROOT), "rev-parse", "HEAD"],
        text=True,
    ).strip()


def model_to_dict(obj: Any) -> Any:
    if hasattr(obj, "model_dump"):
        return obj.model_dump(mode="json")
    if hasattr(obj, "dict"):
        return obj.dict()
    return obj


def extract_simulations(results: Any) -> list[Any]:
    for attr in ("simulations", "results", "runs"):
        value = getattr(results, attr, None)
        if isinstance(value, list):
            return value

    dumped = model_to_dict(results)

    if isinstance(dumped, dict):
        for key in ("simulations", "results", "runs"):
            value = dumped.get(key)
            if isinstance(value, list):
                return value

    return []


def nested_get(obj: Any, *path: str, default=None):
    cur = obj

    for key in path:
        if cur is None:
            return default

        if isinstance(cur, dict):
            cur = cur.get(key)
        else:
            cur = getattr(cur, key, None)

    return default if cur is None else cur


def main() -> None:
    RUN_DIR.mkdir(parents=True, exist_ok=False)

    smoke_config = json.loads(
        SMOKE_CONFIG_PATH.read_text(encoding="utf-8")
    )

    task_ids = smoke_config["task_ids"]
    fixed = smoke_config["fixed_run_config"]

    assert smoke_config["status"] == "FROZEN"
    assert task_ids == ["59", "29", "72", "50", "28"]
    assert len(task_ids) == 5

    upstream_commit = get_upstream_commit()
    expected_commit = smoke_config["parent_baseline"]["upstream_commit"]

    if upstream_commit != expected_commit:
        raise RuntimeError(
            "Upstream commit mismatch:\n"
            f"expected={expected_commit}\n"
            f"actual={upstream_commit}"
        )

    # --------------------------------------------------------
    # Explicit NL Judge override.
    #
    # evaluator_nl_assertions.py uses direct imports:
    # from tau2.config import DEFAULT_LLM_NL_ASSERTIONS
    #
    # Therefore both config and evaluator-local bindings
    # are patched in this process.
    # --------------------------------------------------------

    tau2_config.DEFAULT_LLM_NL_ASSERTIONS = JUDGE_MODEL
    tau2_config.DEFAULT_LLM_NL_ASSERTIONS_ARGS = JUDGE_ARGS

    nl_eval.DEFAULT_LLM_NL_ASSERTIONS = JUDGE_MODEL
    nl_eval.DEFAULT_LLM_NL_ASSERTIONS_ARGS = JUDGE_ARGS

    # --------------------------------------------------------
    # Trace every real NL Judge call.
    # --------------------------------------------------------

    judge_trace_path = RUN_DIR / "judge_call_trace.jsonl"
    original_generate = nl_eval.generate

    def traced_generate(*args, **kwargs):
        model = kwargs.get(
            "model",
            args[0] if args else None,
        )
        call_name = kwargs.get("call_name")

        started = {
            "timestamp": datetime.now().isoformat(),
            "event": "NL_JUDGE_CALL_STARTED",
            "model": model,
            "call_name": call_name,
        }

        with judge_trace_path.open("a", encoding="utf-8") as f:
            f.write(
                json.dumps(started, ensure_ascii=False) + "\n"
            )

        try:
            result = original_generate(*args, **kwargs)

            succeeded = {
                "timestamp": datetime.now().isoformat(),
                "event": "NL_JUDGE_CALL_SUCCEEDED",
                "model": model,
                "call_name": call_name,
            }

            with judge_trace_path.open("a", encoding="utf-8") as f:
                f.write(
                    json.dumps(succeeded, ensure_ascii=False) + "\n"
                )

            return result

        except Exception as exc:
            failed = {
                "timestamp": datetime.now().isoformat(),
                "event": "NL_JUDGE_CALL_FAILED",
                "model": model,
                "call_name": call_name,
                "error_type": type(exc).__name__,
                "error": str(exc),
            }

            with judge_trace_path.open("a", encoding="utf-8") as f:
                f.write(
                    json.dumps(failed, ensure_ascii=False) + "\n"
                )

            raise

    nl_eval.generate = traced_generate

    run_manifest = {
        "run_id": RUN_ID,
        "created_at": datetime.now().isoformat(),
        "upstream_commit": upstream_commit,
        "smoke_config_path": str(SMOKE_CONFIG_PATH),
        "smoke_config_sha256":
            smoke_config["smoke_config_sha256"],
        "task_ids": task_ids,

        "agent_llm": fixed["agent_llm"],
        "agent_temperature":
            fixed["agent_temperature"],

        "user_llm": fixed["user_llm"],
        "user_temperature":
            fixed["user_temperature"],

        "nl_judge_llm": JUDGE_MODEL,
        "nl_judge_temperature": 0.0,

        "evaluation_type":
            fixed["evaluation_type"],

        "seed": fixed["seed"],
        "max_steps": fixed["max_steps"],
        "num_trials": fixed["num_trials"],
        "max_concurrency":
            fixed["max_concurrency"],
    }

    write_json(
        RUN_DIR / "run_manifest.json",
        run_manifest,
    )

    summaries = []

    print("=" * 90)
    print("RETAIL SMOKE 5 START")
    print("RUN_DIR =", RUN_DIR)
    print("TASK_IDS =", task_ids)
    print("=" * 90)

    for index, task_id in enumerate(task_ids, start=1):

        print("\n" + "#" * 90)
        print(
            f"SMOKE TASK {index}/5 | TASK_ID={task_id}"
        )
        print("#" * 90)

        task_dir = RUN_DIR / f"task_{task_id}"
        task_dir.mkdir(parents=True, exist_ok=True)

        tasks = get_tasks(
            "retail",
            task_ids=[task_id],
        )

        assert len(tasks) == 1
        assert tasks[0].id == task_id

        task_snapshot = model_to_dict(tasks[0])

        write_json(
            task_dir / "task.json",
            task_snapshot,
        )

        start = time.perf_counter()

        status = "UNKNOWN"
        error = None
        results = None

        try:
            results = run_tasks(
                domain="retail",
                tasks=tasks,

                agent="llm_agent",
                user="user_simulator",

                llm_agent=fixed["agent_llm"],
                llm_args_agent={
                    "temperature":
                        fixed["agent_temperature"]
                },

                llm_user=fixed["user_llm"],
                llm_args_user={
                    "temperature":
                        fixed["user_temperature"]
                },

                num_trials=1,
                max_steps=fixed["max_steps"],
                max_errors=10,

                save_dir=task_dir / "tau2_artifacts",

                console_display=True,

                evaluation_type=
                    EvaluationType.ALL_WITH_NL_ASSERTIONS,

                max_concurrency=1,
                seed=fixed["seed"],
                log_level="INFO",

                verbose_logs=True,

                max_retries=0,
                auto_resume=False,
                auto_review=False,
            )

            status = "COMPLETED"

            write_json(
                task_dir / "returned_results.json",
                model_to_dict(results),
            )

        except Exception as exc:
            status = "FAILED"
            error = {
                "type": type(exc).__name__,
                "message": str(exc),
                "traceback": traceback.format_exc(),
            }

            write_json(
                task_dir / "error.json",
                error,
            )

        elapsed = time.perf_counter() - start

        # Best-effort summary extraction.
        # Raw returned_results.json remains the source of truth.
        summary = {
            "task_id": task_id,
            "status": status,
            "wall_clock_seconds": round(elapsed, 3),
            "error_type":
                error["type"] if error else None,
            "error_message":
                error["message"] if error else None,
        }

        if results is not None:
            simulations = extract_simulations(results)

            summary["simulation_count"] = len(simulations)

            if simulations:
                sim = simulations[0]

                summary["reward"] = nested_get(
                    sim,
                    "reward",
                    default=None,
                )

                summary["termination_reason"] = str(
                    nested_get(
                        sim,
                        "termination_reason",
                        default=None,
                    )
                )

                reward_info = nested_get(
                    sim,
                    "reward_info",
                    default=None,
                )

                summary["reward_breakdown"] = model_to_dict(
                    nested_get(
                        reward_info,
                        "reward_breakdown",
                        default=None,
                    )
                )

                summary["nl_assertions"] = model_to_dict(
                    nested_get(
                        reward_info,
                        "nl_assertions",
                        default=None,
                    )
                )

        summaries.append(summary)

        write_json(
            task_dir / "summary.json",
            summary,
        )

        print("\nTASK SUMMARY:")
        print(
            json.dumps(
                summary,
                ensure_ascii=False,
                indent=2,
                default=str,
            )
        )

    # --------------------------------------------------------
    # Aggregate smoke report
    # --------------------------------------------------------

    completed = [
        x for x in summaries
        if x["status"] == "COMPLETED"
    ]

    failed = [
        x for x in summaries
        if x["status"] == "FAILED"
    ]

    rewards = [
        x.get("reward")
        for x in completed
        if isinstance(x.get("reward"), (int, float))
    ]

    aggregate = {
        "run_id": RUN_ID,
        "task_count": len(summaries),
        "completed_count": len(completed),
        "failed_count": len(failed),

        "observed_rewards": rewards,

        "observed_success_count":
            sum(1 for r in rewards if r == 1.0),

        "observed_failure_reward_count":
            sum(1 for r in rewards if r != 1.0),

        "total_wall_clock_seconds":
            round(
                sum(
                    x["wall_clock_seconds"]
                    for x in summaries
                ),
                3,
            ),

        "tasks": summaries,

        "warning": (
            "This is a deliberately risk-stratified engineering smoke "
            "subset. Its success rate must not be reported as the "
            "20-task baseline success rate."
        ),
    }

    write_json(
        RUN_DIR / "smoke_summary.json",
        aggregate,
    )

    print("\n" + "=" * 90)
    print("RETAIL SMOKE 5 FINISHED")
    print("RUN_DIR =", RUN_DIR)
    print("COMPLETED =", len(completed))
    print("FAILED =", len(failed))
    print("OBSERVED_REWARDS =", rewards)
    print(
        "TOTAL_WALL_CLOCK_SECONDS =",
        aggregate["total_wall_clock_seconds"],
    )
    print("=" * 90)


if __name__ == "__main__":
    main()
