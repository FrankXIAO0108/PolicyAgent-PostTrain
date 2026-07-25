"""
Run or resume Trial-2 for the four representative Retail failure tasks.

This runner deliberately reuses the frozen Trial-1 configuration so that the
agent, user simulator, NL judge, temperatures, seed, maximum steps, retry
policy, and upstream tau2-bench commit remain unchanged.

It writes to a separate experiment directory and never reuses or overwrites
Trial-1 artifacts.
"""

from __future__ import annotations

import json
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any

import tau2.config as tau2_config
import tau2.evaluator.evaluator_nl_assertions as nl_eval
from tau2.evaluator.evaluator import EvaluationType
from tau2.run import get_tasks, run_tasks

import run_retail_baseline20 as base


TASK_IDS = ["59", "98", "95", "107"]
RUN_SUFFIX = "_retail_failure4_trial2_deepseek"


def find_existing_runs() -> list[Path]:
    if not base.EXPERIMENTS_ROOT.exists():
        return []

    return sorted(
        (
            path
            for path in base.EXPERIMENTS_ROOT.glob(f"*{RUN_SUFFIX}")
            if path.is_dir()
        ),
        key=lambda path: path.name,
        reverse=True,
    )


def select_run_dir() -> tuple[Path, bool]:
    """
    Resume the newest incomplete Trial-2 run.

    If a completed Trial-2 already exists, refuse to create another one
    silently. This prevents accidental Trial-3 sampling.
    """
    existing_runs = find_existing_runs()
    incomplete_runs = [
        path
        for path in existing_runs
        if not (path / "trial2_summary.json").exists()
    ]

    if incomplete_runs:
        return incomplete_runs[0], True

    if existing_runs:
        raise RuntimeError(
            "\nA completed four-task Trial-2 already exists:\n"
            f"  {existing_runs[0]}\n\n"
            "Refusing to create an additional sample silently.\n"
        )

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    return base.EXPERIMENTS_ROOT / f"{run_id}{RUN_SUFFIX}", False


def build_trial2_aggregate(
    *,
    run_id: str,
    run_config: dict[str, Any],
    summaries: list[dict[str, Any]],
) -> dict[str, Any]:
    aggregate = base.build_aggregate(
        run_id=run_id,
        run_config=run_config,
        task_ids=TASK_IDS,
        summaries=summaries,
    )

    aggregate["experiment"] = "retail_failure4_trial2_deepseek"
    aggregate["experiment_role"] = "REPRESENTATIVE_TASK_SECOND_SAMPLE"
    aggregate["scope"] = (
        "Trial-2 second sample for frozen Retail development tasks "
        "59, 98, 95, and 107; not an official leaderboard score."
    )
    aggregate["parent_trial"] = "retail_baseline20_trial1_deepseek"
    aggregate["task_ids"] = TASK_IDS
    return aggregate


def configure_nl_judge(run_config: dict[str, Any]) -> None:
    judge_cfg = run_config["nl_judge"]
    judge_model = judge_cfg["model"]
    judge_args = {"temperature": judge_cfg["temperature"]}

    tau2_config.DEFAULT_LLM_NL_ASSERTIONS = judge_model
    tau2_config.DEFAULT_LLM_NL_ASSERTIONS_ARGS = judge_args
    nl_eval.DEFAULT_LLM_NL_ASSERTIONS = judge_model
    nl_eval.DEFAULT_LLM_NL_ASSERTIONS_ARGS = judge_args


def main() -> None:
    baseline_config = base.load_json(base.BASELINE_CONFIG_PATH)
    run_config = base.load_json(base.RUN_CONFIG_PATH)

    if baseline_config["status"] != "FROZEN":
        raise RuntimeError("baseline_20_tasks.json is not FROZEN.")

    if run_config["status"] != "FROZEN":
        raise RuntimeError("baseline_trial1_run_config.json is not FROZEN.")

    frozen_task_ids = [str(task_id) for task_id in run_config["task_ids"]]
    missing_ids = [
        task_id for task_id in TASK_IDS
        if task_id not in frozen_task_ids
    ]

    if missing_ids:
        raise RuntimeError(
            f"Trial-2 task IDs are absent from the frozen baseline: {missing_ids}"
        )

    if run_config["evaluation"]["type"] != "ALL_WITH_NL_ASSERTIONS":
        raise RuntimeError("Unexpected evaluation type in frozen run config.")

    if run_config["runtime"]["num_trials"] != 1:
        raise RuntimeError("Frozen run config must use num_trials == 1.")

    upstream_commit = base.get_upstream_commit()
    expected_commit = run_config["parent_baseline"]["upstream_commit"]

    if upstream_commit != expected_commit:
        raise RuntimeError(
            "\nUpstream commit mismatch.\n"
            f"Expected: {expected_commit}\n"
            f"Actual:   {upstream_commit}\n"
        )

    run_dir, is_resume = select_run_dir()
    run_id = run_dir.name[: -len(RUN_SUFFIX)]

    if is_resume:
        print(f"RESUMING TRIAL-2: {run_dir}")
    else:
        run_dir.mkdir(parents=True, exist_ok=False)

    agent_cfg = run_config["agent"]
    user_cfg = run_config["user"]
    judge_cfg = run_config["nl_judge"]
    runtime_cfg = run_config["runtime"]

    configure_nl_judge(run_config)

    manifest_path = run_dir / "run_manifest.json"
    expected_manifest = {
        "run_id": run_id,
        "created_at": datetime.now().isoformat(),
        "experiment": "retail_failure4_trial2_deepseek",
        "experiment_role": "REPRESENTATIVE_TASK_SECOND_SAMPLE",
        "scope": (
            "Trial-2 second sample for frozen Retail development tasks "
            "59, 98, 95, and 107."
        ),
        "not_official_leaderboard_score": True,
        "parent_trial": "retail_baseline20_trial1_deepseek",
        "upstream_commit": upstream_commit,
        "baseline_config_path": str(base.BASELINE_CONFIG_PATH),
        "run_config_path": str(base.RUN_CONFIG_PATH),
        "baseline_config_sha256": run_config["parent_baseline"][
            "baseline_config_sha256"
        ],
        "run_config_sha256": run_config["run_config_sha256"],
        "task_ids": TASK_IDS,
        "agent": agent_cfg,
        "user": user_cfg,
        "nl_judge": judge_cfg,
        "evaluation": run_config["evaluation"],
        "runtime": runtime_cfg,
        "protocol": run_config["protocol"],
    }

    if manifest_path.exists():
        manifest = base.load_json(manifest_path)

        if [str(item) for item in manifest.get("task_ids", [])] != TASK_IDS:
            raise RuntimeError(
                "Existing Trial-2 manifest has different task IDs."
            )

        if manifest.get("upstream_commit") != upstream_commit:
            raise RuntimeError(
                "Existing Trial-2 manifest has a different commit."
            )

        if manifest.get("run_config_sha256") != run_config["run_config_sha256"]:
            raise RuntimeError(
                "Existing Trial-2 manifest has a different config."
            )
    else:
        base.write_json(manifest_path, expected_manifest)

    print("=" * 100)
    print("RETAIL REPRESENTATIVE FAILURE-4 TRIAL-2")
    print("RUN_DIR =", run_dir)
    print("TASK_IDS =", TASK_IDS)
    print("AGENT =", agent_cfg["model"])
    print("USER =", user_cfg["model"])
    print("NL_JUDGE =", judge_cfg["model"])
    print("UPSTREAM =", upstream_commit)
    print("=" * 100)

    summaries: list[dict[str, Any]] = []

    for index, task_id in enumerate(TASK_IDS, start=1):
        task_dir = run_dir / f"task_{task_id}"
        task_dir.mkdir(parents=True, exist_ok=True)

        returned_results_path = task_dir / "returned_results.json"

        print("#" * 100)
        print(f"TRIAL-2 TASK {index}/{len(TASK_IDS)} | TASK_ID={task_id}")
        print("#" * 100)

        if returned_results_path.exists():
            print("REUSE EXISTING returned_results.json — NO API CALL")

            base.append_jsonl(
                task_dir / "attempt_history.jsonl",
                {
                    "timestamp": datetime.now().isoformat(),
                    "event": "REUSED_EXISTING_RESULT",
                    "task_id": task_id,
                },
            )

            try:
                summary = base.build_task_summary(
                    task_id=task_id,
                    task_dir=task_dir,
                    execution_source="REUSED_EXISTING_RESULT",
                    wall_clock_seconds=None,
                    system_error=None,
                )
            except Exception as exc:
                summary = {
                    "task_id": task_id,
                    "status": "RESULT_PARSE_FAILED",
                    "execution_source": "REUSED_EXISTING_RESULT",
                    "reward": None,
                    "error_type": type(exc).__name__,
                    "error_message": str(exc),
                    "traceback": traceback.format_exc(),
                }

                base.write_json(
                    task_dir / "parser_error.json",
                    summary,
                )
        else:
            tasks = get_tasks("retail", task_ids=[task_id])

            if len(tasks) != 1 or str(tasks[0].id) != task_id:
                raise RuntimeError(
                    f"Could not resolve exactly one task for {task_id}."
                )

            base.write_json(
                task_dir / "task.json",
                base.model_to_dict(tasks[0]),
            )

            base.append_jsonl(
                task_dir / "attempt_history.jsonl",
                {
                    "timestamp": datetime.now().isoformat(),
                    "event": "EXECUTION_STARTED",
                    "task_id": task_id,
                },
            )

            wall_start = time.perf_counter()
            system_error: dict[str, Any] | None = None
            execution_succeeded = False

            try:
                results = run_tasks(
                    domain="retail",
                    tasks=tasks,
                    agent=agent_cfg["implementation"],
                    user=user_cfg["implementation"],
                    llm_agent=agent_cfg["model"],
                    llm_args_agent={
                        "temperature": agent_cfg["temperature"]
                    },
                    llm_user=user_cfg["model"],
                    llm_args_user={
                        "temperature": user_cfg["temperature"]
                    },
                    num_trials=runtime_cfg["num_trials"],
                    max_steps=runtime_cfg["max_steps"],
                    max_errors=runtime_cfg["max_errors"],
                    save_dir=task_dir / "tau2_artifacts",
                    console_display=True,
                    evaluation_type=(
                        EvaluationType.ALL_WITH_NL_ASSERTIONS
                    ),
                    max_concurrency=runtime_cfg["max_concurrency"],
                    seed=runtime_cfg["seed"],
                    log_level="INFO",
                    verbose_logs=runtime_cfg["verbose_logs"],
                    max_retries=runtime_cfg["max_retries"],
                    auto_resume=False,
                    auto_review=runtime_cfg["auto_review"],
                )

                base.write_json(
                    returned_results_path,
                    base.model_to_dict(results),
                )

                execution_succeeded = True

                base.append_jsonl(
                    task_dir / "attempt_history.jsonl",
                    {
                        "timestamp": datetime.now().isoformat(),
                        "event": "EXECUTION_SUCCEEDED",
                        "task_id": task_id,
                    },
                )

            except Exception as exc:
                system_error = {
                    "type": type(exc).__name__,
                    "message": str(exc),
                    "traceback": traceback.format_exc(),
                    "timestamp": datetime.now().isoformat(),
                }

                base.write_json(
                    task_dir / "error.json",
                    system_error,
                )

                base.append_jsonl(
                    task_dir / "attempt_history.jsonl",
                    {
                        "timestamp": datetime.now().isoformat(),
                        "event": "SYSTEM_FAILURE",
                        "task_id": task_id,
                        "error_type": system_error["type"],
                        "error_message": system_error["message"],
                    },
                )

            wall_clock_seconds = time.perf_counter() - wall_start

            try:
                summary = base.build_task_summary(
                    task_id=task_id,
                    task_dir=task_dir,
                    execution_source=(
                        "EXECUTED_NOW"
                        if execution_succeeded
                        else "SYSTEM_FAILED"
                    ),
                    wall_clock_seconds=wall_clock_seconds,
                    system_error=system_error,
                )
            except Exception as exc:
                summary = {
                    "task_id": task_id,
                    "status": (
                        "RESULT_PARSE_FAILED"
                        if returned_results_path.exists()
                        else "SYSTEM_FAILED"
                    ),
                    "execution_source": (
                        "EXECUTED_NOW"
                        if execution_succeeded
                        else "SYSTEM_FAILED"
                    ),
                    "reward": None,
                    "wall_clock_seconds": round(
                        wall_clock_seconds,
                        3,
                    ),
                    "error_type": type(exc).__name__,
                    "error_message": str(exc),
                    "traceback": traceback.format_exc(),
                }

                base.write_json(
                    task_dir / "parser_error.json",
                    summary,
                )

        base.write_json(
            task_dir / "summary.json",
            summary,
        )

        summaries.append(summary)

        print(
            json.dumps(
                summary,
                ensure_ascii=False,
                indent=2,
                default=str,
            )
        )

        progress = build_trial2_aggregate(
            run_id=run_id,
            run_config=run_config,
            summaries=summaries,
        )

        base.write_json(
            run_dir / "progress.json",
            progress,
        )

    aggregate = build_trial2_aggregate(
        run_id=run_id,
        run_config=run_config,
        summaries=summaries,
    )

    base.write_json(
        run_dir / "trial2_summary.json",
        aggregate,
    )

    base.write_baseline_csv(
        run_dir / "trial2_table.csv",
        summaries,
    )

    print("=" * 100)
    print("RETAIL REPRESENTATIVE FAILURE-4 TRIAL-2 FINISHED")
    print("RUN_DIR =", run_dir)
    print("COMPLETED =", aggregate["completed_count"])
    print("SYSTEM_FAILED =", aggregate["system_failed_count"])
    print(
        "RESULT_PARSE_FAILED =",
        aggregate["result_parse_failed_count"],
    )
    print(
        "BUSINESS_SUCCESS =",
        aggregate["business_success_count"],
    )
    print(
        "BUSINESS_FAILURE =",
        aggregate["business_failure_count"],
    )
    print(
        "TOTAL_MODEL_COST_USD =",
        round(aggregate["total_model_cost_usd"], 8),
    )
    print(
        "TRIAL2_JSON =",
        run_dir / "trial2_summary.json",
    )
    print(
        "TRIAL2_CSV =",
        run_dir / "trial2_table.csv",
    )

    if (
        aggregate["completed_count"] == 4
        and aggregate["system_failed_count"] == 0
        and aggregate["result_parse_failed_count"] == 0
        and aggregate["valid_reward_count"] == 4
    ):
        print("RETAIL_FAILURE4_TRIAL2_RUN_OK")
    else:
        print("RETAIL_FAILURE4_TRIAL2_RUN_INCOMPLETE")

    print("=" * 100)


if __name__ == "__main__":
    main()