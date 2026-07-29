"""Run a paid/live Guard V1 A/B follow-up on non-quarantined failures.

This script is intentionally separate from the frozen baseline runners. It is
not executed by tests or by the offline guard audit.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import tau2.config as tau2_config
import tau2.evaluator.evaluator_nl_assertions as nl_eval
from tau2.evaluator.evaluator import EvaluationType
from tau2.run import get_tasks, run_tasks

from src.agents.guarded_llm_agent import register_guarded_llm_agent


PROJECT = Path(r"D:\PolicyAgent-PostTrain")
RUN_CONFIG = PROJECT / "configs" / "baseline_trial1_run_config.json"
DEFAULT_OUTPUT = (
    PROJECT / "experiments" / "20260726_retail_guarded_failure3_deepseek"
)
DEFAULT_TASK_IDS = ["95", "98", "107"]


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _configure_nl_judge(config: dict[str, Any]) -> None:
    judge = config["nl_judge"]
    args = {"temperature": judge["temperature"]}
    tau2_config.DEFAULT_LLM_NL_ASSERTIONS = judge["model"]
    tau2_config.DEFAULT_LLM_NL_ASSERTIONS_ARGS = args
    nl_eval.DEFAULT_LLM_NL_ASSERTIONS = judge["model"]
    nl_eval.DEFAULT_LLM_NL_ASSERTIONS_ARGS = args


def run_guarded(
    *,
    task_ids: list[str],
    output_dir: Path,
) -> None:
    config = _load_json(RUN_CONFIG)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(
            f"Refusing to overwrite non-empty experiment directory: {output_dir}"
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    register_guarded_llm_agent()
    _configure_nl_judge(config)

    manifest = {
        "experiment": "retail_guarded_failure3_deepseek",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "task_ids": task_ids,
        "excluded_quarantined_task_ids": ["59"],
        "parent_config": str(RUN_CONFIG),
        "agent": {
            "implementation": "guarded_llm_agent",
            "model": config["agent"]["model"],
            "temperature": config["agent"]["temperature"],
            "guard_mode": "enforce",
            "guard_max_retries": 1,
        },
        "user": config["user"],
        "nl_judge": config["nl_judge"],
        "evaluation": "ALL_WITH_NL_ASSERTIONS",
        "purpose": (
            "Paid live A/B follow-up; measure whether deterministic interception "
            "recovers official reward without using gold actions at runtime."
        ),
    }
    (output_dir / "run_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    runtime = config["runtime"]
    for task_id in task_ids:
        task_dir = output_dir / f"task_{task_id}"
        task_dir.mkdir(parents=True, exist_ok=False)
        tasks = get_tasks("retail", task_ids=[task_id])
        if len(tasks) != 1 or str(tasks[0].id) != task_id:
            raise RuntimeError(f"Could not resolve frozen Retail task {task_id}")
        run_tasks(
            domain="retail",
            tasks=tasks,
            agent="guarded_llm_agent",
            user=config["user"]["implementation"],
            llm_agent=config["agent"]["model"],
            llm_args_agent={
                "temperature": config["agent"]["temperature"],
                "guard_mode": "enforce",
                "guard_max_retries": 1,
            },
            llm_user=config["user"]["model"],
            llm_args_user={"temperature": config["user"]["temperature"]},
            num_trials=1,
            max_steps=runtime["max_steps"],
            max_errors=runtime["max_errors"],
            max_concurrency=1,
            seed=runtime["seed"],
            max_retries=0,
            verbose_logs=True,
            auto_review=False,
            evaluation_type=EvaluationType.ALL_WITH_NL_ASSERTIONS,
            save_to=task_dir / "returned_results.json",
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run Guard V1 on non-quarantined Retail failures."
    )
    parser.add_argument(
        "--task-ids",
        nargs="+",
        default=DEFAULT_TASK_IDS,
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    run_guarded(task_ids=args.task_ids, output_dir=args.output.resolve())


if __name__ == "__main__":
    main()
