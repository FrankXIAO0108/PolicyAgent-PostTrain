"""Run a paid/live paired Guard A/B on non-quarantined failures.

This script is intentionally separate from the frozen baseline runners. It is
not executed by tests or by the offline Guard audit. Paid execution is refused
unless the repository is clean, the API key exists, and the explicit approval
flag is provided. Invoke it as a module:

    python -m src.run_retail_guarded_failure3 --approve-paid-run
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.guards.online_ab import compare_arms, load_protocol, preflight


PROJECT = Path(__file__).resolve().parents[1]
RUN_CONFIG = PROJECT / "configs" / "baseline_trial1_run_config.json"
AB_PROTOCOL = PROJECT / "configs" / "guard_online_ab_v1.json"
DEFAULT_OUTPUT = (
    PROJECT / "experiments" / "20260730_retail_guard_online_ab_v1"
)
DEFAULT_TASK_IDS = ["95", "98", "107"]


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _git(*args: str) -> str:
    return subprocess.check_output(
        ["git", *args],
        cwd=PROJECT,
        text=True,
    ).strip()


def _configure_nl_judge(config: dict[str, Any]) -> None:
    import tau2.config as tau2_config
    import tau2.evaluator.evaluator_nl_assertions as nl_eval

    judge = config["nl_judge"]
    args = {"temperature": judge["temperature"]}
    tau2_config.DEFAULT_LLM_NL_ASSERTIONS = judge["model"]
    tau2_config.DEFAULT_LLM_NL_ASSERTIONS_ARGS = args
    nl_eval.DEFAULT_LLM_NL_ASSERTIONS = judge["model"]
    nl_eval.DEFAULT_LLM_NL_ASSERTIONS_ARGS = args


def _run_arm(
    *,
    arm_name: str,
    agent_implementation: str,
    guard_mode: str,
    guard_max_retries: int,
    task_ids: list[str],
    output_dir: Path,
    config: dict[str, Any],
) -> None:
    from tau2.evaluator.evaluator import EvaluationType
    from tau2.run import get_tasks, run_tasks

    output_dir.mkdir(parents=True, exist_ok=True)

    runtime = config["runtime"]
    for task_id in task_ids:
        task_dir = output_dir / f"task_{task_id}"
        task_dir.mkdir(parents=True, exist_ok=False)
        tasks = get_tasks("retail", task_ids=[task_id])
        if len(tasks) != 1 or str(tasks[0].id) != task_id:
            raise RuntimeError(f"Could not resolve frozen Retail task {task_id}")
        llm_args_agent: dict[str, Any] = {
            "temperature": config["agent"]["temperature"],
        }
        if agent_implementation == "guarded_llm_agent":
            llm_args_agent.update(
                {
                    "guard_mode": guard_mode,
                    "guard_max_retries": guard_max_retries,
                    "guard_trace_path": str(task_dir / "guard_trace.jsonl"),
                }
            )
        run_tasks(
            domain="retail",
            tasks=tasks,
            agent=agent_implementation,
            user=config["user"]["implementation"],
            llm_agent=config["agent"]["model"],
            llm_args_agent=llm_args_agent,
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
        print(f"COMPLETED_ARM_TASK={arm_name}:{task_id}")


def run_paired_ab(
    *,
    task_ids: list[str],
    output_dir: Path,
    approve_paid_run: bool,
) -> None:
    protocol = load_protocol(AB_PROTOCOL)
    if task_ids != [str(value) for value in protocol["task_ids"]]:
        raise ValueError(
            "Paid A/B task IDs must exactly match the frozen protocol order."
        )
    readiness = preflight(protocol, paid_approval=approve_paid_run)
    if readiness["status"] != "READY":
        raise RuntimeError(
            "Paid A/B preflight is blocked: "
            + ", ".join(readiness["blocking_failure_ids"])
        )
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(
            f"Refusing to overwrite non-empty experiment directory: {output_dir}"
        )
    output_dir.mkdir(parents=True, exist_ok=True)

    config = _load_json(RUN_CONFIG)
    from src.agents.guarded_llm_agent import register_guarded_llm_agent

    register_guarded_llm_agent()
    _configure_nl_judge(config)
    manifest = {
        "schema_version": "retail-guard-online-ab-run-v1.0.0",
        "experiment": "retail_guard_online_ab_v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "RUNNING",
        "project_commit": _git("rev-parse", "HEAD"),
        "project_git_dirty": False,
        "task_ids": task_ids,
        "excluded_quarantined_task_ids": ["59"],
        "protocol": {
            "path": str(AB_PROTOCOL.relative_to(PROJECT)),
            "sha256": _sha256(AB_PROTOCOL),
        },
        "parent_config": {
            "path": str(RUN_CONFIG.relative_to(PROJECT)),
            "sha256": _sha256(RUN_CONFIG),
        },
        "arms": protocol["arms"],
        "controlled_variables": protocol["controlled_variables"],
        "preflight": readiness,
        "purpose": (
            "Paid paired failure-subset A/B; compare Base and Guarded recovery "
            "without reference actions at runtime."
        ),
        "interpretation": protocol["interpretation"],
    }
    manifest_path = output_dir / "run_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    try:
        for arm_name in ("base", "guarded"):
            arm = protocol["arms"][arm_name]
            _run_arm(
                arm_name=arm_name,
                agent_implementation=arm["agent_implementation"],
                guard_mode=arm["guard_mode"],
                guard_max_retries=int(arm["guard_max_retries"]),
                task_ids=task_ids,
                output_dir=output_dir / arm_name,
                config=config,
            )
        comparison = compare_arms(
            protocol,
            base_dir=output_dir / "base",
            guarded_dir=output_dir / "guarded",
        )
        (output_dir / "raw_comparison.json").write_text(
            json.dumps(comparison, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        manifest["status"] = "RAW_ARMS_COMPLETE_V7_PENDING"
        manifest["completed_at"] = datetime.now(timezone.utc).isoformat()
    except Exception:
        manifest["status"] = "PARTIAL_OR_FAILED"
        manifest["failed_at"] = datetime.now(timezone.utc).isoformat()
        raise
    finally:
        manifest_path.write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
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
    parser.add_argument(
        "--approve-paid-run",
        action="store_true",
        help="Required acknowledgement that this command makes paid model calls.",
    )
    args = parser.parse_args()
    run_paired_ab(
        task_ids=[str(value) for value in args.task_ids],
        output_dir=args.output.resolve(),
        approve_paid_run=args.approve_paid_run,
    )


if __name__ == "__main__":
    main()
