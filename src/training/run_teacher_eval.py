import argparse
import json
import os
import time
import traceback
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.training.dirhash import directory_sha256
from src.training.run_retail_agentic_grpo import validate_upstream_checkout
from src.training.run_tau2_teacher_trajectory_smoke import (
    REPO_ROOT,
    git_value,
    sha256,
    write_json,
)

SCOPE_PREFIX = "TEACHER_SFT_BENCHMARK_EVAL"
DEFAULT_CONFIG = REPO_ROOT / "configs" / "retail_teacher_eval_v1.json"


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def entity_overlap(task_payload: dict[str, Any], entity_values: list[str]) -> list[str]:
    """Return teacher-pool entity values found anywhere in a task definition."""
    full_text = json.dumps(task_payload)
    return [value for value in entity_values if value in full_text]


def validate_config(path: Path) -> dict[str, Any]:
    path = path.resolve()
    config = load_json(path)
    if config.get("status") != "FROZEN":
        raise ValueError("Teacher eval config must be FROZEN")
    if not str(config.get("scope") or "").startswith(SCOPE_PREFIX):
        raise ValueError(f"scope must start with {SCOPE_PREFIX!r}")

    agent = config["agent"]
    if agent["implementation"] != "llm_agent":
        raise ValueError("Teacher eval requires the plain llm_agent")
    if float(agent["temperature"]) != 0.0:
        raise ValueError("Teacher eval agent temperature must be 0.0")
    if not agent.get("api_base"):
        raise ValueError("Teacher eval requires an api_base for the served agent model")

    user = config["user"]
    if user["implementation"] != "user_simulator":
        raise ValueError("Teacher eval requires user_simulator")
    if float(user["temperature"]) != 0.0:
        raise ValueError("Teacher eval user temperature must be 0.0")

    judge = config["nl_judge"]
    if not judge.get("model"):
        raise ValueError("Teacher eval requires an nl_judge model")

    evaluation = config["evaluation"]
    if evaluation["type"] != "ALL_WITH_NL_ASSERTIONS":
        raise ValueError("Teacher eval requires ALL_WITH_NL_ASSERTIONS")
    if int(evaluation["num_trials"]) < 1:
        raise ValueError("num_trials must be at least 1")
    if int(evaluation["max_concurrency"]) != 1:
        raise ValueError("max_concurrency must be 1 for reproducibility")

    task_rows = config["tasks"]
    task_ids = [str(row["task_id"]) for row in task_rows]
    if len(task_ids) != len(set(task_ids)):
        raise ValueError("Teacher eval tasks must be unique")
    sources = {str(row.get("source")) for row in task_rows}
    if not sources.issubset({"train_candidates", "test_clean"}):
        raise ValueError("task source must be train_candidates or test_clean")

    model_runs = config["model_runs"]
    run_names = [str(run["name"]) for run in model_runs]
    if len(run_names) != len(set(run_names)):
        raise ValueError("model_runs names must be unique")
    for run in model_runs:
        if not run.get("vllm_model") or not run.get("litellm_model"):
            raise ValueError("each model_run needs vllm_model and litellm_model")

    gate = config["entity_gate"]
    if gate.get("must_be_disjoint") is not True:
        raise ValueError("entity_gate.must_be_disjoint must be true")
    groups_path = (REPO_ROOT / gate["groups_path"]).resolve()
    if not groups_path.is_file():
        raise FileNotFoundError(f"entity gate groups missing: {groups_path}")
    if sha256(groups_path) != str(gate["groups_sha256"]).upper():
        raise ValueError("entity gate groups hash mismatch")
    groups = load_json(groups_path)
    entity_values = [
        str(group).split(":", 1)[1] for group in groups["entity_groups"]
    ]

    tau2_root_value = os.environ.get("POLICYAGENT_TAU2_ROOT")
    if not tau2_root_value:
        raise RuntimeError("Set POLICYAGENT_TAU2_ROOT to the pinned tau2 checkout")
    tau2_root = Path(tau2_root_value).expanduser().resolve()
    tasks_payload = load_json(tau2_root / "data/tau2/domains/retail/tasks.json")
    split = load_json(tau2_root / "data/tau2/domains/retail/split_tasks.json")
    by_id = {
        str(task["id"]): task
        for task in tasks_payload
        if task and task.get("id")
    }
    for row in task_rows:
        task_id = str(row["task_id"])
        if task_id not in by_id:
            raise ValueError(f"unknown Retail task {task_id}")
        source = str(row["source"])
        if source == "train_candidates" and task_id not in split["train"]:
            raise ValueError(
                f"{task_id} marked train_candidates but not in the official train split"
            )
        if source == "test_clean" and task_id not in split["test"]:
            raise ValueError(
                f"{task_id} marked test_clean but not in the official test split"
            )
        hits = entity_overlap(by_id[task_id], entity_values)
        if hits:
            raise ValueError(
                f"task {task_id} overlaps teacher training entities: {hits}"
            )

    return {
        "config": config,
        "config_path": path,
        "config_sha256": sha256(path),
        "task_rows": task_rows,
        "task_ids": task_ids,
        "model_runs": model_runs,
        "model_runs_by_name": {str(run["name"]): run for run in model_runs},
        "entity_values": entity_values,
        "seed": int(evaluation["seed"]),
        "num_trials": int(evaluation["num_trials"]),
    }


def _configure_nl_judge(config: dict[str, Any]) -> None:
    import tau2.config as tau2_config
    import tau2.evaluator.evaluator_nl_assertions as nl_eval

    judge = config["nl_judge"]
    args = {"temperature": judge["temperature"]}
    tau2_config.DEFAULT_LLM_NL_ASSERTIONS = judge["model"]
    tau2_config.DEFAULT_LLM_NL_ASSERTIONS_ARGS = args
    nl_eval.DEFAULT_LLM_NL_ASSERTIONS = judge["model"]
    nl_eval.DEFAULT_LLM_NL_ASSERTIONS_ARGS = args


def probe_vllm(api_base: str, model: str, timeout: float = 60.0) -> None:
    """Send one minimal OpenAI-compatible completion to confirm the endpoint."""
    body = json.dumps(
        {
            "model": model,
            "messages": [{"role": "user", "content": "ping"}],
            "max_tokens": 1,
            "temperature": 0.0,
        }
    ).encode("utf-8")
    request = urllib.request.Request(
        api_base.rstrip("/") + "/chat/completions",
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        payload = json.loads(response.read().decode("utf-8"))
    if not payload.get("choices"):
        raise RuntimeError(f"vLLM probe returned no choices: {payload}")


def validate_checkpoint_binding(model_run: dict[str, Any]) -> dict[str, str]:
    """Fail closed when an evaluation checkpoint is absent or not hash-bound."""
    checkpoint = model_run.get("checkpoint")
    if not checkpoint:
        raise ValueError(
            f"model checkpoint path is not bound for {model_run.get('name')!r}"
        )
    checkpoint_path = Path(checkpoint).expanduser().resolve()
    if not checkpoint_path.is_dir():
        raise FileNotFoundError(f"model checkpoint missing: {checkpoint_path}")
    expected_value = model_run.get("expected_sha256")
    if not expected_value:
        raise ValueError(
            f"model checkpoint hash is not bound for {model_run.get('name')!r}"
        )
    expected = str(expected_value).upper()
    actual = directory_sha256(checkpoint_path)
    if actual != expected:
        raise ValueError(
            f"checkpoint hash mismatch for {model_run.get('name')}: "
            f"{actual} != {expected}"
        )
    return {"path": str(checkpoint_path), "sha256": actual}


def _row_rewards(row: dict[str, Any]) -> list[float]:
    value = row["reward"]
    if isinstance(value, list):
        return [float(item) for item in value]
    return [float(value)]


def build_summary(
    *,
    per_task: list[dict[str, Any]],
    failures: list[dict[str, Any]],
    infrastructure_failures: list[dict[str, Any]],
    validated: dict[str, Any],
    run_name: str,
    model_run: dict[str, Any],
) -> dict[str, Any]:
    sources = ("train_candidates", "test_clean")
    by_source: dict[str, dict[str, Any]] = {}
    for source in sources:
        rows = [row for row in per_task if row["source"] == source]
        succeeded = sum(1 for row in rows if row["success"])
        source_reward_groups = [_row_rewards(row) for row in rows]
        source_trials = sum(len(rewards) for rewards in source_reward_groups)
        source_successes = sum(
            reward == 1.0
            for rewards in source_reward_groups
            for reward in rewards
        )
        by_source[source] = {
            "tasks": len(rows),
            "success": succeeded,
            "success_rate": round(succeeded / len(rows), 4) if rows else None,
            "success_rate_semantics": "tasks successful in every trial",
            "successful_trials": source_successes,
            "total_trials": source_trials,
            "trial_success_rate": (
                round(source_successes / source_trials, 4)
                if source_trials
                else None
            ),
        }
    succeeded = sum(1 for row in per_task if row["success"])
    reward_groups = [_row_rewards(row) for row in per_task]
    total_trials = sum(len(rewards) for rewards in reward_groups)
    successful_trials = sum(
        reward == 1.0 for rewards in reward_groups for reward in rewards
    )
    tasks_with_any_success = sum(
        any(reward == 1.0 for reward in rewards) for rewards in reward_groups
    )
    return {
        "schema_version": "retail-teacher-sft-benchmark-eval-summary-v2",
        "run_name": run_name,
        "model": {
            "vllm_model": model_run["vllm_model"],
            "litellm_model": model_run["litellm_model"],
            "checkpoint": model_run.get("checkpoint"),
            "expected_sha256": model_run.get("expected_sha256"),
        },
        "evaluation": {
            "type": validated["config"]["evaluation"]["type"],
            "num_trials": validated["num_trials"],
            "seed": validated["seed"],
            "temperature": validated["config"]["agent"]["temperature"],
        },
        "per_task": per_task,
        "success_rate": {
            "overall": round(succeeded / len(per_task), 4) if per_task else None,
            "semantics": "fraction of tasks successful in every trial",
            "trial_level": {
                "successful_trials": successful_trials,
                "total_trials": total_trials,
                "rate": (
                    round(successful_trials / total_trials, 4)
                    if total_trials
                    else None
                ),
            },
            "task_level": {
                "any_trial_success": tasks_with_any_success,
                "all_trials_success": succeeded,
                "total_tasks": len(per_task),
            },
            "by_source": by_source,
        },
        "coverage": {
            "expected_tasks": len(validated["task_ids"]),
            "evaluated_tasks": len(per_task),
            "infrastructure_failure_tasks": len(
                {row["task_id"] for row in infrastructure_failures}
            ),
            "system_failure_tasks": len({row["task_id"] for row in failures}),
        },
        "infrastructure_failures": infrastructure_failures,
        "system_failures": failures,
    }


def simulation_infrastructure_failure(
    simulation: Any,
    *,
    task_id: str,
    source: str,
    trial_index: int,
) -> dict[str, Any] | None:
    """Describe an unscored simulation without treating it as model failure."""
    if simulation.reward_info is not None:
        return None
    info = simulation.info if isinstance(simulation.info, dict) else {}
    return {
        "task_id": task_id,
        "source": source,
        "trial_index": trial_index,
        "simulation_id": str(getattr(simulation, "id", "")),
        "termination_reason": getattr(simulation, "termination_reason", None),
        "error_type": info.get("error_type") or "MissingRewardInfo",
        "message": info.get("error") or "Simulation returned no reward_info",
    }


def build_agent_llm_args(agent_config: dict[str, Any]) -> dict[str, Any]:
    """Build litellm kwargs for the served agent model.

    litellm requires credentials even for local OpenAI-compatible servers.
    vLLM ignores the key value, so local endpoints get a placeholder unless
    the config explicitly provides an api_key.
    """
    api_base = str(agent_config["api_base"])
    args: dict[str, Any] = {
        "temperature": agent_config["temperature"],
        "api_base": api_base,
    }
    if "api_key" in agent_config:
        args["api_key"] = str(agent_config["api_key"])
    elif api_base.startswith(("http://localhost", "http://127.0.0.1")):
        args["api_key"] = "EMPTY"
    else:
        raise ValueError(
            "agent.api_key is required when api_base is not a local endpoint"
        )
    return args


def run(
    validated: dict[str, Any],
    output_dir: Path,
    run_name: str,
    *,
    allow_dirty: bool,
) -> dict[str, Any]:
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite existing output: {output_dir}")
    dirty = bool(git_value(REPO_ROOT, "status", "--porcelain"))
    if dirty and not allow_dirty:
        raise RuntimeError("Commit the frozen eval inputs or pass --allow-dirty")

    config = validated["config"]
    model_run = validated["model_runs_by_name"][run_name]
    checkpoint_binding = validate_checkpoint_binding(model_run)

    output_dir.mkdir(parents=True, exist_ok=False)
    upstream = validate_upstream_checkout(
        config["upstream"]["commit"],
        config["upstream"].get("source_package_sha256"),
        config["upstream"].get("required_files"),
    )
    manifest = {
        "schema_version": "retail-teacher-sft-benchmark-eval-run-v1",
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
            "entity_gate": config["entity_gate"],
            "upstream": upstream,
            "checkpoint": checkpoint_binding,
        },
        "model_run": model_run,
        "task_ids": validated["task_ids"],
        "evaluation": config["evaluation"],
        "claims": config["claims"],
    }
    write_json(output_dir / "run_manifest.json", manifest)

    _configure_nl_judge(config)
    probe_vllm(config["agent"]["api_base"], model_run["vllm_model"])

    from tau2.evaluator.evaluator import EvaluationType
    from tau2.run import get_tasks, run_tasks

    agent_llm_args = build_agent_llm_args(config["agent"])
    user = config["user"]
    evaluation = config["evaluation"]

    private_dir = output_dir / "private_evaluation"
    private_dir.mkdir(parents=True, exist_ok=True)
    failures: list[dict[str, Any]] = []
    infrastructure_failures: list[dict[str, Any]] = []
    per_task: list[dict[str, Any]] = []
    for row in validated["task_rows"]:
        task_id = str(row["task_id"])
        task_dir = private_dir / f"task_{task_id}"
        task_dir.mkdir(parents=True, exist_ok=False)
        tasks = get_tasks("retail", task_ids=[task_id])
        if len(tasks) != 1 or str(tasks[0].id) != task_id:
            raise RuntimeError(f"Unable to resolve exactly one Retail task {task_id}")
        write_json(task_dir / "task_snapshot.json", tasks[0].model_dump(mode="json"))
        started = time.perf_counter()
        try:
            results = run_tasks(
                domain="retail",
                tasks=tasks,
                agent=config["agent"]["implementation"],
                user=user["implementation"],
                llm_agent=model_run["litellm_model"],
                llm_args_agent=agent_llm_args,
                llm_user=user["model"],
                llm_args_user={"temperature": user["temperature"]},
                num_trials=validated["num_trials"],
                max_steps=int(evaluation["max_steps"]),
                max_errors=int(evaluation["max_errors"]),
                save_dir=task_dir / "tau2_artifacts",
                console_display=True,
                evaluation_type=EvaluationType.ALL_WITH_NL_ASSERTIONS,
                max_concurrency=int(evaluation["max_concurrency"]),
                seed=validated["seed"],
                log_level="INFO",
                verbose_logs=True,
                max_retries=int(evaluation["max_retries"]),
                auto_review=False,
                save_to=task_dir / "returned_results.json",
            )
            simulations = results.simulations
            task_infrastructure_failures = [
                failure
                for trial_index, sim in enumerate(simulations)
                if (
                    failure := simulation_infrastructure_failure(
                        sim,
                        task_id=task_id,
                        source=str(row["source"]),
                        trial_index=trial_index,
                    )
                )
                is not None
            ]
            if task_infrastructure_failures:
                infrastructure_failures.extend(task_infrastructure_failures)
                continue
            rewards = [float(sim.reward_info.reward) for sim in simulations]
            successful_trials = sum(reward == 1.0 for reward in rewards)
            per_task.append(
                {
                    "task_id": task_id,
                    "source": row["source"],
                    "reward": rewards[0] if len(rewards) == 1 else rewards,
                    "success": all(reward == 1.0 for reward in rewards),
                    "any_success": any(reward == 1.0 for reward in rewards),
                    "successful_trials": successful_trials,
                    "trial_success_rate": round(
                        successful_trials / len(rewards), 4
                    ),
                    "num_trials": len(simulations),
                    "elapsed_seconds": round(time.perf_counter() - started, 2),
                }
            )
        except Exception as exc:  # noqa: BLE001 - recorded as a system failure
            failure = {
                "task_id": task_id,
                "exception_type": type(exc).__name__,
                "message": str(exc),
                "traceback": traceback.format_exc(),
                "elapsed_seconds": round(time.perf_counter() - started, 2),
            }
            failures.append(failure)
            write_json(task_dir / "system_failure.json", failure)

    summary = build_summary(
        per_task=per_task,
        failures=failures,
        infrastructure_failures=infrastructure_failures,
        validated=validated,
        run_name=run_name,
        model_run=model_run,
    )
    write_json(output_dir / "eval_summary.json", summary)
    manifest.update(
        {
            "status": (
                "COMPLETED_WITH_FAILURES"
                if failures or infrastructure_failures
                else "COMPLETED"
            ),
            "completed_at": datetime.now(timezone.utc).isoformat(),
            "task_count": len(validated["task_ids"]),
            "completed_tasks": len(per_task),
            "infrastructure_failures": infrastructure_failures,
            "system_failures": failures,
            "summary_sha256": sha256(output_dir / "eval_summary.json"),
        }
    )
    write_json(output_dir / "run_manifest.json", manifest)
    return manifest


def select_smoke_task(validated: dict[str, Any], smoke_task: str | None) -> dict[str, Any]:
    """Restrict the run to one task id while keeping the validated protocol."""
    if not smoke_task:
        return validated
    rows = [row for row in validated["task_rows"] if str(row["task_id"]) == smoke_task]
    if not rows:
        raise ValueError(
            f"unknown --smoke-task {smoke_task!r}; not in the eval task set"
        )
    filtered = dict(validated)
    filtered["task_rows"] = rows
    filtered["task_ids"] = [str(row["task_id"]) for row in rows]
    return filtered


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Run the teacher-SFT benchmark evaluation."
    )
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--output-dir", help="required unless --validate-only")
    parser.add_argument("--run-name", help="model_run name (base | sft); required unless --validate-only")
    parser.add_argument("--smoke-task", help="run a single task id (smoke) instead of the full task set")
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--allow-dirty", action="store_true")
    args = parser.parse_args(argv)

    validated = validate_config(Path(args.config))
    validated = select_smoke_task(validated, args.smoke_task)
    if args.validate_only:
        print(
            json.dumps(
                {
                    "status": "VALIDATED",
                    "config_path": str(validated["config_path"].relative_to(REPO_ROOT)),
                    "config_sha256": validated["config_sha256"],
                    "task_ids": validated["task_ids"],
                    "task_sources": {
                        source: [
                            row["task_id"]
                            for row in validated["task_rows"]
                            if row["source"] == source
                        ]
                        for source in ("train_candidates", "test_clean")
                    },
                    "model_runs": [run["name"] for run in validated["model_runs"]],
                    "smoke_task": args.smoke_task,
                    "num_trials": validated["num_trials"],
                    "seed": validated["seed"],
                    "entity_gate_verified": True,
                    "entity_gate_groups": len(validated["entity_values"]),
                    "external_api_called": False,
                },
                indent=2,
            )
        )
        return

    if not args.output_dir:
        raise SystemExit("--output-dir is required (unless --validate-only)")
    if not args.run_name:
        raise SystemExit("--run-name is required (unless --validate-only)")
    if args.run_name not in validated["model_runs_by_name"]:
        raise ValueError(
            f"unknown --run-name {args.run_name!r}; "
            f"available: {sorted(validated['model_runs_by_name'])}"
        )
    if args.smoke_task:
        print(f"SMOKE_TASK={args.smoke_task}")
    manifest = run(
        validated,
        Path(args.output_dir),
        args.run_name,
        allow_dirty=args.allow_dirty,
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()

