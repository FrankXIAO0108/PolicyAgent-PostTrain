from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .db_diff import analyze_db_diff
from .failure_attributor import attribute_failure
from .nl_checker import check_recorded_nl_assertions
from .replay_evaluator import replay_results_artifact
from .report_generator import write_evaluation_report
from .taxonomy import build_three_layer_taxonomy


DEFAULT_EXPERIMENT = Path(
    r"D:\PolicyAgent-PostTrain\experiments"
    r"\20260722_110504_retail_baseline20_trial1_deepseek"
)
DEFAULT_OUTPUT = Path(r"D:\PolicyAgent-PostTrain\reports\evaluation")
VERSION = "v7.1.0"


def _git_commit(repo: Path) -> str | None:
    command = [
        "git",
        "-c",
        f"safe.directory={repo.as_posix()}",
        "-C",
        str(repo),
        "rev-parse",
        "HEAD",
    ]
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    return result.stdout.strip() or None


def _git_dirty(repo: Path) -> bool | None:
    command = [
        "git",
        "-c",
        f"safe.directory={repo.as_posix()}",
        "-C",
        str(repo),
        "status",
        "--porcelain",
    ]
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    return bool(result.stdout.strip()) if result.returncode == 0 else None


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _source_hash() -> str:
    digest = hashlib.sha256()
    for path in sorted(Path(__file__).resolve().parent.glob("*.py")):
        digest.update(path.name.encode("utf-8"))
        digest.update(path.read_bytes())
    return digest.hexdigest()


def _recorded_component(replay: Any, name: str) -> float | None:
    reward_info = replay.simulation.reward_info
    if reward_info is None or reward_info.reward_breakdown is None:
        return None
    for key, value in reward_info.reward_breakdown.items():
        if getattr(key, "value", key) == name:
            return float(value)
    return None


def _evaluate_artifact(path: Path, tau2_root: Path) -> dict[str, Any]:
    replay = replay_results_artifact(path, tau2_root=tau2_root)
    state_diff = analyze_db_diff(
        replay.initial_state,
        replay.agent_state,
        replay.gold_state,
    )
    nl_result = check_recorded_nl_assertions(replay.task, replay.simulation)
    attribution = attribute_failure(replay, state_diff, nl_result)
    reward_info = replay.simulation.reward_info
    recorded_reward = float(reward_info.reward) if reward_info else None
    recorded_db = _recorded_component(replay, "DB")
    reconstructed_reward = (
        float(replay.db_match and nl_result.nl_match)
        if nl_result.nl_match is not None
        else None
    )
    signal_consistent = (
        recorded_db is None or bool(recorded_db == 1.0) == replay.db_match
    )

    official_signal = {
        "recorded_reward": recorded_reward,
        "recorded_db_reward": recorded_db,
        "recorded_nl_reward": _recorded_component(replay, "NL_ASSERTION"),
        "termination_reason": replay.simulation.termination_reason.value,
        "db_match": replay.db_match,
        "nl_match": nl_result.nl_match,
        "reconstructed_reward": reconstructed_reward,
        "recorded_vs_reconstructed_db_consistent": signal_consistent,
    }
    taxonomy = build_three_layer_taxonomy(
        official_signal=official_signal,
        detailed_root_causes=attribution["root_causes"],
        state_diff=state_diff.to_dict(),
    )

    return {
        "task_id": replay.task_id,
        "task_success": recorded_reward == 1.0,
        "official_signal": official_signal,
        "taxonomy": taxonomy,
        "replay": replay.to_dict(),
        "state_diff": state_diff.to_dict(),
        "nl_assertions": nl_result.to_dict(),
        **attribution,
        "artifact": str(path),
        "artifact_sha256": _sha256(path),
    }


def evaluate_experiment(
    experiment_dir: str | Path = DEFAULT_EXPERIMENT,
    *,
    tau2_root: str | Path = r"D:\tau2-bench",
    output_dir: str | Path = DEFAULT_OUTPUT,
) -> dict[str, Any]:
    experiment = Path(experiment_dir).expanduser().resolve()
    upstream = Path(tau2_root).expanduser().resolve()
    output = Path(output_dir).expanduser().resolve()
    artifacts = sorted(
        experiment.glob("task_*/returned_results.json"),
        key=lambda path: int(path.parent.name.split("_")[-1]),
    )
    if not artifacts:
        raise FileNotFoundError(
            f"No task_*/returned_results.json artifacts under {experiment}"
        )

    evaluation_started = time.perf_counter()
    tasks = [_evaluate_artifact(path, upstream) for path in artifacts]
    evaluation_elapsed_seconds = time.perf_counter() - evaluation_started
    failures = [task for task in tasks if not task["task_success"]]
    report = {
        "schema_version": VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "experiment": str(experiment),
        "provenance": {
            "project_commit": _git_commit(Path(__file__).resolve().parents[2]),
            "project_worktree_dirty": _git_dirty(
                Path(__file__).resolve().parents[2]
            ),
            "tau2_commit": _git_commit(upstream),
            "nl_mode": "reuse_frozen_official_results",
            "evaluator_source_sha256": _source_hash(),
            "run_manifest_sha256": (
                _sha256(experiment / "run_manifest.json")
                if (experiment / "run_manifest.json").exists()
                else None
            ),
        },
        "summary": {
            "task_count": len(tasks),
            "success_count": len(tasks) - len(failures),
            "failure_count": len(failures),
            "replay_inconsistency_count": sum(
                not task["official_signal"][
                    "recorded_vs_reconstructed_db_consistent"
                ]
                for task in tasks
            ),
            "failure_task_ids": [task["task_id"] for task in failures],
            "evaluation_elapsed_seconds": round(
                evaluation_elapsed_seconds, 6
            ),
        },
        "tasks": tasks,
    }

    write_evaluation_report(report, output)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run Tau2-aligned Hybrid Evaluation v7."
    )
    parser.add_argument("--experiment", type=Path, default=DEFAULT_EXPERIMENT)
    parser.add_argument("--tau2-root", type=Path, default=Path(r"D:\tau2-bench"))
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    report = evaluate_experiment(
        args.experiment,
        tau2_root=args.tau2_root,
        output_dir=args.output,
    )
    print(json.dumps(report["summary"], ensure_ascii=False, indent=2))
    print(f"Saved: {args.output / 'final_report.json'}")
    print(f"Saved: {args.output / 'failure_analysis.md'}")


if __name__ == "__main__":
    main()
