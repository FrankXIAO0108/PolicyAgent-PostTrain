from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any

from src.rl.retail_agentic_env import _ensure_tau2_importable


REPO_ROOT = Path(__file__).resolve().parents[2]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def load_existing(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    return {str(row["task_id"]): row for row in rows}


def save_rows(path: Path, rows: dict[str, dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ordered = [rows[key] for key in sorted(rows, key=int)]
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in ordered),
        encoding="utf-8",
    )


def generate_opening(task: Any, environment: Any, model: str, seed: int) -> dict[str, Any]:
    from tau2.data_model.message import AssistantMessage
    from tau2.runner.build import build_user

    user = build_user(
        "user_simulator",
        environment,
        task,
        llm=model,
        llm_args={"temperature": 0.0},
        solo_mode=False,
    )
    user.set_seed(seed)
    state = user.get_init_state()
    hello = AssistantMessage(
        role="assistant", content="Hi! How can I help you today?", cost=0.0
    )
    opening, _ = user.generate_next_message(hello, state)
    if opening.is_tool_call() or not str(opening.content or "").strip():
        raise RuntimeError(f"Task {task.id} produced an invalid opening utterance")
    return {
        "task_id": str(task.id),
        "task_split": "train",
        "initial_user_message": str(opening.content).strip(),
        "user_seed": seed,
        "user_model": model,
        "temperature": 0.0,
        "usage": opening.usage,
        "cost": opening.cost,
        "hidden_user_scenario_persisted": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--task-split",
        type=Path,
        default=REPO_ROOT / "data" / "retail_agentic_rl_v1" / "task_split.json",
    )
    parser.add_argument("--subset", choices=("rl_train", "rl_validation"), default="rl_train")
    parser.add_argument(
        "--output",
        type=Path,
        default=REPO_ROOT
        / "data"
        / "retail_agentic_rl_v1"
        / "initial_user_messages.jsonl",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=REPO_ROOT
        / "data"
        / "retail_agentic_rl_v1"
        / "initial_user_messages_manifest.json",
    )
    parser.add_argument(
        "--model",
        default=os.environ.get("POLICYAGENT_USER_MODEL", "deepseek/deepseek-chat"),
    )
    parser.add_argument("--seed", type=int, default=20260810)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()

    _ensure_tau2_importable()
    from tau2.registry import registry

    split_path = args.task_split.resolve()
    split = load_json(split_path)
    task_ids = list(split["splits"][args.subset])
    if args.limit is not None:
        task_ids = task_ids[: args.limit]
    tasks = {
        str(task.id): task for task in registry.get_tasks_loader("retail")("train")
    }
    rows = load_existing(args.output.resolve())
    environment_factory = registry.get_env_constructor("retail")
    for index, task_id in enumerate(task_ids):
        expected_seed = args.seed + index
        if task_id in rows:
            existing = rows[task_id]
            bindings = {
                "task_split": "train",
                "user_seed": expected_seed,
                "user_model": args.model,
                "temperature": 0.0,
                "hidden_user_scenario_persisted": False,
            }
            mismatched = {
                key: {"expected": value, "actual": existing.get(key)}
                for key, value in bindings.items()
                if existing.get(key) != value
            }
            if mismatched:
                raise RuntimeError(
                    f"Existing opening for task {task_id} has incompatible bindings: "
                    f"{mismatched}"
                )
            continue
        task = tasks[task_id]
        environment = environment_factory()
        initial_state = task.initial_state
        environment.set_state(
            initialization_data=(
                initial_state.initialization_data if initial_state is not None else None
            ),
            initialization_actions=(
                initial_state.initialization_actions
                if initial_state is not None
                else None
            ),
            message_history=[],
        )
        rows[task_id] = generate_opening(
            task,
            environment,
            args.model,
            expected_seed,
        )
        save_rows(args.output.resolve(), rows)

    selected = {task_id: rows[task_id] for task_id in task_ids if task_id in rows}
    missing = sorted(set(task_ids) - set(selected), key=int)
    if missing:
        raise RuntimeError(f"Opening generation incomplete: {missing}")
    total_cost = sum(float(row.get("cost") or 0.0) for row in selected.values())
    total_usage: dict[str, int] = {}
    for row in selected.values():
        for key, value in (row.get("usage") or {}).items():
            if isinstance(value, int):
                total_usage[key] = total_usage.get(key, 0) + value
    output_path = args.output.resolve()
    manifest = {
        "schema_version": "retail-agentic-rl-user-openings-v1",
        "scope": "ISOLATED_AGENTIC_RL_ENGINEERING",
        "subset": args.subset,
        "rows": len(selected),
        "task_split_path": str(split_path),
        "task_split_sha256": sha256(split_path),
        "output_path": str(output_path),
        "output_sha256": sha256(output_path),
        "user_model": args.model,
        "temperature": 0.0,
        "total_usage": total_usage,
        "total_cost": total_cost,
        "hidden_user_scenarios_persisted": False,
        "training_labels_included": False,
    }
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
