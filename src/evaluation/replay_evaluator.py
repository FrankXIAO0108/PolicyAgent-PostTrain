from __future__ import annotations

import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(slots=True)
class ReplayResult:
    """Outcome of replaying one Tau2 simulation and its gold actions."""

    task_id: str
    domain: str
    db_match: bool
    gold_hash: str | None
    agent_hash: str | None
    gold_user_hash: str | None = None
    agent_user_hash: str | None = None
    replay_errors: list[str] = field(default_factory=list)
    initial_state: dict[str, Any] = field(default_factory=dict, repr=False)
    agent_state: dict[str, Any] = field(default_factory=dict, repr=False)
    gold_state: dict[str, Any] = field(default_factory=dict, repr=False)
    task: Any = field(default=None, repr=False)
    simulation: Any = field(default=None, repr=False)
    raw_results: dict[str, Any] = field(default_factory=dict, repr=False)

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "domain": self.domain,
            "db_match": self.db_match,
            "gold_hash": self.gold_hash,
            "agent_hash": self.agent_hash,
            "gold_user_hash": self.gold_user_hash,
            "agent_user_hash": self.agent_user_hash,
            "replay_errors": self.replay_errors,
        }


class Tau2Runtime:
    """Loads the pinned upstream Tau2 checkout without copying its evaluator."""

    def __init__(self, tau2_root: str | Path = r"D:\tau2-bench") -> None:
        self.root = Path(tau2_root).expanduser().resolve()
        source = self.root / "src"
        if not source.is_dir():
            raise FileNotFoundError(f"Tau2 source directory not found: {source}")
        source_text = str(source)
        if source_text not in sys.path:
            sys.path.insert(0, source_text)

        # Imported lazily so this project can still expose non-replay utilities
        # when the upstream checkout is not installed.
        from tau2.data_model.simulation import Results
        from tau2.registry import registry

        self.Results = Results
        self.registry = registry

    def load_results(self, path: str | Path) -> tuple[Any, dict[str, Any]]:
        artifact_path = Path(path).expanduser().resolve()
        with artifact_path.open("r", encoding="utf-8-sig") as handle:
            raw = json.load(handle)
        return self.Results.model_validate(raw), raw


def _initialization(task: Any) -> tuple[Any, Any, list[Any]]:
    initial = task.initial_state
    if initial is None:
        return None, None, []
    return (
        initial.initialization_data,
        initial.initialization_actions,
        list(initial.message_history or []),
    )


def _environment_state(environment: Any) -> dict[str, Any]:
    state: dict[str, Any] = {}
    if environment.tools is not None and environment.tools.db is not None:
        state["agent"] = environment.tools.db.model_dump(mode="json")
    if environment.user_tools is not None and environment.user_tools.db is not None:
        state["user"] = environment.user_tools.db.model_dump(mode="json")
    return state


def replay_task_simulation(
    task: Any,
    simulation: Any,
    *,
    constructor: Any,
    domain: str,
    raw_results: dict[str, Any] | None = None,
) -> ReplayResult:
    """Reconstruct initial, predicted, and reference states for one simulation."""

    initialization_data, initialization_actions, initial_messages = _initialization(
        task
    )
    state_args = {
        "initialization_data": initialization_data,
        "initialization_actions": initialization_actions,
    }
    replay_errors: list[str] = []

    initial_environment = constructor(solo_mode=False)
    initial_environment.set_state(message_history=initial_messages, **state_args)

    predicted_environment = constructor(solo_mode=False)
    try:
        predicted_environment.set_state(
            message_history=list(simulation.messages or []), **state_args
        )
    except Exception as exc:
        replay_errors.append(f"predicted_replay: {type(exc).__name__}: {exc}")

    gold_environment = constructor(solo_mode=False)
    gold_environment.set_state(message_history=initial_messages, **state_args)
    for action in task.evaluation_criteria.actions or []:
        try:
            gold_environment.make_tool_call(
                tool_name=action.name,
                requestor=action.requestor,
                **action.arguments,
            )
        except Exception as exc:
            replay_errors.append(
                f"gold_action:{action.action_id}:{action.name}: "
                f"{type(exc).__name__}: {exc}"
            )

    agent_hash = predicted_environment.get_db_hash()
    gold_hash = gold_environment.get_db_hash()
    agent_user_hash = predicted_environment.get_user_db_hash()
    gold_user_hash = gold_environment.get_user_db_hash()
    return ReplayResult(
        task_id=str(task.id),
        domain=domain,
        db_match=agent_hash == gold_hash and agent_user_hash == gold_user_hash,
        gold_hash=gold_hash,
        agent_hash=agent_hash,
        gold_user_hash=gold_user_hash,
        agent_user_hash=agent_user_hash,
        replay_errors=replay_errors,
        initial_state=_environment_state(initial_environment),
        agent_state=_environment_state(predicted_environment),
        gold_state=_environment_state(gold_environment),
        task=task,
        simulation=simulation,
        raw_results=raw_results or {},
    )


def replay_results_artifact(
    results_path: str | Path,
    *,
    tau2_root: str | Path = r"D:\tau2-bench",
) -> ReplayResult:
    """Reconstruct predicted and gold state using Tau2's own Environment APIs."""

    runtime = Tau2Runtime(tau2_root)
    results, raw_results = runtime.load_results(results_path)
    if len(results.tasks) != 1 or len(results.simulations) != 1:
        raise ValueError(
            "Expected one task and one simulation in returned_results.json; "
            f"got {len(results.tasks)} tasks and {len(results.simulations)} simulations"
        )

    task = results.tasks[0]
    simulation = results.simulations[0]
    domain = results.info.environment_info.domain_name
    constructor = runtime.registry.get_env_constructor(domain)
    return replay_task_simulation(
        task,
        simulation,
        constructor=constructor,
        domain=domain,
        raw_results=raw_results,
    )
