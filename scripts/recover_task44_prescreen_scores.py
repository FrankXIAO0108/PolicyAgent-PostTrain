"""Recover deterministic scores from frozen Task44 traces. Never calls a model."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys


def load(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def require_same(actual, expected, label):
    if actual != expected:
        raise ValueError(f"Frozen evidence mismatch: {label}")


def content_value(content):
    try:
        return json.loads(content)
    except (json.JSONDecodeError, TypeError):
        return content


def verify_pair(raw, evidence):
    from src.rl.retail_agentic_env import _canonical_sha256, transport_invalid_reasons

    for key in ("task_id", "user_seed", "completion", "evidence_sha256"):
        require_same(raw[key], evidence[key], key)
    require_same(str(raw["task_id"]), "44", "task")
    require_same(
        _canonical_sha256(
            {k: v for k, v in evidence.items() if k != "evidence_sha256"}
        ),
        evidence["evidence_sha256"],
        "evidence content hash",
    )
    for name in ("initial", "final"):
        require_same(
            _canonical_sha256(evidence[f"{name}_state"]),
            evidence["state_hashes"][f"{name}_sha256"],
            f"{name} state hash",
        )
    if transport_invalid_reasons(raw["completion"]):
        raise ValueError("Transport-invalid trajectory cannot be recovered as eligible")


def reconstruct(raw, evidence, config, task, constructor):
    from pydantic import TypeAdapter
    from tau2.data_model.message import Message
    from tau2.evaluator.evaluator_env import EnvironmentEvaluator
    from tau2.evaluator.evaluator_communicate import CommunicateEvaluator
    from src.rl.retail_agentic_env import (
        _canonical_sha256,
        _environment_state,
        _tool_trace,
        one_to_one_action_progress,
        tiered_terminal_process_reward,
    )

    verify_pair(raw, evidence)
    messages = [TypeAdapter(Message).validate_python(m) for m in raw["messages"]]
    require_same(_tool_trace(messages), evidence["tool_trace"], "tool trace")
    initial = task.initial_state
    if initial and initial.message_history:
        raise ValueError(
            "This recovery supports only the original empty-history runner"
        )
    environment = constructor()
    environment.set_state(
        initialization_data=initial.initialization_data if initial else None,
        initialization_actions=initial.initialization_actions if initial else None,
        message_history=[],
    )
    require_same(
        _environment_state(environment), evidence["initial_state"], "initial database"
    )
    checked = []
    # Replay reads as well as writes; upstream evaluator only checks write returns.
    pending = []
    for message in messages:
        if message.role in {"assistant", "user"} and getattr(
            message, "tool_calls", None
        ):
            if pending:
                raise ValueError("New calls before pending tool results")
            pending.extend(message.tool_calls)
        elif message.role == "tool":
            if not pending:
                raise ValueError("Orphan tool response")
            call = pending.pop(0)
            require_same(call.id, message.id, "tool response id/order")
            actual = environment.get_response(call)
            require_same(bool(actual.error), bool(message.error), "tool error status")
            require_same(
                content_value(actual.content),
                content_value(message.content),
                "tool content",
            )
            checked.append({"id": call.id, "name": call.name, "result_matches": True})
        elif pending:
            raise ValueError("Missing tool result before conversation continuation")
    if pending:
        raise ValueError("Unresolved tool call")
    require_same(
        _environment_state(environment), evidence["final_state"], "final database"
    )
    env_info = EnvironmentEvaluator.calculate_reward(
        environment_constructor=constructor,
        task=task,
        full_trajectory=messages,
        solo_mode=False,
        env_kwargs={},
    )
    comm_info = CommunicateEvaluator.calculate_reward(
        task=task, full_trajectory=messages
    )
    progress = one_to_one_action_progress(task, messages)
    spec = config["reward"]["staged_reward_spec"]
    base = tiered_terminal_process_reward(
        task_id="44",
        messages=messages,
        action_progress=progress,
        environment_payload=env_info.model_dump(mode="json"),
        communication_payload=comm_info.model_dump(mode="json"),
        environment_state_reward=float(env_info.reward),
        user_stopped=raw["completion"]["user_stopped"],
        completion=raw["completion"],
        staged_reward_spec=spec,
        skip_semantics=True,
    )
    return {
        "recovered_offline_not_original_score_receipt": True,
        "tool_checks": checked,
        "initial_state_matches": True,
        "final_state_matches": True,
        "replayed_final_state_sha256": _canonical_sha256(
            _environment_state(environment)
        ),
        "environment_evaluator": env_info.model_dump(mode="json"),
        "communication_evaluator": comm_info.model_dump(mode="json"),
        "action_progress": progress,
        "base": base,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError("Refusing to overwrite recovery")

    # Enforced before tau2/LLM imports: even accidental network access must fail.
    def offline_guard(event, values):
        if event in {"socket.connect", "socket.getaddrinfo", "socket.sendto"}:
            raise RuntimeError("Network disabled in offline recovery")

    sys.addaudithook(offline_guard)
    os.environ["LITELLM_LOCAL_MODEL_COST_MAP"] = "True"
    from loguru import logger

    logger.remove()
    from src.rl.retail_agentic_env import _ensure_tau2_importable

    _ensure_tau2_importable()
    from tau2.domains.retail.environment import get_tasks, get_environment
    from src.evaluation.task44_hybrid_reward import score_candidate_response
    from src.evaluation.semantic_shadow_judge import build_packet

    cfg = load(args.run_dir / "config.json")
    fail = load(args.run_dir / "failure_manifest.json")
    runtime = load(args.run_dir / "environment.json")
    upstream = Path(os.environ["POLICYAGENT_TAU2_ROOT"])
    upstream_head = subprocess.check_output(
        [
            "git",
            "-c",
            f"safe.directory={upstream.as_posix()}",
            "-C",
            str(upstream),
            "rev-parse",
            "HEAD",
        ],
        text=True,
    ).strip()
    require_same(upstream_head, cfg["upstream"]["commit"], "upstream commit")
    raw_path, ev_path = (
        args.run_dir / n for n in ("raw_rollouts.jsonl", "rollout_evidence.jsonl")
    )
    for p in (raw_path, ev_path):
        require_same(
            sha(p), fail["partial_artifacts"][p.name]["sha256"].lower(), p.name
        )
    # Exact project-side base reward code must match the deployed run.
    for name in (
        "src/rl/retail_agentic_env.py",
        "src/evaluation/staged_reward_shadow.py",
        "src/evaluation/task44_reward_evidence.py",
    ):
        require_same(sha(name), runtime["hybrid_reward_sources"][name].lower(), name)
    raw = [json.loads(s) for s in raw_path.read_text(encoding="utf-8").splitlines()]
    evidence = [json.loads(s) for s in ev_path.read_text(encoding="utf-8").splitlines()]
    require_same((len(raw), len(evidence)), (4, 4), "frozen n4 batch")
    task = next(t for t in get_tasks("train") if t.id == "44")
    caches = [
        (load(p.with_name("request.json")), load(p), p)
        for p in sorted((args.run_dir / "semantic_cache").glob("*/response.json"))
    ]
    paths = [
        raw_path,
        ev_path,
        args.run_dir / "config.json",
        args.run_dir / "environment.json",
        args.run_dir / "failure_manifest.json",
        Path(__file__),
    ]
    paths += [Path(n) for n in runtime["hybrid_reward_sources"]]
    # Source fingerprints include actual upstream files, not just Git HEAD.
    paths += list((upstream / "src/tau2").rglob("*.py"))
    paths += list((upstream / "data/tau2/domains/retail").glob("*.*"))
    for _, _, p in caches:
        paths += [p, p.with_name("request.json")]
    source_hashes = {str(p): sha(p) for p in paths}
    args.output_dir.mkdir(parents=True)
    results = []
    for i, (row, ev) in enumerate(zip(raw, evidence, strict=True)):
        result = reconstruct(row, ev, cfg, task, get_environment)
        result.update(
            row_index_zero_based=i,
            rule_only_reward=result["base"]["staged_reward"],
            hybrid_reward=None,
            semantic_status="NO_CACHED_RESPONSE",
        )
        for request, response, _ in caches:
            packet = request["packet"]
            if (
                build_packet(row, packet["policy"], row=0)["messages"]
                != packet["messages"]
            ):
                continue
            choice = response["choices"][0]
            require_same(choice["finish_reason"], "stop", "cached extractor completion")
            hybrid = score_candidate_response(
                result["base"],
                row,
                cfg["reward"]["staged_reward_spec"],
                packet,
                choice["message"]["content"],
            )
            hybrid.update(
                used_as_training_reward=False,
                scope="OFFLINE_RECOVERED_BASE_CACHED_SEMANTICS",
            )
            result.update(
                hybrid_reward=hybrid["offline_reward"],
                semantic_status="OLD_RESPONSE_REINTERPRETED",
                semantic_result=hybrid,
            )
        # These are newly derived recovery inputs, not retroactive edits to the old run.
        result["scoring_input"] = {
            "base": result["base"],
            "raw": row,
            "spec": copy.deepcopy(cfg["reward"]["staged_reward_spec"]),
        }
        with (args.output_dir / f"row_{i}.json").open("x", encoding="utf-8") as out:
            json.dump(result, out, ensure_ascii=False, indent=2, allow_nan=False)
        results.append(
            {
                k: result[k]
                for k in (
                    "row_index_zero_based",
                    "rule_only_reward",
                    "hybrid_reward",
                    "semantic_status",
                    "initial_state_matches",
                    "final_state_matches",
                )
            }
        )
    require_same(
        {str(p): sha(p) for p in paths},
        source_hashes,
        "inputs unchanged during recovery",
    )
    manifest = {
        "status": "BASE_RECOVERED_SEMANTICS_INCOMPLETE",
        "command": [sys.executable, *sys.argv],
        "project_head": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip(),
        "upstream_head": upstream_head,
        "source_sha256": source_hashes,
        "external_api_called": False,
        "network_disabled": True,
        "gpu_used": False,
        "training_eligible": False,
        "group_hybrid_reward_std": None,
        "results": results,
    }
    with (args.output_dir / "recovery_manifest.json").open(
        "x", encoding="utf-8"
    ) as out:
        json.dump(manifest, out, ensure_ascii=False, indent=2, allow_nan=False)
    print(json.dumps(results, ensure_ascii=False))


if __name__ == "__main__":
    main()
