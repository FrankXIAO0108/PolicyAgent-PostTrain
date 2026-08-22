from __future__ import annotations

import argparse
import hashlib
import json
import statistics
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from src.verifiers.intent_state import is_write_tool
from src.verifiers.policy_grounding_v2 import verify_trajectory
from src.verifiers.schemas import MessageEvent, ToolCall, Verdict


SCHEMA_VERSION = "teacher-eval-cards-1.1.0"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_call(name: str, arguments: dict[str, Any]) -> str:
    return f"{name}:{json.dumps(arguments, ensure_ascii=False, sort_keys=True)}"


def _max_consecutive_same_name(names: list[str]) -> int:
    maximum = 0
    current = 0
    previous: str | None = None
    for name in names:
        current = current + 1 if name == previous else 1
        maximum = max(maximum, current)
        previous = name
    return maximum


def _message_events(messages: list[dict[str, Any]], source: Path) -> list[MessageEvent]:
    events: list[MessageEvent] = []
    for index, message in enumerate(messages):
        calls = [
            ToolCall(
                id=str(call.get("id", "")),
                name=str(call.get("name", "")),
                arguments=dict(call.get("arguments") or {}),
            )
            for call in message.get("tool_calls") or []
        ]
        events.append(
            MessageEvent(
                index=index,
                role=str(message.get("role", "")),
                content=str(message.get("content") or ""),
                tool_calls=calls,
                tool_call_id=(
                    str(message.get("id"))
                    if message.get("role") == "tool" and message.get("id")
                    else None
                ),
                tool_error=bool(message.get("error", False)),
                source=str(source),
            )
        )
    return events


def _single_simulation(payload: dict[str, Any], source: Path) -> dict[str, Any]:
    simulations = payload.get("simulations") or []
    if len(simulations) != 1:
        raise ValueError(
            f"Expected exactly one simulation in {source}, found {len(simulations)}"
        )
    return simulations[0]


def analyze_task(
    result_path: str | Path,
    *,
    run_name: str,
    source_split: str | None = None,
    is_replacement: bool = False,
) -> dict[str, Any]:
    path = Path(result_path).resolve()
    payload = json.loads(path.read_text(encoding="utf-8"))
    simulation = _single_simulation(payload, path)
    return _analyze_simulation(
        simulation,
        path,
        run_name=run_name,
        source_split=source_split,
        is_replacement=is_replacement,
        trial_index=0,
    )


def _analyze_simulation(
    simulation: dict[str, Any],
    path: Path,
    *,
    run_name: str,
    source_split: str | None,
    is_replacement: bool,
    trial_index: int,
) -> dict[str, Any]:
    task_id = str(simulation.get("task_id"))
    reward_info = simulation.get("reward_info")
    messages = simulation.get("messages") or []
    events = _message_events(messages, path)

    observed_usage: dict[str, dict[str, int]] = {}
    for role in ("assistant", "user"):
        role_messages = [message for message in messages if message.get("role") == role]
        usage_rows = [
            message.get("usage") for message in role_messages if message.get("usage")
        ]
        prompt_tokens = sum(int(row.get("prompt_tokens") or 0) for row in usage_rows)
        completion_tokens = sum(
            int(row.get("completion_tokens") or 0) for row in usage_rows
        )
        observed_usage[role] = {
            "messages_with_usage": len(usage_rows),
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        }

    calls = [call for event in events for call in event.tool_calls]
    call_names = [call.name for call in calls]
    canonical = [_canonical_call(call.name, call.arguments) for call in calls]
    counts = Counter(canonical)
    name_counts = Counter(call_names)
    duplicate_count = sum(count - 1 for count in counts.values())
    consecutive_duplicates = sum(
        left == right for left, right in zip(canonical, canonical[1:])
    )
    read_calls = sum(not is_write_tool(call.name) for call in calls)
    write_calls = len(calls) - read_calls
    tool_errors = sum(event.role == "tool" and event.tool_error for event in events)

    infrastructure_valid = reward_info is not None
    reward = reward_info.get("reward") if infrastructure_valid else None
    raw_action_checks = reward_info.get("action_checks") if reward_info else None
    action_checks = raw_action_checks or []
    action_note = (
        ((reward_info.get("info") or {}).get("action") or {}).get("note")
        if reward_info
        else None
    )
    action_evaluation_status = (
        "evaluated"
        if isinstance(raw_action_checks, list)
        else (
            "no_reference_actions"
            if action_note == "No actions to evaluate"
            else "unavailable"
        )
    )
    matched_actions = sum(bool(check.get("action_match")) for check in action_checks)
    reference_actions = len(action_checks)
    nl_checks = (reward_info.get("nl_assertions") or []) if reward_info else []
    communicate_checks = (
        (reward_info.get("communicate_checks") or []) if reward_info else []
    )
    db_check = reward_info.get("db_check") if reward_info else None

    benchmark_verdict = (
        Verdict.PASS
        if reward == 1
        else Verdict.FAIL if reward == 0 else Verdict.NOT_EVALUATED
    )
    policy = verify_trajectory(
        events,
        task_id=task_id,
        benchmark_verdict=benchmark_verdict,
    ).to_dict()

    return {
        "task_id": task_id,
        "trial_index": trial_index,
        "run_name": run_name,
        "source_split": source_split,
        "artifact": {
            "returned_results": str(path),
            "sha256": _sha256(path),
            "is_replacement": is_replacement,
        },
        "infrastructure": {
            "valid": infrastructure_valid,
            "termination_reason": simulation.get("termination_reason"),
            "duration_seconds": simulation.get("duration"),
        },
        "observed_token_usage": {
            "agent_assistant_messages": observed_usage["assistant"],
            "user_simulator_messages": observed_usage["user"],
            "judge_usage_recorded": False,
            "cost_fields": {
                "agent_cost": simulation.get("agent_cost"),
                "user_cost": simulation.get("user_cost"),
                "billing_reliable": False,
            },
        },
        "outcome": {
            "reward": reward,
            "success": reward == 1,
            "db_match": db_check.get("db_match") if db_check else None,
            "nl_assertions_met": sum(bool(row.get("met")) for row in nl_checks),
            "nl_assertions_total": len(nl_checks),
            "communicate_checks_met": sum(
                bool(row.get("met")) for row in communicate_checks
            ),
            "communicate_checks_total": len(communicate_checks),
        },
        "tool_use": {
            "total_calls": len(calls),
            "read_calls": read_calls,
            "write_calls": write_calls,
            "unique_exact_calls": len(counts),
            "repeated_exact_calls": duplicate_count,
            "repeated_exact_call_rate": (
                duplicate_count / len(calls) if calls else None
            ),
            "consecutive_exact_repeats": consecutive_duplicates,
            "dominant_tool_name": (
                name_counts.most_common(1)[0][0] if name_counts else None
            ),
            "dominant_tool_call_count": (
                name_counts.most_common(1)[0][1] if name_counts else 0
            ),
            "max_consecutive_same_tool_name": _max_consecutive_same_name(call_names),
            "tool_error_results": tool_errors,
            "reference_action_evaluation_status": action_evaluation_status,
            "reference_action_count": reference_actions,
            "matched_reference_actions": matched_actions,
            "reference_action_recall": (
                matched_actions / reference_actions if reference_actions else None
            ),
            "reference_action_density": (
                matched_actions / len(calls) if calls else None
            ),
            "excess_call_proxy": max(len(calls) - reference_actions, 0),
        },
        "policy_diagnostic": {
            "status": "PROVISIONAL_DIAGNOSTIC",
            "verifier_version": policy["metrics"].get("verifier_version"),
            "verdict": policy["verdict"],
            "dimensions": policy["dimensions"],
            "major_finding_count": sum(
                finding["severity"] == "MAJOR" for finding in policy["findings"]
            ),
            "minor_finding_count": sum(
                finding["severity"] == "MINOR" for finding in policy["findings"]
            ),
            "finding_codes": [finding["code"] for finding in policy["findings"]],
        },
        "dimension_card": {
            "task_outcome": {
                "value": (
                    "PASS"
                    if reward == 1
                    else "FAIL" if reward == 0 else "NOT_EVALUATED"
                ),
                "evidence": "Tau2 reward",
                "authority": "BENCHMARK_OUTCOME",
            },
            "intent_grounding": {
                "value": policy["dimensions"].get("latest_intent", "NOT_EVALUATED"),
                "evidence": "Policy Grounding V2.2 latest_intent dimension",
                "authority": "PROVISIONAL_DIAGNOSTIC",
            },
            "reference_action_coverage": {
                "value": (
                    matched_actions / reference_actions
                    if reference_actions and action_evaluation_status == "evaluated"
                    else None
                ),
                "evidence": "Tau2 action_checks",
                "authority": (
                    "BENCHMARK_DIAGNOSTIC"
                    if action_evaluation_status != "unavailable"
                    else "NOT_EVALUATED"
                ),
            },
            "final_state": {
                "value": db_check.get("db_match") if db_check else None,
                "evidence": "Tau2 db_check",
                "authority": "BENCHMARK_OUTCOME" if db_check else "NOT_EVALUATED",
            },
            "final_response": {
                "value": (
                    sum(bool(row.get("met")) for row in nl_checks) / len(nl_checks)
                    if nl_checks
                    else None
                ),
                "evidence": "Tau2 nl_assertions",
                "authority": "BENCHMARK_OUTCOME" if nl_checks else "NOT_EVALUATED",
            },
            "policy_compliance": {
                "value": policy["dimensions"].get("policy_compliance", "NOT_EVALUATED"),
                "evidence": "Policy Grounding V2.2",
                "authority": "PROVISIONAL_DIAGNOSTIC",
            },
            "tool_efficiency": {
                "value": None,
                "evidence": "Call count, repeat, fan-out and error diagnostics",
                "authority": "DIAGNOSTIC_NO_COMPOSITE_SCORE",
            },
        },
        "validity_notes": [
            "reference_action_recall measures coverage of Tau2 reference actions; it is not tool-call precision.",
            "reference_action_density and excess_call_proxy are diagnostics because reference actions may omit legitimate reads and may not be minimal.",
            "exact repeated calls are loop/redundancy candidates, not automatically errors when state changed between calls.",
            "policy_diagnostic uses unadjudicated verifier V2.2 and is not a human-gold policy score.",
            "intent_grounding is a verifier proxy rather than an independently labeled intent-accuracy metric.",
            "message token usage covers agent and user-simulator messages only; NL-judge usage is not recorded in returned_results.",
            "serialized agent_cost/user_cost are not treated as billing evidence.",
        ],
    }


def _task_result_paths(run_dir: Path) -> dict[str, Path]:
    result: dict[str, Path] = {}
    for path in sorted(run_dir.rglob("returned_results.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        simulations = payload.get("simulations") or []
        if not simulations:
            raise ValueError(f"No simulations in {path}")
        task_ids = {str(simulation.get("task_id")) for simulation in simulations}
        if len(task_ids) != 1:
            raise ValueError(f"Multiple task IDs in {path}: {sorted(task_ids)}")
        task_id = task_ids.pop()
        if task_id in result:
            raise ValueError(f"Duplicate task {task_id} under {run_dir}")
        result[task_id] = path
    return result


def _source_splits(run_dir: Path) -> dict[str, str]:
    summary = run_dir / "eval_summary.json"
    if not summary.exists():
        return {}
    payload = json.loads(summary.read_text(encoding="utf-8"))
    return {
        str(row["task_id"]): str(row.get("source"))
        for row in payload.get("per_task", [])
    }


def load_run_cards(
    run_dir: str | Path,
    *,
    run_name: str,
    replacements: dict[str, str | Path] | None = None,
) -> list[dict[str, Any]]:
    root = Path(run_dir).resolve()
    paths = _task_result_paths(root)
    splits = _source_splits(root)
    replacement_ids: set[str] = set()
    for task_id, replacement_dir in (replacements or {}).items():
        replacement_root = Path(replacement_dir).resolve()
        replacement_paths = _task_result_paths(replacement_root)
        if str(task_id) not in replacement_paths:
            raise ValueError(
                f"Replacement directory {replacement_root} has no task {task_id}"
            )
        paths[str(task_id)] = replacement_paths[str(task_id)]
        replacement_ids.add(str(task_id))
        splits.update(_source_splits(replacement_root))
    cards: list[dict[str, Any]] = []
    for task_id in sorted(paths, key=int):
        path = paths[task_id]
        payload = json.loads(path.read_text(encoding="utf-8"))
        simulations = payload.get("simulations") or []
        for trial_index, simulation in enumerate(simulations):
            cards.append(
                _analyze_simulation(
                    simulation,
                    path,
                    run_name=run_name,
                    source_split=splits.get(task_id),
                    is_replacement=task_id in replacement_ids,
                    trial_index=trial_index,
                )
            )
    return cards


def _mean(cards: Iterable[dict[str, Any]], section: str, metric: str) -> float | None:
    values = [
        card[section][metric] for card in cards if card[section].get(metric) is not None
    ]
    return statistics.fmean(values) if values else None


def _summary(cards: list[dict[str, Any]]) -> dict[str, Any]:
    valid = [card for card in cards if card["infrastructure"]["valid"]]
    task_ids = sorted({card["task_id"] for card in cards}, key=int)
    by_task = {
        task_id: [card for card in cards if card["task_id"] == task_id]
        for task_id in task_ids
    }
    action_evaluable = [
        card
        for card in valid
        if card["tool_use"]["reference_action_evaluation_status"] != "unavailable"
    ]
    matched = sum(
        card["tool_use"]["matched_reference_actions"] for card in action_evaluable
    )
    reference = sum(
        card["tool_use"]["reference_action_count"] for card in action_evaluable
    )
    policy_counts = Counter(card["policy_diagnostic"]["verdict"] for card in valid)
    successful_trials = sum(card["outcome"]["success"] for card in valid)
    agent_tokens = sum(
        card.get("observed_token_usage", {})
        .get("agent_assistant_messages", {})
        .get("total_tokens", 0)
        for card in valid
    )
    user_tokens = sum(
        card.get("observed_token_usage", {})
        .get("user_simulator_messages", {})
        .get("total_tokens", 0)
        for card in valid
    )
    return {
        "task_count": len(task_ids),
        "trial_count": len(cards),
        "infrastructure_valid_count": len(valid),
        "success_count": successful_trials,
        "success_rate": (successful_trials / len(valid) if valid else None),
        "success_rate_semantics": "successful infrastructure-valid trials",
        "stable_success_task_count": sum(
            bool(rows)
            and all(card["infrastructure"]["valid"] for card in rows)
            and all(card["outcome"]["success"] for card in rows)
            for rows in by_task.values()
        ),
        "any_success_task_count": sum(
            any(
                card["infrastructure"]["valid"] and card["outcome"]["success"]
                for card in rows
            )
            for rows in by_task.values()
        ),
        "mean_duration_seconds": (
            statistics.fmean(
                card["infrastructure"]["duration_seconds"]
                for card in valid
                if card["infrastructure"].get("duration_seconds") is not None
            )
            if any(
                card["infrastructure"].get("duration_seconds") is not None
                for card in valid
            )
            else None
        ),
        "observed_tokens": {
            "agent_total": agent_tokens,
            "agent_mean_per_trial": agent_tokens / len(valid) if valid else None,
            "user_simulator_total": user_tokens,
            "user_simulator_mean_per_trial": (
                user_tokens / len(valid) if valid else None
            ),
            "judge_total": None,
            "billing_cost": None,
        },
        "mean_tool_calls": _mean(valid, "tool_use", "total_calls"),
        "median_tool_calls": (
            statistics.median(card["tool_use"]["total_calls"] for card in valid)
            if valid
            else None
        ),
        "mean_read_calls": _mean(valid, "tool_use", "read_calls"),
        "mean_write_calls": _mean(valid, "tool_use", "write_calls"),
        "mean_repeated_exact_calls": _mean(valid, "tool_use", "repeated_exact_calls"),
        "mean_consecutive_exact_repeats": _mean(
            valid, "tool_use", "consecutive_exact_repeats"
        ),
        "mean_dominant_tool_call_count": _mean(
            valid, "tool_use", "dominant_tool_call_count"
        ),
        "mean_max_consecutive_same_tool_name": _mean(
            valid, "tool_use", "max_consecutive_same_tool_name"
        ),
        "tool_error_results": sum(
            card["tool_use"]["tool_error_results"] for card in valid
        ),
        "reference_action_micro_recall": matched / reference if reference else None,
        "reference_action_matched": matched,
        "reference_action_total": reference,
        "reference_action_evaluable_task_count": len(action_evaluable),
        "policy_provisional_verdict_counts": dict(sorted(policy_counts.items())),
    }


def _transition(base: dict[str, Any], candidate: dict[str, Any]) -> str:
    base_success = base["outcome"]["success"]
    candidate_success = candidate["outcome"]["success"]
    if base_success and candidate_success:
        return "both_success"
    if not base_success and candidate_success:
        return "improved"
    if base_success and not candidate_success:
        return "regressed"
    return "both_failure"


def _card_key(card: dict[str, Any]) -> tuple[str, int]:
    return card["task_id"], int(card.get("trial_index", 0))


def compare_cards(
    base_cards: list[dict[str, Any]], candidate_cards: list[dict[str, Any]]
) -> dict[str, Any]:
    base_by_id = {_card_key(card): card for card in base_cards}
    candidate_by_id = {_card_key(card): card for card in candidate_cards}
    if len(base_by_id) != len(base_cards) or len(candidate_by_id) != len(
        candidate_cards
    ):
        raise ValueError("Duplicate task/trial card")
    if set(base_by_id) != set(candidate_by_id):
        raise ValueError("Base and candidate task/trial IDs differ")
    pairs = [
        {
            "task_id": key[0],
            "trial_index": key[1],
            "transition": _transition(base_by_id[key], candidate_by_id[key]),
            "base_success": base_by_id[key]["outcome"]["success"],
            "candidate_success": candidate_by_id[key]["outcome"]["success"],
            "base_tool_calls": base_by_id[key]["tool_use"]["total_calls"],
            "candidate_tool_calls": candidate_by_id[key]["tool_use"]["total_calls"],
            "tool_call_delta": (
                candidate_by_id[key]["tool_use"]["total_calls"]
                - base_by_id[key]["tool_use"]["total_calls"]
            ),
            "candidate_is_replacement": candidate_by_id[key]["artifact"][
                "is_replacement"
            ],
        }
        for key in sorted(base_by_id, key=lambda value: (int(value[0]), value[1]))
    ]
    strata: dict[str, Any] = {}
    for transition in ("both_success", "improved", "regressed", "both_failure"):
        rows = [row for row in pairs if row["transition"] == transition]
        strata[transition] = {
            "task_ids": sorted({row["task_id"] for row in rows}, key=int),
            "task_trial_ids": [
                f"{row['task_id']}:{row['trial_index']}" for row in rows
            ],
            "pair_count": len(rows),
            "base_mean_tool_calls": (
                statistics.fmean(row["base_tool_calls"] for row in rows)
                if rows
                else None
            ),
            "candidate_mean_tool_calls": (
                statistics.fmean(row["candidate_tool_calls"] for row in rows)
                if rows
                else None
            ),
            "mean_tool_call_delta": (
                statistics.fmean(row["tool_call_delta"] for row in rows)
                if rows
                else None
            ),
        }
    comparable_action_pairs = [
        (base_by_id[key], candidate_by_id[key])
        for key in sorted(base_by_id, key=lambda value: (int(value[0]), value[1]))
        if base_by_id[key]["tool_use"]["reference_action_evaluation_status"]
        != "unavailable"
        and candidate_by_id[key]["tool_use"]["reference_action_evaluation_status"]
        != "unavailable"
        and base_by_id[key]["tool_use"]["reference_action_count"]
        == candidate_by_id[key]["tool_use"]["reference_action_count"]
    ]
    base_action_matched = sum(
        base["tool_use"]["matched_reference_actions"]
        for base, _ in comparable_action_pairs
    )
    candidate_action_matched = sum(
        candidate["tool_use"]["matched_reference_actions"]
        for _, candidate in comparable_action_pairs
    )
    comparable_action_total = sum(
        base["tool_use"]["reference_action_count"]
        for base, _ in comparable_action_pairs
    )
    top_call_increases = sorted(
        pairs, key=lambda row: row["tool_call_delta"], reverse=True
    )[:5]
    return {
        "base": _summary(base_cards),
        "candidate": _summary(candidate_cards),
        "replacement_task_ids": sorted(
            {row["task_id"] for row in pairs if row["candidate_is_replacement"]},
            key=int,
        ),
        "paired_rows": pairs,
        "strata": strata,
        "paired_reference_action_comparison": {
            "task_ids": [base["task_id"] for base, _ in comparable_action_pairs],
            "trial_count": len(comparable_action_pairs),
            "reference_action_total": comparable_action_total,
            "base_matched": base_action_matched,
            "candidate_matched": candidate_action_matched,
            "base_micro_recall": (
                base_action_matched / comparable_action_total
                if comparable_action_total
                else None
            ),
            "candidate_micro_recall": (
                candidate_action_matched / comparable_action_total
                if comparable_action_total
                else None
            ),
            "excluded_task_ids": sorted(
                {
                    task_id
                    for task_id, trial_index in set(base_by_id)
                    if (task_id, trial_index)
                    not in {_card_key(base) for base, _ in comparable_action_pairs}
                },
                key=int,
            ),
        },
        "top_tool_call_increases": top_call_increases,
        "validity_notes": [
            "Lower tool-call count is not automatically better; early failure can reduce calls.",
            "Both-success is the cleanest efficiency slice, while improved/regressed strata expose outcome-efficiency trade-offs.",
            "Policy verdict counts are provisional diagnostics and are not included in the benchmark success rate.",
        ],
    }


def _fmt(value: float | None, *, percent: bool = False) -> str:
    if value is None:
        return "N/A"
    return f"{value:.1%}" if percent else f"{value:.2f}"


def render_markdown(report: dict[str, Any]) -> str:
    comparison = report["comparison"]
    base = comparison["base"]
    candidate = comparison["candidate"]
    lines = [
        "# Base vs SFT 多维过程评测",
        "",
        "## 结论摘要",
        "",
        "| 指标 | Base | SFT |",
        "|---|---:|---:|",
        f"| Trial 成功率 | {base['success_count']}/{base['infrastructure_valid_count']} = {_fmt(base['success_rate'], percent=True)} | {candidate['success_count']}/{candidate['infrastructure_valid_count']} = {_fmt(candidate['success_rate'], percent=True)} |",
        f"| 全 trial 稳定成功任务 | {base['stable_success_task_count']}/{base['task_count']} | {candidate['stable_success_task_count']}/{candidate['task_count']} |",
        f"| 至少一次成功任务 | {base['any_success_task_count']}/{base['task_count']} | {candidate['any_success_task_count']}/{candidate['task_count']} |",
        f"| 平均工具调用数 | {_fmt(base['mean_tool_calls'])} | {_fmt(candidate['mean_tool_calls'])} |",
        f"| 工具调用中位数 | {_fmt(base['median_tool_calls'])} | {_fmt(candidate['median_tool_calls'])} |",
        f"| 平均读取调用数 | {_fmt(base['mean_read_calls'])} | {_fmt(candidate['mean_read_calls'])} |",
        f"| 平均写入调用数 | {_fmt(base['mean_write_calls'])} | {_fmt(candidate['mean_write_calls'])} |",
        f"| 平均完全重复调用数 | {_fmt(base['mean_repeated_exact_calls'])} | {_fmt(candidate['mean_repeated_exact_calls'])} |",
        f"| 平均同工具最长连续调用 | {_fmt(base['mean_max_consecutive_same_tool_name'])} | {_fmt(candidate['mean_max_consecutive_same_tool_name'])} |",
        f"| 平均 trial 耗时（秒） | {_fmt(base['mean_duration_seconds'])} | {_fmt(candidate['mean_duration_seconds'])} |",
        f"| Agent 观测 token / trial | {_fmt(base['observed_tokens']['agent_mean_per_trial'])} | {_fmt(candidate['observed_tokens']['agent_mean_per_trial'])} |",
        f"| 用户模拟器观测 token / trial | {_fmt(base['observed_tokens']['user_simulator_mean_per_trial'])} | {_fmt(candidate['observed_tokens']['user_simulator_mean_per_trial'])} |",
        "",
        f"SFT 替换重跑任务：{comparison['replacement_task_ids']}。",
        "",
        "## 可比参考动作覆盖",
        "",
        (
            f"在双方 action checks 均可用且参考动作数一致的 "
            f"{comparison['paired_reference_action_comparison']['trial_count']} 个 task-trial 对上，"
            f"Base/SFT 微平均召回分别为 "
            f"{_fmt(comparison['paired_reference_action_comparison']['base_micro_recall'], percent=True)} / "
            f"{_fmt(comparison['paired_reference_action_comparison']['candidate_micro_recall'], percent=True)}；"
            f"排除任务为 {comparison['paired_reference_action_comparison']['excluded_task_ids']}。"
        ),
        "",
        "## 按结果变化分层的工具效率",
        "",
        "| 分层 | task:trial | Base均值 | SFT均值 | 平均变化(SFT-Base) |",
        "|---|---|---:|---:|---:|",
    ]
    for name, row in comparison["strata"].items():
        lines.append(
            f"| {name} | {row['task_trial_ids']} | {_fmt(row['base_mean_tool_calls'])} | "
            f"{_fmt(row['candidate_mean_tool_calls'])} | {_fmt(row['mean_tool_call_delta'])} |"
        )
    lines.extend(
        [
            "",
            "## 工具调用膨胀 Top 5",
            "",
            "| task:trial | 结果变化 | Base | SFT | 增量 |",
            "|---|---|---:|---:|---:|",
        ]
    )
    for row in comparison["top_tool_call_increases"]:
        lines.append(
            f"| {row['task_id']}:{row['trial_index']} | {row['transition']} | {row['base_tool_calls']} | "
            f"{row['candidate_tool_calls']} | {row['tool_call_delta']:+d} |"
        )
    lines.extend(
        [
            "",
            "## 指标边界",
            "",
            "- `reference_action_recall` 是 Tau2 参考动作覆盖率，不是 Tool Precision；额外调用不会被原始 action checker 惩罚。",
            "- 工具调用更少不等于更好，提前失败也会减少调用；`both_success` 是更可信的效率比较切片。",
            "- 完全重复调用只是循环/冗余候选；若中间状态发生变化，重复调用可能合理。",
            "- Policy V2.2 尚未基于独立人工金标完成验证，因此这里只作为 provisional diagnostic，不进入正式成功率。",
            "- 用户意图识别目前只能通过最终断言、参考动作覆盖和 verifier finding 间接观察，不能伪造为独立准确率。",
            "- Token 仅汇总 `returned_results.json` 中消息级 usage：assistant 为本地 Agent，user 为外部用户模拟器；NL Judge token 未记录，不能由此推算完整 API 账单。",
            "- 序列化 cost 字段均不能替代供应商账单，因此不报告美元成本。",
            "",
            "## 可追溯产物",
            "",
            "- 每个任务的输入文件路径、SHA-256、是否重跑，见 `evaluation_cards.json`。",
            "- 每任务配对的成功变化与工具调用差值，见 `comparison.json`。",
            "",
        ]
    )
    return "\n".join(lines)


def build_report(
    *,
    base_dir: str | Path,
    candidate_dir: str | Path,
    output_dir: str | Path,
    replacements: dict[str, str | Path] | None = None,
    overwrite: bool = False,
) -> dict[str, Any]:
    output = Path(output_dir).resolve()
    if output.exists() and any(output.iterdir()) and not overwrite:
        raise FileExistsError(f"Output directory is not empty: {output}")
    output.mkdir(parents=True, exist_ok=True)
    base_cards = load_run_cards(base_dir, run_name="base")
    candidate_cards = load_run_cards(
        candidate_dir,
        run_name="sft",
        replacements=replacements,
    )
    report = {
        "schema_version": SCHEMA_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "inputs": {
            "base_dir": str(Path(base_dir).resolve()),
            "candidate_dir": str(Path(candidate_dir).resolve()),
            "replacements": {
                str(task_id): str(Path(path).resolve())
                for task_id, path in (replacements or {}).items()
            },
        },
        "comparison": compare_cards(base_cards, candidate_cards),
    }
    cards_payload = {"base": base_cards, "candidate": candidate_cards}
    (output / "evaluation_cards.json").write_text(
        json.dumps(cards_payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (output / "comparison.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (output / "analysis.md").write_text(render_markdown(report), encoding="utf-8")
    return report


def _replacement(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("replacement must be TASK_ID=RUN_DIR")
    task_id, path = value.split("=", 1)
    return task_id, Path(path)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build traceable Base-vs-SFT evaluation cards and tool-use metrics."
    )
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--replacement", action="append", type=_replacement, default=[])
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    report = build_report(
        base_dir=args.base,
        candidate_dir=args.candidate,
        output_dir=args.output,
        replacements=dict(args.replacement),
        overwrite=args.overwrite,
    )
    print(json.dumps(report["comparison"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
