from pathlib import Path
import json

RUN_DIR = Path(
    r"D:\PolicyAgent-PostTrain\experiments"
    r"\20260721_151851_retail_smoke5_deepseek"
)

TASK_IDS = ["59", "29", "72", "50", "28"]


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


rows = []

for task_id in TASK_IDS:
    result_path = RUN_DIR / f"task_{task_id}" / "returned_results.json"

    data = load_json(result_path)

    simulations = data.get("simulations", [])

    assert len(simulations) == 1, (
        f"Task {task_id}: expected 1 simulation, "
        f"got {len(simulations)}"
    )

    sim = simulations[0]

    reward_info = sim.get("reward_info") or {}

    reward = reward_info.get("reward")

    reward_breakdown = reward_info.get("reward_breakdown") or {}

    action_checks = reward_info.get("action_checks") or []

    action_total = len(action_checks)

    action_passed = sum(
        1
        for x in action_checks
        if x.get("action_reward") == 1.0
    )

    read_checks = [
        x for x in action_checks
        if x.get("tool_type") == "read"
    ]

    write_checks = [
        x for x in action_checks
        if x.get("tool_type") == "write"
    ]

    read_passed = sum(
        1 for x in read_checks
        if x.get("action_reward") == 1.0
    )

    write_passed = sum(
        1 for x in write_checks
        if x.get("action_reward") == 1.0
    )

    agent_cost = sim.get("agent_cost") or 0.0
    user_cost = sim.get("user_cost") or 0.0

    rows.append({
        "task_id": task_id,

        "reward": reward,

        "db_reward":
            reward_breakdown.get("DB"),

        "nl_assertion_reward":
            reward_breakdown.get("NL_ASSERTION"),

        "duration_seconds":
            sim.get("duration"),

        "termination_reason":
            sim.get("termination_reason"),

        "agent_cost_usd":
            agent_cost,

        "user_cost_usd":
            user_cost,

        "agent_plus_user_cost_usd":
            agent_cost + user_cost,

        "action_passed":
            action_passed,

        "action_total":
            action_total,

        "read_passed":
            read_passed,

        "read_total":
            len(read_checks),

        "write_passed":
            write_passed,

        "write_total":
            len(write_checks),
    })


rewards = [
    r["reward"]
    for r in rows
    if isinstance(r["reward"], (int, float))
]

durations = [
    r["duration_seconds"]
    for r in rows
    if isinstance(r["duration_seconds"], (int, float))
]

total_agent_cost = sum(
    r["agent_cost_usd"]
    for r in rows
)

total_user_cost = sum(
    r["user_cost_usd"]
    for r in rows
)

total_agent_user_cost = (
    total_agent_cost + total_user_cost
)

success_count = sum(
    1
    for r in rows
    if r["reward"] == 1.0
)

aggregate = {
    "source_run_dir": str(RUN_DIR),

    "task_ids": TASK_IDS,

    "task_count": len(rows),

    "observed_success_count": success_count,

    "observed_failure_count":
        len(rows) - success_count,

    "observed_reward_mean":
        sum(rewards) / len(rewards)
        if rewards else None,

    "note_on_success_rate": (
        "This 5-task subset is deliberately risk-stratified "
        "and is NOT a random sample. The observed success "
        "fraction must not be reported as the 20-task baseline "
        "success rate."
    ),

    "total_duration_seconds":
        sum(durations),

    "mean_duration_seconds":
        sum(durations) / len(durations)
        if durations else None,

    "total_agent_cost_usd":
        total_agent_cost,

    "total_user_cost_usd":
        total_user_cost,

    "total_agent_plus_user_cost_usd":
        total_agent_user_cost,

    "mean_agent_plus_user_cost_usd":
        total_agent_user_cost / len(rows),

    "linear_projection_20_tasks": {
        "estimated_duration_seconds":
            (sum(durations) / len(rows)) * 20
            if rows else None,

        "estimated_agent_plus_user_cost_usd":
            (total_agent_user_cost / len(rows)) * 20
            if rows else None,

        "warning": (
            "Linear projection only. Task complexity varies, "
            "and NL Judge cost is NOT included yet."
        ),
    },

    "tasks": rows,
}


output_path = RUN_DIR / "smoke_summary_v1.json"

output_path.write_text(
    json.dumps(
        aggregate,
        ensure_ascii=False,
        indent=2,
    ),
    encoding="utf-8",
)


print("=== SMOKE 5 OFFLINE SUMMARY ===")

for r in rows:
    print(
        f"TASK {r['task_id']:>3} | "
        f"reward={r['reward']} | "
        f"DB={r['db_reward']} | "
        f"NL={r['nl_assertion_reward']} | "
        f"action={r['action_passed']}/{r['action_total']} | "
        f"read={r['read_passed']}/{r['read_total']} | "
        f"write={r['write_passed']}/{r['write_total']} | "
        f"duration={r['duration_seconds']:.2f}s | "
        f"agent=${r['agent_cost_usd']:.6f} | "
        f"user=${r['user_cost_usd']:.6f}"
    )

print("\n=== AGGREGATE ===")

print(
    "OBSERVED_SUCCESS_COUNT =",
    aggregate["observed_success_count"],
)

print(
    "OBSERVED_FAILURE_COUNT =",
    aggregate["observed_failure_count"],
)

print(
    "OBSERVED_REWARD_MEAN =",
    aggregate["observed_reward_mean"],
)

print(
    "TOTAL_DURATION_SECONDS =",
    round(
        aggregate["total_duration_seconds"],
        3,
    ),
)

print(
    "MEAN_DURATION_SECONDS =",
    round(
        aggregate["mean_duration_seconds"],
        3,
    ),
)

print(
    "TOTAL_AGENT_COST_USD =",
    round(
        aggregate["total_agent_cost_usd"],
        8,
    ),
)

print(
    "TOTAL_USER_COST_USD =",
    round(
        aggregate["total_user_cost_usd"],
        8,
    ),
)

print(
    "TOTAL_AGENT_PLUS_USER_COST_USD =",
    round(
        aggregate[
            "total_agent_plus_user_cost_usd"
        ],
        8,
    ),
)

projection = aggregate[
    "linear_projection_20_tasks"
]

print(
    "PROJECTED_20_DURATION_SECONDS =",
    round(
        projection[
            "estimated_duration_seconds"
        ],
        3,
    ),
)

print(
    "PROJECTED_20_AGENT_PLUS_USER_COST_USD =",
    round(
        projection[
            "estimated_agent_plus_user_cost_usd"
        ],
        8,
    ),
)

print("\nOUTPUT =", output_path)

print("\nIMPORTANT:")
print(
    "- 3/5 is an observed smoke result, "
    "NOT the 20-task baseline success rate."
)
print(
    "- Judge cost is NOT included in the cost projection."
)

print("\nSMOKE_5_OFFLINE_SUMMARY_OK")
