"""
Trajectory analysis for PolicyAgent-PostTrain.

Analyze exported tau2 trajectories:
- reward distribution
- success/failure statistics
- message length
- tool usage
- trajectory complexity
"""

import json
from pathlib import Path
from collections import Counter


INPUT_FILE = Path(
    "data/trajectory/retail_baseline20_trial1.jsonl"
)

OUTPUT_DIR = Path("reports/trajectory")


def load_jsonl(path):
    trajectories = []

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            trajectories.append(json.loads(line))

    return trajectories


def analyze(trajectories):

    report = {}

    total = len(trajectories)

    rewards = []
    success = 0
    failure = 0

    message_lengths = []
    tool_counts = []
    tool_names = Counter()

    for traj in trajectories:

        reward = traj.get(
            "reward",
            {}
        ).get(
            "overall",
            None
        )

        if reward is not None:
            rewards.append(reward)

            if reward == 1:
                success += 1
            else:
                failure += 1


        messages = traj.get(
            "messages",
            []
        )

        message_lengths.append(
            len(messages)
        )


        tools = traj.get(
            "tool_calls",
            []
        )

        tool_counts.append(
            len(tools)
        )


        for tool in tools:
            name = tool.get(
                "tool_name",
                ""
            )

            if name:
                tool_names[name] += 1


    report["total_trajectories"] = total

    report["success"] = {
        "count": success,
        "rate": (
            success / total
            if total else 0
        )
    }

    report["failure"] = {
        "count": failure,
        "rate": (
            failure / total
            if total else 0
        )
    }


    report["message_statistics"] = {
        "avg": sum(message_lengths)
        / len(message_lengths)
        if message_lengths else 0,

        "max": max(message_lengths)
        if message_lengths else 0,

        "min": min(message_lengths)
        if message_lengths else 0,
    }


    report["tool_statistics"] = {
        "avg_tool_calls":
            sum(tool_counts)
            /
            len(tool_counts)
            if tool_counts else 0,

        "max_tool_calls":
            max(tool_counts)
            if tool_counts else 0,

        "tool_frequency":
            dict(tool_names)
    }


    report["reward_distribution"] = dict(
        Counter(rewards)
    )


    return report



def save_report(report):

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )


    json_path = (
        OUTPUT_DIR
        /
        "trajectory_summary.json"
    )

    with open(
        json_path,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            report,
            f,
            indent=2,
            ensure_ascii=False
        )


    md_path = (
        OUTPUT_DIR
        /
        "trajectory_summary.md"
    )


    with open(
        md_path,
        "w",
        encoding="utf-8"
    ) as f:

        f.write(
            "# Trajectory Analysis Report\n\n"
        )

        f.write(
            f"Total trajectories: "
            f"{report['total_trajectories']}\n\n"
        )


        f.write(
            "## Success\n\n"
        )

        f.write(
            json.dumps(
                report["success"],
                indent=2
            )
        )

        f.write(
            "\n\n## Failure\n\n"
        )

        f.write(
            json.dumps(
                report["failure"],
                indent=2
            )
        )


        f.write(
            "\n\n## Message Statistics\n\n"
        )

        f.write(
            json.dumps(
                report["message_statistics"],
                indent=2
            )
        )


        f.write(
            "\n\n## Tool Usage\n\n"
        )

        f.write(
            json.dumps(
                report["tool_statistics"],
                indent=2
            )
        )


    print(
        "Saved:"
    )

    print(
        json_path
    )

    print(
        md_path
    )



def main():

    trajectories = load_jsonl(
        INPUT_FILE
    )

    report = analyze(
        trajectories
    )

    save_report(
        report
    )


if __name__ == "__main__":
    main()