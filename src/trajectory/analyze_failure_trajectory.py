import json
from pathlib import Path


INPUT_FILE = Path(
    "data/trajectory/retail_baseline20_trial1.jsonl"
)

OUTPUT_DIR = Path(
    "reports/failure_analysis"
)


def load_jsonl(path):

    data = []

    with open(
        path,
        "r",
        encoding="utf-8"
    ) as f:

        for line in f:
            data.append(
                json.loads(line)
            )

    return data



def extract_failures(data):

    failures = []

    for traj in data:

        reward = traj.get(
            "reward",
            {}
        ).get(
            "overall"
        )

        if reward == 0:

            failures.append(traj)


    return failures



def save_failures(failures):

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )


    summary = []

    for traj in failures:

        task_id = traj.get(
            "task_id"
        )


        output_file = (
            OUTPUT_DIR
            /
            f"task_{task_id}.json"
        )


        with open(
            output_file,
            "w",
            encoding="utf-8"
        ) as f:

            json.dump(
                traj,
                f,
                indent=2,
                ensure_ascii=False
            )


        summary.append(
            {
                "task_id": task_id,
                "reward": traj.get(
                    "reward"
                ),
                "message_count": len(
                    traj.get(
                        "messages",
                        []
                    )
                ),
                "tool_calls": len(
                    traj.get(
                        "tool_calls",
                        []
                    )
                )
            }
        )


    with open(
        OUTPUT_DIR /
        "failure_summary.json",
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            summary,
            f,
            indent=2,
            ensure_ascii=False
        )


    print(
        f"Extracted {len(failures)} failures"
    )



def main():

    data = load_jsonl(
        INPUT_FILE
    )

    failures = extract_failures(
        data
    )

    save_failures(
        failures
    )


if __name__ == "__main__":
    main()