import json
from pathlib import Path


TRAJECTORY_FILE = Path(
    "data/trajectory/retail_baseline20_trial1.jsonl"
)

OUTPUT_DIR = Path(
    "data/verifier"
)


FAILURE_LABELS = {
    "59": {
        "failure_type": "golden_mismatch",
        "rule": "Check whether the agent action matches benchmark expected behavior.",
        "explanation": "Agent behavior differs from benchmark golden trajectory."
    },

    "98": {
        "failure_type": "scope_confirmation_failure",
        "rule": "Check whether agent modifies only the items explicitly requested by user.",
        "explanation": "Agent cancelled or modified more items than requested."
    },

    "95": {
        "failure_type": "variant_understanding_failure",
        "rule": "Check whether selected replacement item satisfies user requested attributes.",
        "explanation": "Agent selected or misunderstood product variant incorrectly."
    },

    "107": {
        "failure_type": "policy_violation",
        "rule": "Check whether agent action follows business policy constraints.",
        "explanation": "Agent action violates policy although tool execution succeeded."
    }
}


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



def save_jsonl(path, data):

    with open(
        path,
        "w",
        encoding="utf-8"
    ) as f:

        for item in data:

            f.write(
                json.dumps(
                    item,
                    ensure_ascii=False
                )
                + "\n"
            )



def build_dataset():

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )


    trajectories = load_jsonl(
        TRAJECTORY_FILE
    )


    success_samples = []
    failure_samples = []


    for traj in trajectories:

        task_id = str(
            traj["task_id"]
        )

        reward = traj["reward"]["overall"]


        if reward == 1.0:

            sample = {

                "task_id": task_id,

                "messages": traj["messages"],

                "label": 1,

                "failure_type": "none",

                "verification_rule":
                    "Verify that the agent completed the task correctly.",

                "explanation":
                    "The trajectory passed benchmark evaluation."

            }


            success_samples.append(sample)


        else:

            info = FAILURE_LABELS.get(
                task_id,
                {
                    "failure_type": "unknown",
                    "rule": "unknown",
                    "explanation": "unknown failure"
                }
            )


            sample = {

                "task_id": task_id,

                "messages": traj["messages"],

                "label": 0,

                "failure_type":
                    info["failure_type"],

                "verification_rule":
                    info["rule"],

                "explanation":
                    info["explanation"]

            }


            failure_samples.append(sample)



    save_jsonl(
        OUTPUT_DIR / "success_cases.jsonl",
        success_samples
    )

    save_jsonl(
        OUTPUT_DIR / "failure_cases.jsonl",
        failure_samples
    )

    save_jsonl(
        OUTPUT_DIR / "verifier_dataset.jsonl",
        success_samples + failure_samples
    )


    print(
        "Verifier dataset generated"
    )

    print(
        f"success: {len(success_samples)}"
    )

    print(
        f"failure: {len(failure_samples)}"
    )



if __name__ == "__main__":

    build_dataset()