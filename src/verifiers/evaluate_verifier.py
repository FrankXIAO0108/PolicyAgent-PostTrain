import json
from pathlib import Path
from collections import Counter


DATASET_FILE = Path(
    "data/verifier/verifier_dataset.jsonl"
)

OUTPUT_FILE = Path(
    "reports/verifier/verifier_eval.json"
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



def rule_based_verifier(sample):

    """
    当前版本:
    使用已有 reward 标签模拟 verifier

    后续替换为:
    - LLM judge
    - reward model
    - policy verifier
    """

    label = sample["label"]

    failure_type = sample["failure_type"]


    return {

        "prediction": label,

        "failure_type": failure_type,

        "reason":
            sample["explanation"]

    }



def evaluate():

    samples = load_jsonl(
        DATASET_FILE
    )


    results=[]


    correct=0


    failure_counter=Counter()


    for sample in samples:


        pred = rule_based_verifier(
            sample
        )


        results.append({

            "task_id":
                sample["task_id"],

            "gold_label":
                sample["label"],

            "prediction":
                pred["prediction"],

            "failure_type":
                pred["failure_type"],

            "reason":
                pred["reason"]

        })


        if (
            pred["prediction"]
            ==
            sample["label"]
        ):
            correct += 1


        if sample["label"] == 0:

            failure_counter[
                sample["failure_type"]
            ] += 1



    accuracy = (
        correct / len(samples)
    )


    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True
    )


    output={

        "total_samples":
            len(samples),

        "accuracy":
            accuracy,

        "failure_distribution":
            dict(failure_counter),

        "results":
            results

    }



    with open(
        OUTPUT_FILE,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            output,
            f,
            indent=2,
            ensure_ascii=False
        )


    print(
        "Verifier evaluation finished"
    )

    print(
        f"Accuracy: {accuracy:.3f}"
    )



if __name__ == "__main__":

    evaluate()