import json
from pathlib import Path


INPUT_FILE = Path(
    "reports/verifier/llm_verifier_result.jsonl"
)

OUTPUT_FILE = Path(
    "reports/verifier/llm_verifier_summary.json"
)


def load_jsonl(path):
    data = []

    with open(
        path,
        "r",
        encoding="utf-8"
    ) as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))

    return data


def main():

    results = load_jsonl(INPUT_FILE)

    total = len(results)

    correct = 0
    wrong = 0

    failure_type_correct = 0

    gold_failures = 0
    pred_failures = 0


    for item in results:

        gold = item.get("gold", {})
        prediction = item.get("prediction", {})


        # overall correctness
        if prediction.get("correct") is True:
            correct += 1
        else:
            wrong += 1


        # failure taxonomy accuracy
        if (
            gold.get("failure_type")
            ==
            prediction.get("failure_type")
        ):
            failure_type_correct += 1


        if gold.get("label") == 0:
            gold_failures += 1

        if prediction.get("failure_type") != "none":
            pred_failures += 1



    summary = {

        "total": total,

        "overall_accuracy": {
            "correct": correct,
            "wrong": wrong,
            "accuracy": correct / total if total else 0
        },


        "failure_type_accuracy": {
            "correct": failure_type_correct,
            "accuracy": (
                failure_type_correct / total
                if total else 0
            )
        },


        "failure_statistics": {

            "gold_failure_cases": gold_failures,

            "predicted_failure_cases": pred_failures

        }

    }


    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True
    )


    with open(
        OUTPUT_FILE,
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
        json.dumps(
            summary,
            indent=2,
            ensure_ascii=False
        )
    )

    print(
        f"Saved: {OUTPUT_FILE}"
    )


if __name__ == "__main__":
    main()