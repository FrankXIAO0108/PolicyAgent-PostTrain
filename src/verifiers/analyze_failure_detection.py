import json
from pathlib import Path


INPUT_FILE = Path(
    "reports/verifier/llm_verifier_result.jsonl"
)

OUTPUT_JSON = Path(
    "reports/verifier/failure_detection_analysis.json"
)

OUTPUT_MD = Path(
    "reports/verifier/failure_detection_analysis.md"
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


    failure_cases = []

    tp = 0
    fp = 0
    fn = 0
    tn = 0


    for item in results:

        task_id = item["task_id"]

        gold = item.get(
            "gold",
            {}
        )

        prediction = item.get(
            "prediction",
            {}
        )


        # gold label:
        # 1 = success
        # 0 = failure

        gold_failure = (
            gold.get("label") == 0
        )


        pred_failure = (
            prediction.get("failure_type")
            != "none"
        )


        if gold_failure and pred_failure:
            tp += 1

        elif not gold_failure and pred_failure:
            fp += 1

        elif gold_failure and not pred_failure:
            fn += 1

        else:
            tn += 1


        if gold_failure:

            failure_cases.append(
                {
                    "task_id": task_id,

                    "gold_failure_type":
                        gold.get(
                            "failure_type"
                        ),

                    "prediction_failure_type":
                        prediction.get(
                            "failure_type"
                        ),

                    "prediction_correct":
                        prediction.get(
                            "correct"
                        ),

                    "reason":
                        prediction.get(
                            "reason"
                        )
                }
            )


    total_failure = tp + fn


    recall = (
        tp / total_failure
        if total_failure
        else 0
    )


    precision = (
        tp / (tp + fp)
        if (tp + fp)
        else 0
    )


    analysis = {

        "confusion_matrix": {

            "true_positive": tp,

            "false_positive": fp,

            "false_negative": fn,

            "true_negative": tn

        },


        "failure_detection": {

            "precision": precision,

            "recall": recall

        },


        "failure_cases": failure_cases

    }


    OUTPUT_JSON.parent.mkdir(
        parents=True,
        exist_ok=True
    )


    with open(
        OUTPUT_JSON,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            analysis,
            f,
            indent=2,
            ensure_ascii=False
        )


    with open(
        OUTPUT_MD,
        "w",
        encoding="utf-8"
    ) as f:

        f.write(
            "# LLM Verifier Failure Detection Analysis\n\n"
        )

        f.write(
            "## Confusion Matrix\n\n"
        )

        f.write(
            f"- True Positive: {tp}\n"
        )

        f.write(
            f"- False Positive: {fp}\n"
        )

        f.write(
            f"- False Negative: {fn}\n"
        )

        f.write(
            f"- True Negative: {tn}\n\n"
        )


        f.write(
            "## Metrics\n\n"
        )

        f.write(
            f"- Precision: {precision:.3f}\n"
        )

        f.write(
            f"- Recall: {recall:.3f}\n\n"
        )


        f.write(
            "## Failure Cases\n\n"
        )


        for case in failure_cases:

            f.write(
                f"### Task {case['task_id']}\n\n"
            )

            f.write(
                f"- Gold: {case['gold_failure_type']}\n"
            )

            f.write(
                f"- Prediction: {case['prediction_failure_type']}\n"
            )

            f.write(
                f"- Correct: {case['prediction_correct']}\n\n"
            )

            f.write(
                f"Reason:\n{case['reason']}\n\n"
            )


    print(
        json.dumps(
            analysis,
            indent=2,
            ensure_ascii=False
        )
    )

    print(
        f"Saved: {OUTPUT_JSON}"
    )

    print(
        f"Saved: {OUTPUT_MD}"
    )


if __name__ == "__main__":
    main()