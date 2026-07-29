import json
from pathlib import Path


INPUT_FILE = Path(
    "reports/verifier/llm_verifier_v2_result.jsonl"
)


OUTPUT_FILE = Path(
    "reports/verifier/llm_verifier_v2_summary.json"
)


def load_jsonl(path):

    data = []

    with open(
        path,
        "r",
        encoding="utf-8"
    ) as f:

        for line in f:
            data.append(json.loads(line))

    return data



def main():

    data = load_jsonl(
        INPUT_FILE
    )


    total = len(data)


    correct = 0

    failure_pred = 0


    dimension = {

        "intent_alignment":0,

        "policy_compliance":0,

        "tool_usage":0,

        "scope_handling":0

    }


    failure_types = {}


    for item in data:


        pred = item["prediction"]


        if pred.get("correct"):

            correct += 1


        else:

            failure_pred += 1



        for k in dimension:

            try:

                dimension[k] += (
                    pred[k]["score"]
                )

            except Exception:

                pass



        ft = pred.get(
            "failure_type",
            "unknown"
        )


        failure_types[ft] = (
            failure_types.get(ft,0)
            +
            1
        )



    result = {


        "total":

            total,


        "overall_accuracy":

        {

            "correct":

                correct,

            "wrong":

                total-correct,

            "accuracy":

                round(
                    correct/total,
                    3
                )

        },


        "failure_prediction":

        {

            "predicted_failure":

                failure_pred,

            "rate":

                round(
                    failure_pred/total,
                    3
                )

        },


        "dimension_accuracy":

        {

            k:

            round(
                dimension[k]/total,
                3
            )

            for k in dimension

        },


        "failure_type_distribution":

            failure_types

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
            result,
            f,
            indent=2,
            ensure_ascii=False
        )


    print(
        json.dumps(
            result,
            indent=2,
            ensure_ascii=False
        )
    )


    print(
        f"Saved: {OUTPUT_FILE}"
    )



if __name__=="__main__":

    main()