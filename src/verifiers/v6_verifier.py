import json

from pathlib import Path


from src.verifiers.evidence_extractor import (
    extract_evidence
)

from src.verifiers.failure_classifier import (
    classify_failure
)



INPUT_FILE = Path(
    "data/trajectory/retail_baseline20_trial1.jsonl"
)


RULE_FILE = Path(
    "configs/failure_rules_v1.json"
)


OUTPUT_FILE = Path(
    "reports/verifier/v6_verifier_result.jsonl"
)



def load_json(path):

    with open(
        path,
        "r",
        encoding="utf-8"
    ) as f:

        return json.load(f)



def load_jsonl(path):

    data=[]

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



def build_trajectory(item):


    text=""


    for msg in item["messages"]:

        text += (
            f"{msg['role']}: "
            f"{msg['content']}\n"
        )


    return text




def main():


    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True
    )


    trajectories = load_jsonl(
        INPUT_FILE
    )


    failure_rules = load_json(
        RULE_FILE
    )



    with open(
        OUTPUT_FILE,
        "w",
        encoding="utf-8"
    ) as f:



        for idx,item in enumerate(
            trajectories,
            1
        ):


            print(
                f"Processing {idx}/{len(trajectories)} "
                f"task={item['task_id']}"
            )


            trajectory = build_trajectory(
                item
            )


            # Stage 1
            evidence = extract_evidence(
                trajectory
            )


            # Stage 2
            prediction = classify_failure(
                evidence,
                failure_rules
            )



            result={


                "task_id":
                    item["task_id"],


                "evidence":
                    evidence,


                "prediction":
                    prediction,


                "gold_reward":
                    item.get(
                        "reward",
                        {}
                    )

            }



            f.write(
                json.dumps(
                    result,
                    ensure_ascii=False
                )
                +
                "\n"
            )



    print(
        "v6.1 verifier finished"
    )


    print(
        f"Saved: {OUTPUT_FILE}"
    )



if __name__=="__main__":

    main()