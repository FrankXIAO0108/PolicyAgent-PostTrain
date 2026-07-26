import json
from pathlib import Path

from src.verifiers.deepseek_client import DeepSeekClient


INPUT_FILE = Path(
    "data/verifier/verifier_dataset.jsonl"
)

OUTPUT_FILE = Path(
    "reports/verifier/llm_verifier_result.jsonl"
)


client = DeepSeekClient()



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



def build_prompt(sample):

    conversation = "\n".join(
        [
            f'{m["role"]}: {m["content"]}'
            for m in sample["messages"]
        ]
    )


    return f"""
You are an expert AI agent verifier.

Analyze this customer service agent trajectory.

Conversation:

{conversation}


Determine whether the agent completed the task correctly.

Evaluation criteria:

1. Did the agent understand user intent?
2. Did the agent modify only requested items?
3. Did the agent follow policy?
4. Did the final tool action match the request?


Return ONLY JSON:

{{
    "correct": true,
    "failure_type": "none",
    "reason": ""
}}


Allowed failure_type:

none
golden_mismatch
scope_confirmation_failure
variant_understanding_failure
policy_violation
unknown

"""



def call_verifier(prompt):

    response = client.chat(
        prompt
    )


    try:

        return json.loads(
            response
        )


    except Exception:

        return {

            "correct": None,

            "failure_type":
                "parse_error",

            "reason":
                response

        }



def main():

    samples = load_jsonl(
        INPUT_FILE
    )


    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True
    )


    with open(
        OUTPUT_FILE,
        "w",
        encoding="utf-8"
    ) as f:


        for i, sample in enumerate(samples):

            print(
                f"Evaluating {i+1}/{len(samples)} "
                f"task={sample['task_id']}"
            )


            prompt = build_prompt(
                sample
            )


            prediction = call_verifier(
                prompt
            )


            result = {

                "task_id":
                    sample["task_id"],


                "gold":
                    {
                        "label":
                            sample["label"],

                        "failure_type":
                            sample["failure_type"]
                    },


                "prediction":
                    prediction

            }


            f.write(
                json.dumps(
                    result,
                    ensure_ascii=False
                )
                + "\n"
            )


    print(
        "LLM verifier evaluation finished"
    )

    print(
        "Saved:",
        OUTPUT_FILE
    )



if __name__ == "__main__":
    main()