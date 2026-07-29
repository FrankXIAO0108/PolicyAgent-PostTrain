import json
from pathlib import Path

from src.verifiers.deepseek_client import DeepSeekClient


INPUT_FILE = Path(
    "data/trajectory/retail_baseline20_trial1.jsonl"
)

OUTPUT_FILE = Path(
    "reports/verifier/llm_verifier_v2_result.jsonl"
)


client = DeepSeekClient()


SYSTEM_PROMPT = """
You are a policy-aware evaluator for customer service agent trajectories.

Evaluate whether the agent behavior is correct.

You must analyze four dimensions.

1. Intent Alignment

Check:
- Did the agent understand the user's actual request?
- Did it process only requested items?
- Did it avoid misunderstanding user goals?


2. Policy Compliance

Important:

A tool action being executable does NOT mean it is allowed.

Check:
- Did the agent follow business rules?
- Did it violate exchange/cancellation/refund policies?
- Did it perform actions that policy forbids?


3. Tool Usage

Check:
- Were the correct tools used?
- Were parameters correct?
- Did tool execution match user intent?


4. Scope Handling

Check:
- Did the agent modify only requested items?
- Did it cancel/exchange unrelated items?
- Did it confirm ambiguous requests?


Return JSON only.

Format:

{
 "correct": true/false,

 "intent_alignment": {
    "score": 0/1,
    "reason": ""
 },

 "policy_compliance": {
    "score": 0/1,
    "reason": ""
 },

 "tool_usage": {
    "score": 0/1,
    "reason": ""
 },

 "scope_handling": {
    "score": 0/1,
    "reason": ""
 },

 "failure_type": "",

 "reason": ""
}


Failure types:

none

golden_mismatch

variant_understanding_failure

scope_confirmation_failure

policy_violation
"""



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



def build_prompt(item):

    text = ""

    for message in item["messages"]:

        text += (
            message["role"]
            +
            ": "
            +
            message["content"]
            +
            "\n"
        )


    return f"""
Evaluate this trajectory:

{text}

Return JSON only.
"""



def call_verifier(item):

    prompt = f"""

{SYSTEM_PROMPT}


{build_prompt(item)}

"""


    response = client.chat(
        prompt
    )


    response = response.strip()


    try:

        return json.loads(
            response
        )


    except Exception:


        return {

            "correct": False,

            "intent_alignment": {
                "score":0,
                "reason":"JSON parse failed"
            },

            "policy_compliance":{
                "score":0,
                "reason":"JSON parse failed"
            },

            "tool_usage":{
                "score":0,
                "reason":"JSON parse failed"
            },

            "scope_handling":{
                "score":0,
                "reason":"JSON parse failed"
            },

            "failure_type":
                "parse_error",

            "reason":
                response
        }




def main():


    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True
    )


    data = load_jsonl(
        INPUT_FILE
    )


    with open(
        OUTPUT_FILE,
        "w",
        encoding="utf-8"
    ) as f:


        for idx,item in enumerate(
            data,
            1
        ):

            print(
                f"Evaluating {idx}/{len(data)} "
                f"task={item['task_id']}"
            )


            prediction = call_verifier(
                item
            )


            result = {

                "task_id":
                    item["task_id"],


                "prediction":
                    prediction

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
        "LLM verifier v2 finished"
    )


    print(
        f"Saved: {OUTPUT_FILE}"
    )



if __name__ == "__main__":

    main()