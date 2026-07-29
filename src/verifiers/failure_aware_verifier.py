import json
from pathlib import Path

from src.verifiers.deepseek_client import DeepSeekClient


INPUT_FILE = Path(
    "data/trajectory/retail_baseline20_trial1.jsonl"
)

DEMO_FILE = Path(
    "configs/failure_demonstrations.json"
)

OUTPUT_FILE = Path(
    "reports/verifier/failure_aware_v5_result.jsonl"
)


client = DeepSeekClient()



SYSTEM_PROMPT = """
You are a benchmark failure verifier.

Your task is NOT to judge whether an agent is good or bad in the real world.

Your task is:

Detect whether the agent trajectory contains failures defined by the benchmark.

You must follow these rules:

1. Only classify failure when trajectory evidence matches known failure patterns.

2. Do NOT introduce external business knowledge.

Example:

Wrong:
"Most companies require return windows, therefore failure."

Correct:
"The benchmark policy explicitly requires this and the agent violated it."

3. Tool success does not guarantee correctness.

A tool can execute successfully while the agent violates benchmark policy.

4. Normal limitations are NOT failures.

Examples:

- Agent cannot provide unknown shipping information.
- Agent asks clarification.
- Agent explains unavailable information.

These are not failures unless they match benchmark failure patterns.

Return JSON only:

{
    "has_failure": true/false,
    "intent_summary": "",
    "matched_pattern": "",
    "evidence": [],
    "failure_type": "",
    "reason": ""
}

"""



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



def build_demo_context():

    demos = load_json(
        DEMO_FILE
    )

    return json.dumps(
        demos,
        ensure_ascii=False,
        indent=2
    )



def call_verifier(item):

    trajectory = build_trajectory(
        item
    )

    demonstrations = build_demo_context()


    prompt = f"""

{SYSTEM_PROMPT}


Known failure demonstrations:

{demonstrations}


Now analyze the following trajectory:


================
Trajectory
================

{trajectory}


Steps:

1. Extract user intent.
2. Identify agent actions.
3. Compare with failure patterns.
4. Only output evidence-backed failure.


Return JSON only.

"""


    response = client.chat(
        prompt
    )


    response=response.strip()


    try:

        return json.loads(
            response
        )

    except Exception:


        return {

            "has_failure": False,

            "intent_summary": "",

            "matched_pattern": "",

            "evidence": [],

            "failure_type": "parse_error",

            "reason": response

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


            result={

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
        "Failure-aware verifier v5 finished"
    )

    print(
        f"Saved: {OUTPUT_FILE}"
    )



if __name__ == "__main__":

    main()