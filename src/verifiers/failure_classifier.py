import json

from src.verifiers.deepseek_client import DeepSeekClient


client = DeepSeekClient()


PROMPT = """
You are a benchmark failure classifier.

Your task is to determine whether an Agent trajectory contains
a benchmark-defined failure.

You are NOT judging whether the agent is good according to
real-world business knowledge.

You must strictly follow the provided failure rules.

Decision process:

Step 1:
Read extracted evidence.

Step 2:
For each possible failure type:
- check whether required evidence exists
- check whether failure conditions are satisfied

Step 3:
Only classify failure when there is explicit evidence.

If evidence is insufficient:
return has_failure=false.


Important constraints:

1. Tool success does NOT mean the action is correct.

2. Do NOT introduce external policies.

3. Do NOT assume common business practices.

4. The benchmark rules are the only source of truth.


Possible failure types:

- none
- golden_mismatch
- variant_understanding_failure
- scope_confirmation_failure
- policy_violation


Return JSON only:

{
    "has_failure": true/false,

    "failure_type": "",

    "matched_rule": "",

    "evidence": [],

    "reason": ""
}

"""


def classify_failure(
    evidence,
    failure_rules
):


    prompt = f"""

{PROMPT}


====================
Benchmark Failure Rules
====================

{json.dumps(
    failure_rules,
    ensure_ascii=False,
    indent=2
)}



====================
Extracted Evidence
====================

{json.dumps(
    evidence,
    ensure_ascii=False,
    indent=2
)}



Now classify this case.

Remember:

You must match evidence with rules.

Do not judge based on general intuition.

Return JSON only.

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

            "has_failure": False,

            "failure_type": "parse_error",

            "matched_rule": "",

            "evidence": [],

            "reason": response

        }