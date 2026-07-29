import json

from src.verifiers.deepseek_client import DeepSeekClient


client = DeepSeekClient()



PROMPT = """
You are an evidence extraction module for Agent trajectory evaluation.

Your task is NOT to judge whether the agent is correct.

Only extract factual information from the trajectory.

Extract:

1. User intent:
- What does the user want?
- Which items/actions are requested?

2. Agent actions:
- What did the agent decide?
- What operations were performed?

3. Tool actions:
- What tools were called?
- What parameters were used?

4. Constraints:
- Explicit user limitations
- Required conditions


Do not:
- infer hidden policies
- judge success/failure
- use external knowledge


Return JSON:

{
"user_intent":[],
"agent_actions":[],
"tool_actions":[],
"constraints":[]
}

"""


def extract_evidence(trajectory):

    prompt = f"""

{PROMPT}


Trajectory:

{trajectory}


Return JSON only.

"""


    response = client.chat(
        prompt
    )


    try:

        return json.loads(
            response
        )

    except Exception:

        return {
            "user_intent":[],
            "agent_actions":[],
            "tool_actions":[],
            "constraints":[]
        }