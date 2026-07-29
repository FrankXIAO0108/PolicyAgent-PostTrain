from src.rules.variant_checker import (
    check_variant_understanding
)


test = {


    "user_intent":[
        "User wants laptop with i7 CPU, 8GB RAM and 1TB SSD"
    ],


    "agent_actions":[
        "Agent selected laptop with i5 CPU, 8GB RAM and 512GB SSD"
    ],


    "tool_actions":[

    ],


    "constraints":[

    ]

}



result = check_variant_understanding(
    test
)


print(result)