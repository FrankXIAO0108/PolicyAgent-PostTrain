from src.rules.scope_checker import (
    check_scope_confirmation
)


test = {

    "user_intent":[
        "User only wants to cancel skateboard"
    ],


    "agent_actions":[
        "Agent cancelled entire order"
    ],


    "tool_actions":[
        "cancel_order(order)"
    ]

}


result = check_scope_confirmation(
    test
)


print(result)