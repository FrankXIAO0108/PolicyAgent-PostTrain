"""
Scope Rule Checker

负责检测：
用户明确要求范围
vs
Agent实际执行范围

不使用LLM。

输入：
Evidence Extractor输出

输出：
rule-based judgement
"""


def normalize_items(items):

    """
    统一item表示方式
    """

    if items is None:
        return []

    if isinstance(items, str):
        return [items]

    return items



def extract_requested_scope(evidence):

    """
    从Evidence中提取用户要求范围

    注意：
目前是简单版本，
后续可以根据tau2格式增强
    """

    requested = []


    for item in evidence.get(
        "user_intent",
        []
    ):

        text = item.lower()


        if "only" in text:
            requested.append(text)


        if "exchange" in text:
            requested.append(text)


        if "cancel" in text:
            requested.append(text)


    return requested



def extract_agent_scope(evidence):

    """
    提取agent执行范围

    第一版通过agent_actions和tool_actions判断
    """

    actions=[]


    for item in evidence.get(
        "agent_actions",
        []
    ):

        actions.append(
            item.lower()
        )


    for item in evidence.get(
        "tool_actions",
        []
    ):

        actions.append(
            item.lower()
        )


    return actions



def check_scope_confirmation(evidence):

    """
    Scope checker主函数


    返回:

    {
        "has_scope_failure": bool,
        "evidence": [],
        "reason": ""
    }

    """

    requested_scope = (
        extract_requested_scope(
            evidence
        )
    )


    agent_scope = (
        extract_agent_scope(
            evidence
        )
    )


    result = {

        "has_scope_failure": False,

        "evidence": [],

        "reason": ""

    }



    # 没有足够信息

    if not requested_scope:

        result["reason"] = (
            "No explicit user scope found."
        )

        return result



    # 简单规则：
    # 用户明确only
    # agent进行了更宽泛操作

    for req in requested_scope:


        if "only" in req:


            for action in agent_scope:


                if (
                    "all" in action
                    or
                    "entire order" in action
                    or
                    "all items" in action
                ):


                    result["has_scope_failure"] = True


                    result["evidence"] = [
                        req,
                        action
                    ]


                    result["reason"] = (
                        "Agent action exceeded "
                        "user confirmed scope."
                    )


                    return result



    result["reason"] = (
        "No scope mismatch detected."
    )


    return result