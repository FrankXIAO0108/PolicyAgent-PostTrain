"""
Variant Rule Checker

检测产品variant理解错误。

核心逻辑：

requested attributes
        VS
selected attributes

如果关键属性不一致：
认为存在variant理解失败。


输入:
Evidence Extractor输出

输出:
rule judgement
"""


def normalize_dict(data):

    if data is None:
        return {}

    if isinstance(data, dict):
        return data

    return {}



def extract_requested_attributes(evidence):

    """
    提取用户要求的产品属性

    第一版：
    从constraints和user_intent中寻找

    后续可以改成结构化extractor
    """

    attributes = {}


    texts = (
        evidence.get("constraints", [])
        +
        evidence.get("user_intent", [])
    )


    for text in texts:

        text = text.lower()


        # 示例规则
        if "i7" in text:
            attributes["cpu"] = "i7"


        if "i5" in text:
            attributes["cpu"] = "i5"


        if "8gb" in text:
            attributes["ram"] = "8gb"


        if "16gb" in text:
            attributes["ram"] = "16gb"


        if "1tb" in text:
            attributes["storage"] = "1tb"


        if "512gb" in text:
            attributes["storage"] = "512gb"


    return attributes



def extract_selected_attributes(evidence):

    """
    提取agent最终选择variant属性

    第一版同样基于文本。
    """


    attributes = {}


    texts = (
        evidence.get("agent_actions", [])
        +
        evidence.get("tool_actions", [])
    )


    for text in texts:

        text=text.lower()


        if "i7" in text:
            attributes["cpu"]="i7"


        if "i5" in text:
            attributes["cpu"]="i5"


        if "8gb" in text:
            attributes["ram"]="8gb"


        if "16gb" in text:
            attributes["ram"]="16gb"


        if "1tb" in text:
            attributes["storage"]="1tb"


        if "512gb" in text:
            attributes["storage"]="512gb"


    return attributes



def check_variant_understanding(evidence):

    """
    主检查函数

    返回:

    {
        has_variant_failure,
        evidence,
        reason
    }

    """


    requested = (
        extract_requested_attributes(
            evidence
        )
    )


    selected = (
        extract_selected_attributes(
            evidence
        )
    )


    result = {

        "has_variant_failure": False,

        "evidence": [],

        "reason": ""

    }



    if not requested:

        result["reason"] = (
            "No explicit variant requirement found."
        )

        return result



    mismatches=[]


    for key,value in requested.items():


        if key in selected:

            if selected[key] != value:

                mismatches.append(
                    {
                        "attribute":key,
                        "requested":value,
                        "selected":selected[key]
                    }
                )



    if mismatches:


        result["has_variant_failure"]=True

        result["evidence"]=mismatches

        result["reason"]=(
            "Selected product variant "
            "does not satisfy user requirements."
        )


    else:

        result["reason"]=(
            "No variant mismatch detected."
        )


    return result