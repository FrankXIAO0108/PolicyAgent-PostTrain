import json
from pathlib import Path


SUMMARY_FILE = Path(
    "reports/verifier/llm_verifier_summary.json"
)

FAILURE_FILE = Path(
    "reports/verifier/failure_detection_analysis.json"
)

OUTPUT_FILE = Path(
    "reports/verifier/verifier_evaluation_report.md"
)


def load_json(path):

    with open(
        path,
        "r",
        encoding="utf-8"
    ) as f:
        return json.load(f)


def main():

    summary = load_json(
        SUMMARY_FILE
    )

    failure = load_json(
        FAILURE_FILE
    )


    acc = summary["overall_accuracy"]

    taxonomy = summary[
        "failure_type_accuracy"
    ]

    matrix = failure[
        "confusion_matrix"
    ]

    metrics = failure[
        "failure_detection"
    ]


    lines = []


    lines.append(
        "# PolicyAgent-PostTrain LLM Verifier Evaluation 复盘报告\n"
    )


    lines.append(
        """
## 一、实验背景

本实验目标是在 PolicyAgent-PostTrain 项目中构建一个基于 LLM 的 trajectory verifier，
用于自动评估 Agent 执行轨迹是否符合用户意图、工具调用结果以及任务约束。

整体流程：

Agent trajectory
→ Trajectory Parser
→ LLM Verifier
→ 正确性判断与错误类型分析

本阶段主要验证 LLM verifier 对已有 baseline trajectory 的评价能力。
"""
    )


    lines.append(
        """
## 二、实验数据

本次实验使用：

- 数据集：Retail domain baseline trajectory
- 样本数量：20 条 trajectory
- 成功案例：16 条
- 失败案例：4 条

失败案例来源于此前人工 audit：
- task 59
- task 95
- task 98
- task 107
"""
    )


    lines.append(
        """
## 三、LLM Verifier 方法

LLM verifier 输入：

- 用户请求
- Agent 对话轨迹
- Tool 调用过程
- 最终执行结果

输出：

- trajectory 是否正确
- failure type
- 错误原因解释


Failure 类型包括：

- golden_mismatch
- variant_understanding_failure
- scope_confirmation_failure
- policy_violation
"""
    )


    lines.append(
        f"""
## 四、整体评价结果

### 1. 正确性判断

|指标|结果|
|-|-|
|总样本数|{summary['total']}|
|判断正确|{acc['correct']}|
|判断错误|{acc['wrong']}|
|Accuracy|{acc['accuracy']:.3f}|


LLM verifier 在整体 trajectory 正确性判断任务上取得：

**{acc['accuracy']:.2%} accuracy**

说明模型能够较好判断 Agent 行为是否符合任务目标。
"""
    )


    lines.append(
        f"""
## 五、Failure 类型识别能力

Failure taxonomy 分类结果：

|指标|结果|
|-|-|
|分类正确数量|{taxonomy['correct']}|
|分类准确率|{taxonomy['accuracy']:.3f}|


整体分类准确率：

**{taxonomy['accuracy']:.2%}**

相比于简单判断 trajectory 是否正确，
错误类型分类需要理解更细粒度的：

- 用户真实意图
- policy 约束
- tool 行为
- benchmark 标注规则

因此难度更高。
"""
    )


    lines.append(
        f"""
## 六、Failure Detection 分析


混淆矩阵：

| |预测失败|预测成功|
|-|-|-|
|真实失败|{matrix['true_positive']}|{matrix['false_negative']}|
|真实成功|{matrix['false_positive']}|{matrix['true_negative']}|


Failure detection:

- Precision: {metrics['precision']:.3f}
- Recall: {metrics['recall']:.3f}


分析：

当前 verifier 对失败案例识别较为保守：

优点：
- 几乎不会误判正常 trajectory
- Precision 较高

不足：
- 对真实 failure case 存在漏检
- Failure recall 仍需提升
"""
    )


    lines.append(
        """
## 七、失败案例分析

### Task 59

问题类型：

gold:
golden_mismatch

verifier:
variant_understanding_failure

分析：

LLM 能够识别 trajectory 存在问题，
但对于 benchmark 标注中的具体失败原因理解不足。

---

### Task 95

问题类型：

gold:
variant_understanding_failure

verifier:
none

分析：

LLM 认为 Agent 转人工处理符合要求，
但 benchmark 认为 Agent 在商品 variant 理解阶段已经出现错误。

说明 verifier 需要进一步结合 policy 和任务目标判断。


---

### Task 98

问题类型：

gold:
scope_confirmation_failure

verifier:
none

分析：

LLM 更关注最终结果是否完成，
但忽略了执行过程中是否严格限制用户指定范围。


---

### Task 107

问题类型：

gold:
policy_violation

verifier:
none

分析：

该案例反映：

tool execution success
不代表
policy compliance。

Verifier 需要增强对于 policy constraint 的检查能力。
"""
    )


    lines.append(
        """
## 八、问题总结与优化方向

当前 LLM verifier 已具备：

- trajectory 正确性判断能力
- 基础 failure 分类能力
- 自动生成解释能力


主要问题：

1. Failure recall 较低

需要增加：

- failure-oriented prompting
- policy constraint checklist


2. 对 benchmark golden 标注理解不足

需要强化：

- expected behavior
- forbidden behavior
- tool action scope


3. 缺少显式 policy grounding

后续可以加入：

trajectory
+
policy rules
+
tool logs

联合验证。


## 九、实验结论

本实验成功构建了一个基于 LLM 的 Agent trajectory verifier。

实验结果表明：

- 整体正确性判断 Accuracy 达到较高水平
- 对正常 trajectory 具有稳定判断能力
- 对复杂 failure case 仍存在漏检问题

后续优化方向主要集中在：
提升 failure detection recall，
以及增强 policy-aware verification 能力。
"""
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

        f.write(
            "\n".join(lines)
        )


    print(
        f"Saved: {OUTPUT_FILE}"
    )


if __name__ == "__main__":
    main()
    