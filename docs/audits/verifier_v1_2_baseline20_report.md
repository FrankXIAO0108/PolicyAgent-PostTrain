# Retail Baseline20 Verifier V1.2 验证记录

## 一、实验目的

本实验将 Policy Grounding Verifier V1.2 集成到 PolicyAgent 项目中，对 Retail Baseline20 的 20 个任务进行自动验证。

验证重点包括：

- Latest Intent（最新用户意图）
- Explicit Confirmation（显式确认）
- Policy Compliance（策略遵循）
- Action Result Truthfulness（执行结果真实性）

目标是验证 Verifier 是否能够发现 Benchmark Reward 无法反映的策略问题。

---

## 二、实验配置

### 数据集

Retail Baseline20（20 个固定任务）

### 实验目录

```
experiments/20260722_110504_retail_baseline20_trial1_deepseek
```

### Verifier

```
Policy Grounding Verifier V1.2
```

### 运行命令

```bash
python -m src.verifiers.policy_grounding_v1 experiments/20260722_110504_retail_baseline20_trial1_deepseek
```

### 输出结果

```
data/verifier_gold/verifier_v1_2_baseline20.json
```

---

## 三、验证结果统计

共验证任务：

```
20
```

统计结果如下：

| Verdict | 数量 |
| -------- | ----: |
| PASS | 0 |
| REVIEW | 11 |
| FAIL | 9 |

可以看出，Verifier 的评估标准明显比 Benchmark Reward 更严格。

Benchmark Reward 主要关注任务是否完成，而 Verifier 更关注策略执行是否符合规范。

---

## 四、与 Benchmark Reward 对比

Baseline Trial1 的结果如下：

| Reward | 数量 |
| ------- | ----: |
| Success | 16 |
| Failure | 4 |

Verifier V1.2 的结果如下：

| Verdict | 数量 |
| -------- | ----: |
| PASS | 0 |
| REVIEW | 11 |
| FAIL | 9 |

说明：

- 即使任务最终完成，也可能存在策略违规，因此会被判定为 REVIEW 或 FAIL。
- Verifier 与 Benchmark Reward 属于两个不同维度的评估指标。

---

## 五、典型案例分析

### Task 59

Benchmark Reward：

FAIL

人工分析：

该任务主要属于 User Simulator 与 Golden Answer 不一致，并非 Agent 本身的策略错误。

Verifier 结果：

REVIEW

分析：

Verifier 没有直接判定为 FAIL，与人工分析基本一致。

---

### Task 95

Benchmark Reward：

FAIL

人工分析：

Agent 错误理解了商品规格信息，提前转人工，属于真实 Agent 推理错误。

Verifier 结果：

REVIEW

分析：

Verifier 只发现了工具调用规范问题，没有识别商品规格理解错误。

说明当前版本更关注流程规范，而不是商品语义理解。

---

### Task 98

Benchmark Reward：

FAIL

人工分析：

用户只要求取消滑板订单，Agent 却取消了整个订单。

Verifier 结果：

FAIL

分析：

Verifier 成功识别出了策略违规。

---

### Task 107

Benchmark Reward：

FAIL

人工分析：

Agent 执行了违反 Retail Policy 的换货操作。

Verifier 结果：

FAIL

分析：

Verifier 成功识别了策略违规。

---

## 六、能力分析

目前能够覆盖：

- Latest Intent 检查
- Explicit Confirmation 检查
- Tool 调用规范检查
- Policy Compliance 检查
- Action Result Truthfulness 检查

目前仍存在不足：

- 商品规格理解能力不足
- 商品语义推理能力不足
- SKU / Variant 等细粒度语义无法识别
- 无法判断复杂业务逻辑中的推理错误

---

## 七、本阶段结论

本次实验完成了 Verifier V1.2 在 Retail Baseline20 上的集成与验证。

主要完成内容包括：

- 完成 Verifier V1.2 集成
- 完成命令行运行验证
- 完成 Baseline20 全量验证
- 成功生成验证结果文件
- 完成统计分析
- 完成典型案例分析

实验结果表明，Verifier 能够有效发现策略执行过程中的违规行为，但对于涉及商品语义理解和复杂推理的问题仍存在一定局限。

后续可考虑结合语义推理或 LLM Judge，对 Verifier 进行进一步增强。

---

## 八、相关产物

实验目录：

```
experiments/20260722_110504_retail_baseline20_trial1_deepseek
```

验证结果：

```
data/verifier_gold/verifier_v1_2_baseline20.json
```

Verifier 版本：

```
Policy Grounding Verifier V1.2
```