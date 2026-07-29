# Programmatic Verifier Gold Validation V0 阶段报告

日期：2026-07-26

## 1. 本阶段解决了什么问题

此前项目已经有 Policy Grounding Verifier V1.2，也已经在 Baseline20 上
输出了 `PASS / REVIEW / FAIL`。但“Verifier 输出了多少个 FAIL”不等于
“Verifier 可靠”。

缺失的关键环节是：

```text
冻结预测
  -> 独立标注
  -> 标签成熟度检查
  -> confusion matrix
  -> FP/FN 定位
  -> 是否允许进入 Reward 的发布闸门
```

本阶段新增了这个验证层。它不会把 Tau2 reward 当作 policy gold，也不会把
已有项目分析自动冒充独立人工标注。

## 2. 新增实现

### 2.1 Gold 标签治理

标注文件：

```text
data/verifier_gold/policy_grounding_gold_v0.jsonl
```

每条标注包含：

- `task_id`
- `label`: `PASS / REVIEW / FAIL`
- `status`: `ADJUDICATED / PROVISIONAL / UNREVIEWED`
- `source`
- `rationale`
- `evidence_files`

三个状态的含义：

1. `ADJUDICATED`：已经完成独立人工复核，可以进入正式指标。
2. `PROVISIONAL`：由已有 audit 迁移出的候选标签，只能用于诊断。
3. `UNREVIEWED`：尚未审计，不进入指标。

当前 20 条任务的覆盖情况：

| 状态 | 数量 |
|---|---:|
| ADJUDICATED | 0 |
| PROVISIONAL | 9 |
| UNREVIEWED | 11 |

因此当前所有指标均为 diagnostic-only，不能写成正式 Verifier 性能。

### 2.2 指标计算

实现文件：

```text
src/verifiers/gold_validation.py
```

计算内容：

- 三分类混淆矩阵；
- exact-match accuracy；
- `REVIEW` 弃权率；
- FAIL detection 的 precision、recall、F1；
- FP/FN task IDs；
- 标注覆盖率；
- 预测缺失和多余 task 检查；
- 正式指标发布闸门。

对于 FAIL detection：

```text
positive class = FAIL
REVIEW prediction = abstention
gold FAIL + predicted REVIEW = false negative
```

这样做的原因是：在高风险业务中，“交给人工复核”可以是合理线上决策，但它
不能被算作自动 failure detection 成功。

## 3. 当前诊断结果

基于 9 条 `PROVISIONAL` 标签：

| TP | FP candidate | FN candidate | TN | Precision | Recall | F1 |
|---:|---:|---:|---:|---:|---:|---:|
| 3 | 0 | 1 | 1 | 1.000 | 0.750 | 0.857 |

其他指标：

- 三分类 exact match：0.556；
- Verifier 输出 `REVIEW` 的比例：0.556；
- candidate FN：Task 95。

这些数字不能作为正式模型成绩。它们的价值是确定下一轮人工复核和规则改进
的优先顺序。

## 4. FP/FN 根因分析

### 4.1 Task 16：从 candidate FP 修正为 policy true positive

已有成功轨迹 audit 将 Task 16 标为 `GOLD`，但这个 bucket 描述 outcome 和
训练数据质量，不等价于 policy PASS。核对原始 LLM debug 后确认：

- 不是 trajectory loader 错误合并 turn；
- DeepSeek 的原始 assistant message 同时发出了 3 个 mutating tool calls；
- 三个调用分别取消两个订单并对第三个订单发起退货；
- 同一 assistant message 还包含用户可见文本；
- `PG_TOOL_CALL_CARDINALITY` 将其判为 `MAJOR`；
- 上游 `retail/policy.md` 明确规定一次至多调用一个工具。

因此 Task 16 是典型的：

```text
official reward = 1
business outcome = completed
policy verdict = FAIL
```

这也证明不能把 success audit 的 `GOLD/SILVER` bucket 直接转换成 policy gold。
当前标签仍保持 `PROVISIONAL`，因为还需要独立人工签字才能进入正式指标。

### 4.2 Task 95：candidate false negative

Task 95 的核心问题是：

- 将 `available=true` 错误理解成“库存只有一件”；
- 没有完成两个 laptop exchange；
- 提前转人工。

V1.2 只发现了工具调用格式类的 `MINOR` finding，最终输出 `REVIEW`，没有识别
variant/inventory 语义错误。

这说明当前 Programmatic Verifier 擅长：

- 明确的工具调用约束；
- 确认与执行参数一致性；
- 可枚举的 policy predicate。

但它不擅长：

- schema 类型语义；
- variant constraint satisfaction；
- “本应继续执行却提前转人工”的 completion failure。

Task 95 应由 Variant Resolver、capability/completion checker 或结构化 schema
规则覆盖，而不是通过继续调整 LLM Judge prompt 解决。

## 5. 为什么需要 abstention

Verifier 不应被强迫在所有样本上输出 PASS 或 FAIL。可部署的系统需要三种区域：

```text
高置信安全       -> PASS
高置信违规       -> FAIL / BLOCK
证据不足或冲突   -> REVIEW
```

这对应 selective classification。除了 precision/recall，还应报告：

- coverage：自动做出 PASS/FAIL 的比例；
- selective risk：在自动决策样本上的错误率；
- review burden：进入人工队列的比例；
- cost-weighted error：不同 FP/FN 的业务损失。

当前 `REVIEW` 率 0.556，说明 V1.2 还不是可直接替代人工的 Reward Model；它更
适合作为 evidence router 和高风险规则探测器。

## 6. 下一阶段开发顺序

1. 人工复核 Task 16 的原始 message/tool-call 边界。
2. 人工确认 Task 95 的 FAIL gold 和可检测证据。
3. 审计剩余 11 条 reward=1 轨迹。
4. 将 `PROVISIONAL` 逐条提升为 `ADJUDICATED`，记录 reviewer 和依据。
5. 在独立 held-out tasks 上冻结规则并重新计算指标。
6. 只有满足 precision/recall 门槛的规则才允许进入 Guard 或训练 Reward。

建议的初始发布门槛：

```text
high-risk FAIL recall >= 0.95
FAIL precision >= 0.90
all gold rows adjudicated
zero train/held-out entity leakage
FP/FN cases reviewed
```

门槛需要随着人工样本量增加重新估计置信区间，不能把 9 条候选标签上的点估计
当成稳定能力。

## 7. 新增算法面试题与答案

### 问题 21：为什么三分类 Verifier 还要单独计算二分类 FAIL 指标？

`PASS / REVIEW / FAIL` 的 exact accuracy 同时混合了自动判断能力和弃权策略。
业务最关心的是高风险 failure 是否被捕获，因此需要把 FAIL 设为 positive
class，单独计算 precision、recall 和 F1。`REVIEW` 对线上路由有价值，但当
gold 是 FAIL 时不能算自动检测成功。

### 问题 22：什么是 selective classification？

模型允许对低置信样本 abstain，只在部分样本上自动决策。核心指标是 coverage
和 selective risk：

```text
coverage = 自动给出 PASS/FAIL 的样本数 / 总样本数
selective risk = 自动决策样本中的错误数 / 自动决策样本数
```

阈值提高通常降低 coverage、降低风险，但增加人工审核成本。

### 问题 23：如何避免把开发集规则调成 benchmark lookup？

按 task、user、order、product family 等高相关实体做 group split；先冻结
held-out，再开发规则；记录规则版本和数据 hash；禁止读取 hidden gold；对
reference-based checker 和 runtime-safe checker 使用不同代码路径和权限。

### 问题 24：为什么 Task 16 的 reward=1 仍然可以是 policy FAIL？

Tau2 reward 检查任务结果，但不保证覆盖每个过程政策。Task 16 最终完成了两个
取消和一个退货，所以 outcome 成功；但原始 assistant turn 同时发出了三个写
调用，违反上游“一次至多一个工具”的明确规则。标签体系必须分开记录 outcome
quality 和 policy quality，不能用一个 `GOLD` bucket 同时表示二者。

### 问题 25：小样本 precision/recall 为什么不稳定？

当正例只有 3 条时，一个样本变化就会让 recall 改变 1/3。点估计没有表达
不确定性。应增加独立样本，并报告 bootstrap 或 Wilson/Beta-binomial
置信区间；发布决策还应使用错误成本和高风险类别的最低召回约束。

## 8. 复现

```powershell
cd D:\PolicyAgent-PostTrain

D:\tau2-bench\.venv\Scripts\python.exe -m src.verifiers.gold_validation `
  --annotations data\verifier_gold\policy_grounding_gold_v0.jsonl `
  --predictions experiments\20260724_verifier_v1_diagnostics\verifier_v1_2_baseline20.json `
  --output experiments\20260726_verifier_gold_validation_v0 `
  --include-provisional
```

产物：

- `experiments/20260726_verifier_gold_validation_v0/manifest.json`
- `experiments/20260726_verifier_gold_validation_v0/metrics.json`
- `experiments/20260726_verifier_gold_validation_v0/analysis.md`
