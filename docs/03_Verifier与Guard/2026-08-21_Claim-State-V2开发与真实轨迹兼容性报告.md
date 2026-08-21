# Claim-State V2 开发与真实轨迹兼容性报告

日期：2026-08-21

状态：开发版完成；尚未进行新的 untouched holdout 评测

## 1. 修改目标

V1 对抗评测暴露两类结构性错误：金额 claim 依赖固定语序，以及相邻订单共享 120 字符窗口造成
跨实体污染。V2 不扩大窗口或追加 task67 专用正则，而是改为局部 claim span：

- 按句号、问号、感叹号、分号和换行切分局部 span；
- 只有一个订单的 span 才允许自动绑定状态或金额；
- 金额可以位于订单号之前或之后；
- 一个 span 有多个订单或多个金额时默认 `REVIEW`；
- 否定表达和比较选择继续 `REVIEW`；
- V1 函数保留，历史实验仍可复现。

## 2. 开发集结果

V2 使用单独的 20 条 development cases，覆盖语序、多订单、多金额、否定、比较、缺失事实和组合
claim。该数据明确标记：

```text
training_allowed = false
rule_tuning_allowed = true
```

结果为 `20/20` 精确匹配。这个数字只说明代码符合当前开发规格，不能解释为 verifier 泛化率，
也不能替代新的 holdout。

开发集 SHA-256：

```text
950598CACCCEF248C38DA9AA18C5A2E7306D9E48BE7F60F747789965FC79FFF1
```

## 3. 60 条冻结真实轨迹兼容性诊断

V2 只读回放 seed19/seed20 的 60 条既有轨迹，不重新调用模型：

| 指标 | 结果 |
| --- | ---: |
| 轨迹总数 | 60 |
| Tau2 成功轨迹 | 32 |
| PASS | 1 |
| REVIEW | 42 |
| NOT_APPLICABLE | 17 |
| FAIL | 0 |
| 成功轨迹被判 FAIL | 0 |
| 翻转任务 | 8 |
| 可评价翻转任务 | 4 |
| 成功轨迹优先 | 1（task67） |

正面结果是 V2 没有在这 32 条成功轨迹上制造假 FAIL。负面结果同样重要：它极度保守，60 条中
只有 1 条 `PASS`，没有任何 `FAIL`。因此它目前不能提供足够密度的负 Reward，也不能因为“零假
FAIL”就宣称可用。

这些轨迹已经参与过 Reward discovery，不是 untouched holdout；上述数字只能检查兼容性，不能
估计 Precision、Recall 或泛化能力。

## 4. 对抗式审查

V2 当前仍有四个主要风险：

1. 基于标点切 span，长句、列表和缺标点回答可能大量进入 `REVIEW`；
2. 同一 span 多订单一律 abstain，牺牲了多实体覆盖率；
3. 状态词仍依赖有限模式，不能覆盖自由表达；
4. 真实轨迹中没有自动 `FAIL`，说明信号稀疏，直接接 Reward 可能几乎不起作用。

因此 `ready_for_reward_penalty=false` 保持不变。

## 5. 下一步门禁

下一步必须新建一套未参与 V2 开发的 holdout，并由项目所有者审阅预期标签。该集合至少需要：

- 新措辞的正确/错误状态和金额；
- 无标点、列表、长句和 Markdown；
- 单句多实体但可明确配对的表达；
- 否定、条件、计划与已完成动作的区别；
- 缺失或冲突 Tool observation；
- 正确流程中的高风险假阳性检查。

只有新 holdout 达到门禁，才讨论把高置信 `FAIL` 接成小权重 penalty。即使通过，也必须继续报告
覆盖率，因为大量 `REVIEW` 代表信号可能过稀，无法支撑有效 GRPO。

## 6. 证据与测试

- 实现：`src/training/teacher_evidence_pack.py`
- 开发集：`data/claim_state_v2_development.json`
- 开发评测：`experiments/20260821_claim_state_v2_development/development_evaluation.json`
- Manifest：`experiments/20260821_claim_state_v2_development/manifest.json`
- 真实轨迹诊断器：`src/evaluation/claim_state_trajectory_audit.py`
- 聚焦测试：`32 passed, 3 subtests passed`
- 全量回归：`286 passed, 2 skipped, 1 warning, 12 subtests passed`

warning 为 Python `audioop` 弃用提示，与本轮实现无关。
