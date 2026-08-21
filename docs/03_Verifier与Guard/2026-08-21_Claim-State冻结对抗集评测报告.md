# Claim-State 冻结对抗集评测报告

日期：2026-08-21

状态：v1 checker 未通过接入过程 Reward 的门禁

## 1. 为什么做这次评测

过程 Reward 离线审计发现，task67 的成功与失败轨迹在数据库、动作和 Tool error 上完全相同，
差异只存在于最终自然语言中的订单选择。现有 claim-state checker 在该 pair 上可以把成功轨迹
判为 `PASS`、失败轨迹路由为 `REVIEW`，但单个已知 case 不能证明规则可以泛化。

本轮因此冻结 checker，不继续围绕 task67 调正则；另建一个训练禁用的合成对抗集，检查：

- 明确订单状态的正确与错误陈述；
- 明确订单金额的正确与错误陈述；
- 否定表达、缺失事实和多笔支付；
- “最新订单”等比较选择；
- 多订单局部绑定；
- 不包含可核验 claim 的文本。

## 2. 数据和口径

- 数据：`data/claim_state_holdout_v1.json`；
- 样本数：24；
- 数据 SHA-256：`D49CA938D8624FDD770A4867B76487EB58550AAC7A1C9F15CC5B7DBFDFF97C93`；
- 训练使用：禁止；
- 规则调参使用：禁止；
- 外部模型调用：0；
- checker：`src.training.teacher_evidence_pack.claim_state_consistency`；
- 评测器：`src.evaluation.claim_state_holdout`。

预期标签由项目实现者根据样本中显式给出的状态和金额构造，不是独立业务专家 gold。因此本报告
只能作为程序规则的工程对抗测试，不能作为生产 Precision/Recall 结论。

## 3. 结果

| 指标 | 结果 |
| --- | ---: |
| 精确标签匹配 | 20/24（83.33%） |
| FAIL Precision | 71.43% |
| FAIL Recall | 83.33% |
| FAIL F1 | 76.92% |
| 可判定样本覆盖率 | 84.62% |
| 可判定样本选择性准确率 | 90.91% |
| REVIEW 路由召回率 | 88.89% |
| 假 FAIL | 2 |
| 应 REVIEW 却自动 PASS | 0 |

预设门禁要求 FAIL Precision ≥ 95%、FAIL Recall ≥ 90%、REVIEW 路由率 ≥ 90%、
假 FAIL 为 0、应 REVIEW 却 PASS 为 0。本轮只有最后一项通过：

```text
ready_for_reward_penalty = false
```

## 4. 四个错误及根因

### 4.1 两个金额 claim 漏检

`amount_pass_tool_observation_06` 与 `amount_fail_wrong_comma_12` 都采用：

```text
The amount paid for order #W... was $...
```

当前金额匹配先从订单号开始切局部 segment，只检查订单号之后的金额关键词。金额语义出现在订单号
之前，因此正确和错误金额都被判为 `NOT_APPLICABLE`。这说明当前 parser 依赖固定表述顺序，存在
确定性的召回缺口。

### 4.2 两个多订单窗口污染

`review_partial_multi_order_19` 和 `status_pass_multi_order_20` 包含相邻两个订单。当前状态检查为每个
订单截取前后 120 个字符，再在窗口中搜索状态短语；窗口覆盖了相邻订单的句子，导致一个订单的
状态被错误绑定到另一个订单：

- 一条本应 `REVIEW` 的部分可评价文本被判 `FAIL`；
- 一条两个状态都正确的文本也被判 `FAIL`。

这是 verifier 假阳性，若直接作为负 Reward，会惩罚正确的多订单回答。

## 5. 结论与决策

本轮不修 checker，也不重跑同一对抗集后宣称通过。原因是该数据已经暴露过错误，继续针对它修改
会把 holdout 变成开发集。

当前可靠结论是：

1. v1 checker 对简单单订单显式状态和“订单号在金额前”的金额表达具有一定检测能力；
2. 它尚不具备稳健的语序泛化和多实体局部绑定能力；
3. `CONTRADICTED` 暂不能接入 GRPO 标量 Reward；
4. Agentic GRPO 启动门禁继续关闭。

## 6. 下一步

下一版本应在独立 development cases 上改为实体级 claim span 解析，而不是扩大字符窗口：

1. 同时支持“金额在订单号前/后”的局部关系抽取；
2. 按句子或结构化 span 绑定订单、金额和状态，避免跨订单污染；
3. 保留 `REVIEW` 作为无法安全判定时的默认结果；
4. 冻结 v2 后重新建立新的、未参与修改的 holdout，禁止用本 v1 对抗集调阈值或宣称泛化；
5. 只有新 holdout 通过门禁，才讨论把高置信 `FAIL` 作为 Reward penalty。

## 7. 证据

- Manifest：`experiments/20260821_claim_state_holdout_v1/manifest.json`
- 完整逐条结果：`experiments/20260821_claim_state_holdout_v1/evaluation.json`
- 结果 SHA-256：`CDBD1162B87B6310FB323A2E18A6983C5A2E0C2EA454EDDFC47CE5DCFDB48F85`
- 全量回归：`282 passed, 2 skipped, 1 warning, 12 subtests passed`

warning 为 Python `audioop` 弃用提示，与本轮代码无关。
