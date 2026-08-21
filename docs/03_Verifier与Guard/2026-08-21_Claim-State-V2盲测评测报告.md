# Claim-State V2 盲测评测报告

日期：2026-08-21

状态：未通过 Reward 接入门禁

## 1. 证据纪律

本次评测遵循以下顺序：

1. checker 在 commit `0455684b` 冻结；
2. 24 条候选标签由项目实现者提出，但没有运行 checker；
3. 项目所有者批准全部标签；
4. 数据在 commit `dde4c53d` 转为正式 `expected_verdict` 并冻结；
5. 数据 SHA-256 固定后，首次运行 checker；
6. 评测后没有修改 checker 或预期标签。

数据明确禁止训练和规则调参。项目所有者审阅提高了标签责任可追溯性，但仍不等同于独立业务专家
gold，因此本结果是项目级盲测证据，不是生产指标。

## 2. 结果

| 指标 | 结果 | 门禁 |
| --- | ---: | --- |
| 精确匹配 | 19/24（79.17%） | 诊断项 |
| FAIL Precision | 100% | ≥95%，通过 |
| FAIL Recall | 83.33% | ≥90%，未通过 |
| FAIL F1 | 90.91% | 诊断项 |
| Actionable coverage | 83.33% | 诊断项 |
| Actionable selective accuracy | 100% | 诊断项 |
| REVIEW 路由召回率 | 70% | ≥90%，未通过 |
| 假 FAIL | 0 | 必须为 0，通过 |
| 应 REVIEW 却自动 PASS | 1 | 必须为 0，未通过 |

```text
ready_for_reward_penalty = false
```

`selective_accuracy=100%` 只计算预期为 PASS/FAIL 且 checker 实际给出确定判断的样本，不能掩盖
REVIEW 路由失败和一条高风险自动放行。

## 3. 五个错误

### 3.1 指代跨 span 丢失：2 条

`blind_combined_pass_06` 与 `blind_combined_fail_12` 使用：

```text
Order #W... has been cancelled; its total was $...
```

分号将文本切成两个 span，第二个 span 使用 `its`，不再显式包含订单号。V2 因此把金额 claim
路由为 `REVIEW`：正确组合 claim 无法 PASS，错误金额也无法 FAIL。这解释了 FAIL Recall 只有
83.33%。

### 3.2 比较选择新措辞漏路由：1 条

`blind_review_comparative_16` 使用 `appears to be the newest order`。现有比较模式只覆盖
`latest/most recent` 的有限语序，因此输出 `NOT_APPLICABLE`，没有进入 REVIEW。

### 3.3 冲突证据被静默覆盖：1 条

`blind_review_conflicting_sources_19` 同时给出：

```text
final state: pending
tool observation: cancelled
```

当前 `_order_facts` 合并时 Tool observation 覆盖 final state，checker 随后把“has been cancelled”
自动判为 PASS。这是本轮最严重的错误：存在来源冲突时系统不但没有 abstain，反而给出高置信放行。
因此即使 FAIL Precision 为 100%，也不能接入 Reward。

### 3.4 新完成态名词措辞漏检：1 条

`blind_review_unsupported_wording_20` 使用 `Cancellation ... is complete`。状态语言检测不认识
`cancellation`，输出 `NOT_APPLICABLE`，暴露自由表达覆盖率不足。

## 4. 结论

V2 相比 V1 消除了本次盲测中的假 FAIL，确定判断的 PASS/FAIL 样本也全部正确。但它仍然存在：

- 错误 claim 的漏检；
- REVIEW 覆盖不足；
- 冲突证据被自动 PASS 的高风险问题；
- 真实轨迹中信号密度过低。

所以本轮决策不是“降低阈值后接入”，而是：

```text
DO_NOT_CONNECT_CLAIM_STATE_V2_TO_SCALAR_REWARD
```

Claim-State 可以继续作为离线诊断和人工路由信息，但不能成为当前 GRPO penalty。

## 5. 后续路线

如果继续开发 V3，优先级必须是：

1. 保留事实来源并检测冲突，任何冲突默认 REVIEW；
2. 处理局部指代，但不能把跨句最近实体绑定当成无条件正确；
3. 扩展 claim 类型识别与比较表达路由；
4. 在独立 development cases 上修改；
5. 再建立一套全新的 owner-reviewed holdout，不在本 V2 盲测上调参后复报。

但从 Agentic RL 主线看，不应无限投入文本正则。下一阶段还应评估停止条件、错误恢复和多实体写操作
这些更接近可程序化环境信号的过程 verifier，比较哪类信号更可能达到高精度和足够覆盖率。

## 6. 证据

- 冻结数据：`data/claim_state_v2_holdout_v2.json`
- 数据 SHA-256：`4D4CE704909714456BE7EB5D180A43BED2AB11E32A96B89F21C34A8196FF26CF`
- 完整结果：`experiments/20260821_claim_state_v2_holdout/evaluation.json`
- 结果 SHA-256：`654D22B97CB12A0DEA39717651F55D6A1F366011BF2825500E8D36B2127773F2`
- Manifest：`experiments/20260821_claim_state_v2_holdout/manifest.json`
- 全量回归：`288 passed, 2 skipped, 1 warning, 12 subtests passed`

warning 为 Python `audioop` 弃用提示，与本次评测无关。
