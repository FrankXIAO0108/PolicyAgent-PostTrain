# Retail Guard 合成场景诊断 V1 阶段报告

日期：2026-07-30

## 1. 目标

冻结 Baseline20 的 Guard 审计证明 Task 95、98、107 的已知危险动作可以被离线
识别，但这些规则来自同一批开发失败，不能证明场景迁移能力。

本阶段新增一套与 Baseline20 原始轨迹分离的合成场景诊断，检查同一组
runtime-safe predicate 在新订单号、新商品、新表述和安全负对照上的确定性行为。

## 2. 场景范围

15 个开发者构造场景覆盖：

- 商品级请求扩大为整单取消；
- 订单状态与 Tool 不匹配；
- 未归属支付方式；
- same-item exchange；
- 不可用 variant；
- 跨 product replacement；
- 同轮多个写操作；
- 同订单二次修改；
- 存在可行动 variant 时提前转人工；
- 合法整单取消、合法退货、合法换货和合法转人工等负对照；
- 多只读调用、Tool Call 与文本同轮等非阻断 minor finding。

输入位于：

```text
configs/guard_synthetic_diagnostic_v1.json
```

运行入口：

```powershell
python -m src.guards.scenario_evaluation `
  --suite configs\guard_synthetic_diagnostic_v1.json `
  --output experiments\20260730_guard_synthetic_diagnostic_v1
```

## 3. 结果

| 指标 | 结果 |
|---|---:|
| Case 数 | 15 |
| Exact case accuracy | 15/15 |
| Decision accuracy | 15/15 |
| Blocking-rule exact match | 15/15 |
| 风险场景 TP | 9 |
| 安全对照 TN | 6 |
| FP / FN | 0 / 0 |
| 新增 LLM 调用 | 0 |
| 使用 reference action / gold DB | 否 |

## 4. 工程设计

`src/guards/scenario_evaluation.py` 完成：

1. 校验 case ID 唯一性；
2. 强制 `official_metric=false`；
3. 拒绝 runtime 场景携带 reference action；
4. 从 JSON 恢复 `GuardContext` 和 `ToolProposal`；
5. 计算 decision、blocking-rule exact match 和阻断二分类 P/R/F1；
6. 保存 suite snapshot、results、analysis 和带 SHA-256 的 manifest；
7. 任一 case 不匹配时以非零状态退出。

## 5. 解释边界

这不是独立 held-out，也不是人工 gold：

- 场景是在 Guard V1 和 Policy Grounding V2.2 已存在后由开发者构造；
- 标签是确定性期望，不是业务 reviewer 独立裁决；
- 15/15 证明的是规则回归和有限场景迁移，不是生产 precision/recall；
- 没有执行 LLM regeneration，因此不能证明在线恢复率；
- 不能据此打开 SFT、DPO 或 RLHF/GRPO 门禁。

该结果的正确用途是：

```text
新规则提交前的 deterministic regression
+ runtime/reference 隔离检查
+ 在线 A/B 前的低成本工程预检
```

## 6. 下一步

1. 冻结本次代码、配置和产物；
2. 在不调整 Guard 规则的前提下运行小规模在线 A/B；
3. 测量 blocked proposal 后的模型恢复率、误拦截、最终 task success、延迟和成本；
4. 独立人工 Policy Gold 仍是正式 Verifier 和训练数据发布的必要条件。
