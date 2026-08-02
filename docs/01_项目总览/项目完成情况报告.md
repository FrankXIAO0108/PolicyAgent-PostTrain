# PolicyAgent-PostTrain 项目完成报告

日期：2026-07-29

## 1. 项目定位

本项目是建立在上游 `sierra-research/tau2-bench` Retail 环境之上的 Tool
Agent 可靠性工程。上游提供任务、策略、工具、数据库、环境和基础评测；本项目
负责实验冻结、轨迹审计、状态重放、失败归因、政策验证、执行前防护和后训练
数据治理。

项目解决的核心问题是：

> 最终 reward 正确，不代表执行轨迹经过授权、符合政策、没有扩大操作范围，
> 也不代表 Agent 对最终状态的描述真实。

## 2. 已完成的可验证成果

### 2.1 冻结开发基线

- 20 个 Retail 开发任务全部产生有效结果；
- 16 个业务成功，4 个业务失败，0 个系统失败；
- 失败任务为 59、95、98、107；
- 模型、温度、种子、任务集合、项目版本、上游版本和原始输出均有记录；
- 该结果是开发基线，不是官方排行榜成绩。

### 2.2 确定性 Hybrid Evaluation V7

V7 重放 Agent 和目标工具操作，重建两份最终数据库状态，再结合冻结的官方
NL assertion 结果输出：

1. 官方失败信号；
2. 结构化状态差异；
3. 主要因果根因和次要发现；
4. 客户与业务影响；
5. 是否建议隔离。

在同一冻结开发集上，历史 V6 轨迹 LLM pipeline 漏报 4 个失败，V7 重放结果
与 20 个冻结官方结果一致。该数字证明的是冻结数据重放一致性，不是未见任务
泛化能力。

### 2.3 三个代表性业务案例

| Task | 失败 | 根因 | 业务风险 |
|---|---|---|---|
| 95 | 两个换货目标未完成 | 把布尔型 availability 误解为库存数量 | 错误能力判断、请求未完成、沟通遗漏 |
| 98 | 支付方式写错，并存在商品请求扩大为整单取消的风险 | payment 与 scope 错误 | 钱款流向错误、越权影响订单 |
| 107 | 换成错误变体，并提交新旧商品相同的换货 | variant 与 policy 错误 | 错商品、业务规则被工具绕过 |

Task 59 没有被强行当作普通负样本。它包含用户最终意图与静态 gold 的冲突，
因此被标记为数据对齐风险并建议隔离。

### 2.4 Programmatic Policy Verifier V2.2

Verifier 已覆盖授权、确认、工具参数、范围、变体、重复写操作、最终陈述与工具
状态一致性等维度。20 条开发轨迹已有分析员 provisional 标签，V2.2 与它们
一致。

但所有标签仍为 `PROVISIONAL`，独立人工裁决数量为 0，且规则在同一开发集上
迭代。因此本项目不把这一结果声明为 held-out 精度或人工金标性能。

### 2.5 Deterministic Pre-action Guard

Guard 位于 LLM 工具提案与环境执行之间：

```text
LLM proposal
-> deterministic Guard
-> ALLOW / REQUIRE_CONFIRMATION / REGENERATE / BLOCK / TRANSFER
-> tool execution
```

Runtime-safe 模式只读取用户范围、已观察工具状态和政策，不读取 gold。离线审计
中，它拦截了 Task 95、98、107 三个非隔离失败。Reference diagnostic 与运行时
模式严格隔离，不能被描述为可部署规则。

离线拦截只证明原错误操作可以在执行前被识别，不证明模型重新生成后一定成功。

### 2.6 后训练数据治理

项目已实现：

- 轨迹质量裁决；
- 修正轨迹完整性和哈希检查；
- 双人审批与作者回避；
- TRAIN/VALIDATION 实体泄漏检测；
- 官方测试集隔离；
- assistant-only loss mask；
- SFT 数据发布门禁；
- SFT、DPO、RLHF/GRPO readiness gate。

这些门禁当前保持关闭，因为没有独立人工政策金标。项目没有运行或虚构 SFT、
DPO、RLHF、GRPO 结果。

## 3. 当前完成边界

| 能力 | 状态 |
|---|---|
| 开发基线与原始轨迹 | 完成 |
| 失败分类与业务影响 | 完成 |
| 确定性状态重放 | 完成 |
| Programmatic Verifier 开发版 | 完成 |
| Runtime Guard 原型与离线审计 | 完成 |
| 后训练数据门禁 | 完成 |
| 独立人工政策金标 | 未获得 |
| 正式 SFT 数据集 | 门禁关闭 |
| SFT 与冻结重评测 | 未运行 |
| DPO / RLHF / GRPO | 未运行且当前无充分依据 |

因此，项目当前完成的是一个可复现的 Tool Agent Reliability System，而不是
一个已经证明后训练提升的项目。

## 4. 工程证据

- 冻结开发实验：
  `experiments/20260722_110504_retail_baseline20_trial1_deepseek`
- V7 结果：`reports/evaluation/final_report.json`
- V6/V7 对比：
  `experiments/20260726_v6_vs_v7_evaluation/comparison.json`
- Guard 审计：
  `experiments/20260726_pre_action_guard_v1/guard_audit.json`
- 一键展示入口：`python -m src.portfolio_demo`
- 自动化测试：74 项通过

## 5. 后续路线

### 可立即推进

1. 保持当前冻结结果不变，完善演示和面试表达；
2. 增加新的、严格隔离的测试任务，验证 V7/Verifier/Guard 泛化；
3. 在有预算时进行 Guard 在线 A/B，测量拦截后的恢复率和业务效用；
4. 补充 token、延迟、成本和人工复核时间等生产指标。

### 获得可靠监督后推进

1. 收集独立政策金标；
2. 验证 Verifier 的 precision、recall、F1、FP 和 FN；
3. 裁决轨迹质量并发布严格拆分、可追溯的 SFT 数据；
4. 执行 SFT 和冻结 Base-vs-SFT 对比；
5. 只有当残余错误和奖励可靠性支持时，才考虑 DPO 或 GRPO。

## 6. 求职价值

这个项目可以证明候选人不仅会调用 Agent 框架，还能处理真实系统中更难的
问题：

- 状态一致性与工具副作用；
- 用户授权和操作范围；
- Policy、Tool 与 Evaluator 之间的覆盖缺口；
- 错误数据对后训练的放大风险；
- 可复现实验和可信结果边界；
- 在监督信号不足时做出正确的 go/no-go 决策。
