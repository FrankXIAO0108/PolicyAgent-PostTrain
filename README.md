# PolicyAgent-PostTrain

面向政策约束型工具智能体（Tool Agent）的可靠性评测与执行防护项目，构建于
上游 [tau2-bench](https://github.com/sierra-research/tau2-bench) Retail
环境之上。

本项目关注一个真实业务问题：

> 对话听起来合理、工具返回成功，甚至最终 reward 为 1，都不等于 Agent
> 获得了正确授权、执行了正确副作用、遵守了业务政策并如实告知用户。

因此，项目没有直接跳到 SFT 或强化学习，而是先建立可信的证据链：

```text
冻结基线
→ 轨迹与最终状态重放
→ 结构化差异
→ 失败根因与业务影响
→ Programmatic Verifier
→ 执行前 Guard
→ 训练数据质量门禁
→ 具备可靠监督后再进入 SFT / DPO / GRPO
```

## 30 秒离线演示

演示不需要 API Key、不调用模型、不付费，也不需要重新运行 tau2。它只读取
仓库中冻结并带哈希的报告：

```powershell
cd <PolicyAgent-PostTrain 仓库路径>
.\demo.ps1
```

等价的 Python 命令：

```powershell
python -m src.portfolio_demo
```

演示内容包括：

- 20 个冻结任务的基线结果；
- V6 轨迹 LLM pipeline 与 V7 确定性重放对比；
- Task 95、98、107 三个代表性业务失败；
- Runtime-safe Guard 的具体拦截规则；
- SFT、DPO、RLHF/GRPO 尚未运行的原因；
- 所有结果的实验边界。

求职材料：

- [项目完成报告](docs/PROJECT_COMPLETION_REPORT.md)
- [面试讲解手册与中英文简历表述](docs/INTERVIEW_PLAYBOOK.md)
- [中文冻结演示输出](reports/portfolio_demo_zh/demo.md)

## 项目与上游的边界

本仓库没有从零实现 tau2-bench。

上游 tau2-bench 提供：

- Retail 任务与政策；
- 工具、数据库和环境；
- Agent/User 交互编排；
- 官方基础评测与 Gym 接口。

本项目实现：

- 风险分层任务选择与冻结实验；
- 原始轨迹、版本和配置哈希管理；
- Agent/Gold 工具操作重放；
- 最终数据库结构化差异；
- 官方失败信号重建；
- 失败根因与业务影响映射；
- Programmatic Policy Verifier；
- Runtime-safe Pre-action Guard；
- 轨迹修正、SFT 发布与训练阶段门禁；
- 面向真实业务的案例审计和可复现报告。

## 当前可验证结果

冻结的 20 任务 Retail 开发基线：

- 20/20 产生有效结果；
- 16 个业务成功；
- 4 个业务失败；
- 0 个系统失败；
- 失败任务：59、95、98、107；
- 总模型成本：`$0.0447107304`；
- 总模拟时间：`506.2374537` 秒。

这是从训练集选取的冻结开发基线，不是官方排行榜成绩。

### V6 与 V7 对比

| 评测方法 | 准确率 | 失败召回率 | FP | FN | 新增 LLM 调用 |
|---|---:|---:|---:|---:|---:|
| V6 轨迹 + LLM pipeline | 75% | 0% | 1 | 4 | 40 |
| V7 确定性状态重放 | 100% | 100% | 0 | 0 | 0 |

这些数字的解释边界：

- V6 使用原始 `prediction.has_failure`，没有使用事后修正标签；
- V7 的 100% 表示冻结开发集上的重放一致性；
- 它不表示未见任务上的泛化准确率；
- 四个失败根因在开发中参与过审计，不是 held-out 结果；
- V6 历史产物没有 token 和延迟数据，因此不能补造成本指标。

## 核心架构

### 离线可靠性评测

```text
冻结 returned_results.json
          │
          ├─ 重放 Agent 工具调用 ──→ Agent 最终数据库
          │
          └─ 重放目标工具调用 ────→ 目标最终数据库
                                      │
两份数据库 ───────→ 结构化状态差异 ──┤
                                      │
冻结 NL assertion ───────────────────┘
                  │
                  ▼
          官方失败信号重建
                  │
                  ▼
          主要因果根因 / 次要发现
                  │
                  ▼
              业务影响
```

V7 不使用 LLM 重建确定性 reward。LLM 可以辅助解释语义证据，但不能覆盖状态
重放得到的官方结果信号。

### 在线执行防护

```text
用户请求与已观察状态
          │
          ▼
      LLM 工具提案
          │
          ▼
  Deterministic Pre-action Guard
          │
          ├─ ALLOW ─────────────→ 执行工具
          ├─ REQUIRE_CONFIRMATION → 询问用户
          ├─ REGENERATE ─────────→ 结构化反馈给模型
          ├─ BLOCK ──────────────→ 拒绝危险操作
          └─ TRANSFER ───────────→ 转人工
```

Runtime-safe 模式只使用用户范围、已经观察到的工具状态和业务政策，绝不读取
gold。Reference diagnostic 允许在离线研究中比较 gold，但禁止进入部署路径。

## 三层失败模型

项目把三个容易混淆的概念分开：

1. **官方信号**：`db_mismatch`、`nl_failure`；
2. **失败根因**：`variant_error`、`scope_error`、`payment_error`、
   `policy_error`、`missing_action` 等；
3. **业务影响**：错商品、错退款渠道、扩大订单副作用、请求未完成和政策风险。

主要因果根因与次要政策发现也会分开。数据库中多个字段不一致，可能只是同一个
错误操作的连锁后果，不能把每个字段都当成独立根因。

## 四个失败案例

| Task | 官方信号 | 主要诊断 | 处理方式 |
|---|---|---|---|
| 59 | DB + NL 失败 | 用户最终意图与静态 gold 冲突，同时存在缺失操作 | 隔离，不作为普通训练负样本 |
| 95 | DB + NL 失败 | 把 availability 布尔值误解为库存数量，导致换货缺失 | Guard 阻止无必要转人工 |
| 98 | DB 失败 | 支付方式写错，并存在商品请求扩大为整单操作的风险 | Guard 阻止范围扩大并串行化写操作 |
| 107 | DB 失败 | 变体错误，且新旧商品相同的换货违反政策 | Guard 在工具执行前拦截 |

### 为什么 Task 59 必须隔离

Task 59 的静态目标与用户最后确认的意图存在冲突。把它直接当成模型负样本会让
模型学习错误目标。因此项目将 `dataset_alignment_error` 与普通 Agent failure
分离，并设置 `quarantine_recommended=true`。

## Deterministic Pre-action Guard

离线审计结果：

- Runtime-safe Guard 拦截了 3/4 个官方失败；
- 唯一未按普通失败处理的是被隔离的 Task 59；
- Runtime + reference diagnostic 覆盖 3/3 个非隔离失败；
- 新增 LLM 调用为 0。

Runtime-safe Guard 的代表性规则：

- `goal.transfer_with_actionable_variant`
- `scope.item_request_would_cancel_whole_order`
- `protocol.one_tool_call_per_turn`
- `policy.exchange_requires_different_option`

离线拦截不等于在线恢复成功。要证明真实收益，还需要 A/B 实验测量：

- 模型重新生成后的恢复率；
- 额外 token、延迟和成本；
- Guard 误拦截；
- 最终业务成功率；
- 转人工比例。

## Programmatic Policy Verifier V2.2

Verifier 检查：

- 用户身份与授权；
- 写操作前的明确确认；
- 工具名称和参数；
- 商品、变体、地址和支付方式；
- 商品级请求是否扩大为订单级副作用；
- 重复或并行的一次性写操作；
- 工具结果与最终陈述是否一致。

目前 20 条轨迹都有分析员 `PROVISIONAL` 标签，V2.2 与这批开发标签一致。

但：

- 独立人工裁决标签为 0；
- 规则在相同开发集上迭代；
- 数据中没有可用于正式验证的独立 gold；
- 因此不把该结果声明为 held-out 精度或生产性能。

## 后训练数据治理

项目已经实现但保持关闭的门禁：

- 轨迹质量独立裁决；
- 环境污染与 benchmark 冲突隔离；
- 修正轨迹哈希；
- 行为改变后的环境重放；
- 双人审批与作者回避；
- TRAIN/VALIDATION 用户、订单和商品实体泄漏检测；
- 官方测试集隔离；
- assistant-only loss mask；
- SFT 数据发布；
- Base/SFT 对比绑定；
- DPO 和 RLHF/GRPO readiness 判断。

当前状态：

| 阶段 | 状态 |
|---|---|
| Prompt Baseline | 已完成 |
| 轨迹审计与失败分类 | 已完成 |
| 确定性 Hybrid Evaluation | 已完成 |
| Verifier 开发版 | 已完成 |
| Runtime Guard 原型与离线审计 | 已完成 |
| 独立人工政策金标 | 未获得 |
| 正式 SFT 数据集 | 门禁关闭 |
| SFT 与冻结重评测 | 未运行 |
| DPO | 未运行 |
| RLHF / GRPO | 未运行 |

这是一个明确的工程判断：监督信号不可靠时，不应为了补齐流程而训练。

## 关键代码

```text
src/
├─ portfolio_demo.py               零 API 求职演示
├─ evaluation/
│  ├─ replay_evaluator.py          重放 Agent 与目标工具操作
│  ├─ db_diff.py                   最终数据库结构化差异
│  ├─ nl_checker.py                读取冻结 NL assertion
│  ├─ failure_attributor.py        失败根因
│  ├─ taxonomy.py                  官方信号/根因/业务影响
│  ├─ report_generator.py          JSON 与 Markdown 报告
│  └─ pipeline.py                  V7 评测入口
├─ guards/
│  ├─ retail_pre_action.py         Runtime-safe Retail Guard
│  └─ offline_audit.py             冻结轨迹反事实审计
├─ agents/
│  └─ guarded_llm_agent.py         tau2 兼容的 Guard Agent 适配器
├─ verifiers/
│  ├─ policy_grounding_v2.py       Programmatic Verifier V2
│  ├─ adjudication.py              双人审阅与冲突裁决
│  └─ review_submission.py         审阅提交预检
└─ training/
   ├─ correction_validation.py     修正轨迹校验
   ├─ quality_adjudication.py      轨迹质量裁决
   ├─ sft_decision_builder.py      SFT 决策数据构建
   ├─ sft_release.py               SFT 发布门禁
   └─ readiness_gate.py            SFT/DPO/RL 阶段门禁
```

## 复现

环境：

- Windows
- Python 3.12
- 当前冻结实验在 Windows/Python 3.12 环境生成；
- 完整重放需要单独安装上游 tau2-bench；
- 上游运行 commit：
  `58e5e1ace69302e6982d27014569c03e0ffccdd2`

### 运行测试

```powershell
python -m pytest -q
```

当前结果：74 项测试通过，另有一个来自上游 `audioop` 的 Python 3.13
弃用警告。

### 重放 V7

该命令需要本地上游 tau2 环境：

```powershell
$tau2Root = "D:\path\to\tau2-bench"
python -m src.evaluation.pipeline `
  --experiment experiments\20260722_110504_retail_baseline20_trial1_deepseek `
  --tau2-root $tau2Root `
  --output reports\evaluation
```

输出：

- `reports/evaluation/final_report.json`
- `reports/evaluation/failure_analysis.md`

### 运行 Guard 离线审计

```powershell
python -m src.guards.offline_audit
```

输出：

- `experiments/20260726_pre_action_guard_v1/guard_audit.json`
- `experiments/20260726_pre_action_guard_v1/analysis.md`

### 可选的付费在线 A/B

```powershell
python src\run_retail_guarded_failure3.py
```

该命令会产生新的 Agent、用户模拟器和 NL Judge API 调用，默认不会自动执行。
结果必须重新经过 V7 才能形成恢复率结论。

## 证据入口

- 基线：
  `experiments/20260722_110504_retail_baseline20_trial1_deepseek`
- V7：
  `reports/evaluation/final_report.json`
- V6/V7 对比：
  `experiments/20260726_v6_vs_v7_evaluation/comparison.json`
- Guard：
  `experiments/20260726_pre_action_guard_v1/guard_audit.json`
- Verifier V2.2：
  `experiments/20260727_policy_grounding_v2_2`
- Post-training readiness：
  `experiments/20260728_post_training_readiness_v0/readiness_report.json`

## 后续计划

1. 在严格隔离的新任务上验证 V7、Verifier 和 Guard 的泛化；
2. 运行 Guard 在线 A/B，补齐恢复率、延迟、成本和误拦截；
3. 获得独立政策 gold 后验证 Verifier 的 precision、recall、F1、FP 和 FN；
4. 裁决并发布严格拆分、带哈希的 SFT 数据；
5. 执行 Base 与 SFT 的冻结对比；
6. 只有残余错误和奖励可靠性充分时，才决定 DPO 或 GRPO。

## 结论

本项目当前完成的是一个可复现的 Tool Agent Reliability System，而不是一个
已经证明后训练提升的项目。

它展示的核心能力不是“调用框架跑一次训练”，而是：

- 识别 reward 与真实业务正确性之间的缺口；
- 重建工具调用对持久状态的真实影响；
- 把失败映射到可行动的工程根因；
- 在副作用发生之前建立确定性防护；
- 防止错误标签和 benchmark 冲突进入后训练；
- 对实验结果保持可复现、可审计、不过度声明。
