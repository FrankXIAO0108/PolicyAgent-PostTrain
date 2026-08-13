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
- 正式 Retail SFT、DPO、RLHF/GRPO 门禁仍关闭的原因；
- 所有结果的实验边界。

求职材料：

- [全部文档的中文分类导航](docs/README.md)
- [项目完成报告](docs/01_项目总览/项目完成情况报告.md)
- [面试讲解手册与中英文简历表述](docs/05_求职与面试/面试讲解手册.md)
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

### 合成场景诊断 V1

为了检查规则是否只记住原 20 条轨迹，项目增加了 15 个使用新订单、商品和表述的
合成诊断场景，覆盖 scope、order state、payment、variant、protocol、
one-shot mutation、premature transfer，以及 6 个安全负对照。

- 15/15 场景的 Guard decision 与预期一致；
- 9 个风险场景被阻断，6 个安全对照保持允许；
- blocking rule exact match 为 15/15；
- 新增 LLM 调用为 0；
- 不读取 reference action、gold DB 或官方 Tau2 test。

该结果是开发者构造的 deterministic regression，不是独立 held-out 人工 gold，
不能作为生产 precision/recall，也不能打开 SFT 或 RL 门禁。详细报告见
`docs/03_Verifier与Guard/2026-07-30_Guard合成场景诊断报告.md`。

### Guard 规则族消融 V1

项目进一步固定同一套 15 个合成场景，比较 no-Guard、full-Guard 和七个
leave-one-family-out 变体：

- no-Guard 检出 0/9 个风险场景；
- full-Guard 检出 9/9，6/6 安全对照保持允许；
- 去掉 variant 语义规则后检出降为 6/9；
- 去掉 scope、order-state、payment、protocol、one-shot 或 goal-completion
  任一规则族后检出均为 8/9；
- 新增 LLM 调用为 0。

这只量化开发者构造套件中的规则覆盖贡献，不代表生产精度或在线恢复率。报告见
`docs/03_Verifier与Guard/2026-08-02_Guard规则族消融实验报告.md`。

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
| 正式 Retail SFT 与冻结重评测 | 门禁关闭，未运行 |
| 正式 Retail DPO | 门禁关闭，未运行 |
| 正式 Retail RLHF / GRPO | 门禁关闭，未运行 |
| 隔离合成 SFT→DPO→GRPO 工程实操 | 已在单卡 RTX 4090 完成并自动验收 |
| Qwen3-4B 隔离 Tool SFT warmup | 80-step QLoRA、merge 与 20 条同分布 holdout 已完成；不代表业务提升 |
| 隔离真实多轮 Retail Agentic RL | 1-step 工程 sanity 与 Base 模型 32 条无更新 rollout 已完成；尚无有效 RL 改善证据 |

这是一个明确的工程判断：监督信号不可靠时，不应为了补齐流程而训练。

为补齐真实后训练操作经验，仓库另行提供与 frozen Retail 数据完全隔离的
SFT→DPO→GRPO GPU 工程实操包，使用开发者合成工具调用数据、Qwen2.5-0.5B、LoRA、
偏好对和三维程序化 Verifier Reward。该隔离实验已在单卡 RTX 4090 上完成
30-step SFT、20-step DPO、10-step GRPO 及各阶段 merge/冻结评测，自动验收结果为
`verified_complete=true`。实验同时发现 DPO 将精确动作匹配从 SFT 的 100% 降至
50%，GRPO 虽产生非零组内 reward 方差和参数更新，但未恢复该指标。该结果证明真实
工程实操，不代表正式 Retail 业务提升。详见
[云端后训练完整实跑报告](docs/2026-08-02_posttrain_cloud_run_report.md)与
[SFT→DPO→GRPO 工程实操执行手册](docs/04_数据治理与后训练/2026-08-02_SFT-DPO-GRPO工程实操执行手册.md)。

2026-08-11 新增真实多轮 Retail Agentic RL 路线：使用 tau2 Retail 的动态客户、
数据库工具和终态检查，将一对一必要动作进度、沟通完成度、工具错误、重复调用与非预期
写操作组成轨迹级过程奖励。44 条 train、10 条 validation、既有 20 条 development audit
已严格拆分，官方 test 保留。云端已完成 1-step 工程 sanity，以及 Qwen3-4B Base 的
8-task、每任务 4 候选、共 32 条无权重更新 rollout 诊断。该诊断出现 0/32 工具调用，
因此不支持直接扩大 GRPO。随后完成隔离 Tool SFT warmup；其 20 条同分布合成 holdout
工具协议指标由 0% 提升到 100%，但 Tool-SFT 后的 32-rollout 复测尚未获得本地完成证据。
因此当前只能声称“真实环境、过程奖励、诊断和协议 SFT 已实跑”，不能声称 Agentic GRPO
带来业务提升。详见
[Retail 智能体强化学习设计](docs/04_数据治理与后训练/2026-08-11_Retail智能体强化学习设计.md)和
[Agent RL 云端运行手册](docs/04_数据治理与后训练/2026-08-11_Agent-RL云端运行手册.md)。

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
├─ rl/
│  ├─ retail_agentic_env.py        真实多轮 Retail RL 环境与过程奖励
│  ├─ task_split.py                44/10/20 冻结任务拆分
│  └─ prepare_user_openings.py     动态客户首轮话术冻结与成本记录
├─ verifiers/
│  ├─ policy_grounding_v2.py       Programmatic Verifier V2
│  ├─ adjudication.py              双人审阅与冲突裁决
│  └─ review_submission.py         审阅提交预检
└─ training/
   ├─ correction_validation.py     修正轨迹校验
   ├─ quality_adjudication.py      轨迹质量裁决
   ├─ sft_decision_builder.py      SFT 决策数据构建
   ├─ sft_release.py               SFT 发布门禁
   ├─ readiness_gate.py            SFT/DPO/RL 阶段门禁
   └─ run_retail_agentic_grpo.py   Agentic GRPO 预检、训练与证据保存
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
& "D:\tau2-bench\.venv\Scripts\python.exe" -m pytest -q
```

完整 V7 重放测试需要使用已安装上游依赖的 tau2 虚拟环境；仅运行不依赖 tau2 的
单元测试时也可使用当前项目 Python。2026-08-13 使用当前本机 Python 复核：114 项
通过、9 个子测试通过、8 项失败。8 项均依赖本地 tau2 导入链，首个根因是缺少
`addict`；云端固定环境中的相关专项测试此前为 19/19 通过。不能把本机结果表述成
“全部测试通过”。

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
python -m src.run_retail_guarded_failure3 --approve-paid-run
```

该命令会运行同配置的 Base 与 Guarded 两个 arm，产生新的 Agent、用户模拟器和
NL Judge API 调用。执行前必须同时满足 clean Git、API Key 已配置和显式
`--approve-paid-run`；默认不会自动执行。Guarded arm 额外记录 intervention
trace。所有 raw 结果必须重新经过 V7 才能形成恢复率结论。

当前零调用 Preflight 为 `BLOCKED`，没有发生付费调用。详细协议与阻塞项见：

- `docs/03_Verifier与Guard/2026-07-30_Guard在线配对AB预检报告.md`
- `experiments/20260730_guard_online_ab_preflight_v3`

## 证据入口

- 基线：
  `experiments/20260722_110504_retail_baseline20_trial1_deepseek`
- V7：
  `reports/evaluation/final_report.json`
- V6/V7 对比：
  `experiments/20260726_v6_vs_v7_evaluation/comparison.json`
- Guard：
  `experiments/20260726_pre_action_guard_v1/guard_audit.json`
- Guard 合成场景诊断：
  `experiments/20260730_guard_synthetic_diagnostic_v1`
- Guard 规则族消融：
  `experiments/20260802_guard_ablation_v1`
- Guard 在线 A/B 预检：
  `experiments/20260730_guard_online_ab_preflight_v3`
- Verifier V2.2：
  `experiments/20260727_policy_grounding_v2_2`
- Post-training readiness：
  `experiments/20260728_post_training_readiness_v0/readiness_report.json`
- 隔离对抗评测集 V2：
  `data/posttrain_adversarial_holdout_v2/manifest.json`

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

## 2026-08-13 多轮 Tool-SFT 诊断更新

Qwen3-4B Tool-SFT 后的冻结 32-rollout 诊断已完成，证据位于
`experiments/20260813_qwen3_4b_tool_sft_rollout_diagnostic_v5/`。工具调用轨迹由 Base 的
`0/32` 变为 `32/32`，平均必要动作召回为 `0.32232`，说明单步工具协议 warmup 有效；
但 29/32 条轨迹只执行 2 次工具调用，32/32 均未正常完成交互。逐任务审计仅有
`1/8` 个任务存在 reward 与 Action Recall 联合组内方差，因此暂不启动 GRPO 权重更新。

详见：
[`docs/04_数据治理与后训练/2026-08-13_Qwen3-4B多轮Tool-SFT诊断复盘.md`](docs/04_数据治理与后训练/2026-08-13_Qwen3-4B多轮Tool-SFT诊断复盘.md)。

用户模拟器调用链现已增加运行前最小 API 健康检查、运行中有限重试、独立系统失败
artifact 与 secret 脱敏。系统失败会立即终止运行，且明确标记为不可进入 reward 或训练数据。
设计与边界见：
[`docs/04_数据治理与后训练/2026-08-13_用户模拟器Fail-Fast设计与实现.md`](docs/04_数据治理与后训练/2026-08-13_用户模拟器Fail-Fast设计与实现.md)。

针对旧 Tool-SFT 只监督“单条用户消息 → 首次工具调用”的缺口，项目新增多步工具轨迹数据协议：
24 条合成训练轨迹被切为 136 个 next-assistant 决策点，6 条独立留出轨迹被切为 34 个决策点；
上下文保留真实 chat-template 格式的 assistant tool-call 与 tool result，loss 仅作用于下一次
assistant 工具调用。数据和训练仍属于隔离工程实验，不是 Retail 人工金标。设计、质量门和指标见：
[`docs/04_数据治理与后训练/2026-08-13_多步工具轨迹SFT设计与数据审计.md`](docs/04_数据治理与后训练/2026-08-13_多步工具轨迹SFT设计与数据审计.md)。
