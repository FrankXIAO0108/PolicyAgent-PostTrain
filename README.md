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

## 离线证据摘要

演示不需要 API Key、不调用模型、不付费，也不需要重新运行 tau2。它只读取
仓库中冻结并带哈希的报告：

```powershell
cd <PolicyAgent-PostTrain 仓库路径>
.\demo.ps1
```

等价的 Python 命令：

```powershell
python -m src.project_summary
```

演示内容包括：

- 20 个冻结任务的基线结果；
- V6 轨迹 LLM pipeline 与 V7 确定性重放对比；
- Task 95、98、107 三个代表性业务失败；
- Runtime-safe Guard 的具体拦截规则；
- 正式 Retail SFT、DPO、RLHF/GRPO 门禁仍关闭的原因；
- 所有结果的实验边界。

项目文档：

- [全部文档的中文分类导航](docs/README.md)
- [项目完成报告](docs/01_项目总览/项目完成情况报告.md)
- [项目完整工程复盘](docs/01_项目总览/项目完整工程复盘.md)

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

2026-08-21 起，独立业务专家不再作为**个人项目开发训练**的硬阻塞。开发数据采用
“确定性校验与状态回放 → Codex 预标注 → 项目所有者逐条复核 → 修正后重验”的闭环。
首批 10 条 Base/SFT 配对轨迹已完成单人复核，分为 5 条 `RAW_GOLD` 和 5 条
`CORRECTION_REQUIRED`。这些标签不是业务专家 gold，不能用于发布正式 Verifier
Precision/Recall/F1 或生产可靠性结论。治理边界见
[单人复核与开发训练门禁决策](docs/04_数据治理与后训练/2026-08-21_单人复核与开发训练门禁决策.md)。

## 后训练数据治理

项目已经实现分层门禁：开发训练可在项目所有者复核、修正回放、哈希和防泄漏检查通过后
放行；独立 gold 指标与生产结论门禁继续保持关闭。相关能力包括：

- 轨迹质量独立裁决；
- 环境污染与 benchmark 冲突隔离；
- 修正轨迹哈希；
- 行为改变后的环境重放；
- 项目所有者逐条复核；双人审批作为未来正式 gold 的可选高标准协议保留；
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
| 开发级教师 SFT 数据 | 34 条 TRAIN、13 条实体隔离 VALIDATION；项目所有者复核，不是独立专家 gold |
| Qwen3-4B 教师 SFT | 三个训练 seed、80 steps、merge 与 30-task 开发重评测已完成 |
| 教师 SFT 行为结果 | seed18/19/20 开发视图为 17/30、16/30、16/30；多 seed 逐任务一致率 73.3% |
| 过程 Reward 离线审计 | 8 个翻转对中成功轨迹排序更高 7/8；task67 标量分数并列 |
| Claim-State 对抗评测 | 24 条冻结合成样本精确匹配 20/24；FAIL F1 76.92%，Reward 接入门禁未通过 |
| Claim-State V2 开发版 | 20/20 开发规格；60 条既有轨迹中 0 个假 FAIL，但仅 1 PASS、42 REVIEW，等待全新 holdout |
| Claim-State V2 盲测 | owner-reviewed 24 条首次评测为 19/24；FAIL P/R/F1=100%/83.33%/90.91%，REVIEW 召回 70%，禁止接入 Reward |
| 停止条件离线诊断 | 冻结 60 条轨迹中 59 PASS、1 个异常终止 FAIL；3 条成功人工转接序列均正确，真实违规负例覆盖不足，尚未接入 Reward |
| Tool error 分层诊断 | 31 次错误分为 13 次身份查询未命中、3 次其他读取错误、15 次写工具错误；豁免身份查询的反事实排序仍为 7/8，不修改 Reward |
| 正式 Retail DPO | 未运行；偏好数据与独立验证门禁未通过 |
| 正式 Retail Agentic GRPO | 未运行；Reward holdout 与抗钻空子门禁未通过 |
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

2026-08-11 起，项目建立真实多轮 Retail Agentic RL 路线：使用 tau2 Retail 的动态客户、
数据库工具和终态检查，将一对一必要动作进度、沟通完成度、工具错误、重复调用与非预期
写操作组成轨迹级过程奖励。44 条 train、10 条 validation、既有 development audit
严格拆分，官方 test 保留。后续完成了 Tool-SFT、多步 Tool-SFT、教师轨迹 SFT、多 seed
30-task 评测和过程 Reward 离线审计。当前 Reward 在 8 个观察到 outcome 翻转的任务中正确
排序 7 个，task67 因最终自然语言目标选择不同而并列；claim-state 诊断能够将其路由为复核。
停止条件诊断在 60 条冻结轨迹中识别出 1 个 `too_many_errors` 异常终止，且未误罚 32 条成功
轨迹；但真实停止违规负例仍不足。两类诊断均未进入标量 Reward。因此当前不能声称
Agentic GRPO 带来业务提升。进一步的 Tool error 分层显示，统一按次扣分会误罚用户逐步纠正
身份信息的查询，但豁免这类错误不能改善翻转排序，因此没有修改线上 Reward。详见
[Retail 智能体强化学习设计](docs/04_数据治理与后训练/2026-08-11_Retail智能体强化学习设计.md)和
[过程 Reward 离线正向验证报告](docs/04_数据治理与后训练/2026-08-21_过程Reward离线正向验证报告.md)。

## 关键代码

```text
src/
├─ project_summary.py              零 API 冻结证据摘要
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
单元测试时也可使用当前项目 Python。2026-08-21 使用 `D:\tau2-bench\.venv` 完整回归：
`297 passed, 2 skipped, 1 warning, 12 subtests passed`。唯一 warning 为 Python `audioop`
弃用提示；2 个 skip 需按各测试自身条件解释，不能写成 296/296。

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

1. 暂停把 Claim-State V2 接入 Reward；它只作为离线诊断和人工路由信号；
2. 停止条件与 Tool error 分层已完成首轮离线诊断；下一步优先验证多实体写操作和联合约束；
3. 若开发 Claim-State V3，必须先解决事实来源冲突，再在全新 holdout 上评测；
4. 做 SFT 数据规模与多 seed 消融，区分数据不足、行为方差和模型容量限制；
5. 只有高精度且覆盖率足够的过程信号通过独立验证后，才启动小规模 Agentic GRPO。

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

多步 Tool-SFT 已在 Qwen3-4B 上完成 136-step QLoRA、merge 与 34 个同分布留出决策点评测；
post-tool Tool Match 从 42.86% 升至 78.57%。但同协议的 32-rollout 真实多轮复验出现退化：
平均 Action Recall 从 0.3223 降至 0.1914，工具错误从 1 增至 15，excess duplicate 从 0 增至 42，
且仍有 32/32 unfinished。因此当前不启动 GRPO。诊断门已增加相对 baseline 的回归约束，避免把
错误路径产生的方差误判为可训练信号。详见：
[`docs/04_数据治理与后训练/2026-08-13_Qwen3-4B多步Tool-SFT实跑结果.md`](docs/04_数据治理与后训练/2026-08-13_Qwen3-4B多步Tool-SFT实跑结果.md)。

## 2026-08-21 教师 SFT 与过程 Reward 更新

同一份 34 条教师轨迹、80 steps 的三个训练 seed，实体隔离 validation assistant loss
为 `0.4105 / 0.4114 / 0.4107`。两个新增 checkpoint 在扩窗组合口径下均为 `16/30`，
但逐任务一致率只有 `22/30 = 73.3%`，说明 token-level loss 平台不等于 Agent 行为稳定。
历史 Base 为 `12/30`，三个 SFT 开发视图为 `17/30 / 16/30 / 16/30`；这些是不同上下文
上限组合出的开发证据，不是正式业务指标。

过程 Reward 已离线回放到 seed19/seed20 的 60 条冻结轨迹。修复 task108 无序
`item_ids` 的 action matcher 假阳性后，8 个成功/失败翻转对中有 7 个成功轨迹得分更高；
task67 因 DB、动作和 Tool error 相同而仍然并列。显式订单—金额/状态 claim-state
诊断已完成首轮验证，但尚未进入 Reward，GRPO 门禁继续关闭。

详见：

- [教师 SFT 多种子稳定性与扩窗补跑报告](docs/04_数据治理与后训练/2026-08-21_教师SFT多种子稳定性与扩窗补跑报告.md)
- [过程 Reward 离线正向验证报告](docs/04_数据治理与后训练/2026-08-21_过程Reward离线正向验证报告.md)
