# PolicyAgent-PostTrain 项目表达、模拟面试与知识地图

# 第七部分：3 分钟项目介绍

## 可直接练习的版本

我做的是一个面向电商售后场景的 Tool Agent 可靠性评测和后训练数据治理项目，
底层使用 tau2-bench Retail 环境。上游提供任务、Policy、Tool 和基础 evaluator，
我负责的是实验冻结、轨迹审计、状态重放、Policy Verifier、Pre-action Guard
和训练阶段门禁。

这个项目起因是我发现 Tool Agent 的 benchmark reward 和真实业务正确性之间有
明显缺口。Agent 的对话可能很流畅、Tool 也返回成功、甚至最终 reward 是 1，
但它仍可能扩大用户授权范围、违反业务政策、写入错误支付方式，或者对最终状态
作出不真实陈述。

我先建立了一个冻结的 20-task Retail 开发基线，固定 task、模型、temperature、
seed、上游 commit 和配置 hash。结果是 16 个业务成功、4 个业务失败、0 个系统
失败。随后我没有直接优化 Prompt，而是逐条审计四个失败。结果发现 Task 59 是
User Simulator 与 static golden 冲突；Task 95 是把 boolean availability 误解
为单件库存并提前转人工；Task 98 同时包含支付方式、操作范围和最终陈述风险；
Task 107 则是同 item 换同 item，Tool 还接受了违反 Policy 的动作。

这个分析让我意识到，raw reward 不能直接当训练标签。所以我把评测拆成
Outcome、Policy、State 和 Claim 四层。早期 trajectory-only LLM verifier 在
20 条上漏掉了全部 4 个 outcome failure。我随后实现了确定性状态重放：分别
重放 Agent 与目标 Tool 操作，比较最终数据库，再复用冻结 NL assertion。新
pipeline 与 20 个冻结结果全部一致，消除了旧方案的 4 个漏报，而且不需要新增
LLM 调用。这里的 100% 只表示冻结 artifact 的重放一致性，不是未见任务泛化。

在执行侧，我把高风险 predicate 做成不读取 gold 的 Pre-action Guard，例如
商品级请求不能扩大成整单取消、换货新旧 variant 不能相同、存在可行动方案时
不能提前转人工。离线审计中，它覆盖了 3 个非隔离失败。之后我又让离线
Verifier 和在线 Guard 共享同一规则核，避免线上阻止、离线不计错、训练 reward
反而鼓励同一行为。

最后，我实现了轨迹质量、修正 hash、双人审批、实体级 split 泄漏和
SFT/DPO/GRPO readiness gate。当前 20 条 Policy 标签仍是 provisional，独立
adjudicated gold 为 0，因此我没有虚构 SFT 或 GRPO 提升，而是主动关闭训练
门禁。

我认为项目的创新点不在于又套了一个 Agent 框架，而在于把“最终结果正确”
与“执行过程安全、Policy 合规、状态真实、可用于训练”分开，并用可复现证据
决定何时该训练、何时不该训练。下一步是获得独立 gold，在新的 held-out 任务
验证 Verifier/Guard，并做 Base-vs-SFT 冻结比较；只有残余错误和 reward 可靠性
证明有必要时，才进入 DPO 或 GRPO。

## 30 秒压缩版

我基于 tau2-bench Retail 做了 Tool Agent 可靠性项目。冻结 20 个开发任务得到
16 成功、4 失败，但审计发现 reward=0 可能是 benchmark 冲突，reward=1 也可能
违反 Policy。于是我实现了确定性状态重放、Policy Verifier 和不读取 gold 的
Pre-action Guard。旧轨迹 LLM pipeline 漏掉全部 4 个 outcome failure，重放
方案与 20 个冻结结果一致且不新增 LLM 调用；Guard 离线覆盖 3 个非隔离失败。
我还实现了 SFT/DPO/GRPO 数据门禁，因为独立 gold 尚未完成，所以当前没有虚构
训练提升。

---

# 第八部分：30 个项目模拟面试问题

## A. 基础问题（1—10）

### 1. 这个项目一句话解决什么问题？

**关注点**：能否抽象核心矛盾。
**优秀回答**：解决 Tool Agent 的 benchmark outcome 与真实业务正确性不一致：
把授权、Policy、Tool 副作用、最终状态和陈述一致性纳入可复现评测，并在可靠
监督建立前阻止脏数据进入后训练。

### 2. 哪些代码是你写的，哪些来自上游？

**关注点**：项目边界与诚信。
**优秀回答**：tau2-bench 提供 Retail 任务、Policy、Tools、DB、交互编排和
基础 evaluator；本仓库实现 runner/manifest、audit/taxonomy、deterministic
replay、Verifier、Guard 和 training gates。

### 3. 为什么先做 5-task Smoke？

**关注点**：工程实验意识。
**优秀回答**：先验证 Runtime、日志、Judge、错误恢复和成本记录；它是风险分层
样本，不是代表性 baseline，因此没有把 3/5 当正式能力指标。

### 4. 20-task Baseline 如何冻结？

**关注点**：可复现性。
**优秀回答**：固定 task IDs、Agent/User/Judge model、temperature=0、seed=300、
max steps、上游 commit、配置 hash 和 no-task-replacement/no-prompt-tuning
协议，官方 test 保持未使用。

### 5. 80% success rate 能说明什么？

**关注点**：统计边界。
**优秀回答**：只说明这 20 个冻结开发任务的单 trial outcome；样本小、不是
leaderboard、没有多 seed CI，不能外推生产成功率。

### 6. 为什么 reward=0 不等于 Agent failure？

**关注点**：标签噪声。
**优秀回答**：Task 59 的动态用户最终意图与 static golden 冲突，Agent 行为
不能被简单标成负例；Task 98 还是 mixed case。

### 7. 为什么 reward=1 也不等于好轨迹？

**关注点**：outcome/process 区分。
**优秀回答**：Task 16/29/76/109 outcome 成功但有并行写或 once-per-order
Policy 违规；SFT 会模仿过程，所以不能直接作为 positive demonstration。

### 8. Verifier 输出为什么有 REVIEW？

**关注点**：selective classification。
**优秀回答**：证据不足时强制 PASS/FAIL 会制造错误；REVIEW 是 abstention。
需要同时报告 coverage、selective risk 和 review burden。

### 9. Guard 与 Verifier 的区别？

**关注点**：在线/离线角色。
**优秀回答**：Guard 在 Tool 执行前决定是否允许、确认、重生成或阻止；Verifier
事后读取轨迹做评价。两者输入适配不同，但违规 predicate 应共享。

### 10. 为什么没有直接做 SFT/GRPO？

**关注点**：是否诚实理解项目状态。
**优秀回答**：独立 policy gold 为 0，正式 SFT 数据未发布，reward 未在 held-out
验证；直接训练会放大标签噪声。Readiness gate 因此保持关闭。

## B. 中等问题（11—20）

### 11. Task 95 为什么不是简单“推理能力不足”？

**关注点**：根因粒度。
**优秀回答**：它是 schema type semantics：`available: bool` 被推成库存数量 1，
进而导致 goal completeness 失败和 premature transfer。可通过 schema、variant
resolver、goal ledger 和 Guard 分层解决。

### 12. Task 107 暴露哪四层问题？

**关注点**：系统思维。
**优秀回答**：Policy 要求不同 option；Agent 没 ground；Tool 没 enforce；
Evaluator 只检查发生 exchange，没有覆盖 variant/policy。

### 13. 为什么 V6 LLM Verifier 漏掉四个失败？

**关注点**：observability。
**优秀回答**：它只看 trajectory，没有 gold final-state transition；流畅文本
掩盖副作用。缺失事实不是靠更强 prompt 就能恢复的。

### 14. V7 如何重建 outcome？

**关注点**：确定性评测逻辑。
**优秀回答**：从 artifact 恢复初始环境，分别重放 Agent 和 target Tool actions，
规范化 final DB 做 recursive diff，再结合冻结 NL assertion 重建官方信号。

### 15. V7 的 100% 为什么不能写成 Verifier 泛化准确率？

**关注点**：指标解释。
**优秀回答**：它是对同一冻结 artifact 的 replay fidelity；root-cause 规则也
来自这四个已审计失败，没有独立 held-out。

### 16. Task 98 如何做 Claim Consistency？

**关注点**：证据链。
**优秀回答**：绑定 latest authorized scope -> tool arguments -> tool result ->
final DB -> final response；例如实际整单取消/退款金额必须与回复一致。

### 17. V1.2 的假阳性从哪来？

**关注点**：多轮状态。
**优秀回答**：过度做最后一轮字面匹配，要求用户重复内部 order/item/payment ID；
实际用户用商品规格、卡尾号和上下文确认。修复方式是 entity aliases +
ConfirmationSnapshot。

### 18. 为什么 severity 要分 MAJOR/MINOR？

**关注点**：业务风险。
**优秀回答**：多个写调用可能造成部分成功和不可逆副作用，属于 MAJOR；多个只读
或 message/tool mixing 风险较低，可 REVIEW。否则所有告警同权导致过度拦截。

### 19. 如何避免 Guard 偷看 gold？

**关注点**：leakage。
**优秀回答**：runtime API 不接受 reference；`GuardContext` 的 reference 默认为
空且不 enforce；输出记录 `uses_reference_actions=false`；相同 observed
trajectory 配不同 reference 时 runtime verdict 应不变。

### 20. 训练数据为什么要做实体级 split？

**关注点**：隐性泄漏。
**优秀回答**：同一 user/order/product family 跨 train/validation 会让模型记忆
实体或模板，产生虚高指标；应 group split 并冻结 hash。

## C. 困难问题（21—30）

### 21. 如果让你把 Verifier 变成 GRPO reward，怎么设计？

**关注点**：reward engineering。
**优秀回答**：先保持多分量：outcome、policy、invalid action、state/claim、
cost/length；hard violation 用 gate/large penalty，soft quality 用 learned
score；在 held-out 验证 precision/recall、critical recall、calibration 和
adversarial gaming，再考虑 normalization。当前项目还不满足前置条件。

### 22. 为什么不把所有规则直接写进 Prompt？

**关注点**：Prompt 与 enforcement 边界。
**优秀回答**：Prompt 是软约束，长上下文可能遗忘；高风险不变量应在 Tool/backend
或 Guard enforce。Prompt 用于引导，Verifier 用于测量，Guard 用于阻止。

### 23. Guard 会不会降低任务成功率？

**关注点**：安全-效用 trade-off。
**优秀回答**：会，错误阻止会增加 transfer 或 failure。离线“拦住错误”不等于
在线收益；需要 A/B 测 recovery rate、false block、latency、token cost、最终
success 和人工转接率。

### 24. 如何给 Verifier 做正式评测？

**关注点**：严谨实验。
**优秀回答**：预先定义 labeling guide；两位 blind reviewer + 第三人裁决；
按实体 group split；冻结 rules 和 held-out；报告三分类 matrix、FAIL P/R/F1、
coverage、selective risk、风险加权 FP/FN 和置信区间。

### 25. 如果 V2.2 development accuracy=100%，下一步最优先是什么？

**关注点**：是否继续刷开发集。
**优秀回答**：停止调同一池；获得 independent adjudicated gold，在新的隔离任务
验证。继续修开发集只会增强过拟合。

### 26. 如何区分 primary root cause 和 secondary finding？

**关注点**：因果推理。
**优秀回答**：primary 是导致官方 outcome 失败的最早必要偏离，例如 Task 98 的
wrong payment；scope/policy 可能是真风险但未导致当前 DB reward，列为 secondary，
避免把多个 diff 字段都当独立原因。

### 27. 为什么 DPO 可能比 GRPO 更适合第一步？

**关注点**：算法选择。
**优秀回答**：若能得到同 prompt 下的 adjudicated corrected vs failed pairs，
但在线 reward 尚不稳定，DPO 更简单；GRPO 要可靠程序奖励和多 rollout。仍应先
做 SFT 与冻结重评测。

### 28. Tool/backend、Guard、Verifier 三层是否重复？

**关注点**：纵深防御。
**优秀回答**：不重复。Backend enforce 不变量；Guard 在执行前给模型可恢复反馈；
Verifier 离线测量和审计。三者共享政策语义，但失败模式和职责不同。

### 29. 如何处理动态用户意图与 static golden 冲突？

**关注点**：benchmark 设计。
**优秀回答**：记录 latest explicit intent，比较 simulator/golden divergence；
冲突样本 quarantine；不能训练模型服从过时 golden。长期应让 task generation
约束 simulator 不偏离可接受目标，或使用动态 evaluator。

### 30. 这个项目最大的不足是什么？

**关注点**：反思能力。
**优秀回答**：任务量小、单 trial、规则和 provisional 标签在同一开发池迭代、
0 independent adjudication、Guard 只有离线 counterfactual audit、未做真实
SFT/RL。优点是这些限制都被 manifest/gate 明确记录，没有包装成完成结果。

---

# 第九部分：项目知识地图

```text
PolicyAgent-PostTrain
|
|-- 1. Agent System
|   |-- LLM decision loop
|   |-- Tool schema / function calling
|   |-- Multi-turn state tracking
|   |-- Goal completeness
|   |-- Error recovery / transfer
|   `-- Long-horizon side effects
|
|-- 2. Retail Business Semantics
|   |-- Identity / authorization
|   |-- Order / item scope
|   |-- Product / variant constraints
|   |-- Payment / refund binding
|   |-- One-shot mutation policy
|   `-- Claim-state truthfulness
|
|-- 3. Agent Evaluation
|   |-- Outcome: DB + NL assertion
|   |-- Action: tool name + arguments
|   |-- Policy: authorization + order + constraints
|   |-- State: replay + structured diff
|   |-- Claim: final response vs tool/DB
|   |-- LLM-as-a-Judge limitations
|   `-- Selective classification / REVIEW
|
|-- 4. Verifier
|   |-- Rule-based
|   |   |-- cardinality
|   |   |-- confirmation
|   |   |-- same-item exchange
|   |   |-- scope expansion
|   |   `-- one-shot mutation
|   |-- State-based
|   |   |-- deterministic replay
|   |   |-- DB diff
|   |   `-- result/claim consistency
|   |-- LLM-based
|   |   `-- semantic explanation, not outcome authority
|   `-- Hybrid
|       |-- hard rules
|       |-- soft semantic score
|       `-- human abstention
|
|-- 5. Guard / Safety
|   |-- ALLOW
|   |-- REQUIRE_CONFIRMATION
|   |-- REGENERATE
|   |-- BLOCK
|   |-- TRANSFER
|   |-- Runtime-safe vs reference diagnostic
|   `-- Offline audit vs online A/B
|
|-- 6. Failure Taxonomy
|   |-- Agent reasoning failure
|   |-- Tool misuse
|   |-- Policy violation
|   |-- Environment corruption
|   |-- Benchmark alignment
|   |-- Evaluator blind spot
|   `-- Mixed badcase
|
|-- 7. Experiment Discipline
|   |-- Smoke / development / final split
|   |-- Manifest / commit / config hash
|   |-- Seed / temperature / model
|   |-- Raw artifact preservation
|   |-- No post-hoc test tuning
|   |-- Group leakage
|   `-- Confidence interval / multi-seed
|
|-- 8. Post-training Data
|   |-- GOLD / SILVER / SUSPECT
|   |-- NEGATIVE / MIXED / EXCLUDED
|   |-- Corrected target
|   |-- Human adjudication
|   |-- Assistant-only loss mask
|   |-- Entity-level split
|   `-- Dataset release hash
|
`-- 9. Alignment Algorithms
    |-- SFT
    |   |-- imitation learning
    |   `-- data quality dominates
    |-- Reward Model
    |   |-- pairwise preference
    |   `-- calibration / hacking
    |-- DPO
    |   |-- chosen/rejected
    |   `-- reference-policy regularization
    |-- PPO
    |   |-- policy/value/reference/reward
    |   |-- advantage / clipping
    |   `-- KL control
    `-- GRPO
        |-- group sampling
        |-- relative advantage
        |-- no separate value model
        `-- reliable programmatic reward
```

## 掌握优先级

### 一级：必须能脱稿讲

- 项目边界与 20-task baseline；
- Task 59/95/98/107；
- reward=1/0 为什么都不能直接当训练标签；
- V6 为什么失败、V7 如何重放；
- Verifier/Guard/Reward Model 的区别；
- 为什么当前没有 SFT/DPO/GRPO。

### 二级：能回答追问

- V1/V1.2/V2.0/V2.1/V2.2 演进；
- ConfirmationSnapshot、entity alias、runtime-safe context；
- Gold/Silver/Suspect/Mixed/Excluded；
- precision/recall/F1、REVIEW/coverage/selective risk；
- DPO/PPO/GRPO 的适用条件。

### 三级：用于高分扩展

- reward hacking 与 specification gaming；
- long-horizon credit assignment；
- group split 与 benchmark contamination；
- risk-weighted metrics 和 confidence interval；
- Guard 在线 A/B 的恢复率、误拦截、延迟与成本；
- backend enforcement、Guard、Verifier 的纵深防御。

## 建议练习顺序

1. 先背熟 30 秒与 3 分钟版本；
2. 用 2 分钟分别讲四个失败；
3. 不看文档回答 30 题；
4. 对每个指标主动补一句“这个数字的边界是什么”；
5. 最后再练 DPO/PPO/GRPO 扩展，避免算法概念盖过真实项目证据。
