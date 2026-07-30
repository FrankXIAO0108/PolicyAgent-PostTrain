# PolicyAgent-PostTrain：Verifier 深度复盘与大模型面试基础

# 第五部分：Verifier 设计深度复盘

## 1. 为什么需要 Verifier

Tau2 原始 reward 主要回答“任务结果是否满足 DB/NL 目标”，但业务还需要回答：

- 写操作是否得到用户明确授权？
- Tool 参数是否与最新确认一致？
- 是否违反一次一个 Tool、一次性订单修改等 Policy？
- Tool 返回成功后，状态是否真的正确？
- Agent 的最终回复是否真实描述 Tool/DB？

Verifier 将这些问题转为带证据的可回归 predicate。

## 2. LLM Judge 的问题

### 不稳定

不同模型、温度、提示词或采样可能改变 verdict。即使 temperature=0，服务端实现、
上下文截断和模型版本也可能漂移。

### 证据不完备

V6 只看 trajectory 语义，无法恢复没有提供的 gold DB transition，因此漏掉
59、95、98、107 四个 outcome failure。

### 流畅性偏差

Agent 最终回答如果表达自信、结构清楚，Judge 可能忽视错误 Tool 参数和副作用。

### 细粒度业务约束难稳定覆盖

`available: bool`、same-item exchange、one-shot order mutation 等适合写成明确
predicate，而不是依赖开放式语言判断。

### 可解释但不可执行

LLM 能说“可能存在风险”，却未必输出稳定 rule ID、severity、证据位置和可用于
Guard 的 action。

因此本项目采用：

```text
确定性事实 -> 程序重放/规则
语义解释 -> LLM 可辅助
不确定证据 -> REVIEW / 人工
```

## 3. Verifier 版本演进

### 3.1 Policy Grounding V0

**设计思想**：先覆盖确定性、低歧义的结构错误。

**检测维度**：

- benchmark reward（仅独立报告）；
- explicit confirmation；
- tool-call cardinality；
- message/tool exclusivity；
- Tool 明确报错后仍声称成功；
- latest intent 暂不做开放式语义判断，写操作存在时可 abstain 为 REVIEW。

**代码结构**：

- `policy_grounding_v0.py` 遍历标准化 `MessageEvent`；
- `trajectory_loader.py` 从 frozen artifact 恢复事件；
- `schemas.py` 定义 Dimension、Severity、Finding、Verdict。

**输入/输出**：

```text
输入：MessageEvent[] + task_id + benchmark_verdict
输出：VerificationResult
     {verdict, dimensions, findings, metrics, notes}
```

**实际发现**：多 Tool call、Tool 与文本同轮、未经确认写入和明显虚假成功声明。

### 3.2 初始 Policy Grounding V1

**设计思想**：把 latest intent 从“无法判断”升级为“冻结确认状态后核对写参数”。

**核心过程**：

```text
assistant action summary
-> next user explicit confirmation
-> ConfirmationSnapshot
-> material write arguments
-> confirmed / missing / review
```

**新增能力**：比较 order、item、payment、address 等 material arguments 是否
出现在确认上下文中；后一次确认覆盖旧约束。

**实际问题**：过度依赖字面匹配，用户用“Visa 尾号 6593”确认时，规则却要求
内部 `payment_method_id`；最终确认未重复 order ID 也可能误报。

### 3.3 关于 V1.1

仓库历史中没有一份独立冻结、带正式指标的 `Verifier V1.1` 产物。可以把早期
V1 到 V1.2 之间的内部增量理解为“V1.1 阶段”，但面试时应明确：

> 正式可复现的早期产物是初始 V1 与 V1.2，我不会为没有冻结 artifact 的 V1.1
> 编造指标。

### 3.4 Policy Grounding V1.2

**设计思想**：加入用户可见实体与内部 ID 的 alias grounding，并把 severity
纳入 verdict 聚合。

**检测维度**：

- Latest Intent；
- Explicit Confirmation；
- Policy Compliance；
- Action Result Truthfulness；
- Benchmark Reward（独立，不参与 policy proxy）。

**代码结构**：

- `intent_state.entity_aliases_before` 从 Tool 结果提取 ID、名称、规格、卡品牌/
  尾号等别名；
- `confirmation_snapshot_before` 维护 proposal/confirmation；
- `audit_call_against_latest_intent` 核对 material fields；
- major finding -> FAIL，minor-only -> REVIEW。

**输入/输出**：仍为标准化事件到 `VerificationResult`，额外输出 audited/passed/
failed/review write call 数。

**实际发现**：

- 20 条：0 PASS、11 REVIEW、9 FAIL；
- 成功发现 Task 98/107 的部分政策问题；
- Task 95 只到 REVIEW，漏掉 schema semantics；
- 定向审计暴露 Task 1/37/72 latest-intent FP、Task 21/107 FN。

### 3.5 V1.3（当前 V1 系列源码）

当前 `policy_grounding_v1.py` 内部版本为 1.3，继续修复实体 alias、前一确认摘要
继承和增量确认。它不是项目最终主线，主线已进入组合 Guard 的 V2。

### 3.6 Policy Grounding V2.0

**设计思想**：离线 Verifier 与在线 Guard 共享同一套 runtime-safe rule core。

**检测内容**：

- V1 系列 confirmation/structure；
- variant 可用性和 same-item exchange；
- payment；
- item-to-order scope expansion；
- one-shot order mutation；
- actionable solution 存在时的 premature transfer。

**代码结构**：

- `policy_grounding_v2.py` 回放 observed messages；
- `retail_pre_action.py` 维护 `GuardContext` 和纯规则；
- GuardFinding 映射为 Verifier Finding；
- runtime 模式不读取 reference action 或 gold DB。

**实际发现**：补回 Task 95，解决 Guard 能拦而 Verifier 漏报的语义漂移。

### 3.7 V2.1 与 V2.2

V2.1：

- 用户可见 item/address/payment 已确认时，不要求重复内部 order ID；
- 增加跨 turn 的同订单第二次 modify/exchange 检查；
- Task 37 从错误 FAIL 降到 REVIEW。

V2.2：

- 保留紧邻、已经确认的完整 action summary；
- 用户随后只做 scope reconfirmation 时不丢失先前 material arguments；
- Task 1 假升级被修复；
- 20 条开发 provisional 标签全部匹配：12 REVIEW、8 FAIL。

边界：0 adjudicated、0 provisional PASS、规则和标签均在同一开发集迭代，因此
不能报告 held-out 100%。

## 4. Verifier 与 Reward Model 的区别

| 维度 | Programmatic Verifier | Reward Model |
|---|---|---|
| 实现 | 规则、状态机、重放、结构化比较 | 通常是从人类偏好/标签训练的神经模型 |
| 输出 | verdict、finding、证据、severity，可选标量 | 通常是标量 score 或 preference probability |
| 优点 | 可解释、确定、易回归、低成本 | 能覆盖难以枚举的开放语义和总体偏好 |
| 缺点 | 覆盖有限、规则维护成本、容易遗漏新表达 | 会偏移、可被 reward hacking、需要高质量数据 |
| 本项目状态 | 已实现开发版本，未通过独立 gold 发布门 | 未训练 |

关系不是二选一。未来可以采用 hybrid reward：

```text
hard safety constraints -> deterministic Verifier/Guard
soft semantic quality -> learned Reward Model
uncertain/conflict -> human review
```

在进入 RL 前必须在 held-out 上评估 reward 的 precision、recall、critical recall、
calibration 和 attack robustness。

---

# 第六部分：结合项目的大模型面试八股

## 1. 什么是 Agent？

### 面试问题

什么是大模型 Agent？

### 标准答案

Agent 是以模型作为决策核心，感知环境状态，通过规划、工具调用和多轮反馈完成
目标的系统。其闭环通常是 observation -> reasoning/planning -> action ->
new observation。

### 结合项目回答

Tau2 Retail Agent 根据用户消息和 Tool 返回持续决策，通过取消、退货、换货等
工具改变数据库。PolicyAgent 评估的不只是最终文本，而是完整闭环中的授权、
动作、状态和最终声明。

## 2. Agent 和普通 Chatbot 有什么区别？

### 面试问题

为什么 Tool Agent 风险高于普通 Chatbot？

### 标准答案

Chatbot 主要生成文本；Agent 还会选择动作、调用外部工具、读写持久状态。错误
可能从“回答不准确”升级为真实副作用。

### 结合项目回答

Task 98 可能把商品级取消扩大到整单并影响退款；Task 107 的非法换货被 Tool
接受。此类错误不能只靠回答相似度评估。

## 3. ReAct 是什么？

### 面试问题

解释 ReAct，以及它在本项目中的体现。

### 标准答案

ReAct 交替进行 Reasoning 与 Acting：模型根据当前 observation 形成推理并采取
动作，再读取环境反馈继续决策。它提高可交互性，但长链路会积累状态和推理误差。

### 结合项目回答

Tau2 轨迹就是消息、Tool Call、Tool Result 的多轮交替。项目不依赖暴露私有
chain-of-thought，而是审计可观察的 action summary、tool arguments 和 state。

## 4. Tool Calling 如何实现？

### 面试问题

LLM Tool Calling 的工程流程是什么？

### 标准答案

向模型提供工具名称和参数 schema；模型输出结构化 function call；执行器校验并
调用工具；结果以 tool message 返回模型；模型继续决策或生成最终回答。

### 结合项目回答

项目额外在 proposal 与 execution 之间加入 `retail_pre_action.py`：
proposal 先经 Guard，输出 ALLOW、REQUIRE_CONFIRMATION、REGENERATE、BLOCK 或
TRANSFER，防止副作用发生后才发现错误。

## 5. Agent 为什么会失败？

### 面试问题

Agent failure 的主要来源有哪些？

### 标准答案

包括意图理解、规划、知识、工具选择、参数、状态跟踪、长上下文、环境异常、
策略遵循、错误恢复和评测缺陷。

### 结合项目回答

Task 95 是 schema semantics + goal completeness；98 是 payment/scope/claim；
107 是 policy grounding + tool gap；59 甚至不是纯 Agent failure，而是 benchmark
alignment。项目用 taxonomy 避免一律归为“推理不行”。

## 6. LLM-as-a-Judge 有什么问题？

### 面试问题

什么时候不该使用 LLM Judge？

### 标准答案

当真值可程序计算、Judge 缺少关键状态、对一致性要求高或成本/延迟敏感时，不应
让 LLM 取代确定性检查。LLM Judge 还存在位置偏差、长度偏差、自偏好、随机性和
prompt sensitivity。

### 结合项目回答

V6 漏掉全部四个 outcome failure。V7 直接重放状态，无需 40 次 LLM 调用。
LLM 仍可解释 root cause，但不能覆盖 replay 得到的 outcome。

## 7. Reward Model 是什么？

### 面试问题

Reward Model 的训练目标是什么？

### 标准答案

Reward Model 学习人类偏好或任务质量，把 prompt-response/trajectory 映射到
标量奖励。常见 Bradley-Terry 目标对 chosen/rejected score 做 logistic
preference loss。

### 结合项目回答

本项目没有训练 Reward Model。Programmatic Verifier 只提供可解释规则信号；
在独立 gold 和 held-out validation 完成前，不能直接作为 GRPO reward。

## 8. Verifier 和 Reward Model 有什么区别？

### 面试问题

Verifier 是否等于 Reward Model？

### 标准答案

不等于。Verifier 通常验证可判定约束，输出结构化判定；Reward Model 学习总体
偏好并输出连续分数。Verifier 可成为 reward 的一个组件。

### 结合项目回答

PolicyAgent 用 Verifier 检查 same-item exchange、confirmation、scope 等 hard
constraints；开放式服务质量未来更适合 learned reward。两者应 hybrid。

## 9. RLHF 流程是什么？

### 面试问题

完整说明经典 RLHF。

### 标准答案

预训练模型 -> SFT -> 收集偏好对 -> 训练 Reward Model -> 使用 PPO 等算法在
KL 约束下优化 policy -> 安全/质量评测。工程上还需数据治理、reward validation
和防 reward hacking。

### 结合项目回答

本项目目前停在可靠数据/reward 准备阶段。没有独立 gold、SFT 比较和 held-out
reward，所以 readiness gate 正确阻止 RLHF。

## 10. DPO 是什么？

### 面试问题

DPO 相比 PPO-based RLHF 有什么特点？

### 标准答案

DPO 通过偏好对直接优化 policy 相对于 reference model 的对数概率差，隐式对应
KL-regularized reward optimization，无需显式 Reward Model 和在线 RL rollout。
它更简单稳定，但依赖高质量 chosen/rejected pairs，且是离线偏好优化。

### 结合项目回答

Task 95/107 可能形成 corrected chosen vs failed rejected，但必须先独立裁决、
绑定同一上下文并防 benchmark noise。Task 59 不能直接做 DPO pair。

## 11. PPO 为什么需要 Value Model？

### 面试问题

PPO 中 Value Model 的作用是什么？

### 标准答案

Value Model 估计状态期望回报，作为 baseline 计算 advantage，降低 policy
gradient 方差。PPO 再用 clipped surrogate objective 限制更新幅度，并常加入
KL penalty 约束偏离 reference policy。

### 结合项目回答

若未来做长链路 Agent PPO，reward 延迟到任务结束，credit assignment 很难；
Value Model 可估计中间状态价值。但当前 reward 可靠性都未通过，先训练 value
只会拟合错误目标。

## 12. GRPO 是什么？

### 面试问题

GRPO 与 PPO 的核心区别是什么？

### 标准答案

GRPO 对同一 prompt 采样一组输出，用组内 reward 的相对均值/方差估计 advantage，
通常不需要独立 Value Model，并配合 clipped objective 与 KL 正则。它降低
value 训练成本，但依赖多样 rollout 和可靠、可区分的 reward。

### 结合项目回答

Tool Agent GRPO 可对同一任务采样多条 trajectory，用 outcome、policy、invalid
action、false success 等信号评分。但 Verifier 必须先在 held-out 上可靠，否则
模型会学会绕过规则或优化 benchmark proxy。

## 13. Agent Evaluation 为什么困难？

### 面试问题

为什么不能只用 exact match？

### 标准答案

Agent 有多条等价动作路径、动态环境、延迟副作用、部分完成、工具错误和长链路
credit assignment。结果正确、过程正确和安全合规是不同维度。

### 结合项目回答

Task 16 outcome 成功但 policy fail；Task 59 outcome fail 但标签有冲突；Task 21
tool arguments 正确但状态污染。因此项目分 Outcome/Policy/State/Claim。

## 14. Benchmark 为什么需要固定 test set？

### 面试问题

为什么看过 badcase 后不能继续报告同一 test 分数？

### 标准答案

因为规则、Prompt 或训练数据已针对 test 信息优化，导致 test contamination 和
乐观偏差。应冻结 development/validation/test，记录 hash，并让最终 test 只使用
一次或严格限制访问。

### 结合项目回答

Trial-1 禁止换题和 prompt tuning，官方 test 未使用。V2.2 的 100% 只称开发
integration diagnostics，因为规则和 provisional 标签都在同一池迭代。

## 15. Data Leakage 是什么？

### 面试问题

Agent/Post-training 中有哪些泄漏？

### 标准答案

包括样本重复、同用户/实体跨 split、test 被用于规则开发、gold/reference 进入
runtime、模型见过 benchmark、派生数据跨 split 等。

### 结合项目回答

项目检查 user/order/product 等实体级 group leakage，并把 runtime-safe Guard
与 reference diagnostic 物理隔离；V2 显式记录
`uses_reference_actions=false`。

## 16. Policy Grounding 是什么？

### 面试问题

Policy Grounding 与普通 instruction following 有何区别？

### 标准答案

Policy Grounding 是把抽象政策结合当前用户授权、环境状态和候选动作，落实为
当下可执行/禁止的约束。它要求理解条件、例外、时序和副作用，而不是复述政策。

### 结合项目回答

Task 107 中模型能谈换货，却没有把“different product option”落到
`old_item_id != new_item_id`；Task 72/109 涉及 once-per-order。Verifier/Guard
将自然语言 policy 转成可回归 predicate。
