# Failure Taxonomy v1 + Training Data Eligibility Matrix v1

## 1. 实验背景

来源实验：Retail Prompt Base / Trial-1

- 总任务数：20
- Raw 成功：16
- Raw 失败：4
- Raw Success Rate：80%

> 重要：人工审计不会回改冻结的 Raw Baseline。
> Raw Baseline 仍然保持 16/20 = 80%。

人工审计的目标不是“提高分数”，而是判断 Reward=0 到底代表：

- 真正的 Agent Failure；
- Benchmark / Evaluator 对齐问题；
- 混合型 Badcase；
- Policy / Tool 语义尚未解决。

---

## 2. Failure Taxonomy v1

| Task | Raw Reward | 人工审计分类 | 主标签 | 是否真实 Agent Failure | 是否 Benchmark 可疑 | 训练状态 |
|---|---:|---|---|---|---|---|
| 59 | 0.0 | BENCHMARK_ALIGNMENT_FAILURE | USER_SIMULATOR_GOLD_MISMATCH | NO | YES | EXCLUDE_FROM_RAW_NEGATIVE_POOL |
| 98 | 0.0 | MIXED_BADCASE | DYNAMIC_INTENT_STATIC_GOLD_MISMATCH | YES | YES | SEGMENT_AND_RELABEL |
| 95 | 0.0 | VALID_AGENT_FAILURE | ENVIRONMENT_STATE_SEMANTICS_MISUNDERSTANDING | YES | NO | ELIGIBLE_AFTER_CORRECTION |
| 107 | 0.0 | UNRESOLVED_POLICY_TOOL_SEMANTICS | CONDITIONAL_BRANCH_STATIC_GOLD_MISMATCH | UNRESOLVED | YES | HOLD_UNTIL_POLICY_VERIFIED |

---

## 3. Training Data Eligibility Matrix v1

| Task | 整条作为负 SFT | 修正后正向 SFT | Preference | Verifier | RL | 优先级 |
|---|---|---|---|---|---|---|
| 59 | NO | POSSIBLE_AFTER_ADJUDICATION | NO_AS_RAW_REWARD_PAIR | YES_FOR_BENCHMARK_ALIGNMENT_AND_FINAL_INTENT | NO | AUDIT_ONLY |
| 98 | NO | SEGMENT_LEVEL_ONLY | YES_AFTER_SEGMENTATION_AND_RELABELING | YES_HIGH_VALUE | NOT_YET | HIGH |
| 95 | YES_WITH_CORRECTED_TARGET | YES | YES | YES_HIGH_VALUE | NOT_YET | VERY_HIGH |
| 107 | NO | HOLD | HOLD | YES_FOR_BRANCH_AND_POLICY_CHECKING | NO | VERIFY_FIRST |

---

## 4. 四条失败的人工审计结论

### Task 59

**分类：** `BENCHMARK_ALIGNMENT_FAILURE`

**主标签：** `USER_SIMULATOR_GOLD_MISMATCH`

**结论：**

User Simulator explicitly instructed the Agent to cancel #W2702727 and leave #W8268610 unchanged, while Static Golden expected cancellation of #W8268610 and address modification of #W2702727. The Agent followed the final explicit user authorization after clarification and confirmation.

**第一次偏离点：**

The divergence begins in the generated user trajectory: the User Simulator identifies #W2702727 as the older order to cancel, conflicting with the Static Golden branch.

**推荐用途：**

- Benchmark / evaluator auditing
- User Simulator consistency checking
- Final authorized intent tracking
- Reward-label-noise detection

**禁止直接做的事情：**

- Do not treat Reward=0 as direct Agent negative label
- Do not train the Agent to ignore the user's final explicit intent

---

### Task 98

**分类：** `MIXED_BADCASE`

**主标签：** `DYNAMIC_INTENT_STATIC_GOLD_MISMATCH`

**结论：**

All three write tools actually executed successfully. The two exchange actions mismatched Static Golden only because the Agent used the Visa card explicitly confirmed by the user, while Static Golden expected a different card. Separately, the Agent made a real production-risk error: cancel_pending_order cancelled the entire order and refunded $1058.79, but the Agent described it as cancelling only the skateboard with a $202.13 refund.

**第一次偏离点：**

Benchmark divergence occurs when the user explicitly confirms credit_card_3951670 while Static Golden expects credit_card_8105988. The first real Agent error occurs after cancellation when the Agent misrepresents whole-order cancellation as single-item cancellation.

**推荐用途：**

- Action-scope verifier
- Post-tool state grounding
- Claim-action consistency verifier
- Final authorized payment-method tracking
- Mixed reward-label-noise analysis

**禁止直接做的事情：**

- Do not treat the entire Reward=0 trajectory as a single negative
- Do not penalize the user-confirmed payment method solely because Static Golden differs

---

### Task 95

**分类：** `VALID_AGENT_FAILURE`

**主标签：** `ENVIRONMENT_STATE_SEMANTICS_MISUNDERSTANDING`

**结论：**

The Agent correctly found the target laptop variant 9844888101, but incorrectly interpreted one matching Variant as one physical inventory unit. It therefore concluded that only one of the two laptops could be exchanged and prematurely escalated to a human. Static Golden shows that both orders should use the same target variant.

**第一次偏离点：**

After finding item_id=9844888101 with available=true, the Agent incorrectly infers that only one physical unit exists and that a second exchange to the same Variant is impossible.

**推荐用途：**

- SFT badcase correction
- Environment-schema grounding
- Tool-state semantic reasoning
- Premature-escalation verifier
- Multi-goal completeness verifier
- Preference pair construction

**禁止直接做的事情：**

- Do not confuse Variant identity with physical inventory count
- Do not escalate without evidence of a real capability boundary

---

### Task 107

**分类：** `UNRESOLVED_POLICY_TOOL_SEMANTICS`

**主标签：** `CONDITIONAL_BRANCH_STATIC_GOLD_MISMATCH`

**结论：**

The user first requested a fresh pair of hiking boots with the same specs. Only if that exchange was not allowed should the fallback be size 9, leather, waterproof. The Agent performed a same-variant exchange, which the Tool accepted successfully. Static Golden instead fixes the fallback variant 8106223139. Policy semantics must be verified before declaring the Agent wrong.

**第一次偏离点：**

The Agent chooses the primary same-spec replacement branch because the original variant is available and the Tool accepts same-item exchange, while Static Golden assumes the fallback branch.

**推荐用途：**

- Conditional-intent tracking
- Branch-aware evaluator research
- Policy/tool semantic consistency audit
- Same-variant replacement rule verification

**禁止直接做的事情：**

- Do not use as a negative Agent sample before Policy verification
- Do not assume DB=0 proves the Tool execution was incorrect

---

## 5. 当前最重要的项目结论

### 5.1 Reward=0 不等于 Agent Failure

四条 Raw Failure 经人工审计后：

- Task 59：主要是 User Simulator / Static Golden 对齐问题；
- Task 98：Benchmark mismatch 与真实 Agent 错误并存；
- Task 95：当前最干净、最明确的真实 Agent Failure；
- Task 107：需要先验证 Policy / Tool 对 same-variant exchange 的真实语义。

因此不能直接执行：

```text
Reward=1 -> 正样本
Reward=0 -> 负样本
```

这种粗粒度数据构造会把 Evaluator Label Noise 注入后训练数据。

### 5.2 当前最有价值的 SFT Badcase

Task 95。

核心能力缺口：

```text
Environment Schema Understanding
-> Variant / item_id 语义
-> Capability Boundary Judgment
-> Multi-goal Completion
-> Avoid Premature Escalation
```

### 5.3 当前最有价值的 Verifier Badcase

Task 98。

核心需要验证：

```text
User Authorized Scope
-> Tool Actual Scope
-> Tool Result / Final DB State
-> Agent Final Claim
```

尤其需要检测：

```text
Tool refund = $1058.79
Agent claim = $202.13
```

这种 Claim-State Inconsistency 是真实业务 Agent 的高风险问题。

### 5.4 当前暂时不能用于训练的样本

Task 59 和 Task 107。

原因不是它们没有研究价值，恰恰相反：

它们主要用于：

- Benchmark Audit；
- Evaluator Alignment；
- Policy / Tool Semantics Verification；
- Reward Label Noise Detection。

在人工裁决完成前，不应直接作为 Agent 负样本。

---

## 6. 对后训练路线的影响

当前证据只支持：

```text
Prompt Base
-> Failure Audit
-> Data Cleaning
-> SFT / Verifier Dataset Construction
```

目前还不能证明必须直接进入 RL。

RL 是否必要，需要后续通过：

- 更大规模 Base failure distribution；
- SFT 后残余失败；
- Verifier 能否解决核心错误；
- Preference / policy-compliance error 是否仍持续存在；

再决定。

因此当前结论是：

> 先做好数据裁决、SFT 和 Verifier，再决定 RL。

---

## 7. 当前版本边界

Failure Taxonomy v1 只基于当前冻结 20-task Trial-1 中的 4 个 Reward=0 样本。

它不能直接代表整个 tau2-bench 的全局失败分布。

后续随着：

- 更多任务；
- 多 Trial 稳定性实验；
- 16 条 Reward=1 成功轨迹质量审计；
- Policy 源码核验；

Taxonomy 需要继续升级为 v2、v3。
