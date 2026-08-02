# Failure Taxonomy v2 + Training Data Eligibility Matrix v2

## 1. 冻结实验结果

- 实验：Retail Prompt Base / Trial-1
- 总任务数：20
- Raw Success：16
- Raw Failure：4
- Raw Success Rate：80%

> Failure Taxonomy 的人工审计不会回改冻结的 Raw Baseline。
> 原始结果始终保留为 16/20 = 80%。

---

## 2. Failure Taxonomy v2

| Task | Raw | 人工审计分类 | 主标签 | 真实 Agent Failure | Benchmark 可疑 | Tool Enforcement Gap |
|---|---:|---|---|---|---|---|
| 59 | 0.0 | BENCHMARK_ALIGNMENT_FAILURE | USER_SIMULATOR_GOLD_MISMATCH | NO | YES | NO |
| 98 | 0.0 | MIXED_BADCASE | CLAIM_ACTION_INCONSISTENCY | YES | YES | NO |
| 95 | 0.0 | VALID_AGENT_FAILURE | ENVIRONMENT_STATE_SEMANTICS_MISUNDERSTANDING | YES | NO | NO |
| 107 | 0.0 | VALID_AGENT_FAILURE | POLICY_GROUNDING_FAILURE | YES | NO | YES |

### 当前结构

```text
4 个 Raw Reward=0

├── Benchmark Alignment Failure
│   └── Task 59
│
├── Mixed Badcase
│   └── Task 98
│
└── Valid Agent Failure
    ├── Task 95：Environment Semantics
    └── Task 107：Policy Grounding
```

---

## 3. Training Data Eligibility Matrix v2

| Task | 负 SFT | 修正 SFT | Preference | Verifier | RL | 状态 |
|---|---|---|---|---|---|---|
| 59 | EXCLUDE | POSSIBLE_AFTER_ADJUDICATION | EXCLUDE_AS_RAW_PAIR | YES | NO | EXCLUDE_FROM_RAW_NEGATIVE_POOL |
| 98 | SEGMENT_ONLY | YES_AFTER_SEGMENTATION | YES_AFTER_RELABELING | YES_HIGH_VALUE | NOT_YET | SEGMENT_AND_RELABEL |
| 95 | YES_WITH_CORRECTED_TARGET | YES | YES | YES_HIGH_VALUE | NOT_YET | ELIGIBLE_AFTER_CORRECTION |
| 107 | YES | YES | YES | YES_VERY_HIGH_VALUE | NOT_YET | ELIGIBLE_AFTER_CORRECTION |

---

## 4. 各任务最终裁决

### Task 59

**分类：** `BENCHMARK_ALIGNMENT_FAILURE`

**主标签：** `USER_SIMULATOR_GOLD_MISMATCH`

**结论：**

The User Simulator explicitly authorized cancellation of #W2702727 and no change to #W8268610, while Static Golden expected cancellation of #W8268610 and modification of #W2702727. The Agent followed the final explicit user intent.

**第一次偏离点：**

The generated user trajectory diverged from Static Golden before the Agent made the disputed write action.

**推荐用途：**

- Benchmark audit
- User Simulator / Golden consistency checking
- Final authorized intent tracking
- Reward-label-noise detection

---

### Task 98

**分类：** `MIXED_BADCASE`

**主标签：** `CLAIM_ACTION_INCONSISTENCY`

**结论：**

All three write tools executed successfully. Two exchange actions mismatched Static Golden because the Agent used the payment method explicitly confirmed by the user. Separately, the Agent made a real production-risk error: the cancellation tool cancelled the entire order and refunded $1058.79, but the Agent described it as cancelling only the skateboard with a $202.13 refund.

**第一次偏离点：**

Benchmark divergence begins at payment-method confirmation. The first genuine Agent failure occurs after the cancellation Tool result, when the Agent misstates the actual action scope and refund amount.

**推荐用途：**

- Action-scope verifier
- Post-tool state grounding
- Claim-action consistency verifier
- Final authorized intent tracking

---

### Task 95

**分类：** `VALID_AGENT_FAILURE`

**主标签：** `ENVIRONMENT_STATE_SEMANTICS_MISUNDERSTANDING`

**结论：**

The Agent correctly found target Variant 9844888101 but misinterpreted one matching Variant as one physical inventory unit. It incorrectly concluded that only one of two laptops could use that Variant and prematurely escalated to a human.

**第一次偏离点：**

After observing item_id=9844888101 with available=true, the Agent incorrectly inferred that only one physical unit was available.

**推荐用途：**

- SFT badcase correction
- Environment schema grounding
- Premature-escalation verifier
- Multi-goal completeness verifier
- Preference pair construction

---

### Task 107

**分类：** `VALID_AGENT_FAILURE`

**主标签：** `POLICY_GROUNDING_FAILURE`

**结论：**

The Agent incorrectly treated same-variant exchange 1615379700 -> 1615379700 as Policy-compliant. Retail Policy requires exchange to a different product option. The Tool nevertheless accepted the invalid same-item exchange, revealing a Policy-Tool Enforcement Gap.

**第一次偏离点：**

The Agent states that because the same Hiking Boots Variant is available, it can exchange the item for the same item. This violates the Retail exchange Policy requirement that the replacement use a different product option.

**推荐用途：**

- Policy-grounding SFT
- Policy-aware preference pair
- Pre-tool Policy verifier
- Policy-Tool consistency testing
- Tool guardrail regression test

---

## 5. Task 107 源码核验后的关键结论

Task 107 在 v1 中属于待核验案例。

源码核验后确认：

```text
Retail Policy
要求 exchange 使用 different product option

        ↓

Agent
却执行 old_item_id == new_item_id

        ↓

exchange_delivered_order_items
没有阻止这一非法动作

        ↓

Tool 返回成功

        ↓

NL Evaluator
只检查两个订单是否发生 exchange
没有检查 Policy compliance
```

因此 Task 107 是一个完整的四层失配案例：

```text
Policy Rule
    ↓
Agent Policy Grounding
    ↓
Tool Enforcement
    ↓
Evaluator Coverage
```

最终标签：

- `POLICY_GROUNDING_FAILURE`
- `POLICY_TOOL_ENFORCEMENT_GAP`
- `INVALID_ACTION_ALLOWED_BY_TOOL`
- `NL_EVALUATOR_COVERAGE_GAP`

---

## 6. 当前最优先的训练样本

### 第一优先级：Task 95

训练能力：

- Environment Schema Understanding
- Variant / Inventory 语义区分
- Capability Boundary Judgment
- Avoid Premature Escalation
- Multi-goal Completion

### 第一优先级：Task 107

训练能力：

- Policy Grounding
- Policy-aware Tool Calling
- Conditional Intent / Fallback Handling
- Pre-tool Policy Verification
- Tool Enforcement Gap Detection

### Verifier 高价值样本：Task 98

重点：

```text
User Authorized Scope
→ Tool Actual Scope
→ Tool Result
→ Final DB State
→ Agent Final Claim
```

### 排除普通负样本池：Task 59

原因：

Simulator / Static Golden 与最终用户意图发生冲突。

---

## 7. 当前后训练结论

当前证据支持的路线是：

```text
Prompt Base
    ↓
Trajectory Audit
    ↓
Failure Taxonomy
    ↓
Training Data Cleaning
    ↓
Corrected SFT
    +
Verifier
    ↓
重新评测
    ↓
再决定是否需要 Preference / RL
```

目前没有足够证据支持直接跳到 RL。

RL 必须由 SFT / Verifier 后仍持续存在的系统性失败来证明必要性。

---

## 8. 版本边界

Failure Taxonomy v2 仅来自当前冻结的 20-task Trial-1。

它不能代表整个 tau2-bench 的全局失败分布。

后续仍需要：

- 对成功轨迹做质量审计；
- 做多 Trial 稳定性实验；
- 扩展训练数据；
- 重新运行 Base / SFT / Verifier 对照实验。
