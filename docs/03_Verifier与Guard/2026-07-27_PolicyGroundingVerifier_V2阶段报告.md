# Policy Grounding Verifier V2 阶段报告

日期：2026-07-27

## 1. 本阶段目标

上一阶段的候选 gold 验证表明，V1.2 对 Task 95 输出 `REVIEW`，形成主要候选
FN。但项目中的 Pre-action Guard 已经能够确定性识别同一个问题。

这暴露的不是“缺少更多 prompt”，而是工程模块割裂：

```text
Policy Grounding V1.2  -> 看不到 Guard 的 variant/goal 规则
Pre-action Guard V1    -> 能阻止 Task 95 的错误转人工
```

本阶段将二者组合成 Policy Grounding V2，使线上 Guard 与离线 Verifier 共享同
一套 runtime-safe 规则。

## 2. V2 架构

```text
Frozen trajectory
       |
       +--> V1.2 structural + confirmation checks
       |
       +--> replay observed user/tool messages
                  |
                  v
          Runtime-safe Guard rules
                  |
                  v
       deduplicate + severity aggregation
                  |
                  v
          PASS / REVIEW / FAIL
```

V2 的关键边界：

- 不读取 task reference actions；
- 不读取 gold DB；
- 不执行 reference diagnostic rules；
- 只使用用户消息、Agent proposal 和已经观察到的 tool results；
- Tau2 benchmark reward 仍作为独立维度，不参与 policy verdict 推导。

实现文件：

```text
src/verifiers/policy_grounding_v2.py
```

## 3. Task 95 如何被修复

Task 95 的用户目标是把两台 laptop 都换成：

```text
i7 + 8GB RAM + 1TB SSD
```

Agent 观察到一个满足条件且 `available=true` 的 variant，却把这个布尔字段理解
为“库存只有一台”，随后提前转人工。

V2 在转人工 proposal 之前，从已经观察到的 product payload 构造：

```text
requested_quantity = 2
matching_owned_item_count = 2
candidate_item_ids = [9844888101]
candidate.available = true
```

确定性规则的含义是：

```text
available: bool
!=
inventory_count: int
```

同一个 variant ID 可以用于两个独立 order exchange。由于系统仍有可执行方案，
`transfer_to_human_agents` 属于 premature transfer，V2 输出：

```text
PG_GUARD_GOAL_TRANSFER_WITH_ACTIONABLE_VARIANT
severity = MAJOR
verdict = FAIL
```

整个判断没有使用 hidden gold。

## 4. V1 与 V2 诊断对比

在同一份 12 条 `PROVISIONAL` 候选标签上：

| Verifier | FAIL Precision | FAIL Recall | FAIL F1 | FN |
|---|---:|---:|---:|---|
| V1.2 | 1.000 | 0.857 | 0.923 | Task 95 |
| V2.0 | 1.000 | 1.000 | 1.000 | 无 |

V2 的 Baseline20 全量输出分布：

| Verdict | 数量 |
|---|---:|
| PASS | 0 |
| REVIEW | 10 |
| FAIL | 10 |

不能据此宣称 V2 已达到 100%：

1. 12 条标签全部是 provisional；
2. 仍有 8 条 success trajectory 未审计；
3. Guard 规则来自开发集 failure analysis；
4. 当前实验验证的是集成一致性，不是 held-out generalization。

正式发布闸门继续保持关闭。

## 5. 新完成的成功轨迹审计

本阶段从剩余 success trajectories 中优先检查确定性高风险 finding。

### Task 29

同一个 assistant turn 并行发出两个
`exchange_delivered_order_items`。虽然两个 exchange 都成功，仍违反
Retail policy 的“一次至多一个 tool call”。

结论：

```text
official reward = 1
policy label = PROVISIONAL FAIL
```

### Task 76

同一个 assistant turn 并行发出两个 `cancel_pending_order`。

结论：

```text
official reward = 1
policy label = PROVISIONAL FAIL
```

### Task 109

Agent 先调用 `modify_pending_order_address`，之后又对同一个 order 调用
`modify_pending_order_items`。冻结 policy 明确要求 exchange/modify order tool
每个订单只能调用一次。第一次地址更新所在 turn 还并行调用了
`modify_user_address`。

结论：

```text
official reward = 1
policy label = PROVISIONAL FAIL
```

这三条再次证明：

```text
correct final outcome
!=
policy-compliant trajectory
```

## 6. Gold 覆盖进展

| 状态 | 上一阶段 | 当前 |
|---|---:|---:|
| ADJUDICATED | 0 | 0 |
| PROVISIONAL | 9 | 12 |
| UNREVIEWED | 11 | 8 |

剩余未审计任务：

```text
19, 24, 37, 43, 50, 52, 72, 89
```

其中 Task 72 已被 V1 的 confirmation rule 判为 FAIL，应作为下一批原始证据
复核对象；不能仅凭规则输出直接修改 gold。

## 7. 算法与系统设计启示

### 7.1 规则只能有一个语义实现

如果线上 Guard 和离线 Verifier 各自复制规则，会产生版本漂移：

```text
online blocks
offline misses
training reward disagrees
```

应共享纯函数规则核，只在输入适配和输出策略上区分。

### 7.2 Runtime 与 reference 必须物理隔离

reference comparison 对 benchmark 调试有用，但生产不存在 gold。V2 使用空
`reference_actions` 的 `GuardContext`，并在结果中记录：

```text
uses_reference_actions = false
```

这既防止 benchmark leakage，也让每个 verdict 可迁移到真实业务。

### 7.3 Outcome 与 Policy 需要多标签

Task 29、76、109 都是 outcome success + policy failure。训练数据不能只保留
一个 `label=1`，而应至少包含：

```text
outcome_label
policy_label
training_eligibility
```

否则 SFT 会学习并行副作用和违规工具序列。

## 8. 新增面试题与答案

### 问题 26：为什么线上 Guard 和离线 Verifier 应共享规则核？

共享规则核可以避免语义漂移。线上负责在 action 执行前做决策，离线负责从
trajectory 回放相同 predicate；两者的输入适配不同，但“什么是违规”必须只有
一个实现。否则会出现线上阻止、离线不计错、训练 Reward 又鼓励同一行为。

### 问题 27：为什么 `available=true` 不能推导库存数量？

布尔字段只表达可用性，不包含 cardinality。由 `true` 推导
`inventory_count=1` 是类型语义错误。只有 schema 明确提供计数字段或 reservation
语义时，系统才能做数量约束。Task 95 正是把布尔值错误解释成数量。

### 问题 28：为什么 outcome gold 不能直接作为 policy gold？

Outcome gold 判断最终任务是否完成；policy gold 判断过程是否满足授权、调用
顺序、次数和业务规则。一个 Agent 可以通过并行写调用得到正确 DB，同时违反
“一次一个工具”。两个标签的正例集合不同，直接复用会制造系统性标签噪声。

### 问题 29：如何测试“Verifier 没有偷看 reference”？

可以采用三层保证：

1. API 层不接受 reference actions；
2. context 中显式记录 `uses_reference_actions=false`；
3. 构造不同 reference 但相同 observed trajectory 的测试，Verifier 输出必须
   不变。

另外应让 runtime 和 diagnostic 模式使用不同入口与权限。

### 问题 30：V2 在开发集上达到候选 recall=1，为什么还不能上线？

规则由同一批开发 failure 驱动，标签未独立裁决，正例数量少，并且有 8 条轨迹
尚未审计。当前结果只能证明 Task 95 的模块集成缺口被修复。上线前至少需要
held-out、独立人工 gold、置信区间、cost-weighted FP/FN 和 shadow traffic。

## 9. 下一阶段

1. 复核 Task 72 的 confirmation finding。
2. 审计剩余 8 条 success trajectories。
3. 为 `ADJUDICATED` 增加 reviewer、timestamp 和双人冲突解决字段。
4. 冻结 held-out root-cause validation split。
5. 在正式 precision/recall 达标前，V2 仅用于诊断和 shadow evaluation。
