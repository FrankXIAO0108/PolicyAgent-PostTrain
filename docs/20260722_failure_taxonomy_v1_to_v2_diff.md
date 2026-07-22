# Failure Taxonomy v1 → v2 变更记录

v1 JSON 是否存在：`True`

## 核心变更

只有 Task 107 的最终裁决发生实质变化。

### v1

```text
UNRESOLVED_POLICY_TOOL_SEMANTICS
CONDITIONAL_BRANCH_STATIC_GOLD_MISMATCH
HOLD_UNTIL_POLICY_VERIFIED
```

### 源码核验

冻结上游 commit：`58e5e1ace69302e6982d27014569c03e0ffccdd2`

核验对象：

- `D:\tau2-bench\data\tau2\domains\retail\policy.md`
- `D:\tau2-bench\src\tau2\domains\retail\tools.py`

核验结论：

1. Retail Policy 要求 exchange 使用不同 product option。
2. Agent 执行了 old_item_id == new_item_id。
3. exchange Tool 未强制阻止该非法动作。
4. NL Evaluator 只检查是否发生 exchange，没有检查 Policy compliance。

### v2

```text
VALID_AGENT_FAILURE
POLICY_GROUNDING_FAILURE
POLICY_TOOL_ENFORCEMENT_GAP
ELIGIBLE_AFTER_CORRECTION
```

## 未发生变化

- Task 59：仍为 Benchmark Alignment Failure。
- Task 98：仍为 Mixed Badcase。
- Task 95：仍为 Valid Agent Failure。
- Raw Baseline：仍为 16/20 = 80%。

## 版本原则

不删除历史结论。

v1 用于保留第一次人工审计过程；
v2 用于记录源码核验后的最新裁决。
