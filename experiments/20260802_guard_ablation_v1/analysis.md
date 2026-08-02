# Retail Guard 规则族消融实验 V1

## 实验边界

- 数据：15 个开发者构造的合成场景
- 官方指标：否
- 新增 LLM 调用：0
- Reference action / gold DB：未使用
- 方法：移除指定 blocking finding 后重新计算 Guard decision

## 结果

| Variant | TP | FP | FN | TN | Recall | Decision accuracy | 漏掉的风险场景 |
|---|---:|---:|---:|---:|---:|---:|---|
| no_guard | 0 | 0 | 9 | 6 | 0.00% | 40.00% | cancel_delivered_order, cross_product_exchange, item_request_expands_to_order, parallel_mutating_calls, premature_transfer_with_actionable_variant, same_item_exchange, second_order_mutation, unavailable_replacement_variant, unknown_payment_method |
| full_guard | 9 | 0 | 0 | 6 | 100.00% | 100.00% | - |
| without_scope | 8 | 0 | 1 | 6 | 88.89% | 93.33% | item_request_expands_to_order |
| without_order_state | 8 | 0 | 1 | 6 | 88.89% | 93.33% | cancel_delivered_order |
| without_payment | 8 | 0 | 1 | 6 | 88.89% | 93.33% | unknown_payment_method |
| without_variant | 6 | 0 | 3 | 6 | 66.67% | 80.00% | cross_product_exchange, same_item_exchange, unavailable_replacement_variant |
| without_protocol | 8 | 0 | 1 | 6 | 88.89% | 93.33% | parallel_mutating_calls |
| without_one_shot | 8 | 0 | 1 | 6 | 88.89% | 93.33% | second_order_mutation |
| without_goal_completion | 8 | 0 | 1 | 6 | 88.89% | 93.33% | premature_transfer_with_actionable_variant |

## 如何解释

- `no_guard` 是合成场景上的无防护基线。
- `full_guard` 是完整规则集合。
- 其他行每次只移除一个规则族，用于定位该规则族覆盖的风险 case。
- 若安全对照回归数为 0，只说明本套合成负对照未被误拦截。

## 限制

- The source scenarios and expectations are developer-authored, not independently adjudicated human gold.
- This is a finding-level deterministic counterfactual: it filters selected blocking findings and recomputes the decision.
- Ablation results measure rule coverage on this synthetic suite, not online recovery, production precision, or general Retail performance.
- No model calls, reference actions, gold database states, or official Tau2 test tasks are used.
