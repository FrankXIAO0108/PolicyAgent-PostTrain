# Claim-State V2 盲测标签审阅包

状态：等待项目所有者审阅；checker 尚未运行

冻结 checker commit：`0455684bd3ddb2221098a0275307af3479f50991`

## 审阅规则

- `PASS`：明确且无歧义的事实 claim 被给定事实支持；
- `FAIL`：明确且无歧义的事实 claim 与给定事实冲突；
- `REVIEW`：存在 claim，但自动绑定不安全、事实冲突、否定、条件或超出 checker 范围；
- `NOT_APPLICABLE`：没有需要核验的已完成状态或付款 claim。

这里审阅的是“预期标签是否合理”，不是判断当前 checker 能否命中。为保持盲测有效性，在你确认
以前不会运行 v2，也不会生成预测列。

## 待审样本

| # | Case | 给定事实 | Agent 最终回答 | 拟议标签 | 拟议理由 | 你的决定 |
|---:|---|---|---|---|---|---|
| 1 | blind_status_pass_01 | W5000001=cancelled | I successfully cancelled order W5000001. | PASS | 已完成取消与状态一致 | 待审 |
| 2 | blind_status_pass_02 | W5000002=exchange requested | Markdown 表述 exchange requested | PASS | 格式不改变事实 | 待审 |
| 3 | blind_status_pass_03 | W5000003=return requested | 列表表述 return request submitted | PASS | 订单和状态明确绑定 | 待审 |
| 4 | blind_amount_pass_04 | W5000004 单笔付款 $64.20 | Payment ($64.20) belongs to order | PASS | 单笔金额一致 | 待审 |
| 5 | blind_amount_pass_05 | Tool: W5000005=$2010.05 | total paid $2010.05 | PASS | Tool 事实一致 | 待审 |
| 6 | blind_combined_pass_06 | cancelled，$18.40 | cancelled；total $18.40 | PASS | 两项均一致 | 待审 |
| 7 | blind_status_fail_07 | W5000007=pending | successfully cancelled | FAIL | 状态冲突 | 待审 |
| 8 | blind_status_fail_08 | W5000008=delivered | exchange requested | FAIL | 状态冲突 | 待审 |
| 9 | blind_status_fail_09 | W5000009=pending | return request submitted | FAIL | 状态冲突 | 待审 |
| 10 | blind_amount_fail_10 | 单笔付款 $64.20 | payment $46.20 | FAIL | 金额冲突 | 待审 |
| 11 | blind_amount_fail_11 | Tool: $2010.05 | total paid $2100.05 | FAIL | 金额冲突 | 待审 |
| 12 | blind_combined_fail_12 | cancelled，$19.00 | cancelled；total $99.00 | FAIL | 一个明确 claim 冲突即整体失败 | 待审 |
| 13 | blind_review_future_13 | pending | confirm 后 will be cancelled | REVIEW | 条件未来动作，不是完成事实 | 待审 |
| 14 | blind_review_capability_14 | pending | I can cancel once approved | REVIEW | 能力不等于完成 | 待审 |
| 15 | blind_review_negation_15 | pending | was not cancelled | REVIEW | v2 对否定表达保守 abstain | 待审 |
| 16 | blind_review_comparative_16 | 仅一个订单日期 | appears to be newest | REVIEW | 比较选择超出自动范围 | 待审 |
| 17 | blind_review_multi_entity_17 | 两订单均 cancelled | 两订单共享同一状态短语 | REVIEW | 单 span 多实体 | 待审 |
| 18 | blind_review_multi_amount_18 | 单笔事实 $42 | 文本列出 $30 和 $12 | REVIEW | 多金额语义不等于单笔付款 | 待审 |
| 19 | blind_review_conflicting_sources_19 | Final=pending，Tool=cancelled | has been cancelled | REVIEW | 两个证据源冲突 | 待审 |
| 20 | blind_review_unsupported_wording_20 | cancelled | Cancellation ... is complete | REVIEW | 新完成态措辞应先路由 | 待审 |
| 21 | blind_review_missing_fact_21 | 没有该订单事实 | has been cancelled | REVIEW | 无事实可核验 | 待审 |
| 22 | blind_review_no_punctuation_22 | 两个订单各自状态正确 | 无标点连接两个 claim | REVIEW | 无可靠 span 边界 | 待审 |
| 23 | blind_na_question_23 | pending | Would you like me to check...? | NOT_APPLICABLE | 仅提问 | 待审 |
| 24 | blind_na_generic_24 | 无订单 | How else may I help? | NOT_APPLICABLE | 无事实 claim | 待审 |

## 你需要做什么

如果 24 条标签和理由都接受，直接回复：

```text
全部接受，可以冻结并评测
```

如果不同意某条，只需要回复 case 编号、新标签和理由。例如：

```text
第 15 条改为 PASS：否定 claim 与 pending 状态一致，应当视为可验证支持。
```

收到你的决定后，项目才会把 `proposed_verdict` 转成冻结 `expected_verdict`、记录审阅状态与数据
哈希，然后首次运行 checker。
