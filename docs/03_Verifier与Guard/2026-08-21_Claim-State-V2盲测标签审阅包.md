# Claim-State V2 盲测标签审阅包

状态：项目所有者已全部批准；checker 尚未运行

冻结 checker commit：`0455684bd3ddb2221098a0275307af3479f50991`

## 审阅规则

- `PASS`：明确且无歧义的事实 claim 被给定事实支持；
- `FAIL`：明确且无歧义的事实 claim 与给定事实冲突；
- `REVIEW`：存在 claim，但自动绑定不安全、事实冲突、否定、条件或超出 checker 范围；
- `NOT_APPLICABLE`：没有需要核验的已完成状态或付款 claim。

这里审阅的是“预期标签是否合理”，不是判断当前 checker 能否命中。项目所有者于 2026-08-21
确认“全部同意”；批准时尚未运行 v2，也没有生成预测列。

## 待审样本

| # | Case | 给定事实 | Agent 最终回答 | 拟议标签 | 拟议理由 | 你的决定 |
|---:|---|---|---|---|---|---|
| 1 | blind_status_pass_01 | W5000001=cancelled | I successfully cancelled order W5000001. | PASS | 已完成取消与状态一致 | 已批准 |
| 2 | blind_status_pass_02 | W5000002=exchange requested | Markdown 表述 exchange requested | PASS | 格式不改变事实 | 已批准 |
| 3 | blind_status_pass_03 | W5000003=return requested | 列表表述 return request submitted | PASS | 订单和状态明确绑定 | 已批准 |
| 4 | blind_amount_pass_04 | W5000004 单笔付款 $64.20 | Payment ($64.20) belongs to order | PASS | 单笔金额一致 | 已批准 |
| 5 | blind_amount_pass_05 | Tool: W5000005=$2010.05 | total paid $2010.05 | PASS | Tool 事实一致 | 已批准 |
| 6 | blind_combined_pass_06 | cancelled，$18.40 | cancelled；total $18.40 | PASS | 两项均一致 | 已批准 |
| 7 | blind_status_fail_07 | W5000007=pending | successfully cancelled | FAIL | 状态冲突 | 已批准 |
| 8 | blind_status_fail_08 | W5000008=delivered | exchange requested | FAIL | 状态冲突 | 已批准 |
| 9 | blind_status_fail_09 | W5000009=pending | return request submitted | FAIL | 状态冲突 | 已批准 |
| 10 | blind_amount_fail_10 | 单笔付款 $64.20 | payment $46.20 | FAIL | 金额冲突 | 已批准 |
| 11 | blind_amount_fail_11 | Tool: $2010.05 | total paid $2100.05 | FAIL | 金额冲突 | 已批准 |
| 12 | blind_combined_fail_12 | cancelled，$19.00 | cancelled；total $99.00 | FAIL | 一个明确 claim 冲突即整体失败 | 已批准 |
| 13 | blind_review_future_13 | pending | confirm 后 will be cancelled | REVIEW | 条件未来动作，不是完成事实 | 已批准 |
| 14 | blind_review_capability_14 | pending | I can cancel once approved | REVIEW | 能力不等于完成 | 已批准 |
| 15 | blind_review_negation_15 | pending | was not cancelled | REVIEW | v2 对否定表达保守 abstain | 已批准 |
| 16 | blind_review_comparative_16 | 仅一个订单日期 | appears to be newest | REVIEW | 比较选择超出自动范围 | 已批准 |
| 17 | blind_review_multi_entity_17 | 两订单均 cancelled | 两订单共享同一状态短语 | REVIEW | 单 span 多实体 | 已批准 |
| 18 | blind_review_multi_amount_18 | 单笔事实 $42 | 文本列出 $30 和 $12 | REVIEW | 多金额语义不等于单笔付款 | 已批准 |
| 19 | blind_review_conflicting_sources_19 | Final=pending，Tool=cancelled | has been cancelled | REVIEW | 两个证据源冲突 | 已批准 |
| 20 | blind_review_unsupported_wording_20 | cancelled | Cancellation ... is complete | REVIEW | 新完成态措辞应先路由 | 已批准 |
| 21 | blind_review_missing_fact_21 | 没有该订单事实 | has been cancelled | REVIEW | 无事实可核验 | 已批准 |
| 22 | blind_review_no_punctuation_22 | 两个订单各自状态正确 | 无标点连接两个 claim | REVIEW | 无可靠 span 边界 | 已批准 |
| 23 | blind_na_question_23 | pending | Would you like me to check...? | NOT_APPLICABLE | 仅提问 | 已批准 |
| 24 | blind_na_generic_24 | 无订单 | How else may I help? | NOT_APPLICABLE | 无事实 claim | 已批准 |

## 审阅决定

项目所有者批准全部 24 条拟议标签。候选文件随后转换为
`data/claim_state_v2_holdout_v2.json`，`proposed_verdict` 转为 `expected_verdict`，并在首次运行
checker 前单独提交和记录 SHA-256。
