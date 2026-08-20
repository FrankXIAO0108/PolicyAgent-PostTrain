# Base vs SFT 多维过程评测与 Tool Use 分析

## 1. 评测对象与证据

本报告比较同一组 30 个 Retail 任务上的 Base 与教师 SFT 模型。它是项目内部冻结开发评测，不是 tau2 官方榜单成绩。

- Base 原始产物：`_local_private_runs/teacher_eval_base_v1`
- SFT 原始产物：`_local_private_runs/teacher_eval_sft_v1`
- SFT task 100 重跑：`_local_private_runs/teacher_eval_sft_task100_rerun_20260820_v1`
- SFT task 107 重跑：`_local_private_runs/teacher_eval_sft_task107_rerun_20260820_v1`
- 本次逐任务 Evaluation Card：`_local_private_runs/teacher_eval_process_analysis_20260820_v2/evaluation_cards.json`
- 本次结构化对比：`_local_private_runs/teacher_eval_process_analysis_20260820_v2/comparison.json`

task 100、107 使用上下文窗口修正后的重跑产物替换原基础设施失败，因此当前为可比的 30/30。每个任务卡记录实际使用的 `returned_results.json` 路径、SHA-256 和是否为替换重跑。

## 2. 为什么不能只看总 Reward

Tau2 总 reward 主要回答“最终数据库状态和自然语言断言是否满足”，不能完整回答：

1. Agent 是否准确跟随了用户最后意图；
2. 是否选择了正确且必要的工具；
3. 是否发生无效查询、循环或穷举；
4. 写操作前是否获得确认；
5. 工具失败后是否仍然谎称成功。

因此本次为每个任务建立分层 Evaluation Card，但没有把各维度武断加权成一个新总分：权重尚无人类金标或业务损失数据支持。

| 维度 | 当前信号 | 权威级别 |
|---|---|---|
| 任务完成 | Tau2 reward | Benchmark outcome |
| 最终状态 | `db_check.db_match` | Benchmark outcome |
| 最终回复 | `nl_assertions` | Benchmark outcome |
| 参考动作覆盖 | `action_checks` | Benchmark diagnostic |
| 最新意图跟随 | Policy Grounding V2.2 | Provisional diagnostic |
| Policy 合规 | Policy Grounding V2.2 | Provisional diagnostic |
| 工具效率 | 调用数、错误数、重复、同工具连续扇出 | Diagnostic |

## 3. 核心结果

| 指标 | Base | SFT | 解释 |
|---|---:|---:|---|
| 成功任务 | 12/30 | 17/30 | SFT 提升 5 个任务 |
| 成功率 | 40.0% | 56.7% | 提升 16.7 个百分点 |
| 平均工具调用 | 6.63 | 9.43 | SFT 增加 2.80 次 |
| 工具调用中位数 | 6 | 7 | 非纯粹由单个极端值造成 |
| 平均读取调用 | 5.37 | 7.97 | 主要膨胀来自查询，而非写操作 |
| 平均写入调用 | 1.27 | 1.47 | 小幅增加 |
| 平均完全重复调用 | 0.13 | 0.47 | 循环/冗余候选增加 |
| 平均同工具最长连续调用 | 2.23 | 3.20 | 查询扇出增加 |
| Tool error 结果数 | 15 | 19 | 工具错误未随 SFT 减少 |

结论不能写成“SFT 后工具调用下降”。事实相反：SFT 提高了任务成功率和动作覆盖，但过程效率、错误恢复和停止能力变差。

## 4. 配对分层分析

平均调用数必须按结果变化分层，否则“提前失败导致调用少”会被误认为高效率。

| 分层 | 任务数 | Base 平均调用 | SFT 平均调用 | SFT-Base |
|---|---:|---:|---:|---:|
| 双方都成功 | 9 | 5.11 | 6.44 | +1.33 |
| Base 失败、SFT 成功 | 8 | 6.38 | 7.88 | +1.50 |
| Base 成功、SFT 失败 | 3 | 6.00 | 8.00 | +2.00 |
| 双方都失败 | 10 | 8.40 | 13.80 | +5.40 |

即使只看双方都成功的 9 个任务，SFT 仍多调用 1.33 次。因此“调用膨胀”不是单纯由 SFT 完成了更多任务造成的。

## 5. 参考动作覆盖的正确口径

Tau2 `action_checks` 对每个参考动作检查是否在实际调用中出现，额外调用不会被惩罚。因此：

- 可以称为 `reference_action_recall`；
- 不能称为 Tool Precision；
- 不能单独用于判断“不多也不少”。

task 19 的 SFT 轨迹因 `too_many_errors` 提前终止，评价器未生成 action checks。若把它误当成“0 个参考动作”，会虚高 SFT 的聚合召回。排除这一不可比任务后，在共同可评的 29 个任务、83 个参考动作上：

- Base 匹配 59 个，微平均召回 71.1%；
- SFT 匹配 64 个，微平均召回 77.1%。

这支持“SFT 学到了更多参考动作”，但不支持“SFT 学到了更精简的工具路径”。

## 6. 代表性失败

### task 36：查询扇出爆炸

- Base/SFT 均失败，工具调用从 10 增至 45；
- SFT 共调用 `get_item_details` 35 次，最长连续同工具调用 12 次；
- 最终自然语言断言 2/2 满足，但数据库状态不匹配；
- 这不是相同参数的机械重复，而是对大量候选 item 逐个穷举。

它说明现有 SFT 数据教会了模型继续查询，却没有充分教会候选剪枝、信息充分后的停止条件和写操作前确认。

### task 19：错误恢复失败

- SFT 调用 13 次，其中 `exchange_delivered_order_items` 连续调用 5 次；
- 产生 5 个 tool error，最终 `too_many_errors`；
- action checks、DB check 和 NL check 均未执行。

它说明 Agent 遇到工具错误后没有切换策略、向用户澄清或安全停止。

### task 66：从成功退化为失败

- Base 成功，SFT 失败；调用从 6 增至 10；
- SFT 出现 tool error、数据库不匹配；
- V2.2 诊断发现未确认写操作、工具失败后成功宣称等问题。

这是后续数据修复优先级高于普通 `both_failure` 的 case，因为它直接暴露 SFT 引入的行为退化。

## 7. 对训练路线的含义

当前证据不支持立即把“工具调用越少”直接写成 RL reward。原因是必要的读取调用随任务复杂度变化，粗暴惩罚调用数可能鼓励 Agent 提前作答或漏查状态。

更合理的顺序是：

1. 针对 task 19、36、38、66 等 case 构造或修正多步 SFT 轨迹；
2. 明确训练以下过程行为：候选剪枝、信息充分即停止、工具错误后的策略切换、写操作确认、写成功后终止；
3. 在冻结任务上重跑，要求成功率不下降，同时降低双方成功切片的调用数和错误数；
4. 只有当过程 verifier 经人工金标验证后，再把硬约束和分段过程奖励接入 GRPO。

## 8. 当前可用于简历的诚实表述

> 在冻结的 30 个 Retail 开发任务上构建逐任务 Evaluation Card，将任务完成、最终状态、自然语言断言、参考动作覆盖、Policy 诊断和工具效率分层评估；教师 SFT 将成功率由 40.0% 提升至 56.7%，在 29 个可比任务上的参考动作微平均召回由 71.1% 提升至 77.1%，同时识别出平均工具调用由 6.63 增至 9.43、task 36 出现 45 次调用的查询扇出退化，为后续停止条件、错误恢复数据治理与过程奖励设计提供依据。

这里不能写“Tool Precision 提升”，也不能写“Policy 合规率提升”，因为前者没有可靠的多余调用金标，后者的 V2.2 仍是未经独立人工金标验证的 provisional verifier。
