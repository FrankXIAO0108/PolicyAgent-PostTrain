# SFT v3 到 Agentic RL 协议桥接设计

日期：2026-08-24
状态：20-step 协议 SFT 与 16 条动态 rollout 诊断已完成；RL 优化门禁保持关闭

## 1. 问题定义

SFT v3 使用 tau2 原生教师消息：客户沟通是普通 `assistant` 文本，下一条客户回复是
`user` 消息。当前 TRL Agentic 环境的动作空间不同：模型必须调用
`respond_to_user(message=...)`，下一条客户回复才会作为工具结果返回。无更新诊断中，
SFT v3 的 16/16 completion 都生成了正常自然语言，但 0/16 生成结构化工具调用，导致
环境没有收到客户消息、0/16 产生客户继续交互、所有 reward 和组内方差均为 0。

这首先是动作协议不一致，不能解释为 Reward 无效，也不能通过调高 GRPO steps 修复。

## 2. 转换原则

数据源固定为：

`_local_private_runs/merged_sft_v3_wave_a_owner_reviewed/release_v2/sft_dataset.jsonl`

源数据 SHA-256：

`DFB76C036B17226E211A93BECDC7F518C07AB4B16B2666BE7787185192885146`

转换只改变交互表示：

1. 删除 tau2 固定开场白 `Hi! How can I help you today?`；
2. 第一条真实客户需求保留为普通 `user`，与 RL opening 对齐；
3. 后续普通 assistant 客户沟通改为 `respond_to_user` 工具调用；
4. 紧随其后的真实客户回复改为该工具调用的 tool result；
5. 原业务工具名、参数和返回值原样保留；
6. 观察到 `###STOP###` 或 `###TRANSFER###` 后，追加固定非工具 completion
   `Interaction complete.`，监督模型终止 TRL 工具循环；
7. TRAIN/VALIDATION、task、candidate、来源哈希均保持不变。

固定终止句是协议监督，不是业务事实，也不作为业务质量结论。

## 3. 本地数据审计

| 项目 | TRAIN | VALIDATION | 合计 |
|---|---:|---:|---:|
| 轨迹 | 58 | 22 | 80 |
| task | - | - | 55 |
| 保留的真实业务工具调用 | 485 | 173 | 658 |
| 新包装的 `respond_to_user` | 385 | 147 | 532 |
| 终止监督 | 58 | 22 | 80 |

转换后数据 SHA-256：

`572706FA3D4D9F88C03E02B8ED50E0BACE832BEDF28FA9FDD42F75FE1E2B74AB`

门禁结果：

- task 跨 split 泄漏：0；
- candidate 重复：0；
- 每条轨迹都以真实 user opening 开始：通过；
- 每条轨迹都以非工具终止目标结束：通过；
- 终止目标前残留普通 assistant 文本：0；
- PII 门禁继承自绑定的源数据，源报告 `pii_hits=0`、`pii_errors=0`。

数据仍是所有者复核的开发数据，不是独立专家金标。

## 4. 运行顺序与判据

1. 云端 `--validate-only`；
2. 2-step smoke，只验证序列长度、显存、mask、loss 和产物归档；
3. smoke 通过后运行 20-step 增量协议 SFT，起点固定为已选择的 SFT v3 80-step merged model；
4. 使用与 2026-08-24 无更新诊断相同的 4 task × 4 rollout 协议复测；
5. 只有工具调用、客户继续交互、正常终止、Action Recall 和组内 Reward 方差均出现
   可审计改善，才考虑冻结小步 GRPO 优化配置。

20 steps 与 `5e-5` 学习率是成本受控的工程起点，不是已证明最优超参数。

## 5. 结论边界

协议 SFT 若成功，只证明模型能把已有业务行为映射到 Agentic 环境动作空间。它不证明
业务成功率提高，也不证明 RL 已有收益。若复测仍无组内信号，应继续诊断动作终止、
上下文长度或轨迹覆盖，而不是强行开启权重更新。

## 6. 实际运行结果

运行目录：

`/root/autodl-tmp/policyagent-runs/20260824-sft-v3-agentic-protocol-bridge-s20-v1`

绑定与结果：

| 项目 | 结果 |
|---|---:|
| 项目 commit | `b17a675c7d98973076b77dce7a98716dac18074e` |
| config SHA-256 | `83757478B9F369EC3BA0C6623ED9C3AC0C61AF42D37C1F0B00F5E585C83D75F5` |
| 最大序列长度 | 14,266 tokens |
| 训练步数 | 20 |
| 训练累计 token | 730,600（日志显示值） |
| train loss | 0.259592 |
| Base validation loss | 0.428757 |
| SFT validation loss | 0.409007 |
| validation loss 相对下降 | 4.61% |
| 总墙钟 | 389.72 秒 |
| adapter SHA-256 | `488146860A18E5FCC0F351F3423B502678859360A16007465344FCC40D89EFDE` |
| merged model SHA-256 | `0A2E06C9BCA6082F3FE6723EC54A46D4CE1D37A8C6DD6B9BC116BBA4BAB16576` |
| run manifest SHA-256 | `56A28EF5040E84E157FDE8EA486CB4225FFC55FC6CFEC41DBE5DB86B146A71CF` |

2-step smoke 的 Base/SFT validation loss 为 0.428757/0.427405，正式 20-step 的
SFT validation loss 降至 0.409007。该结果支持“协议监督被模型学习”，但动态行为是否
改善仍必须由冻结的无更新 rollout 诊断回答。

## 7. 冻结动态 rollout 诊断

运行目录：

`/root/autodl-tmp/policyagent-runs/20260824-sft-v3-protocol-bridge-s20-agentic-rollout-diagnostic-v4`

关键绑定：

| 项目 | 结果 |
|---|---|
| 项目 commit | `f7d7dfa310030875dea59ac03ba2f541d89913ad` |
| config SHA-256 | `B27EF44063129BBE7C3A469E3E395CCC9C81AF89AEAC300B66E0BCA27BBF7F17` |
| 起始模型 SHA-256 | `0A2E06C9BCA6082F3FE6723EC54A46D4CE1D37A8C6DD6B9BC116BBA4BAB16576` |
| raw rollout SHA-256 | `E20F581FD53B193E331E5A4A1FAA15FCE51BAD99ED9148D7C73CC503640FE465` |
| run manifest SHA-256 | `FE415AEE47D5FB663B4D64007EFFD3C135C0D51902E1F421342F322F9A7CD863` |
| user simulator | `deepseek/deepseek-v4-flash`，API preflight 通过 |
| 轨迹规模 | 4 tasks × 4 rollouts = 16 |
| 墙钟 | 368.10 秒 |

相同任务、rollout 数和 reward 口径下，与协议桥 SFT 前的诊断比较：

| 指标 | 原 SFT v3 | 协议桥 SFT 20-step |
|---|---:|---:|
| 有业务工具调用的 rollout | 0/16 | 14/16 |
| 有客户继续交互的 rollout | 0/16 | 14/16 |
| 平均业务工具调用数 | 0.0000 | 1.5625 |
| 平均 Action Recall | 0.0000 | 0.2625 |
| 正 reward rollout | 0/16 | 4/16 |
| 平均 reward | 0.0000 | 0.0750 |
| 正常终止 rollout | 0/16 | 0/16 |
| reward 与 Action Recall 均有组内方差的 task | 0/4 | 0/4 |
| system failure | 0 | 5 条 sidecar 记录 |

这证明协议桥 SFT 已学到 `respond_to_user` 和基础身份认证工具链；它没有证明模型已能完成
端到端业务任务。task 10 的 4 条 reward 均为 0.3，其余 task 均为 0；没有任何 task
同时具备 reward 与 Action Recall 的组内方差，所以 GRPO 相对优势仍为零。

5 条 system failure 均发生在 task 0、seed 20260810，包括 1 条
`INVALID_RESPONSE` 和 4 条 `UPSTREAM_REQUEST_FAILED`。它们由用户模拟器收到空 assistant
消息触发，不能混入普通模型 0 分。诊断分析器已要求正常终止，并将独立 sidecar failure
纳入门禁。

## 8. Completion 预算诊断

当前动态诊断使用 `max_completion_length=384`。TRL 1.9 的工具循环将模型输出、工具调用和
工具返回共同计入该预算；超过预算时会回滚本轮工具结果并退出循环。本轮 8 个 step 的
显式 `clipped_ratio` 为 `0, 0.5, 0, 0.5, 0, 0.5, 0, 0`，即 3/16 rollout
被明确截断。原始 completion 还存在停在半个工具名、半个 JSON 或工具返回后的情况。

对桥接训练集中同 task 的已审核完整轨迹重新使用运行时 tokenizer 计数：

| task | 初始 prompt tokens | 完整 completion tokens | 轨迹条数 |
|---|---:|---:|---:|
| 7 | 3,835 | 6,316–7,116 | 3 |
| 10 | 3,816 | 2,379 | 1 |
| 15 | 3,862 | 5,472 | 1 |

task 0 不在桥接 SFT 数据中，因此没有对应教师轨迹长度，不能推断。384-token 动态预算
至少比现有完整教师 completion 小约 6 倍，复杂任务小约 18 倍。单卡 RTX 4090 不应直接
把 GRPO completion 提到 7k 后硬跑；下一阶段应先定义可审计的阶段化 rollout 单元，或在
保持环境语义的前提下减少工具返回冗余，再做无更新诊断。任何方案都需先证明正常终止、
无 system failure 且至少两个 task 具有联合组内方差，才允许冻结优化配置。

## 9. 身份认证阶段化 rollout（已实现，待运行）

为避免在单卡 RTX 4090 上把完整任务 completion 盲目扩到 2k–7k tokens，新增
`IDENTITY_AUTHENTICATION` 阶段诊断。它保留真实 tau2 Retail 任务、冻结 opening、动态
用户模拟器和真实工具执行，仅把本轮训练目标收窄为完整业务轨迹的身份认证前缀：

1. 信息不足时通过 `respond_to_user` 询问客户；
2. 在 `find_user_id_by_email` 与 `find_user_id_by_name_zip` 中选择正确工具；
3. 参数与任务隐藏 expected action 精确匹配；
4. 成功返回用户 ID 后停止，不继续读取订单或执行写操作。

任务 0、7、10、15 的上游 evaluator 各含且仅含一个身份认证 expected action，因此当前
四任务满足阶段奖励的结构前提。expected action 名称和参数只在环境内用于打分，不进入
policy prompt。

阶段奖励是严格二元过程奖励：只有“正确认证动作 + 正确参数 + 恰好一次业务工具调用 +
无工具错误”同时成立才得到 1；错参数、先错后对、认证后继续调用其他业务工具均为 0。
既有工具错误、重复调用和意外写操作诊断仍保留。该分数不调用 LLM Judge，也不计算完整
任务的 DB final state 或 communication reward。

冻结诊断配置：

`configs/retail_agentic_qwen3_4b_identity_auth_rollout_diagnostic_v1.json`

规模仍为 4 tasks × 4 rollouts，共 16 条，`learning_rate=0`、`beta=0`、
`max_completion_length=384`。分析器把 `stage_complete` 与完整任务的 `user_stopped` 分开
统计，禁止将阶段完成描述为业务任务完成。

运行前状态：`PREPARED_NOT_RUN`。只有满足以下条件才进入最小 GRPO 权重更新：

- 16 条原始 rollout 与 sidecar system failure 完整归档；
- 至少 2 个 task 同时存在组内 reward 方差和身份认证 Action Recall 方差；
- 至少出现一次 `stage_complete=true`；
- 无 system failure、无“无业务工具却获得正奖励”等 reward 泄漏；
- 人工抽查正负轨迹确认奖励方向与真实行为一致。

该阶段通过后只能说明身份认证前缀具备可优化信号，不能声明完整电商任务或 Retail
benchmark 得分提高。
