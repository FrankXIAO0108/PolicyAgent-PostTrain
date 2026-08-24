# SFT v3 到 Agentic RL 协议桥接设计

日期：2026-08-24
状态：20-step 协议 SFT 已完成，动态 rollout 复测待运行

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
