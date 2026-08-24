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
无 system failure 且至少两个 task 具有与当前 rollout 阶段一致的组内方差，才允许冻结优化配置。完整任务要求 reward 与 Action Recall 同时存在组内方差；阶段任务要求 reward 与 `stage_complete` 同时存在组内方差。

## 9. 身份认证阶段化 rollout（已实现并完成两轮诊断）

为避免在单卡 RTX 4090 上把完整任务 completion 盲目扩到 2k–7k tokens，新增
`IDENTITY_AUTHENTICATION` 阶段诊断。它保留真实 tau2 Retail 任务、冻结 opening、动态
用户模拟器和真实工具执行，仅把本轮训练目标收窄为完整业务轨迹的身份认证前缀：

1. 信息不足时通过 `respond_to_user` 询问客户；
2. 在 `find_user_id_by_email` 与 `find_user_id_by_name_zip` 中选择正确工具；
3. 参数与任务隐藏 expected action 精确匹配；
4. 成功返回用户 ID 后停止，不继续读取订单或执行写操作。

任务 0、7、10、11、13、15、20、22 的上游 evaluator 各含且仅含一个身份认证 expected
action，因此满足阶段奖励的结构前提。expected action 名称和参数只在环境内用于打分，
不进入 policy prompt。

阶段奖励是严格二元过程奖励：只有“正确认证动作 + 正确参数 + 恰好一次业务工具调用 +
无工具错误”同时成立才得到 1；错参数、先错后对、认证后继续调用其他业务工具均为 0。
既有工具错误、重复调用和意外写操作诊断仍保留。该分数不调用 LLM Judge，也不计算完整
任务的 DB final state 或 communication reward。

首轮冻结诊断配置：

`configs/retail_agentic_qwen3_4b_identity_auth_rollout_diagnostic_v1.json`

规模仍为 4 tasks × 4 rollouts，共 16 条，`learning_rate=0`、`beta=0`、
`max_completion_length=384`。分析器把 `stage_complete` 与完整任务的 `user_stopped` 分开
统计，禁止将阶段完成描述为业务任务完成。

扩展诊断配置：

`configs/retail_agentic_qwen3_4b_identity_auth_rollout_diagnostic_v2.json`

扩展规模为 8 tasks × 4 rollouts，共 32 条，仍保持 `learning_rate=0`、`beta=0`，不更新
模型权重。只有满足以下条件才进入最小 GRPO 权重更新：

- 预期原始 rollout 与 sidecar system failure 完整归档；
- 完整任务至少 2 个 task 同时存在组内 reward 与 Action Recall 方差；身份认证阶段至少
  2 个 task 同时存在组内 reward 与 `stage_complete` 方差；
- 至少出现一次 `stage_complete=true`；
- 无 system failure、无“无业务工具却获得正奖励”等 reward 泄漏；
- 人工抽查正负轨迹确认奖励方向与真实行为一致。

该阶段通过后只能说明身份认证前缀具备可优化信号，不能声明完整电商任务或 Retail
benchmark 得分提高。

## 10. 2026-08-24 实测结果与当前决策

扩展诊断在提交 `fb05d3565b7345013987d83e89f76947475c03cd` 上完成，原始 32 条
rollout 的 SHA-256 为
`36032C1A5C18C8F5C7D26907DD42FA851FF282B831D71DC810AE1852C84A3C09`。
模型权重未更新，运行清单明确记录 `execution_mode=ROLLOUT_DIAGNOSTIC` 与
`optimization_enabled=false`。

实测结果：

- 8 个任务均为 4 条 rollout，共 32 条；
- Action Recall 均值为 0.9375，30/32 条调用过业务工具；
- 严格阶段奖励为 1 的轨迹共 3 条，平均 reward 为 0.09375；
- 任务 7、10、15 均出现 `[0, 0, 0, 1]` 排列的组内 reward 方差；这些任务的
  Action Recall 全为 1，差异来自正确认证后是否立即停止；
- 29/32 条没有完成阶段目标，主要模式是认证成功后继续调用 `get_user_details` 或
  `get_order_details`；
- 无正奖励泄漏：不存在无工具调用或无认证动作进展却获得正奖励的轨迹；
- `system_failures.jsonl` 共 4 条，全部绑定任务 0、seed 20260810，其中 1 条
  `INVALID_RESPONSE`、3 条 `UPSTREAM_REQUEST_FAILED`。因此扩展诊断整体仍为
  `ready_to_consider_optimization=false`。

现有通用分析器原先要求 reward 与 Action Recall 同时变化。这会错误拒绝阶段目标的理想
信号：认证动作保持正确，而停止决策产生优劣差异。分析口径已改为按阶段选择门禁，完整
任务继续使用 `REWARD_AND_ACTION_PROGRESS`，身份认证阶段使用
`REWARD_AND_STAGE_COMPLETION`。该修正不改变奖励或原始结果。修正后 3 个任务达到阶段
信号阈值 2，但任务 0 的系统失败门禁仍关闭。

当前决定是隔离任务 0，不将受用户模拟器失败污染的轨迹用于优化；下一轮预注册其余 7 个
任务的清洁诊断。只有新诊断无 system failure 且阶段信号门槛继续成立，才运行最小 GRPO
权重更新。

## 11. 清洁诊断与 GRPO 资格证据

7-task 清洁诊断排除 task 0 的唯一依据是该任务在扩展诊断中 4/4 均为冻结用户模拟器
system failure；筛选没有使用 reward。清洁诊断目录为：

`/root/autodl-tmp/policyagent-runs/20260824-identity-auth-rollout-diagnostic7-clean-v3-rerun1`

该轮 7 tasks × 4 rollouts 共 28 条，无 system failure，raw rollout SHA-256 为
`760784DF31C6D78F8158E79D6309624D68EB0E5B5B58F1EB6ADC0F0074B4ACC2`。
单轮仅 task 15 出现阶段目标方差，未达到 2-task 启动阈值。

为避免因单轮小样本错误否定已观察到的稳定信号，随后仅合并两次独立诊断中相同的 7 个
任务；task 0 的轨迹全部按 system failure 排除，不按 reward 选行。合并池共 56 条、每个
任务 8 条，raw SHA-256 为
`F1C7C87AC78982621BFAE6AA06DC5C945336D7D5C31F5C0A1E4013B57A4970EC`。
task 7、10、15 共 3 个任务具有 reward 与 `stage_complete` 方差，超过预注册阈值 2；
system failure、无工具正奖励和无动作进展正奖励均为 0。该证据只开放最小身份认证阶段
GRPO 工程实验，不开放完整 Retail 或业务收益门禁。

## 12. 28-step 身份认证阶段 GRPO

正式配置：

`configs/retail_agentic_qwen3_4b_identity_auth_grpo_v1.json`

运行目录：

`/root/autodl-tmp/policyagent-runs/20260824-identity-auth-grpo-s28-v1`

| 项目 | 结果 |
|---|---:|
| 项目 commit | `dd25be621fd64d3e432b8c53582cf2f8c82563e5` |
| config SHA-256 | `1617F9460C57AD02F99D736027D4EA7A231513A94BCCEFAF62ABD79EB067AA91` |
| 起始 SFT 模型 SHA-256 | `0A2E06C9BCA6082F3FE6723EC54A46D4CE1D37A8C6DD6B9BC116BBA4BAB16576` |
| 训练配置 | 28 steps，2 generations，QLoRA NF4，LR `5e-6`，DR-GRPO，`beta=0` |
| 在线 rollout | 56 条，7 tasks × 8 |
| raw rollout SHA-256 | `E2794D3F34E2711BEFE17DBD60C17A4860435FCE751911ACD40EEFBC5D2D000C` |
| 正奖励 rollout | 10/56 |
| 有组内 reward 方差的 step | 6/28（step 5/7/17/21/23/28） |
| 非零 step grad norm | 0.22–0.38 |
| train loss | -0.024823 |
| 训练墙钟 | 1291.20 秒 |
| system failure | 0 |
| adapter SHA-256 | `09144162CEFB06FF9286D25BC08C830B90E6921B82FEDA5C13E1813155411A36` |
| merged model SHA-256 | `AF9CDEE0DAEE8DD9701AB0CF7DF7FDB4A576335879E55DE1632BD48EA075C72E` |
| run manifest SHA-256 | `F0D27E7EBD6F0DCC7E806EDEA8976C068A3EDB654FE2534325A55392725691F5` |

训练前另完成 1-step smoke，验证在线生成、反向传播、adapter 保存和 7.5GB merged 模型
写出。smoke 的两条 reward 均为 0、grad norm 为 0，因此只证明工程链路，不计作有效学习
证据。正式训练中 6 个 step 同时具有 `reward_std=0.7071` 和非零 grad norm，证明本轮确实
执行了组相对策略更新；训练 rollout 上的 reward 不能用于声明模型改善。

运行中 CUDA allocator 多次报告约 4GB 临时分配失败后自动回退，进程持续完成且无
traceback。单卡峰值接近 24GB，这构成可复现性和吞吐风险，但不是本轮 system failure。

## 13. 同协议 post-GRPO 诊断与结论

后评测配置只替换起始模型为 GRPO merged 模型；7 个任务、opening、seed、temperature、
reward、stage、2 generations 和 28-rollout 规模均与 SFT clean-v3 相同。配置固定
`learning_rate=0`、`beta=0`，不更新权重。

运行目录：

`/root/autodl-tmp/policyagent-runs/20260824-identity-auth-post-grpo-s28-diagnostic-v1`

| 指标 | SFT clean-v3 | GRPO 后 | 方向 |
|---|---:|---:|---|
| 轨迹数 / task 数 | 28 / 7 | 28 / 7 | 可比 |
| 正 reward / stage complete | 2 | 0 | 回退 |
| 平均 Action Recall | 0.8929 | 0.8571 | 回退 |
| 有业务工具调用的 rollout | 25 | 25 | 持平 |
| 有客户继续交互的 rollout | 26 | 26 | 持平 |
| 平均业务工具调用数 | 1.7143 | 1.7857 | 略增 |
| tool error | 0 | 0 | 持平 |
| 未完成 rollout | 26 | 28 | 回退 |
| system failure | 0 | 0 | 持平 |

post-GRPO raw rollout SHA-256 为
`5F0BFAFED9AB8AB2959C4983D117E8BBF317B8C6ABB176E5BA16ED466D9A1730`，run manifest
SHA-256 为
`4D60BF9146DAAA1A4B415CD45B894E8BB537618452EFA081A7B98ED27D2A159C`。

本次实验的工程闭环成立：真实 tau2 环境、动态用户模拟器、在线工具执行、程序化阶段奖励、
GRPO 权重更新、adapter/merge 和独立无更新后评测均完成。但现有证据不支持行为提升，
观测结果反而低于 28 条 SFT 基线。由于基线仅有 2 个正例，样本量不足以区分稳定退化与
采样波动；不得据此宣称 GRPO 必然有害。

轨迹级失败归因进一步表明，回退主要发生在认证后的停止决策，而不是工具执行错误。SFT
基线含 2 条正确停止、23 条“正确认证后继续调用其他工具”和 3 条认证动作缺失；GRPO 后
分别为 0、24、4 条。训练中的 6 个非零优势 group 仅覆盖 task 13、20、22，其中 4 个
直接比较“一次正确认证”与“认证后多调用”，2 个比较“一次正确认证”与“未完成认证”；
SFT 基线唯一出现正确停止的 task 15 没有产生非零优势 group。reward 的比较方向没有反，
但有效 group 的数量和任务覆盖不足以稳定学习并保持跨任务停止行为。

最可能的限制包括：奖励极稀疏，28 个 step 只有 6 个具有非零组内优势；group size 为 2，
估计方差高；`beta=0` 没有 KL 漂移约束；训练和评测任务规模过小。当前不应继续盲目增加
GRPO steps。下一步应先扩大只读诊断或提高阶段奖励密度，并以独立冻结轨迹验证；只有在
更可靠的开发评测上证明收益后，才把该 checkpoint 作为后续业务主线起点。

## 14. GRPO 训练信号离线审计

新增只读审计器 `src/evaluation/grpo_training_audit.py`。它在统计前校验 config 与 raw
rollout 的 SHA-256 是否和 `run_manifest.json` 一致，并按冻结配置中的
`num_generations` 对连续 rollout 分组。当前 runner 没有在 raw row 中记录显式 group ID，
因此报告将 `sequential_rows_by_num_generations` 明确记录为分组方法，并要求同组 task ID
一致；这项限制不能被隐去。

复现命令：

```powershell
python -m src.evaluation.grpo_training_audit `
  --run-dir output/20260824-identity-auth-grpo-s28-v1 `
  --config configs/retail_agentic_qwen3_4b_identity_auth_grpo_v1.json `
  --output output/20260824-identity-auth-grpo-s28-v1/training_audit.json
```

审计结果与原始日志一致：

- 28 个 group 中 20 个全 0、2 个全 1、6 个有组内差异；
- 22/28 step 的 `reward_std=0`，只有 task 13、20、22 产生有效比较；
- 6 个非零 reward-std step 均有非零梯度，grad norm 范围 0.21875--0.375；
- completion step 均值为 162.125 token，第一步 118、最后一步 206；
- 12/28 step 出现截断，`clipped_ratio` 的 step 均值为 0.267857，最大值为 1；
- entropy 从 0.092663 到 0.224084，范围 0.060421--0.338915，现有数据不支持
  entropy collapse 判断；
- `beta=0`，因此不存在可供分析的 KL 曲线；
- 日志只有整体 `step_time`，没有 rollout 与 model update 的独立计时，不能回答二者耗时占比。

该审计只证明出现过真实策略更新，并固定训练信号稀疏、截断和监控缺口；它不评估行为
提升，也不能单独证明 post-GRPO 回退的因果根因。下一项可比实验仍是独立 seed 的无更新
后评测，而不是追加训练 step。
