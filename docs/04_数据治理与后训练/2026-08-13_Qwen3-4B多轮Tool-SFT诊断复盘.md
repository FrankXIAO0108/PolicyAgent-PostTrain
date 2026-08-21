# Qwen3-4B 多轮 Tool-SFT 诊断复盘

> 日期：2026-08-13  
> 范围：隔离的 tau2 Retail train-split Agentic RL 工程诊断  
> 结论：Tool-SFT 已修复工具协议启动能力，但多步执行与 GRPO 组内信号仍不足；暂不启动权重更新。

## 1. 本轮要回答的问题

Base Qwen3-4B 在同一组 8 个 Retail 任务、每任务 4 次采样的无更新诊断中，32 条轨迹均未调用工具。随后完成的 80-step Tool-SFT 只证明模型能在单轮合成样本中输出工具格式，尚未证明它能在真实多轮环境里持续执行。

本轮固定任务、opening、上游 tau2、rollout 参数与采样规模，只把起点模型替换为 Tool-SFT merged model，并保持 `learning_rate=0`，目标是判断：

1. 模型是否开始调用真实 Retail 工具；
2. 是否能消费工具结果并继续执行；
3. reward 与 Action Recall 是否在同一任务的多次采样中产生差异；
4. 当前信号是否足以支持 GRPO。

这不是 RL 权重更新，也不产生新的 adapter。

## 2. 冻结输入与产物

- 云端项目提交：`d8ad0d6e4140e94d8f9bea58f62dc6132fe88a1c`
- 配置哈希：`39F095130DA66E6D5DA3B9DEAE5433F0788A8E4BFD5CA86FAAA0D08CCEFF3AE4`
- 任务切分哈希：`2B5793FD836C1876109E3C265A9F193386126A2B1A504889C2ADDEB44D33CE99`
- opening 哈希：`8F23026B875C977EC6DD380893C9759CA1B6E8808881CD64187F36B19D3BAD20`
- Tool-SFT 起点模型哈希：`2028C809473FA52434BE00583D3C0E5C29633C447A479B17A51E28EEE9D0B4D0`
- 任务：`0, 7, 10, 11, 13, 15, 20, 22`
- 规模：8 个任务 × 4 条轨迹 = 32 条
- GPU：RTX 4090 24GB，4-bit NF4
- 运行时间：约 555.88 秒
- 原始轨迹哈希：`C82191540F3C140C68FD2AB79F21C8B22C6CDC3E24F35166AE880B8B68ECCDA1`

本地产物位于：

`experiments/20260813_qwen3_4b_tool_sft_rollout_diagnostic_v5/`

## 3. 结果

| 指标 | Base 旧诊断 | Tool-SFT 诊断 |
|---|---:|---:|
| rollout | 32 | 32 |
| 工具调用轨迹 | 0/32 | 32/32 |
| 平均工具调用数 | 0 | 2.09375 |
| 顾客继续交互轨迹 | 0/32 | 32/32 |
| 平均 Action Recall | 0 | 0.32232 |
| 正 reward 轨迹 | 4/32 | 6/32 |
| 平均 reward | 0.08472 | 0.03819 |
| 工具错误轨迹 | 0 | 1/32 |

平均 reward 不可直接作 Base/SFT 效果比较。Base 使用的旧 reward 没有 `environment_state_action_progress_gate=multiply`；Task 10 在零工具、零 Action Recall 时仍因初始 DB 状态获得 `0.6778`。新版 reward 会用 Action Recall 门控环境状态分，避免把“不行动”当作成功。因此本轮能可靠比较的是行为结构，不是两个不同 reward 版本下的均值。

## 4. 为什么不能根据自动报告直接启动 GRPO

原始 v1 自动报告检查的是全局方差：只要不同任务之间的 reward 或 Action Recall 不同，就会把方差门判为通过。但 GRPO 的相对优势来自同一 prompt 或同一任务组内候选之间的差异。跨任务难度差异不能替代组内学习信号。

对 8 个任务分别计算 4 次采样的组内方差后：

- reward 有组内方差：`1/8` 个任务；
- Action Recall 有组内方差：`1/8` 个任务；
- 两者同时有组内方差：`1/8 = 12.5%`；
- 其余 `7/8` 个任务四次采样的 reward 和 Action Recall 完全一致。

训练日志也显示，16 个 batch 中只有 2 个 batch 的 `reward_std` 非零，其余 batch 均为 `frac_reward_zero_std=1.0`。如果此时打开学习率，多数 step 的组内相对优势为零，扩大 steps 只会重复生成弱信号轨迹。

因此新增 `retail-agentic-rollout-diagnostic-v2`：报告逐任务组内方差，并使用“至少 2 个任务且至少四分之一任务具有 reward/Action Recall 联合组内方差”的诊断性 heuristic。本次 v5 的后验报告为：

```text
sufficient_task_groups_have_joint_variance = false
ready_to_consider_optimization = false
```

该阈值是工程诊断门槛，不是 GRPO 理论定理。

## 5. 提前终止的根因

32 条轨迹的终止分布：

- 29 条只有 2 次工具调用；
- 3 条有 3 次工具调用；
- 31 条只有 1 次顾客继续轮；
- 32/32 都带 unfinished penalty；
- 0/32 是顾客正常停止。

代表轨迹的共同模式是：

```text
询问身份
→ find_user_id
→ get_user_details 或 get_order_details
→ 收到工具结果后停止
```

Runner 配置允许最多 40 次工具迭代，环境允许最多 32 次工具调用，并未设置“两步后强制终止”。Tool-SFT 数据构造则明确是：

```text
system + 单条 user_message → 单个 expected tool_call
```

数据没有包含：

```text
tool result → 下一次 assistant planning/tool call
```

因此当前最有证据支持的解释是：单步 Tool-SFT 成功教会了工具格式和初始路由，但没有监督模型学习如何消费工具结果、维护多步状态并持续推进任务。现有证据不支持把问题优先归因于 rollout 循环上限。

## 6. DeepSeek API 与故障经验

本轮先遇到旧 Key 无效。tau2 用户模拟器在 API 认证失败后没有立即中止整场实验，而是产生回退对话，存在把外部服务故障误写成正常轨迹的风险。该问题应增加 fail-fast gate：用户模拟器 API 出现认证、配额或连续调用异常时，整条 rollout 标为系统失败，不进入 reward 与训练数据。

有效 v5 轨迹记录：

- API usage 消息：33 条；
- 输入 token：18,734；
- 输出 token：2,664；
- 轨迹内记录成本：约 `$0.0014193424`。

`total_tokens` 字段未填充，因此总 token 采用输入与输出分项相加；最终费用以 DeepSeek 控制台为准。

## 7. 下一步

1. 增加外部用户模拟器 API 的 fail-fast 与系统失败标记，防止异常对话污染 reward。
2. 构造严格隔离、带工具结果的多步 SFT 样本，至少覆盖身份认证、订单读取、商品读取、确认、写操作与最终沟通链路。
3. 先做 1–2 个任务的多步 SFT smoke，验证模型收到 tool result 后能继续调用，而非先扩大数据与训练步数。
4. 在同一代码、同一 reward 版本上重跑 Base 与多步 SFT 诊断。
5. 只有多个任务组出现稳定、可归因的组内 reward 方差，才启动小步 GRPO。

正式 Retail SFT/GRPO 门仍关闭。本轮只证明隔离工程路线中的协议学习与多轮诊断能力，不支持业务成功率提升声明。

## 8. 结果解释边界

> 我没有在看到训练脚本可运行后就直接扩大 GRPO，而是先做了 8 个任务、每任务 4 次采样的无更新 rollout 诊断。Tool-SFT 把工具调用从 0/32 提升到 32/32，平均必要动作召回达到 0.322，但 29/32 条轨迹只执行两步，且只有 1/8 个任务存在组内 reward 方差。我进一步追到 SFT 数据只监督“单条用户消息到单次工具调用”，没有包含工具结果后的持续决策，所以模型学会了协议启动，却没有学会长链路执行。我修正了把全局方差误当组内方差的自动门禁，暂停盲目 GRPO，下一步先构造多步轨迹 SFT 和外部模拟器 fail-fast，再重新验证 RL 信号。
