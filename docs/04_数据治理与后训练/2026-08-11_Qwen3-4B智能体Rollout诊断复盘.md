# Qwen3-4B 智能体 Rollout 诊断复盘

> 日期：2026-08-11  
> 范围：隔离的 Retail train split 工程诊断  
> 结论：32 条 rollout 已完成，但 Tool SFT 与 reward 修正之前不应启动 GRPO 权重更新。

## 1. 为什么先做无权重更新诊断

Qwen2.5-0.5B 的单任务 sanity 已证明训练程序能够运行，但 4 条轨迹全部为零奖励，不能证明
模型真的学到了 Agent 行为。升级到 Qwen3-4B 后，如果立即训练，可能把显存、API 费用和
训练时间消耗在一个没有工具调用能力或存在奖励捷径的系统上。

因此本轮把学习率和 KL 系数都固定为 0，仅使用 TRL 的 Agentic GRPO 生成与评分链路采样：

```text
8 个冻结 Retail train 任务
× 每个任务 4 条轨迹
= 32 条 rollout
→ 工具行为、用户续轮、action progress、reward 方差诊断
→ 门禁判断
```

这不是 RL 训练结果，也不产生 adapter 或 merged model。

## 2. 冻结输入与环境

- 项目运行提交：`3bdf21a7e8f0377ea687314b844e72d4a7c70857`
- tau2 提交：`58e5e1ace69302e6982d27014569c03e0ffccdd2`
- 模型：`Qwen/Qwen3-4B-Instruct-2507`
- 模型目录哈希：`F34D14D691C94109D42451A4C9E63391FCEDEFFA43571AF6B63EF2F421EF1737`
- 配置哈希：`F934D6B447C89B4E33D7366F2254CC7B84974F31EA8D9EC4051754159190DADB`
- task split 哈希：`2B5793FD836C1876109E3C265A9F193386126A2B1A504889C2ADDEB44D33CE99`
- opening 哈希：`8F23026B875C977EC6DD380893C9759CA1B6E8808881CD64187F36B19D3BAD20`
- 任务 ID：`0, 7, 10, 11, 13, 15, 20, 22`
- GPU：RTX 4090 24GB
- PyTorch / CUDA / Transformers / TRL：`2.8.0+cu128 / 12.8 / 5.14.1 / 1.9.0`

8 条 opening 由 `deepseek/deepseek-v4-flash` 在温度 0 下生成，消耗 3,955 个输入 token、
1,117 个输出 token，记录费用为 `$0.0005854744`。隐藏用户场景和训练标签没有写入 opening。
正式 rollout 没有调用 `respond_to_user`，因此没有发生额外顾客模拟 API 调用。

## 3. 两次 OOM 不是无效试错

### Attempt 1：BF16，4 generations

- batch：4
- num_generations：4
- max steps：8
- 结果：第 0 步 prefill OOM；请求额外 7.01GiB，当时仅剩 5.90GiB。

这说明 0.5B 配置不能直接搬到 4B。长 Retail policy、完整工具 schema、4 路并行生成和
训练态模型共同决定显存，而模型参数量不是唯一变量。

### Attempt 2：BF16，2 generations

- batch：2
- num_generations：2
- max steps：16，总轨迹仍为 32
- gradient checkpointing：开启
- 结果：第 0 步仍 OOM；进程已占用约 23.49GiB，又请求 898MiB。

只减并行数仍不足，说明底座 BF16 权重和训练态缓存必须继续压缩。

### Attempt 3：4-bit NF4 QLoRA

- batch：2
- num_generations：2
- max steps：16
- 4-bit NF4 + double quantization
- 结果：32/32 完成；训练阶段耗时 196.31 秒；峰值观察约 23.24GB。

该方案保留 4B 模型、8 个任务和 32 条总轨迹，只把组内候选数从 4 调为 2。它证明单卡
4090 可以承载该规模，但余量很小。两次失败证据分别保存在：

- `experiments/20260811_qwen3_4b_rollout_diagnostic_oom_attempt1/`
- `experiments/20260811_qwen3_4b_rollout_diagnostic_oom_attempt2/`

## 4. 32 条结果

完成产物位于 `experiments/20260811_qwen3_4b_rollout_diagnostic_v3/`。

| 指标 | 结果 |
|---|---:|
| rollout | 32/32 |
| 覆盖任务 | 8/8，每个任务 4 条 |
| 工具调用轨迹 | 0/32 |
| 顾客继续交互轨迹 | 0/32 |
| 平均 action recall | 0 |
| 原始正奖励轨迹 | 4/32 |
| 原始 reward 均值 | 0.08472 |
| 原始 reward 总体方差 | 0.05024 |
| GRPO loss / grad norm | 0 / 0 |

模型并非没有生成文本。`log_history.json` 显示 completion 长度约 24--56 token，但
`tools/call_frequency` 始终为 0。completion 日志显示模型直接使用普通客服文本继续追问，
没有把客户沟通包装成 `respond_to_user` 工具调用。因此动态顾客没有收到消息，轨迹在第一轮
就结束。这反映的是**工具协议行为缺失**，而不是复杂业务推理已经失败。

## 5. 发现的 reward shortcut

4 条正奖励全部来自 Task 10，且共同满足：

- tool calls = 0；
- action recall = 0；
- 预期动作数 = 5；
- tau2 `db_match = true`；
- 原始 environment state reward = 1。

Task 10 的环境终态检查在无动作时已经返回 1；旧奖励把它按归一化后的 0.7778 权重计入，
再减去 0.1 unfinished penalty，得到 `0.6778`。这不是 Agent 成功，而是“初始状态满足宽松
终态检查”的奖励捷径。如果此时训练，GRPO 会把“不调用工具、直接回复”当成优选行为。

## 6. reward v1.1 修正

修正采用 action-progress gating：当任务存在预期动作时，环境状态分乘以 action recall。

```text
gated_environment_reward = raw_environment_reward × action_recall
```

选择乘法门控而不是“只要调用过一个工具就放行”，原因是后者仍可能鼓励 Agent 做一次容易的
读操作后获得过高终态分。乘法门控要求环境状态和过程进度共同增长；没有预期动作的任务仍保留
原环境分。

对已完成的 32 条轨迹做 post-hoc 检查，4 条正奖励全部被识别为：

- positive without tool；
- positive without action progress。

修正已经进入代码和单元测试，但还没有在新模型轨迹与人工审计集上验证，所以它仍是候选奖励，
不能宣称为正式可靠 reward。

## 7. 为什么现在不能直接做 GRPO

GRPO 需要同一 prompt 的候选之间存在可归因的质量差异。当前候选只有普通文本差异：

- 没有工具调用；
- 没有环境交互；
- 没有过程进度；
- 修正 reward 后将全部为 0。

此时 advantage 为 0，梯度也为 0。增大 steps 只会重复生成无训练信号的数据。正确顺序是：

1. 先构建经审计的 Retail tool-call SFT 数据；
2. 用小规模 Tool SFT 让模型学会 `respond_to_user` 和单工具调用协议；
3. 在同一 8-task 协议上重新做无更新 rollout 诊断；
4. 只有工具调用率、customer continuation、action progress 和 reward 方差同时通过，才做 GRPO；
5. 正式 Retail 改进结论仍需独立金标与冻结评测，当前门禁保持关闭。

## 8. 结果解释边界

可以如实表述：

> 我先把 4B 模型放进真实 Retail 多轮工具环境做无更新 rollout 诊断，而不是直接跑 RL。
> 过程中两次遇到 24GB 显存 OOM，通过减少组内并行并改成 NF4 QLoRA，在不缩减 32 条总
> 轨迹的情况下完成运行。诊断发现模型 32 条轨迹均未调用工具，却有 4 条获得 0.6778 正奖励。
> 我追到是初始 DB 状态命中了宽松终态检查，属于 reward shortcut，于是设计 action-progress
> 乘法门控。修正后 reward 会归零，也证明当前应该先做 Tool SFT，而不是盲目增加 GRPO steps。

不应表述为“Qwen3-4B 已完成有效 Agent RL”或“Retail 成功率已提升”。
