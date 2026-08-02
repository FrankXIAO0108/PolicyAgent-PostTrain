# Guard 在线配对 A/B V1 执行前报告

日期：2026-07-30

## 1. 为什么需要重新设计 A/B

旧的 `run_retail_guarded_failure3.py` 只计划重跑 Guarded arm。如果直接拿它与
历史 Trial-1 比较，会把采样波动、运行时间变化和 Guard 影响混在一起。

本阶段把实验改成真正的配对协议：

```text
同一任务 95 / 98 / 107
同一 Agent 模型与温度
同一 User Simulator
同一 NL Judge
同一 seed / max_steps / evaluation

唯一主变量：
Base llm_agent
vs
Guarded llm_agent + deterministic Guard + one retry
```

Task 59 因 simulator/static-gold 冲突继续隔离。

## 2. 新增工程能力

### 冻结协议

`configs/guard_online_ab_v1.json` 固定：

- 两个实验 arm；
- 任务和受控变量；
- primary/secondary metrics；
- 付费执行门禁；
- failure-selected subset 的解释边界；
- 0.10 美元成本复核阈值；当前 runner 尚未实现覆盖 Agent、User 与 Judge
  全链路成本的硬停止，因此不能把该阈值描述为运行时硬预算；
- 官方 test split 不使用。

### 零调用 Preflight

`src/guards/online_ab.py` 检查：

- 三个任务属于冻结 parent config；
- Task 59 未混入；
- 两个 arm 只有 Guard 变量不同；
- Git 工作区已提交；
- API Key 只检查是否存在，不读取或记录其值；
- 用户是否显式提供 `--approve-paid-run`；
- 结果是否明确限制为 failure-selected subset。

### 配对比较器

当两个 arm 产生 raw outputs 后，比较器逐 task 计算：

- Base/Guarded reward、DB reward、NL reward；
- recovery、regression、both success、both failure；
- duration 和已知 Agent/User cost；
- Guard intervention 和 additional retry calls；
- raw artifact SHA-256。

所有 raw outcome 在正式报告前仍必须经过 V7 replay。

### Guard intervention telemetry

`src/agents/guarded_llm_agent.py` 新增 `guard_trace.jsonl`：

- 每次 proposal 的 decision；
- retry index；
- 是否允许执行；
- Tool proposal；
- blocking findings。

它不会记录 API Key。没有该 trace 就不能报告 intervention count。

### 付费 Runner 防误触

`src/run_retail_guarded_failure3.py` 已升级为配对 runner：

- 默认不执行付费调用；
- 必须显式传入 `--approve-paid-run`；
- Git 必须 clean；
- 必须检测到 `DEEPSEEK_API_KEY`；
- 输出目录非空时拒绝覆盖；
- 中途失败时保留 partial artifact；
- 双 arm 完成后状态仍为 `RAW_ARMS_COMPLETE_V7_PENDING`。

## 3. 当前 Preflight 结果

当前状态：`BLOCKED`，且没有发生模型调用。

| 检查 | 结果 |
|---|---|
| 任务属于冻结 parent | 通过 |
| 官方 test 未使用 | 通过 |
| 受控变量完整 | 通过 |
| failure-selected 边界已披露 | 通过 |
| Git 工作区 clean | 未通过 |
| API Key 已配置 | 未通过 |
| 显式付费批准 | 未通过 |

当前阻塞是预期行为：本轮代码和复盘文档尚未提交，用户也没有授权付费执行。

## 4. 产物

```text
experiments/20260730_guard_online_ab_preflight_v3/
├─ protocol_snapshot.json
├─ preflight.json
├─ analysis.md
└─ manifest.json
```

V1 与 V2 是协议收紧过程中的历史 Preflight，继续保留以保证审计链完整；
V3 是与当前配置及模块化运行入口一致的最新快照。

## 5. 解释边界

即使未来得到：

```text
Base: 0/3
Guarded: 3/3
```

也只能说明已知 failure-selected subset 的恢复，不能表述为 Retail success rate
提升。还需要：

1. V7 重放确认最终状态；
2. 分析 Guard intervention 与恢复的因果关系；
3. 检查 Guard 是否引入新 policy/state/claim 错误；
4. 在更广、冻结的新任务上评估误拦截和效用。

该实验仍不能替代独立 Policy Gold，也不会直接打开 SFT、DPO 或 GRPO 门禁。
