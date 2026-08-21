# SFT 瓶颈判定与 DPO / GRPO 分流协议

日期：2026-08-21  
状态：协议与本地计划工具已实现，GPU 消融尚未启动

## 1. 当前证据不足以宣称 SFT 达到瓶颈

当前只有一个 Qwen3-4B 教师 SFT 主运行：34 条 TRAIN 候选、80 steps、单一训练 seed。
内部 13 条实体隔离 VALIDATION 的 assistant loss 从 0.9278 降到 0.4105；30-task
tau2 开发对比从 12/30 提升到 17/30，但只有一个 trial。同时平均工具调用 6.63 → 9.43，
工具错误 15 → 19。

这些证据证明 SFT 有效但不充分，也说明模型“更愿意调用工具”不等于“过程更可靠”。它们
不能区分三种原因：训练步数不足、训练数据不足、SFT 目标本身无法表达负反馈。

## 2. 两条正交消融

### 2.1 优化步数曲线

固定完整 34 条 TRAIN、同一 seed 和其余超参数，只改变：

```text
20 steps → 40 steps → 80 steps
```

目的：判断 80 steps 是否仍处于明显下降区间。若 40 → 80 仍持续改善，优先继续 SFT，
不能声称平台期。

### 2.2 数据规模曲线

按 task 做确定性嵌套抽样，使用 25% / 50% / 100% TRAIN task。VALIDATION 13 条保持
逐字一致；steps 按实际训练行数相对 34 条全量线性缩放，使各候选经历近似相同 epoch。

目的：判断更多高质量正轨迹是否仍有收益。若 50% → 100% 明显改善，说明当前优先级仍是
扩充或修正 SFT 数据，而不是 RL。

## 3. 分阶段控制成本

Stage 1 只比较实体隔离验证集的 assistant loss，复用既有 80-step 全量运行，并新增四次
训练：full-20、full-40、data-25%-equal-epoch、data-50%-equal-epoch。该阶段不调用
DeepSeek 用户模拟器或 Judge，也不支持业务正确率结论。

Stage 2 只有 Stage 1 曲线趋平时才启动：对最多两个 checkpoint family 补足三个训练 seed，
再运行冻结 tau2 30-task 开发评测。不能把 10 条人工复核评测任务回灌训练。

现有全量 80-step 训练约 18 分钟。四个 Stage-1 新训练的纯优化步数合计相当于约 120 个
full-data steps，但每次还包含 Base validation、merge 和 SFT validation 固定开销；因此
实际墙钟时间不能简单按 120/80 线性外推，必须以云端预检后记录为准。

## 4. 项目级平台候选判据

以下只是预注册的个人项目启发式，不是通用统计定理：

1. Tool 协议能力已稳定，不再大量出现零工具调用或 malformed JSON；
2. 增加数据与 40 → 80 steps 的 validation loss 相对改善均低于 5%；
3. 三个训练 seed 的 30-task 结果中，增加 SFT 暴露的中位净增不超过 1 个任务；
4. 严重过程错误没有随更多 SFT 数据下降；
5. 残余问题主要属于偏好排序、禁止行为或长程信用，而不是缺少可写出的正确示范。

满足这些条件只能称为 `SFT_PLATEAU_CANDIDATE`，不能声称已经数学证明平台期。

## 5. DPO / GRPO 分流

| 残余监督信号 | 下一步 |
|---|---|
| 能写出明确的正确多步轨迹 | 继续 SFT |
| 同一状态有多条可行轨迹，能稳定排序 chosen/rejected | DPO |
| 必须在环境中探索，动作影响后续状态，且 Reward 可程序化验证 | Agentic GRPO |
| Verifier 误报高或 Reward 可被钻空子 | 暂停 RL，先修评测 |

RL 不是天然的“负反馈”。只有工具错误、重复调用、非预期写操作、终态错误等信号经过
逐例复核后，才能成为开发 reward。未经独立 gold 的 Policy V2.2 仍只能作为诊断项。

## 6. 工程入口

- 冻结协议：`configs/retail_teacher_sft_plateau_v1.json`
- 计划生成器：`src/training/prepare_sft_plateau_plan.py`
- 当前权威本地计划：`_local_private_runs/teacher_sft_plateau_v1_r1/`

只生成计划，不启动训练：

```powershell
D:\tau2-bench\.venv\Scripts\python.exe -m src.training.prepare_sft_plateau_plan `
  --protocol configs/retail_teacher_sft_plateau_v1.json `
  --output-dir _local_private_runs/teacher_sft_plateau_v1_r1
```

生成的 `plan_manifest.json` 明确记录 `PREPARED_NOT_RUN`、所有配置/数据哈希、Stage-1 命令
和条件式 Stage-2 命令。任何 GPU 运行必须写入新目录，禁止覆盖既有 80-step 产物。
历史 80-step 证据直接绑定原 `run_manifest.json`；云端 LF config 哈希与 Windows 工作区
可能出现的 CRLF 文件哈希分别记录，不把语义一致误写成字节哈希一致。

## 7. 本地计划生成结果

真实私有数据预检已完成：

| 变体 | TRAIN task | TRAIN 行 | steps | 状态 |
|---|---:|---:|---:|---|
| full-20 | 26 | 34 | 20 | Stage 1 待运行 |
| full-40 | 26 | 34 | 40 | Stage 1 待运行 |
| full-80 | 26 | 34 | 80 | 复用既有运行 |
| data-25%-equal-epoch | 7 | 8 | 19 | Stage 1 待运行 |
| data-50%-equal-epoch | 13 | 20 | 47 | Stage 1 待运行 |
| full-80 seed 20260819 | 26 | 34 | 80 | Stage 2 条件式 |
| full-80 seed 20260820 | 26 | 34 | 80 | Stage 2 条件式 |

- 25% task 集是 50% task 集的严格子集；
- 两个数据缩放变体均原样保留 13 条 VALIDATION；
- `plan_manifest.json` SHA-256：
  `FBE021C1B0DA1E0B17F8076B02DA055B9964FD83EC1910E10AB7829A4454F686`；
- `gpu_training_started=false`、`tau2_evaluation_started=false`、
  `sft_plateau_established=false`、`rl_justified=false`。
