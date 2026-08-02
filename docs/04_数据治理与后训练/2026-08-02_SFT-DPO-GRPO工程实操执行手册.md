# SFT→DPO→GRPO 工程实操执行手册

## 1. 这次实操解决什么问题

当前正式 Retail 后训练门禁仍然关闭：20 条开发轨迹的政策标签均为
`PROVISIONAL`，独立裁决金标为 0，因而不能把它们释放成正式 SFT、DPO 或 GRPO
数据。另一方面，求职作品需要证明本人真正操作过三类训练，而不只是写过门禁和概念文档。

本仓库因此新增一条与正式实验严格隔离的工程实操路线：

```text
开发者合成工具调用样本
        ↓
LoRA SFT：学习 JSON 工具调用目标
        ↓ 合并 SFT adapter
DPO：偏好合规 action，拒绝越权/缺参 action
        ↓ 合并 DPO adapter
GRPO：按 JSON、工具、参数三维程序化奖励做组内相对优化
        ↓
同一份隔离 holdout 上比较 Base / SFT / DPO / GRPO
        ↓
验证 checkpoint、loss、配置、代码 commit、数据哈希和运行环境
```

这条路线用于证明训练工程能力，不打开正式 Retail 数据门禁，也不证明真实业务指标提升。

## 2. 训练内容

### SFT

- 输入：24 条开发者合成的 `prompt/completion` 样本。
- 目标：让模型学习只输出一个结构化 JSON action。
- 损失：completion-only 自回归交叉熵。
- 参数更新：LoRA，覆盖 Qwen Attention 与 MLP 投影层。
- 产物：`sft_adapter/`、`sft_merged/`、训练 loss 和冻结 holdout 结果。

### DPO

- 输入：同一训练域内 24 对 `prompt/chosen/rejected`。
- `chosen`：查询、补参、确认或转人工等合规 action。
- `rejected`：未经确认直接写入、猜测参数或错误工具等 action。
- 起点：已经合并 SFT 权重的模型，而不是重新从 Base 开始。
- 产物：`dpo_adapter/`、`dpo_merged/`、偏好训练 loss 和冻结 holdout 结果。

### GRPO

- 输入：与 SFT/DPO scenario ID 不重叠的 8 条合成 prompt。
- 每个 prompt 生成 4 个候选，构成组内相对优势。
- 奖励由三个独立程序函数组成：
  - 可解析 JSON：0.25；
  - 工具名正确：0.35；
  - 参数完全正确：0.40。
- 起点：已经合并 DPO 权重的模型。
- 产物：`grpo_adapter/`、`grpo_merged/`、reward/loss 和冻结 holdout 结果。

这里的程序化奖励是可审计的 synthetic verifier reward，不是已经通过独立人工 gold
校准的正式 Reward Model。

## 3. 数据隔离

数据位于 `data/posttrain_engineering_smoke_v1/`，由
`src/training/engineering_smoke_data.py` 确定性生成。

- SFT/DPO：24 条训练 scenario；
- GRPO：8 条独立 RL scenario；
- 冻结评测：8 条独立 holdout scenario；
- 三组 scenario ID 零交叉；
- 不读取 tau2 frozen task、原始轨迹、provisional 标签或用户隐私数据；
- 每个 JSONL 都在 `manifest.json` 中记录 SHA-256。

## 4. 建议服务器

这是 0.5B 模型的 LoRA 工程实操，不需要租高端多卡机器。建议使用：

- Linux；
- 单张 24 GB NVIDIA GPU；
- Python 3.10～3.12；
- CUDA 版 PyTorch 2.6 或更高；
- 至少 40 GB 可用磁盘，用于模型缓存、三个合并模型与临时文件。

16 GB 显存可能也能完成，但 GRPO 同时生成 4 个候选，24 GB 更稳。配置默认只跑
30/20/10 个 optimizer step，目标是验证完整闭环，不是追求收敛或业务 SOTA。

## 5. 单命令运行

服务器先克隆已经提交本执行包的代码，然后在干净 worktree 中运行：

```bash
cd PolicyAgent-PostTrain
bash scripts/run_posttrain_engineering_smoke.sh \
  /root/autodl-tmp/policyagent-runs/20260802-posttrain-v1
```

脚本会依次：

1. 安装锁定版本依赖；
2. 检查 Git commit、worktree、配置和数据哈希；
3. 检查 GPU；
4. 先用 TRL 官方 2.43M 参数 Tiny Qwen2 各跑 1 step，验证三类 Trainer 的接口与计算图；
5. Tiny API sanity 通过后，再跑 0.5B 主实操；
6. 跑 Base holdout；
7. 跑 SFT、合并 adapter、评测；
8. 跑 DPO、合并 adapter、评测；
9. 跑 GRPO、合并 adapter、评测；
10. 对三阶段 checkpoint、loss 和绑定关系做自动验收；
11. 输出轻量证据压缩包。

完整模型不应提交到 GitHub。应下载完整运行目录到本地备份，仓库只提交轻量证据包中的
manifest、验证报告、环境版本和四份评测 JSON。

### AutoDL 分阶段执行

正式云端实操不使用上述一键连续训练，而是逐阶段执行并检查产物：

```bash
export POLICYAGENT_PYTHON=/root/autodl-tmp/venvs/policyagent/bin/python
export HF_HOME=/root/autodl-tmp/huggingface
RUN_DIR=/root/autodl-tmp/policyagent-runs/20260802-posttrain-v1
CONFIG=configs/posttrain_engineering_smoke_v1.json

bash scripts/run_posttrain_stage.sh "$CONFIG" "$RUN_DIR" base
bash scripts/run_posttrain_stage.sh "$CONFIG" "$RUN_DIR" sft
bash scripts/run_posttrain_stage.sh "$CONFIG" "$RUN_DIR" dpo
bash scripts/run_posttrain_stage.sh "$CONFIG" "$RUN_DIR" grpo
```

每一阶段只在前置阶段完成且 Git/config/data hash 未变化时运行，并分别保存：

- 控制台日志；
- 每 step 的 `log_history.jsonl`；
- LoRA adapter；
- 合并模型；
- 含 optimizer/scheduler 状态的最终 checkpoint；
- adapter、merged model、checkpoint 和 loss history 的 SHA-256；
- 对应阶段的冻结 holdout evaluation。

## 6. 什么条件下可以写“已完成”

只有 `verification_report.json` 同时满足：

```json
{
  "verified_complete": true,
  "completion_claim_allowed": true
}
```

才允许在简历中使用：

> 在隔离合成工具调用数据上完成 SFT→DPO→GRPO 工程闭环实操：基于 Qwen2.5-0.5B
> 依次进行 LoRA SFT、合规偏好 DPO 与多维程序化 Verifier Reward 的 GRPO，固化各阶段
> checkpoint、训练 loss、数据/配置哈希及 Base/SFT/DPO/GRPO 冻结对比；该实验用于验证
> 后训练工程链路，不等同于正式 Retail 业务指标提升。

在实际运行和自动验收之前，只能写：

> 已搭建可复现的 SFT→DPO→GRPO 隔离工程实操链路，待 GPU 实跑验收。

## 7. 面试时如何说明与正式主线的关系

可以直接回答：

> 我把“算法实操”和“业务结论”分开处理。正式 Retail 数据因为独立人工裁决门禁尚未
> 打开，我没有拿 provisional 标签硬训；为了验证自己能真正操作后训练栈，我另外构造了
> 不接触 frozen task 的 synthetic sandbox，按 SFT、DPO、GRPO 顺序跑通 checkpoint 继承、
> 偏好数据、程序化奖励、冻结评测和证据验收。这样既有真实训练操作，也不污染正式实验。

这比直接声称“正式闭环完成”更能体现数据治理、因果边界和工程可信度。
