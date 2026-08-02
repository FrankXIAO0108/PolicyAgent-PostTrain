# 2026-08-02 云端 SFT→DPO→GRPO 完整实跑报告

## 1. 执行结论

项目已经完成一次真实、可验证、可恢复的 LLM 后训练工程闭环：

```text
Qwen/Qwen2.5-0.5B-Instruct
  ↓ Base 冻结评测
LoRA SFT（30 steps）
  ↓ merge + 冻结评测
LoRA DPO（20 steps）
  ↓ merge + 冻结评测
LoRA GRPO（10 steps，num_generations=4）
  ↓ merge + 冻结评测
证据完整性自动验收
```

最终 `verification_report.json`：

```json
{
  "verified_complete": true,
  "completion_claim_allowed": true
}
```

因此可以准确表述：

> 已完成隔离合成工具调用数据上的 SFT→DPO→GRPO 工程闭环实操，保存各阶段
> LoRA adapter、合并模型、可恢复 checkpoint、逐 step loss/reward、冻结评测、
> 环境快照及配置/数据/模型哈希。

不能表述为：

- 正式 Retail 后训练门禁已经打开；
- 后训练已提升真实 Retail 业务成功率；
- 当前程序化 reward 已通过独立人工 gold 校准。

正式 Retail 主线仍受独立人工裁决门禁约束。本次实验使用不接触 frozen Retail
任务的隔离合成数据，解决的是“是否真正操作过完整后训练栈”的工程证明问题。

## 2. 服务器与环境

| 项目 | 实际配置 |
|---|---|
| 云平台 | AutoDL |
| GPU | NVIDIA GeForce RTX 4090 24GB，单卡 |
| CPU | 16 核 Xeon Gold 6430 |
| 内存 | 120GB（平台配置） |
| 系统 | Ubuntu 22.04 / Linux 5.15 |
| 数据盘 | 50GB，最终约使用 5GB |
| Python | 3.12.3 |
| PyTorch | 2.8.0+cu128 |
| CUDA runtime | 12.8 |
| Transformers | 5.14.1 |
| TRL | 1.9.0 |
| PEFT | 0.19.1 |
| Datasets | 5.0.0 |
| Accelerate | 1.14.0 |
| bitsandbytes | 0.49.2 |
| 精度 | BF16 |

没有安装 `flash-attn`，没有升级 CUDA，没有替换镜像自带 PyTorch。运行时观察到
GRPO 显存约 3.9GB；这是轮询快照，不宣称为严格峰值。单张 4090 对该 0.5B
LoRA 工程实验有充足余量，无需双卡。

DeepSeek API 未参与模型下载、训练、reward 或评测，消耗 token 为 **0**。

## 3. 冻结绑定

| 绑定项 | 值 |
|---|---|
| 训练代码 commit | `df92970f739b42836cd5e2bdd807a71fec9fc1ab` |
| 分支 | `main` |
| worktree | 运行开始时 clean |
| 主模型 | `Qwen/Qwen2.5-0.5B-Instruct` |
| revision | `main` |
| seed | `20260802` |
| 配置 SHA-256 | `99D34DC9E8059A52BBC7E60681662E5E76AA1C5E2D0DE0C8D2E17442B35C3F84` |
| 数据 manifest SHA-256 | `F9D4AA7F4E4DAAD7D6318D1424FF872A04B3BF40A43963A39333AB6B5411207E` |
| max_length | 512 |

数据规模：24 条 SFT、24 对 DPO preference、8 条 GRPO prompt、8 条独立 ID
holdout。三组 scenario ID 无交叉，但任务结构同分布且高度模板化，因此评测仅用于
工程验证，不能作为业务泛化指标。

## 4. 实际运行命令

### 环境变量

```bash
export PATH=/root/autodl-tmp/venvs/policyagent/bin:/root/miniconda3/bin:$PATH
export HF_HOME=/root/autodl-tmp/huggingface
export HF_ENDPOINT=https://hf-mirror.com
cd /root/autodl-tmp/PolicyAgent-PostTrain
```

### Tiny API sanity

```bash
RUN_DIR=/root/autodl-tmp/policyagent-runs/api-sanity-v1-attempt4
CONFIG=configs/posttrain_engineering_api_sanity_v1.json

bash scripts/run_posttrain_stage.sh "$CONFIG" "$RUN_DIR" base
bash scripts/run_posttrain_stage.sh "$CONFIG" "$RUN_DIR" sft
bash scripts/run_posttrain_stage.sh "$CONFIG" "$RUN_DIR" dpo
bash scripts/run_posttrain_stage.sh "$CONFIG" "$RUN_DIR" grpo
```

Tiny 随机模型的 GRPO 四个 completion 全为零 reward，导致组内 reward 方差、
advantage、loss 和梯度均为零，GRPO merged hash 与 DPO 相同。该结果验证了 API
计算链路，也验证了“没有 reward 方差就没有 GRPO 学习信号”。

### 主实验

```bash
RUN_DIR=/root/autodl-tmp/policyagent-runs/20260802-posttrain-v1
CONFIG=configs/posttrain_engineering_smoke_v1.json

bash scripts/run_posttrain_stage.sh "$CONFIG" "$RUN_DIR" base
bash scripts/run_posttrain_stage.sh "$CONFIG" "$RUN_DIR" sft
bash scripts/run_posttrain_stage.sh "$CONFIG" "$RUN_DIR" dpo
bash scripts/run_posttrain_stage.sh "$CONFIG" "$RUN_DIR" grpo
```

各阶段分别执行并检查，未用单命令盲跑整个链路。后续阶段只有在前一阶段完成且
Git/config/data 哈希不变时才允许启动。

## 5. 四阶段结果

### 5.1 冻结 holdout

| 阶段 | 可提取 JSON | 严格单 JSON | 工具匹配 | 参数匹配 | 精确动作 |
|---|---:|---:|---:|---:|---:|
| Base | 100% | 0% | 0% | 25% | 0% |
| SFT | 100% | 100% | 100% | 100% | 100% |
| DPO | 100% | 87.5% | 50% | 87.5% | 50% |
| GRPO | 100% | 87.5% | 50% | 87.5% | 50% |

原始 evaluator 的 `valid_json_rate` 实际表示“能从回答中提取到 JSON 对象”。
后验只读格式审计增加了“整段输出必须恰好为一个 JSON object”的严格指标：Base
八条均在 JSON 后继续生成说明文本，DPO/GRPO 各有一条以逗号开头。这一发现不
修改冻结原结果，而是纠正指标解释边界。

### 5.2 SFT

- 30 optimizer steps，5 epochs；
- 平均 train loss：`0.174169`；
- 第 1 step loss：`1.685806`；
- 第 30 step loss：`0.000549`；
- 冻结精确动作：`0% → 100%`；
- 训练用时指标：约 `14.05s`，不包含模型下载、merge 和四阶段评测。

SFT 清晰学会了严格 JSON 工具调用协议。但 24 条模板数据被重复 5 个 epoch，loss
快速接近 0，说明存在强记忆效应。100% 只能说明同分布工程 holdout 被掌握。

### 5.3 DPO

- 20 optimizer steps，约 3.33 epochs；
- 平均 train loss：`0.127908`；
- 第 1 step loss：`0.693147`；
- 从第 2 step 起，训练 batch 的 preference accuracy 为 100%；
- 第 20 step reward margin：`5.094648`；
- 冻结精确动作：`100% → 50%`。

DPO 的内部偏好目标明显优化，但模型产生了
`get_order_missing_information`、`check_variant_availability.` 等不在工具注册表中的
名称。该阶段是失败实验：chosen/rejected 被拉开并不等于 Agent action 可执行。
小数据、高重复与偏好对设计可能共同造成过优化和协议漂移。

### 5.4 GRPO

- 10 optimizer steps，`num_generations=4`；
- 三个程序化 reward：可提取 JSON `0.25`、工具匹配 `0.35`、参数匹配 `0.40`；
- 平均 train loss：`-0.051366`；
- 训练 reward 范围：`0.0625–0.9`；
- 10 个 step 的 `frac_reward_zero_std` 均为 0；
- GRPO merged hash 与 DPO merged hash 不同，证明参数发生更新；
- 冻结精确动作仍为 50%，没有恢复 DPO 退化。

这是“有效训练但没有取得冻结指标提升”，不是空跑，也不是成功纠偏。10-step
on-policy 更新、8 条 RL prompt 和当前 reward 设计不足以保证 holdout 恢复。若继续
优化，应先扩充更难、更多样的 prompt 与错误工具负例，分析 per-category reward，
再做学习率/步数消融；不能直接增加步数后只汇报最优结果。

## 6. 产物与哈希

| 阶段 | Adapter SHA-256 | Merged model SHA-256 | Checkpoint SHA-256 |
|---|---|---|---|
| SFT | `557BA7F8...6620` | `FCCF589D...10C1` | `FE85D007...EB3A` |
| DPO | `36C5877A...B319` | `32806C9A...FEEF` | `4D303907...CF09` |
| GRPO | `03A9ECC5...E7BB` | `181EEB90...BC00` | `DB6176A2...24B9` |

完整哈希见 `experiments/20260802_posttrain_cloud_run_v1/run_manifest.json`。

远端完整产物保存在：

```text
/root/autodl-tmp/policyagent-runs/20260802-posttrain-v1
```

其中每阶段均包含 adapter、merged model、最终可恢复 checkpoint（含 optimizer、
scheduler、RNG 和 trainer state）与逐 step log history。主运行目录约 2.2GB；HF
缓存及其他运行目录计入后，数据盘约使用 5GB/50GB。Git 仅保存轻量证据，不保存
模型权重。

## 7. 遇到的问题与处理

### 7.1 Windows/Linux 数据哈希不一致

首次云端 preflight 因 Git checkout 将 CRLF 转为 LF，导致 JSONL 字节哈希变化。
修复方式是生成器显式写 LF 字节，并用 `.gitattributes` 固定训练数据与 shell 脚本
换行。重新提交后云端哈希预检通过。

### 7.2 官方 Hugging Face 端点不可达

AutoDL 到 `huggingface.co:443` 超时，而 `hf-mirror.com` 可返回模型元数据和权重。
处理方式是只设置 `HF_ENDPOINT=https://hf-mirror.com`，不更换模型、不改变 revision。

### 7.3 本地 SSH 超时会中断远端下载

直接以前台 SSH 运行时，本地命令 124 秒超时关闭了 HTTP client。后续改为远端
`nohup` 后台作业、独立日志目录和短连接轮询。失败目录保留，不覆盖证据。

### 7.4 控制台日志触发输出目录防覆盖

若先把重定向日志创建在新的实验目录内，runner 会正确判定目录非空并拒绝启动。
后续把外层控制台日志放入独立 `policyagent-run-logs`，实验目录只由 runner 创建。

### 7.5 Tiny GRPO 零方差

随机 tiny 模型的四个候选 reward 全为 0，导致 GRPO 无更新。没有把它包装成训练
效果，而是保留为 sanity 诊断。主 Qwen 在 SFT/DPO 后产生了非零 reward 方差，
随后才形成真实 GRPO 参数更新。

### 7.6 DPO 与 GRPO 没有带来最终提升

DPO 优化了 preference margin，却破坏工具名；GRPO 有正确 reward 信号和参数更新，
但 10 steps 没能恢复。这说明本实验的价值是跑通并诊断后训练闭环，而非证明每个
阶段单调提升。最优 checkpoint 是 SFT，而不是链路末端 GRPO。

## 8. 测试与验收

- 云端 focused tests：6 passed；
- 本地新增严格 JSON 审计测试后，相关测试：8 passed；
- 本地完整回归：89 passed、9 subtests passed、4 failed；4 个失败均来自依赖外部
  `D:\tau2-bench` 的 replay 集成测试，首个根因是当前本地 Python 环境缺少上游可选
  包 `toml`，随后同一 pytest 进程出现 tau2 部分初始化的循环导入；本次修改未触碰
  replay 代码；
- 主实验自动验收：`verified_complete=true`；
- adapter、merged model、checkpoint、loss history、四阶段评测、环境及绑定哈希均
  已核验。

## 9. 面试表达

### 90 秒版本

> 我在 PolicyAgent-PostTrain 里把正式业务数据治理和算法实操分开处理。正式
> Retail 数据因为独立人工金标门禁没打开，我没有拿 provisional 标签硬训；另建了
> 一个不接触 frozen task 的合成工具调用 sandbox，在单卡 4090 上真实跑了
> Qwen2.5-0.5B 的 LoRA SFT、merge、DPO、merge、GRPO、merge，并给每个阶段保存
> checkpoint、逐 step 日志、配置和数据哈希，再在同一冻结 holdout 上比较。
>
> Base 能提取 JSON，但工具匹配为 0；SFT 把精确动作提到 100%，不过小数据上有
> 明显过拟合。更有价值的是 DPO 失败：训练 preference margin 已经到 5.09，冻结
> 精确动作却降到 50%，因为工具名发生协议漂移。GRPO 的四候选组内 reward 有真实
> 方差，模型哈希也改变了，但 10 steps 后仍是 50%。这让我验证了一个业务结论：
> 后训练 loss 或 preference accuracy 变好，不代表 Agent action 可执行，所以必须
> 用独立 verifier 和冻结评测守住工具协议。整个闭环自动验收通过，但我不会把合成
> 结果宣称为 Retail 业务提升。

### 高频追问

**为什么 SFT 最好，DPO 反而下降？**

SFT 直接监督精确 action，目标与 holdout 指标一致；DPO 优化的是 chosen 相对
rejected 的序列偏好。数据量小且重复多时，模型可以扩大偏好 margin，却未必保持
工具 schema 的逐字符约束。这属于目标错位和小数据过优化。

**GRPO 是不是白跑？**

不是。主实验每一步都有组内 reward 方差，loss 非零，adapter 和 merged model
哈希均与 DPO 不同，说明实际发生了 on-policy 更新。但冻结指标没提升，所以应说
“真实训练完成、纠偏失败”，不能说“GRPO 提升成功”。

**为什么不用正式 Retail 数据训练？**

正式数据的 20 条政策标签仍为 provisional，独立裁决 gold 为 0。把可能错误的
标签放进 SFT/DPO/RL 会固化 benchmark 冲突和 verifier 偏差。因此先用隔离 sandbox
证明工程能力，正式业务结论继续遵守数据门禁。

**下一步怎么做？**

第一，构建实体和措辞真正不同的 adversarial holdout；第二，为 DPO 加入工具名
字符级/注册表约束和 hard negative，做 beta 与 step 消融；第三，把 GRPO reward
从“可提取 JSON”升级为“严格唯一 JSON + schema 校验 + policy/state verifier”，
并报告 reward 方差与各类别指标；第四，只有独立 gold 到位后才迁移到 Retail 正式
训练。

## 10. 证据索引

- 主 manifest：`experiments/20260802_posttrain_cloud_run_v1/run_manifest.json`
- 自动验收：`experiments/20260802_posttrain_cloud_run_v1/verification_report.json`
- 严格格式审计：`experiments/20260802_posttrain_cloud_run_v1/format_audit.json`
- 四阶段逐样本评测：同目录 `evaluation_base/sft/dpo/grpo.json`
- 逐 step 日志：同目录 `logs/`
- Tiny API sanity：同目录 `api_sanity/`
- 冻结配置：`configs/posttrain_engineering_smoke_v1.json`
- 合成数据 manifest：`data/posttrain_engineering_smoke_v1/manifest.json`

本报告中的指标均来自上述冻结文件；训练效果结论限定于隔离合成工程实验。
