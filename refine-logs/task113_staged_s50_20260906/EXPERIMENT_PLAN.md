# Task113：新 SFT 起点的分层 GRPO，50 次更新

日期：2026-09-06。状态：用户要求暂停并修复，监督PID3480与训练PID6043已终止；旧运行到15次更新、60条完整轨迹，但实际每更新只有一组n4，不符合此方案的两组累积要求。保留checkpoint-10与原始证据，不作为batch8实验验收。修复与真实框架计数验证见ACCUMULATION_FIX.md；尚未重启正式训练。

## 1. 目标与边界

只执行一轮：新 SFT checkpoint-30 → Task113 纯规则分层 GRPO → 同协议前后评测 → 四张训练曲线与 badcase。
不运行 Terminal 训练对照，不改训练数据，不引入 LLM 评分，不扩大任务集。
本轮回答“该开发任务的完成稳定性是否改善”，不能证明分层奖励优于 Terminal、batch 优于旧值或跨任务泛化。
Task113 不在新 88 条 SFT 数据内，但历史上已用于开发，绝不是干净最终测试任务。
以下参数是可检验的工程选择，不是已证明最优值，也不承诺正向结果。

## 2. 证据与起点

- 当前项目 main，HEAD `5e50138f13c4fcc408c714d074449c1ef05549dc`，工作区 dirty；现有改动均保留。
- 上游 tau2 `58e5e1ace69302e6982d27014569c03e0ffccdd2`。
- 模型：`/root/autodl-tmp/policyagent-deployments/20260905-teacher-refresh-sft100-v1/training/teacher_sft_merged`。
- 模型哈希：`2215F09DF8A8FF03B1CED725E7F87F7F953B4D0B91F5DDE1618628C1B036963C`。
- 实际选择的是 checkpoint-30 的合并模型，不是第 100 步模型。使用新建 BF16 LoRA，r=16、alpha=32、dropout=0.05、七个投影层；参考策略为同一 SFT，不是裸 Qwen base。
- 输入沿用已验证的 split、SFT manifest 和 Task113 opening。精确路径与哈希由训练 JSON 冻结。
- 新预采样：`_local_private_runs/tr0905/s100/probe_v2/results_v1/task113_n4/`，终态分数 `[0,1,0,1]`，无截断和系统失败。两条失败分别没有工具调用、查询后不取消；均记录提前 EOS。
- 历史反例：`_local_private_runs/task113_prescreen_20260902/passk-n4-v1/`。与本批不是同一模型，不混作效果对照。

## 3. 分层公式：保留 v7 权重，局部修正条件句

基础分：

`S = 0.35 E + 0.10 U + 0.08 I + 0.12 T + 0.20 A + 0.15 C`

| 分量 | 当前代码的实际含义 |
|---|---|
| E | 上游环境终态检查为 1；不是全过程合规证明 |
| U | 用户模拟器已输出停止标记 |
| I | 成功查找用户身份后读取该用户资料，资料覆盖四个绑定订单 |
| T | 四笔订单详情成功查询的覆盖比例，乘 I；不是工具次数越多分越高 |
| A | 两个目标取消操作均已被验证、I 成立后，乘确认诊断值：PASS=1、REVIEW=0.5、FAIL=0 |
| C | 两个目标写操作均验证、上游 communicate 检查完整、最后写结果之后有非空回复。Task113 没有 communicate_info，因此 C 本身不检查所有事实真实性 |

按现有 v7 顺序对基础分取上限，再扣分并裁剪到 [0,1]：

- 无任何验证写操作：最高 0.25。
- 本轮新增 `authorization_scope=every_observed_write`：任何已观察到的写调用都检查授权，包括只取消了一笔就结束。身份/确认诊断 FAIL：最高 0；确认 REVIEW：最高 0.75。A 分量仍要求全部目标写完成。
- 触发已配置退款声明冲突或声明检查 ERROR：最高 0.75；REVIEW：最高 0.90。
- 终态成功但上游必需沟通不完整：最高 0.90。
- 工具错误每次扣 0.03、最多 0.09；重复调用每次扣 0.02、最多 0.06；客户回合/工具调用预算触顶扣 0.10。
- 额外非目标写操作：最终最高 0。
- 显式启用 completion-budget-as-terminal-failure：纯生成预算耗尽按 0 保留，不能伪造 EOS；未解决工具调用、工具输出放不下、上下文越界、API/框架失败仍停止。

v7.1 有两项明确变更：正确的“如果原支付方式是礼品卡则立即退款”条件句不误罚；授权检查覆盖部分写调用且 FAIL 封顶从 0.15 改为 0。条件排除不得跨逗号/句号吞掉另一条错误声明；使用有限局部模式和反例测试。六项权重与其他上限不变，历史配置不变。

这是有限模式识别，不是通用自然语言验证器。claim PASS 仅表示没命中已知冲突；授权规则对指代表述的误判、退款替代表述及省略退款说明仍须在 badcase 中单独复核。规则判 FAIL 不等于独立人工认定违规，不能把总分 1 当独立人工金标。

**本设计是分层轨迹评分汇总成一个标量，再进行组内标准化；不是每个动作独立 advantage。**正 advantage 不等于整条轨迹所有动作都正确。

## 4. 离线复算

| 数据批次 | 原 v7 | 本轮 v7.1 |
|---|---|---|
| 新 SFT30 四条 | [0, 0.75, 0.2, 1] | [0, 1, 0.2, 1] |
| 历史四条 | [0, 0.75, 1, 0.12] | [0, 0.75, 1, 0] |

新批次一条正确条件句误罚得到修复；历史上的真实立即退款错误仍扣分；历史第4条因确认诊断 FAIL 使用新的0分门禁。本批失败轨迹的 advantage 均为负。这不是整体 verifier 误判率估计。
最终证据：`_local_private_runs/task113_staged_acc8_20260906/offline_audit_v3/report.json`；早期 v1/v2 报告保留，不能作为最终配置的验收。

## 5. 冻结训练配置

配置：`configs/retail_agentic_qwen3_4b_task113_sft30_staged_v7_1_n4_acc8_s50_v1.json`。

| 参数 | 值 |
|---|---|
| num_generations | 4 |
| per_device_train_batch_size | 1 |
| steps_per_generation | 4 |
| gradient_accumulation_steps | 8 |
| 单卡有效更新 batch | 8 条 = 两个独立 n4 group |
| max_steps | 50 次优化器更新，100 group，400 条训练轨迹 |
| learning_rate | 峰值 2e-6 |
| scheduler / warmup | linear / 2 次更新；不是全程恒定 LR |
| beta / max_grad_norm | 0.02 / 1.0 |
| temperature / top_p / top_k | 0.8 / 1.0 / 0 |
| num_iterations / scale_rewards | 1 / group |
| loss_type | dr_grpo；8192 同时参与该损失的长度归一化，不能中途随意改 |
| max_completion_length | 8192，累计模型生成 token 预算 |
| customer turns / tool calls / tool iterations | 8 / 24 / 32 |
| optimizer | adamw_torch |
| train seed / user seed | 2026090601 / opening 中的 20260902 |
| logging / save | 每次更新记录；每 10 次更新保存，原 runner 最多保留两个中间 checkpoint，另存最终 adapter/merged |

两组分别计算 `(r - group_mean)/(sample_std + 1e-4)`，再做原生梯度累积；不是把八条合成一组，也不手动再次除梯度。组全同分仍无 reward advantage；KL 梯度可能存在。不得将 reward_std 或 advantage 大小叫作实际梯度范数。

## 6. 冻结评测与预算

- 每臂：greedy 1 条 + 八次独立 n4，共 33 条；SFT 与最终 GRPO 共 66 条。
- stochastic model seeds：2026090611 到 2026090618；greedy seed：2026090619。
- opening、user seed、用户模拟器、生成预算、温度和评分代码相同；两臂只换 checkpoint。
- DeepSeek 只作用户模拟器，请求别名沿用 `deepseek/deepseek-chat`、既有 llm_args 设置；启动时记录实际返回模型标识，不保证服务商后端永久冻结。不调用 LLM judge。
- 32 条是同一开发任务的随机重复，报告分母和不确定性，不称 32 个独立任务，更不称泛化测试集。
- 主指标：终态任务完成率；另报提前 EOS、未授权/错误取消、声明冲突和必要说明缺失。训练总分为辅助指标。
- 先运行并保存 SFT 基线；训练结束后评估预先指定的 step50，不反复挑中间模型以迎合最终评测。
- 总预算：400 训练 + 66 评测 = 466 条轨迹，另有连接/API 预检；不自动扩步数、不自动失败重跑。无授权不发送任何数据。
- 尚未测过此配置的 backward 显存和步时，因此不提供伪精确 GPU 小时或费用。首个实际更新记录峰值显存和耗时，再估算剩余时间。

## 7. 开云后的顺序

1. 用户确认端点及本轮上传/API/466 条轨迹授权；旧 SSH 端口不能默认仍有效。
2. 独立目录建议 `/root/autodl-tmp/policyagent-deployments/20260906-task113-staged-acc8-s50-v1/`；不覆盖原仓库和旧结果。同步本轮必要代码、配置及所引用的 split/manifest/opening；不上传密钥。
3. 复核当前 GPU、磁盘、SFT 模型哈希、tau2 版本、TRL 1.9.0 及已审阅源码。单进程、单 GPU；不自动升级软件。
4. 运行完整 preflight，不调用用户 API；然后经授权做用户 API 可用性检查和 SFT 冻结评测。
5. 后台运行唯一 50-step 任务。第一次更新检查两个 group 属于同一 optimizer_step、8条输出与原生梯度累积、finite loss/grad/KL 和显存。它属于这 50 步，不另造两步结果。
6. API/数值/环境完整性异常停止并保留状态；普通提前 EOS、预算负例或单个零方差组不退出。不以未上涨为理由自动换 reward/LR。
7. 保存最终权重与 optimizer 更新证据，准备并运行冻结后评测，画图，回传原始轨迹/日志/指标/权重并核对完整性。只有确认无剩余进程且所需产物回传后，提醒用户关 GPU；销毁实例另行确认。

## 8. 命令草案（云端独立目录内，未来授权后执行）

采用现有虚拟环境 `/root/autodl-tmp/venvs/policyagent/bin/python`；凭据由既有安全位置注入，不写入配置或日志。
环境变量：`POLICYAGENT_TAU2_ROOT=/root/autodl-tmp/tau2-bench-58e5e1a`、`CUDA_VISIBLE_DEVICES=0`、`HF_HUB_OFFLINE=1`、`TRANSFORMERS_OFFLINE=1`、`LITELLM_LOCAL_MODEL_COST_MAP=True`、`POLICYAGENT_USER_MODEL=deepseek/deepseek-chat`、`POLICYAGENT_USER_LLM_ARGS_JSON={}`。

```bash
python -m src.training.run_retail_agentic_grpo --config configs/retail_agentic_qwen3_4b_task113_sft30_staged_v7_1_n4_acc8_s50_v1.json --preflight-only --allow-dirty

python scripts/prepare_task113_staged_eval.py --training-config configs/retail_agentic_qwen3_4b_task113_sft30_staged_v7_1_n4_acc8_s50_v1.json --arm SFT --output-dir eval_sft
# 执行 eval_sft/eval_commands.json 中冻结的9条命令；准备脚本本身不启动评测。

python -m src.training.run_retail_agentic_grpo --config configs/retail_agentic_qwen3_4b_task113_sft30_staged_v7_1_n4_acc8_s50_v1.json --output-dir training --allow-dirty

python scripts/prepare_task113_staged_eval.py --training-config configs/retail_agentic_qwen3_4b_task113_sft30_staged_v7_1_n4_acc8_s50_v1.json --arm GRPO --training-run training --output-dir eval_grpo
# 只在完整训练验收后，从实际 manifest 绑定最终模型，再执行9条评测命令。

python scripts/plot_grpo_training_curves.py training --output training/grpo_training_curves.png --label Task113-SFT30-Staged-v7.1
```

四图保持 reward、reward_std、KL、loss。两组在一个优化器日志点上通常取平均；逐组原始分数、样本标准差及 advantage 另保存在 `group_reward_summary.json`。真实模型梯度范数仍看 grad_norm。

## 9. 本地变更与检查

- 不修改原 v7 配置；新配置继承已验证 SFT30 绑定，修改局部退款条件规则及授权作用域/FAIL上限。
- runner 允许显式关闭旧一组限制后使用完整两组更新；校验 50/100/400，拒绝部分生成批次和 num_iterations!=1。
- 显式传递 LR scheduler、warmup、max_grad_norm、scale_rewards、top_p、top_k、optimizer；保存实际 `effective_grpo_config.json`。
- 新 SFT source_stage 接入已有 manifest 校验；不只信任模型目录名。
- 离线脚本只读取历史产物，另目录输出重评分；评测准备脚本只生成配置，拒绝不完整训练和模型哈希不符。
- 最终专项/启动/评分回归160 passed；Retail环境84 passed；已有PyTorch解释器中的张量测试82 passed、4 skipped。完整命令、一次错误解释器调用与未验证项见 `LOCAL_VALIDATION.md`。测试不包含本轮真实 GPU 更新。

## 10. 验收与反证

工程验收：实际50次更新、100组、400条训练轨迹，至少一组非零方差，有限梯度/KL，训练参数发生变化，完整权重/日志/四图和66条冻结评测。
效果判定：按原分母报告前后成功/失败及逐条行为变化；小幅差异可能只是采样噪声，不能只凭一次正差称显著提升。
反证：若 reward 涨而任务成功不涨、或违规/错误声明增加，则本轮优化目标或学习行为不符合目的；如实报告，不换指标解释为成功。
即便已知四条排序正确，仍可能出现失败轨迹相对胜出。最终必须单列 `terminal_failure & advantage>0` 的数量、分量和失败原因，不自动将它等同于 reward 错误或正确学习。
