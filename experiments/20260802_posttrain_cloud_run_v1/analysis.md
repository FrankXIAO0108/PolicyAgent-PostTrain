# Qwen2.5-0.5B 云端后训练实跑分析

## 结论

本实验已在单卡 RTX 4090 上完成并自动验收以下真实计算链路：

```text
Qwen2.5-0.5B-Instruct Base
→ LoRA SFT 30 steps → merge
→ LoRA DPO 20 steps → merge
→ LoRA GRPO 10 steps（每个 prompt 4 个 generation）→ merge
→ Base / SFT / DPO / GRPO 同一 holdout 评测
```

`verification_report.json` 的 `verified_complete` 和
`completion_claim_allowed` 均为 `true`。允许表述为“已完成隔离合成数据上的
SFT→DPO→GRPO 工程闭环实操”，但不得外推为正式 Retail 业务指标提升。

## 冻结评测

| 阶段 | 可提取 JSON | 严格单 JSON | 工具匹配 | 参数匹配 | 精确动作 |
|---|---:|---:|---:|---:|---:|
| Base | 100% | 0% | 0% | 25% | 0% |
| SFT | 100% | 100% | 100% | 100% | 100% |
| DPO | 100% | 87.5% | 50% | 87.5% | 50% |
| GRPO | 100% | 87.5% | 50% | 87.5% | 50% |

“可提取 JSON”沿用冻结运行时 evaluator 的定义：回答中存在一个可解码 JSON
对象即可。`format_audit.json` 是不改写原结果的后验审计，它进一步要求整段回答
只能是一个 JSON 对象。Base 的 100% 可提取率因此不能解释为格式完全合规。

## 训练信号

- SFT：平均 loss `0.174169`，末步 loss `0.000549`，30 steps / 5 epochs。
- DPO：平均 loss `0.127908`；从第 2 step 起训练 batch 的 preference accuracy
  为 100%，末步 reward margin `5.094648`。
- GRPO：平均 loss `-0.051366`；10 个 step 均有非零组内 reward 方差，reward
  范围为 `0.0625` 至 `0.9`，证明不是零 advantage 的空跑。

## 最重要的失败发现

1. SFT 在小规模同分布模板数据上快速达到 100%，存在明显记忆与模板泛化，不能
   当作真实业务泛化证据。
2. DPO 虽然显著拉开 chosen/rejected margin，却把严格工具名学坏，精确动作匹配
   从 100% 降为 50%。偏好优化目标与 Agent 可执行性发生错位。
3. GRPO 的程序化 reward 确实产生方差且模型哈希发生变化，但 10 steps 后没有
   恢复冻结 holdout 指标。短程 on-policy 优化不足以自动修复 DPO 退化。
4. 原 `valid_json_rate` 是 JSON 提取成功率，不是严格格式遵循率。独立格式审计
   发现 Base 为 0%，DPO/GRPO 为 87.5%。

## 证据入口

- `run_manifest.json`：代码、配置、数据、模型、环境、阶段产物哈希与四阶段指标。
- `verification_report.json`：自动完整性验收及允许/禁止声明。
- `evaluation_*.json`：逐样本冻结生成与判定。
- `logs/*_log_history.jsonl`：逐 step loss、DPO margin、GRPO reward 分量。
- `format_audit.json`：严格 JSON 后验格式审计。
- `api_sanity/`：正式运行前的 tiny Trainer/API 计算图验证。

完整 adapter、merged model 和可恢复 checkpoint 保存在 AutoDL 数据盘
`/root/autodl-tmp/policyagent-runs/20260802-posttrain-v1`，未提交 Git。
