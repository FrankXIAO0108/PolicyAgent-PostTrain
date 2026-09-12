# 脚本导航

正式数据与训练逻辑位于 `src/`；本目录提供绘图、启动和轨迹诊断工具。

## 绘图

| 脚本 | 用途 |
|---|---|
| [plot_grpo_training_curves.py](plot_grpo_training_curves.py) | 从 `log_history.json` 绘制 reward、reward_std、KL 和 loss |
| [plot_sft_training_curves.py](plot_sft_training_curves.py) | 从 SFT 原始日志绘制训练与验证指标 |
| [plot_sft_loss_comparison.py](plot_sft_loss_comparison.py) | 对照固定训练集与验证集 loss |

```bash
python scripts/plot_grpo_training_curves.py /path/to/run --output output/figures/grpo.png
```

绘图仅使用日志中实际存在的值，不补造缺失指标。参数说明可通过各脚本的 `--help` 查看。

## 训练与检查

- [run_retail_agentic_grpo.sh](run_retail_agentic_grpo.sh)：调用 Agentic GRPO runner。
- [check_grpo_accumulation_runtime.py](check_grpo_accumulation_runtime.py)：核验真实微批次、采样组和优化器更新的对应关系。
- [prepare_qwen3_4b.py](prepare_qwen3_4b.py)：准备模型；涉及下载与本地存储。
- [smoke_local_tool_model.py](smoke_local_tool_model.py)：本地工具调用检查。

## 实验专用工具

`audit_task*`、`prepare_task*`、`recover_task*` 与 `run_task44_*` 服务于相应的冻结实验、评分审计或语义辅助分支。它们仍被测试或配置引用，保留原位置，不作为默认训练流程。`posttrain_engineering_smoke` 系列是独立的小模型工程实验。

训练依赖、实际输入和运行边界见[安装与运行](../docs/QUICKSTART.md)。
