# 配置导航

配置按数据、模型和任务版本保存。选择配置时以目标、数据路径及模型哈希为准；文件名中的步数不代表已完成运行。

## 主要入口

| 用途 | 配置 | 说明 |
|---|---|---|
| 教师轨迹采集 | [教师 refresh](retail_tau2_teacher_refresh_20260905_v1.json) | Retail 教师与用户模拟器配置 |
| 教师 SFT | [100-step SFT](retail_teacher_refresh_sft_v4_s100.json) | 88 条数据版本、QLoRA、周期验证及 checkpoint 保存 |
| Agentic GRPO | [Task113 staged-v7.1](retail_agentic_qwen3_4b_task113_sft30_staged_v7_1_n4_acc8_s50_v1.json) | SFT checkpoint-30、n=4、两组梯度累积、50 次更新 |
| 保留任务 | [最终任务留出](retail_final_task_holdout_v1.json) | 任务集合及使用边界 |

## 配置中的关键绑定

- `model`：模型路径、来源和预期哈希。
- `data` / `data_dir`：数据发布目录、task split、opening 及其 manifest。
- `upstream`：τ²-bench 提交和关键文件哈希。
- `sft` / `grpo`：训练步数、学习率、采样组大小、累积与 KL 等参数。
- `claims`：实验结果允许支持的结论范围。

训练配置包含特定运行环境的路径与已发布数据哈希，并非 clone 后可直接启动的通用模板。新实验应复制为独立配置、绑定实际模型与数据，保留校验并记录新版本；不要通过删除哈希检查绕过缺失输入。

同一任务的 Terminal / Staged 对照还需固定 opening、checkpoint、采样预算、超参数与评测协议。只更换文件名无法构成公平对照。

其余配置保留于原路径，便于既有脚本、测试和实验 manifest 复现。训练入口与依赖见[安装与运行](../docs/QUICKSTART.md)，实际参数和结果见[技术报告](../TECHNICAL_REPORT.md)。
