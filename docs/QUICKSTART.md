# 安装与运行

[项目首页](../README.md) · [配置导航](../configs/README.md) · [脚本导航](../scripts/README.md)

## 离线示例

在项目根目录使用 Python 3.12：

```bash
python -m src.project_summary
python -m src.project_summary --format json --output output/demo/summary.json
```

此入口只读取仓库中的三份冻结 JSON，展示 Baseline、状态差分和 Guard 审计；不运行模型或重新评分。Windows 也可执行 `.\demo.ps1`。

## Retail 环境与测试

实际重放需要独立的 τ²-bench checkout。以下以 Linux / bash 为例，在项目根目录创建虚拟环境，并把上游放在相邻目录：

```bash
python3.12 -m venv .venv
source .venv/bin/activate
git clone https://github.com/sierra-research/tau2-bench.git ../tau2-bench
git -C ../tau2-bench checkout 58e5e1ace69302e6982d27014569c03e0ffccdd2
python -m pip install -e "../tau2-bench[dev,gym]"
export POLICYAGENT_TAU2_ROOT="$(cd ../tau2-bench && pwd)"
git -C "$POLICYAGENT_TAU2_ROOT" rev-parse HEAD
```

若上游 checkout 已存在，复用版本一致的环境即可。PowerShell 使用 `.\.venv\Scripts\Activate.ps1` 激活环境，并用 `$env:POLICYAGENT_TAU2_ROOT = "D:\path\to\tau2-bench"` 设置上游路径。

无需 CUDA 的入门回归：

```bash
python -m pytest tests/test_project_summary.py tests/test_strict_task_evaluator.py -q
```

在依赖齐备的环境中运行 `python -m pytest -q` 可检查全套测试；部分用例依赖 PyTorch、TRL、CUDA 或本地实验文件，应按 skip 原因解释覆盖范围。本机环境约定见[易错提醒](../易错提醒.md)。

## 重放公开轨迹

基于同一初始 DB 重放实际和参考操作，输出结构化差分及失败分析；自然语言断言复用冻结结果，不调用新的 Judge：

```bash
python -m src.evaluation.pipeline \
  --experiment experiments/20260722_110504_retail_baseline20_trial1_deepseek \
  --tau2-root "$POLICYAGENT_TAU2_ROOT" \
  --output output/replay-demo
```

输出 `final_report.json` 和 `failure_analysis.md`。对同一组轨迹运行 Guard 离线反事实审计：

```bash
python -m src.guards.offline_audit \
  --experiment experiments/20260722_110504_retail_baseline20_trial1_deepseek \
  --output output/guard-demo
```

输出 `guard_audit.json` 和 `analysis.md`。这些命令写入 `output/`，不会覆盖仓库中的冻结报告；保留多轮结果时使用不同输出目录。

## 训练入口

训练使用 Linux / NVIDIA CUDA 环境，额外依赖见 [requirements-agentic-rl.txt](../requirements-agentic-rl.txt)。在匹配 GPU 驱动的 PyTorch 环境中安装：

```bash
python -m pip install -r requirements-agentic-rl.txt
```

完整运行需要本地模型、已审核发布的数据目录、对应 manifest 和哈希。具体 checkpoint、环境版本和训练参数见[技术报告](../TECHNICAL_REPORT.md)，配置入口见[配置导航](../configs/README.md)。

以下 `configs/my_sft.json`、`configs/my_prescreen.json` 和 `configs/my_grpo.json` 代表自行准备、绑定实际输入的新配置；它们不是仓库内置文件。预检默认要求干净工作区，应先完成配置与数据版本冻结并提交。

预采样配置可以参考 [Task113 n=4 采样结构](../configs/retail_agentic_qwen3_4b_task113_passk_n4_prescreen_v1.json)：其 `execution_mode=ROLLOUT_DIAGNOSTIC`、`learning_rate=0`、`beta=0`，并有采样预算、opening manifest 与任务绑定。正式优化使用独立 GRPO 配置。参考配置绑定特定 checkpoint；新对照实验必须重新一致绑定模型与数据，不能直接视为当前模型的配对采样。

SFT 输入校验通过后执行：

```bash
python -m src.training.run_teacher_sft --config configs/my_sft.json --validate-only
python -m src.training.run_teacher_sft --config configs/my_sft.json --output-dir output/my-sft
```

GRPO 先校验业务环境，再检查 GPU、模型和训练依赖：

```bash
python -m src.training.run_retail_agentic_grpo --config configs/my_grpo.json --environment-only-preflight
python -m src.training.run_retail_agentic_grpo --config configs/my_grpo.json --preflight-only
```

动态交互还需显式设置用户模拟器。以下为 DeepSeek 路由示例，服务别名不代表固定权重版本；记录运行时返回的模型信息，并在所有对照中固定模型与采样参数：

```bash
export POLICYAGENT_USER_MODEL="deepseek/deepseek-chat"
export POLICYAGENT_USER_LLM_ARGS_JSON='{"temperature": 0.0}'
```

将 `DEEPSEEK_API_KEY` 配置到运行环境后，先对独立采样配置执行预检，再采样和优化：

```bash
python -m src.training.run_retail_agentic_grpo --config configs/my_prescreen.json --preflight-only
python -m src.training.run_retail_agentic_grpo --config configs/my_prescreen.json --sample-only --completion-budget 8192 --groups-per-task 1 --output-dir output/my-prescreen
python -m src.training.run_retail_agentic_grpo --config configs/my_grpo.json --output-dir output/my-grpo
```

示例采样预算为 8192 token；应按选定任务的完整轨迹长度和配置调整。`--sample-only` 和正式 GRPO 都会运行用户模拟器，使用 DeepSeek 路由时需在环境中配置 `DEEPSEEK_API_KEY`，并产生 API 调用和 GPU 开销。SFT 本身不调用用户模拟器。`--user-simulator-api-preflight-only` 也会访问 API，不是离线检查。

## 曲线与结果

从实际训练日志生成 GRPO 四图：

```bash
python scripts/plot_grpo_training_curves.py output/my-grpo --output output/figures/grpo.png
```

SFT 绘图及进一步轨迹审计见[脚本导航](../scripts/README.md)。比较模型时同时查看固定协议下的任务结果、违规行为与 Reward 分项；训练 loss 或 Reward 单独变化不构成效果提升证据。
