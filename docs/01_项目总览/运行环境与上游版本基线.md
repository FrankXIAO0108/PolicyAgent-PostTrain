# 环境与上游基线记录

## 1. 本地环境

- 操作系统：Windows
- Python：3.12.10
- uv：0.11.29
- Git：2.53.0.windows.2

## 2. 上游 Benchmark

官方仓库：

https://github.com/sierra-research/tau2-bench

当前目标业务域：

Retail

## 3. 官方正式版本参考

- Tag：v1.0.0
- Commit：17e07b1da2bbc0cadfddeea36412686e0604127b

## 4. 当前运行基线

- Commit：58e5e1ace69302e6982d27014569c03e0ffccdd2

### 选择该运行基线的原因

官方 v1.0.0 对应的 commit 在本机按照 core + dev + gym 方式安装后，
执行 `import tau2` 和 `tau2 intro` 时，会错误触发可选的 Voice 模块依赖，
最终因为缺少 `scipy` 而无法正常运行文本模式。

经过排查确认，上游后续 commit `58e5e1a` 修复了 Voice 和 Knowledge
相关可选依赖被提前导入的问题。

切换至该 commit 后，在未安装 Voice 依赖的情况下：

- `import tau2` 可以正常执行；
- `tau2 intro` 可以正常启动；
- Retail 文本业务域可以正常注册和创建环境。

同时，通过 Git 差异检查，目前未发现从 v1.0.0 到该运行基线之间，
以下核心路径存在直接修改：

- Retail Domain
- Retail 业务逻辑
- Evaluator
- Environment
- Orchestrator

但 Runner 和 Simulation 相关基础设施存在改动，
因此后续在正式实验前仍需要进一步核验其是否影响文本 Retail 的执行语义。

## 5. 已完成验证

已成功执行：

```powershell
uv sync --extra dev --extra gym