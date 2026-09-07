# Task113 分层50步：本地验证记录

日期：2026-09-06。没有云连接、API调用、rollout生成或参数更新。

## 最终配置与原始复算

- 配置 SHA256：`2EBA9D821E3EB645540235F069ABDC16D8293FA6A09372FD019DB654A1B5A98F`。
- 最终离线报告：`_local_private_runs/task113_staged_acc8_20260906/offline_audit_v3/report.json`。
- 新 SFT30：原 v7 `[0,.75,.2,1]` → 本轮 `[0,1,.2,1]`。
- 历史 SFT：原 v7 `[0,.75,1,.12]` → 本轮 `[0,.75,1,0]`。
- 最后一项降低来自本轮显式的授权 FAIL 零分门禁，不是假称纯条件句修复。规则 FAIL 本身不是独立人工标签。
- v1/v2 输出均保留；以 v3 对应最终配置为准。

## 本地实际测试

tau2 集成相关测试使用 `D:/tau2-bench/.venv/Scripts/python.exe`，设置 `POLICYAGENT_TAU2_ROOT=D:/tau2-bench`、`LITELLM_LOCAL_MODEL_COST_MAP=True`、`PYTHONUTF8=1`。

```powershell
& D:/tau2-bench/.venv/Scripts/python.exe -m pytest tests/test_task113_staged_accumulation.py tests/test_agentic_launch_safety.py tests/test_staged_reward_shadow.py tests/test_task113_prescreen_configs.py tests/test_completion_budget_terminal_failure.py tests/test_sft_development_probe_preflight.py -q -p no:cacheprovider
# 160 passed, 1 warning

& D:/tau2-bench/.venv/Scripts/python.exe -m pytest tests/test_retail_agentic_rl.py -q -p no:cacheprovider
# 84 passed, 1 warning

python -m pytest tests/test_rollout_diagnostics.py -q -p no:cacheprovider --tb=short
# 本机 Python312 已有 torch，用于张量单测：82 passed, 4 skipped

& D:/tau2-bench/.venv/Scripts/python.exe -m ruff check scripts/audit_task113_staged_preparation.py scripts/prepare_task113_staged_eval.py tests/test_task113_staged_accumulation.py src/training/run_retail_agentic_grpo.py src/evaluation/staged_reward_shadow.py
# All checks passed

& D:/tau2-bench/.venv/Scripts/python.exe scripts/audit_task113_staged_preparation.py --config configs/retail_agentic_qwen3_4b_task113_sft30_staged_v7_1_n4_acc8_s50_v1.json --output-dir _local_private_runs/task113_staged_acc8_20260906/offline_audit_v3
# OFFLINE_AUDIT_PASSED_CUDA_UPDATE_NOT_TESTED
```

两套解释器不可混称一套完整训练环境。最初把 rollout_diagnostics 张量测试放进不含 torch 的 tau2 环境，出现24 failed、30 errors、112 passed、4 skipped，错误均提示 torch 未安装；未因此修改业务逻辑或安装包，随后改用本机已有 torch 的解释器运行该测试文件。4项跳过依赖可选 `_policyagent_audited_trl_loop`，必须在云端预检确认，未计为通过。上游 audioop 弃用 warning 不影响 Retail 文本测试。

## 本轮文件范围

修改已有源码两处（两文件此前均 dirty，保留既有修改）：

- `src/evaluation/staged_reward_shadow.py`：局部退款条件排除；可选每个写调用的授权检查；旧spec默认不变。
- `src/training/run_retail_agentic_grpo.py`：两组更新计数、配置透传与实际参数记录、真实分组统计、新SFT阶段的manifest校验。

新增：独立配置、两个本地辅助脚本、专项测试、设计/复核/验证文档及输出登记。
未改训练数据、上游库、旧实验配置或原始轨迹；未add/commit/push、删文件或迁移工作区。

`git diff --check` 通过，仅现有 Windows 行尾提示。上述两个源码相对HEAD的dirty差异包含本轮之前的改动，不能声称全部由本轮新增。

最终补齐实际GRPO配置在成功manifest与失败保留清单中的登记后，重新运行 `tests/test_task113_staged_accumulation.py` 和 `tests/test_agentic_launch_safety.py`：92 passed。此为聚焦复跑，与上面的160项测试重叠，不累加为新增测试数。

## 开云后必须确认的未验证项

- 当前端点、模型文件仍存在；磁盘有足够保存checkpoint、日志和最终合并模型的空间。
- 实际GPU backward与峰值显存；单卡单进程、原生TRL1.9计数、两组共8条一次更新。
- 依赖源码检查和可选原生loop测试；有效max_grad_norm、decode参数、warmup及KL参考正确。
- DeepSeek API可用性及实际用户模拟器版本元数据；本轮上传与466条轨迹须获得授权。
- 首次真实更新、50步训练、66条评测、四图、实际学习效果：均未完成，不得从本地测试推断。
