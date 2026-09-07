# Research Output Manifest

> 本表仅登记本次本地实验准备产物，不代表训练完成。

| Timestamp | Skill | File | Stage | Description |
|-----------|-------|------|-------|-------------|
| 2026-09-06 | experiment-bridge | refine-logs/task113_staged_s50_20260906/EXPERIMENT_PLAN.md | implementation | Task113 分层 n4、两组累积、50步设计及执行边界 |
| 2026-09-06 | experiment-bridge | configs/retail_agentic_qwen3_4b_task113_sft30_staged_v7_1_n4_acc8_s50_v1.json | implementation | 独立训练配置，未部署 |
| 2026-09-06 | experiment-bridge | scripts/audit_task113_staged_preparation.py | implementation | 两批已有开发轨迹离线重评分 |
| 2026-09-06 | experiment-bridge | scripts/prepare_task113_staged_eval.py | implementation | 冻结66条评测配置生成器，不启动评测 |
| 2026-09-06 | experiment-bridge | tests/test_task113_staged_accumulation.py | implementation | 评分局部排除、批次计数与评测协议测试 |
| 2026-09-06 | experiment-bridge | _local_private_runs/task113_staged_acc8_20260906/offline_audit_v1/report.json | implementation | 实际离线复算；GPU和API调用均为false |
| 2026-09-06 | experiment-bridge | _local_private_runs/task113_staged_acc8_20260906/offline_audit_v2/report.json | implementation | 复核修复后的最终离线复算，保留历史报告 |
| 2026-09-06 | experiment-bridge | _local_private_runs/task113_staged_acc8_20260906/offline_audit_v3/report.json | implementation | 最终条件/并列句与部分写授权修复后的配置复算 |
| 2026-09-06 | experiment-bridge | refine-logs/task113_staged_s50_20260906/LOCAL_VALIDATION.md | implementation | 分解释器记录测试、失败原因及未验证CUDA事项 |
| 2026-09-06 | experiment-bridge | refine-logs/task113_staged_s50_20260906/EXPERIMENT_CODE_REVIEW.md | review | 附加Agent代码复核，修复两项实际阻塞 |
