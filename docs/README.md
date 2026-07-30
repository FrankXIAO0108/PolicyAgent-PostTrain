# 项目文档导航

本目录同时保存面向招聘方的中文材料、当前实验协议，以及项目早期的冻结历史
记录。为了保证哈希、时间线和实验复现，已经冻结的历史记录不会仅为翻译而
改写。

## 推荐阅读顺序

1. [完整工程复盘](project_review.md)
2. [失败实验与版本迭代](failure_analysis.md)
3. [Verifier 与大模型面试基础](llm_fundamentals.md)
4. [项目表达、30 题模拟面试与知识地图](interview_questions.md)
5. [项目完成报告](PROJECT_COMPLETION_REPORT.md)
6. [面试讲解手册](INTERVIEW_PLAYBOOK.md)
7. [任务路线审查](20260728_route_alignment_audit.zh-CN.md)
8. [后训练阶段门禁](20260728_post_training_readiness_protocol.zh-CN.md)
9. [轨迹质量政策](20260722_trajectory_quality_taxonomy_v1.md)

## 求职复盘教材

| 文档 | 对应内容 |
|---|---|
| [完整工程复盘](project_review.md) | 项目背景、生态位置、技术路线、时间线与 Baseline |
| [失败实验与版本迭代](failure_analysis.md) | 按现象、原因、方案、代码和效果复盘每轮问题 |
| [Verifier 与大模型面试基础](llm_fundamentals.md) | Verifier 版本演进、Reward Model、RLHF、DPO、PPO、GRPO |
| [项目表达与模拟面试](interview_questions.md) | 30 秒/3 分钟表达、30 道项目题和三级知识地图 |

## 核心研究文档

| 文档 | 内容 |
|---|---|
| [Failure Taxonomy v2](20260722_failure_taxonomy_v2.md) | 四个失败任务的人工分类和训练资格 |
| [Trajectory Quality Taxonomy](20260722_trajectory_quality_taxonomy_v1.md) | GOLD、SILVER、SUSPECT、MIXED 等数据策略 |
| [Verifier V2 阶段报告](20260727_policy_grounding_v2_report.md) | Programmatic Verifier 的开发证据和边界 |
| [独立裁决协议](20260727_independent_adjudication_protocol.zh-CN.md) | 双人盲审、冲突解决和金标发布 |
| [修正轨迹协议](20260728_corrected_trajectory_protocol.md) | 修正完整性、重放与双人审批 |
| [轨迹质量裁决](20260728_trajectory_quality_adjudication_protocol.md) | 政策金标之后的数据质量决策 |
| [SFT 决策构建](20260728_sft_decision_builder_protocol.md) | 标签、修正、split 和实体组装 |
| [SFT 发布协议](20260728_sft_release_protocol.md) | 数据哈希、泄漏和 loss mask 门禁 |
| [后训练阶段门禁](20260728_post_training_readiness_protocol.zh-CN.md) | SFT、DPO、RLHF/GRPO 的 go/no-go 条件 |
| [Guard 合成场景诊断](20260730_guard_synthetic_diagnostic_report.md) | 15 个新场景的确定性回归、结果与边界 |
| [Guard 在线 A/B 预检](20260730_guard_online_ab_preflight_report.md) | 配对协议、付费门禁、比较器和当前阻塞项 |

## 历史与环境

| 文档 | 说明 |
|---|---|
| [迁移上下文](MIGRATION_CONTEXT.md) | 迁移时的证据快照，部分为英文 |
| [环境基线](environment_baseline.md) | Windows、Python 和上游版本选择 |
| [上游架构地图](repo_architecture.md) | tau2-bench 代码结构与本项目边界 |

部分历史文件保留英文标题或英文技术术语，这是为了保留原始实验语义。公共入口、
项目结论、演示、完成报告和面试材料均以中文为主。
