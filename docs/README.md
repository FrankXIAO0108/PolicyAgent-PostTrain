# 项目文档导航

本目录同时保存面向招聘方的中文材料、当前实验协议，以及项目早期的冻结历史
记录。为了保证哈希、时间线和实验复现，已经冻结的历史记录不会仅为翻译而
改写。

## 推荐阅读顺序

1. [项目完成报告](PROJECT_COMPLETION_REPORT.md)
2. [面试讲解手册](INTERVIEW_PLAYBOOK.md)
3. [任务路线审查](20260728_route_alignment_audit.zh-CN.md)
4. [后训练阶段门禁](20260728_post_training_readiness_protocol.zh-CN.md)
5. [轨迹质量政策](20260722_trajectory_quality_taxonomy_v1.md)

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

## 历史与环境

| 文档 | 说明 |
|---|---|
| [迁移上下文](MIGRATION_CONTEXT.md) | 迁移时的证据快照，部分为英文 |
| [环境基线](environment_baseline.md) | Windows、Python 和上游版本选择 |
| [上游架构地图](repo_architecture.md) | tau2-bench 代码结构与本项目边界 |

部分历史文件保留英文标题或英文技术术语，这是为了保留原始实验语义。公共入口、
项目结论、演示、完成报告和面试材料均以中文为主。
