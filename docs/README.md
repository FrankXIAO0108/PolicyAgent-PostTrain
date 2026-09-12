# 文档导航

[项目首页](../README.md) · [安装与运行](QUICKSTART.md) · [技术报告](../TECHNICAL_REPORT.md) · [配置](../configs/README.md) · [脚本](../scripts/README.md)

## 从哪里开始

| 目标 | 入口 |
|---|---|
| 了解项目和系统架构 | [项目首页](../README.md) |
| 运行离线示例、状态重放和测试 | [安装与运行](QUICKSTART.md) |
| 阅读 SFT、GRPO、奖励公式和失败分析 | [技术报告](../TECHNICAL_REPORT.md) |
| 找到对应训练配置 | [配置导航](../configs/README.md) |
| 绘制真实训练曲线或检查轨迹 | [脚本导航](../scripts/README.md) |

## 数据与后训练

- [轨迹质量分类标准](02_评测与失败分析/2026-07-22_轨迹质量分类标准_v1.md)：区分可用正例、待修正、环境异常与冲突样本。
- [单人复核与开发训练](04_数据治理与后训练/2026-08-21_单人复核与开发训练门禁决策.md)：开发数据的审阅、修正与发布规则。
- [修正轨迹协议](04_数据治理与后训练/2026-07-28_修正轨迹协议.md)：行为修正和工具状态重放。
- [SFT 数据发布协议](04_数据治理与后训练/2026-07-28_SFT数据发布协议.md)：划分、哈希与训练目标校验。
- [技术报告](../TECHNICAL_REPORT.md)：教师数据规模、协议桥、SFT loss mask、checkpoint 选择、GRPO 实现和实测结果。

## 评测、Reward 与 Guard

- [严格评测实现](../src/evaluation/strict_task_evaluator.py)与[原子谓词](../src/evaluation/retail_predicates.py)：固定能力组与证据判定。
- [分层 Reward 实现](../src/evaluation/staged_reward_shadow.py)：过程分项、限制项和轨迹总分。
- [退款时效检查](../src/evaluation/refund_timing.py)：事实声明与退款方式的对应规则。
- [状态重放](../src/evaluation/replay_evaluator.py)与[数据库差分](../src/evaluation/db_diff.py)：解释实际执行与目标状态的差异。
- [Runtime Guard](../src/guards/retail_pre_action.py)：工具执行前的规则检查。
- [Guard 合成场景诊断](03_Verifier与Guard/2026-07-30_Guard合成场景诊断报告.md)与[规则族消融](03_Verifier与Guard/2026-08-02_Guard规则族消融实验报告.md)：规则覆盖的开发回归。

## 公开证据

| 内容 | 文件 |
|---|---|
| Retail 冻结基线 | [实验目录](../experiments/20260722_110504_retail_baseline20_trial1_deepseek/) |
| 状态重放结果 | [评测报告](../reports/evaluation/final_report.json) |
| 两种评测路径对照 | [对照结果](../experiments/20260726_v6_vs_v7_evaluation/comparison.json) |
| Guard 离线审计 | [审计结果](../experiments/20260726_pre_action_guard_v1/guard_audit.json) |
| SFT / GRPO 参数、日志和轨迹索引 | [技术报告及附录](../TECHNICAL_REPORT.md) |

部分完整轨迹与模型仅存于本地实验目录，技术报告中的本地路径不代表公开下载地址。离线示例只依赖随仓库提交的证据。

## 专题资料

以下目录保留协议、设计决策和实验报告。带日期文件描述相应实验版本，使用时核对配置和原始产物。

| 目录 | 内容 |
|---|---|
| [工程治理](00_工程治理/) | 执行规范、代码复核与验收 |
| [项目与上游](01_项目总览/) | 上游版本、架构和设计背景 |
| [评测与失败分析](02_评测与失败分析/) | 失败分类、样本审计和原因分析 |
| [Verifier 与 Guard](03_Verifier与Guard/) | 判定规则、对抗检查与执行防护 |
| [数据治理与后训练](04_数据治理与后训练/) | 数据协议、训练设计和实验复盘 |
| [英文协议](06_历史英文原版/) | 协议原文 |
| [上游证据](07_证据索引/) | 上游源码版本绑定 |
