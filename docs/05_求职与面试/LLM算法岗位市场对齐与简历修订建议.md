# LLM 算法岗位市场对齐与简历修订建议

日期：2026-08-02

## 1. 判断依据

本报告同时使用三类信息：

1. 用户提供的真实 LLM/Agent/后训练岗位面经；
2. 当前 `PolicyAgent-PostTrain` 仓库、冻结实验和测试；
3. 用户提供的简历截图。

没有因为岗位出现 Skill、Memory、Multi-Agent、GraphRAG、DPO 或 GRPO 等关键词，
就假设项目已经具备这些能力。

`SeismoSearch` 的描述来自简历截图；该项目不在当前工作区，因此其中数字在正式
投递前仍应使用对应仓库与冻结报告复核。

## 2. 两个项目与岗位方向的真实匹配

| 项目 | Agent | RAG | 后训练 | Evaluation | 数据工程 |
|---|---|---|---|---|---|
| PolicyAgent-PostTrain | 强：Tool Calling、Agent Loop、执行前 Guard | 无，不应硬贴 | 中：完成数据治理与门禁，尚未训练 | 最强：状态重放、Verifier、失败归因、A/B 协议 | 强：审计、分层、哈希、split 泄漏和发布门禁 |
| SeismoSearch | 强：确定性 Planner、工具路由 | 最强：Hybrid Retrieval、Rerank、Evidence Pack | 无现有证据 | 强：Holdout、Badcase、Contract/Citation 指标 | 中：结构化事件查询和证据组织 |

这两个项目组合起来已经覆盖当前岗位最关心的四条主线：

```text
Agent 执行
+
RAG 检索
+
Evaluation / Verifier
+
训练数据治理
```

真正缺失的是“已经完成并有冻结对比的后训练 checkpoint”，而不是缺少更多框架
名词。

## 3. 当前简历的优势

### PolicyAgent-PostTrain

- 能解释 `reward == 1` 为什么仍可能存在政策、授权、状态和最终声明错误；
- 不是只调用 LLM Judge，而是重放工具副作用并比较最终数据库状态；
- 有 Baseline、失败分类、Verifier、Guard、合成回归和规则族消融实验；
- 能明确区分 Policy、Tool enforcement 和 Evaluator coverage；
- 有训练数据分层、隔离、哈希和泄漏门禁，符合岗位对数据构建的关注。

### SeismoSearch

- 不是普通“向量库 + LLM”，而是 Planner、结构化查询、统计工具、文档检索和安全
  路由的组合；
- 有 BM25、Dense、Hybrid、Rerank 的离线比较；
- Evidence Pack、引用约束和 Contract 指标能够体现可评测 RAG；
- 可以承担面试中的 RAG、Tool Selection、Planner 和 Evidence Evaluation 问题。

## 4. 技术漏洞与面试风险

### 风险一：把计划写成已完成实验

简历截图中存在以下表述：

- “建立从 Prompt Baseline 到 SFT/DPO/GRPO 的完整闭环”；
- “完成 LoRA SFT → Rejection Sampling/DPO → Outcome-GRPO → Policy-aware GRPO”；
- 使用 `[Best]`、`[P1]`、`[F1]`、`[X]` 等占位指标。

这些陈述与当前仓库证据冲突。当前没有正式 SFT 数据集、checkpoint、DPO 或 GRPO
训练结果。面试官只要追问训练框架、显存、batch、loss、checkpoint、reward
曲线或冻结对比，就会暴露。

### 风险二：项目标题把“后训练”说得过重

项目确实研究后训练数据和 reward 可靠性，但当前完成的是后训练前的数据治理与
readiness gate。标题建议突出“Agent 可靠性评测”，把“后训练”限定为数据治理，
不要让招聘方误以为已经完成模型训练。

### 风险三：缺少真正的 ablation

当前已有 Baseline、failure case 和 Guard 回归，但此前没有量化“移除某组规则后
会漏掉什么”。本项目因此新增 Guard 规则族消融协议，直接补齐面经高频的
baseline / ablation / failure-case 证据链。

### 风险四：框架关键词不能硬贴

| 市场关键词 | 当前事实 | 面试表述 |
|---|---|---|
| Workflow / DAG | 未实现通用 DAG 编排 | 可以比较设计取舍，不能声称使用 |
| LangGraph | 未使用 | 不写入技术栈 |
| Multi-Agent | Tau2 有 Agent/User/Judge 多角色，但不是自研 MAS | 不包装成多智能体协作框架 |
| Skill 系统 | 未实现 Skill registry/retrieval | 不声称具备 200-Skill 检索 |
| Memory | 只有对话状态和 Guard observed state | 不包装成长短期 Memory 系统 |
| GraphRAG | SeismoSearch 是 Hybrid RAG，不是知识图谱推理 | 不混淆 Hybrid RAG 与 GraphRAG |
| RLHF / DPO / GRPO | 只有门禁和选择条件 | 回答“何时应该用”，不回答“我已经跑过” |

## 5. 建议替换的简历项目描述

### PolicyAgent-PostTrain｜政策约束型 Tool Agent 可靠性评测与后训练数据治理

- 基于 tau2-bench Retail 冻结 20 个开发任务及模型、seed、配置和数据哈希，完成
  Prompt Baseline：16/20 业务成功、0 系统失败，并保留 Tool Trace、最终数据库、
  reward breakdown、成本和异常证据。
- 设计确定性状态重放与分层失败归因，将 Agent/目标工具调用分别重放并比较最终
  DB，把失败拆为 Outcome、Action、Policy、State 和 Claim Consistency；在同一
  冻结开发集上复现 20/20 官方结果，消除旧轨迹 LLM pipeline 的 4 个失败漏报。
- 实现不读取 gold 的 Pre-action Guard，覆盖授权范围、订单状态、支付方式、商品
  变体、单轮工具协议和一次性写操作；离线拦截 Task 95/98/107 三个非隔离失败，
  并以 15 个合成场景和规则族消融检查覆盖贡献，在线配对 A/B 尚待付费执行。
- 实现轨迹审计、GOLD/SILVER/SUSPECT/MIXED/EXCLUDED 分层、修正哈希、实体级
  split 泄漏检查及 SFT/DPO/GRPO readiness gate；由于独立人工金标为 0，正式
  SFT、DPO、GRPO 门禁保持关闭。

### SeismoSearch｜可评测的工具增强型 Agentic RAG

- 设计确定性 Planner，将请求路由至 catalog、concept、mixed、safety 四类，组合
  DuckDB 结构化事件查询、统计工具、文档检索与安全短路，避免精确数值问题被直接
  交给普通文档 RAG。
- 构建 BM25 + Dense Embedding + RRF + Cross-Encoder Rerank 检索链路，并在冻结
  评测集上比较 Keyword、BM25、Dense、Hybrid 等方案；简历中的 Source Hit Rate
  96.15% 和 Requirement Hit Rate 88.46% 应在投递前用仓库报告再次核验。
- 设计 Evidence Pack、严格 JSON Contract 和 Evidence ID 引用约束，建立 Query
  Routing、Tool Selection、Evidence Correctness、Citation Support、Safety
  Refusal 与 Latency 的端到端评测和 Badcase 分析。

## 6. 面试故事应该如何闭环

```text
业务风险
→ 冻结 Baseline
→ 找到 reward / 真实正确性的缺口
→ 状态重放与失败分类
→ Verifier / Guard
→ 消融与失败案例
→ 数据治理门禁
→ 只有监督可靠后才进入 SFT / DPO / GRPO
```

这个故事比“我顺序跑了 SFT、DPO、GRPO”更可信，也更贴合企业正在关注的数据、
评测、Agent 可靠性和业务落地。

## 7. 后续开发优先级

1. **当前完成**：Guard 规则族消融，回答规则贡献与 case 依赖；
2. **下一步**：获得预算后执行 Base/Guarded 在线配对 A/B，测恢复、误拦截、延迟
   和额外调用；
3. **关键阻塞**：获得独立政策 gold，验证 Verifier 的 P/R/F1 和全部 FP/FN；
4. **门禁打开后**：发布严格拆分的 SFT 数据，先做 Base-vs-SFT 冻结对比；
5. **条件性工作**：只有残余系统错误和可靠 reward 支持时再决定 DPO/GRPO；
6. **不优先**：为了关键词单独增加 Memory、Skill、Multi-Agent 或 GraphRAG。
