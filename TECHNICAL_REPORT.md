# PolicyAgent-PostTrain 技术报告

## 电商多步 Tool Agent 的数据治理、SFT、GRPO 与奖励可信度诊断

版本：v1.0｜核验日期：2026-09-07｜读者：算法工程师、技术面试官

本报告回答的是“实现了什么、为什么这样实现、证据支持到哪里”，不是项目宣传稿。事实来自当前源码、运行时源码快照、配置、训练日志和轨迹；推断与未完成事项单独标明。本轮未启动训练、未连接云端、未调用模型 API，未改变历史评分或实验配置。

### 阅读导航

- 第 1—3 节：项目价值、系统架构、上游与自研边界。
- 第 4—5 节：教师数据、SFT 实现与过拟合判断。
- 第 6—8 节：多步 rollout、GRPO 数学与实际奖励公式。
- 第 9—11 节：真实实验、完整案例与分层失败归因。
- 第 12—15 节：复现条件、技术债务、结论边界和面试问答。
- 附录：可打开的源码、配置、原始证据与曲线索引。

## 1. 核心结论与版本边界

1. **已完成真实后训练工程，不等于已证明 GRPO 提升。** 最新 Task113 实验完成 50 次优化器更新、400 条在线轨迹、100 个独立 n=4 组；参数变化、有限梯度、KL 日志和累积计数有证据。计划中的 GRPO 后评测只完成第一组 4 条，不能报告完整 33 条配对评测。
2. **SFT 有拟合证据，但不能据此宣布能力达到上限。** 最新数据为 88 条轨迹，66 条训练、22 条验证。独立 100-step SFT 运行按验证损失选择 checkpoint-30；之后训练损失继续下降，验证损失轻微回升。
3. **分层奖励确实产生组内差异，但不等于正确的逐动作学习信号。** 当前实现把多个检查汇总为轨迹标量，再进行组内标准化。最新训练有 145 条退款声明规则 FAIL，其中 52 条具有正 advantage；这既涉及规则误判，也涉及轨迹总分的相对排序。
4. **已证明一个具体的评分失真机制，而非证明唯一根因。** 正确的礼品卡条件说明会被旧退款规则误伤。修订规则后，对同样 400 条轨迹重评分的均值上升约 4%，但模型没有重新训练，终局成功仍为 360/400；不能称为能力提升 4%。
5. **目前可交付的是可追溯的开发实验与诊断闭环。** 独立专家金标、完整最终冻结评测、跨任务泛化收益、Terminal 与 Staged 的最终公平优势结论，仍未建立。

证据范围：最新 SFT 数据与日志 [E2—E4]，最新 GRPO 原始产物 [E5—E8]，退款审计 [E9—E10]。历史材料只用于解释版本演进，不替代最新实测。

### 1.1 当前仓库与实验源码不是同一个快照

| 对象 | 已核实状态 | 解读 |
|---|---|---|
| 本地仓库 | `main`，HEAD `5e50138f13c4fcc408c714d074449c1ef05549dc` | 核验时工作区非干净；存在 13 个已跟踪修改文件及多项未跟踪内容 |
| 上游 tau2 | `58e5e1ace69302e6982d27014569c03e0ffccdd2` | 使用上游 Retail 环境，不是从零实现 benchmark |
| 最新云端运行记录 | Git 基底 `5be4b9803588aafe25b2436c140f09cc54145d55`，`dirty_at_start=true` | 实际载荷还包含后续代码，不能只用该 commit 复现 |
| 最新退款候选 v2 | 本地代码及离线复算 | 没有进入已完成的 v7.1 训练 |

SFT 部署 manifest 记录的源仓库 HEAD 是 `5e50138`，运行 manifest 记录的是云端共享克隆基底 `5be4b98`。两者有明确区别，运行源码需由载荷文件哈希补足。报告初次核验时 `run_teacher_sft.py` 的 SHA256 与 SFT 部署 manifest 一致；2026-09-07 提交整理删除了一个未使用的导入，现行源码字节已不同，旧实验仍以部署快照为准。本报告解释 GRPO 实际行为时优先读取回收的运行源码，而不是直接套用修改后的本地 scorer。

## 2. 项目究竟解决什么问题

### 2.1 业务目标：结果正确，并且过程有授权、证据和边界

电商 Agent 不仅要“说得像客服”，还要从用户需求出发，查询真实订单及商品，确认修改范围，调用工具改变状态，最后如实说明结果。

例如 Task44 要处理已有订单中的台灯替换及差价退款；Task113 涉及查看多个订单，并取消满足条件的两笔待处理订单。困难不只在工具 JSON 格式，还包括：

- 查询后才能知道哪些操作合法，不能先猜订单状态。
- 用户确认必须约束到当前操作和具体参数，不能把早先泛泛同意当成所有后续写操作授权。
- 工具调用成功，不代表改对了商品、订单或金额。
- 最后 DB 正确，不代表沟通过程没有虚假退款承诺。
- 面向模型的交互终止协议与用户模拟器的停止状态必须一致。

因此区分三个目标：**业务终局、过程合规、交互完整性**。它们相关，但不能互相替代。

### 2.2 为什么考虑 SFT 和 RL，而不是默认串联

从决策过程看，模型观察到的是历史对话和工具结果，而不是全部隐藏业务状态；动作包括业务工具调用和向用户发问。后续观察取决于当前动作，因此具有部分可观测、多步决策的特征。

SFT 用于学习已审核行为及动作协议；GRPO 尝试在同一任务的多条可执行轨迹中优化相对较好的策略。选择 RL 的必要证据是：模型能采到有效行为，并且奖励能够可靠地区分轨迹。任务“很长”本身不构成 RL 有效的证明。

如果问题只是漏学 `respond_to_user`、固定格式或明确业务映射，先修协议或补 SFT 更直接。若已有可靠偏好对，DPO 是另一种可选方法；本报告不把“存在 DPO 入口或计划”写成已完成 Retail DPO 收益。

## 3. 系统架构及贡献边界

### 3.1 两条相连但不同的执行链

```text
教师数据链
tau2 Task + Policy + DB + Tools
  → 教师 Agent 与用户模拟器交互
  → 原始轨迹 → 审计/修正/环境重放 → 审阅与发布门禁
  → 数据划分和哈希 → 协议转换 → QLoRA SFT → 验证/行为评测

在线 GRPO 链
冻结 SFT checkpoint + Policy/工具 schema + 冻结用户 opening
  → TRL 为同一任务采样 n 条轨迹
  → 每条独立 RetailAgenticEnvironment，真实执行工具/用户交互
  → 环境重放与规则打分 → 轨迹标量 reward → 组内 advantage
  → 仅对模型生成 token 更新 LoRA，带固定参考 KL
  → 保存 checkpoint → 同协议无更新评测 → 轨迹审计
```

### 3.2 哪些是上游提供，哪些是本仓库实现

| 层 | 来源 | 本项目实际工作 |
|---|---|---|
| Retail 任务、初始 DB、业务 Policy、工具及状态变更语义 | `sierra-research/tau2-bench` | 固定版本并接入；审计其检查范围与任务冲突 |
| 用户模拟器框架、环境重放及基础 evaluator | tau2 | 冻结 opening、隔离隐藏情境、记录用户模型绑定和失败 |
| Qwen 模型、tokenizer、预训练/指令能力 | 模型上游 | 选择并校验起点，进行项目数据上的适配 |
| LoRA/QLoRA、优化器与训练框架 | PEFT、bitsandbytes、PyTorch、Transformers、TRL | 配置和集成；不声称自行实现这些算法底层 |
| 数据与训练工作流 | 本仓库 | 质量标签、修正重放、发布门禁、数据转换、SFT runner |
| Agentic RL 接口及运行可靠性 | 本仓库 | `RetailAgenticEnvironment`、生成诊断、完整性门禁、累积校验、证据保存 |
| 可靠性评测与训练奖励 | 本仓库 | 状态差分、归因、Guard、任务 rubric、确定性谓词、分层 reward 和离线敏感性审计 |

本项目不是从零开发电商环境，也不是从零实现 GRPO。工程价值在于**把多步交互、训练表示、可核验评分和复评链路连接起来，并能够追查失效来源**。

### 3.3 关键目录

| 目录 | 作用 |
|---|---|
| `src/agents/`、`src/guards/` | 教师/受 Guard 约束的 Agent 与执行前规则 |
| `src/evaluation/`、`src/verifiers/` | 重放、差分、归因、严格评测、评分和审阅工具 |
| `src/training/` | 数据校验/发布、SFT/GRPO runner、生成与优化证据 |
| `src/rl/` | Retail Agentic 环境、任务划分、用户 opening、API 失败处理 |
| `configs/`、`data/` | 版本化实验配置、rubric、split、冻结 opening |
| `experiments/` | 较早基线、审计与实验资料 |
| `_local_private_runs/` | 大量本地原始轨迹、模型/日志、部署清单和派生分析；不能假设 Git clone 后全部可得 |
| `tests/`、`scripts/`、`docs/` | 聚焦回归、运行辅助和带日期决策记录 |

### 3.4 DB、Policy 与 evaluator 的关系

DB 是用户、订单、商品、支付记录等业务状态，不是训练样本表。上游 `EnvironmentEvaluator.calculate_reward` 从相同初始状态构造 predicted 与 gold 两个环境：predicted 重放实际轨迹，gold 执行预设 actions，然后比较 DB 哈希；环境断言按任务 `reward_basis` 加入。

gold actions 因而可以用于构造目标状态，**不代表所有查询和对话必须逐步复制唯一参考路径**。上游另有 Action、Communicate、NL Assertions 等检查；是否参与总分取决于调用的 evaluation type 和 reward basis，不能笼统说“上游完全不检查 Policy”或“上游完整检查了 Policy”。

最新 GRPO wrapper 直接调用 Environment 和 Communicate evaluator，记录 `nl_assertions_used=false`。Task113 的环境输出虽带有 `DB`、`NL_ASSERTION` 等任务元数据，该训练路径并没有因此自动运行 LLM NL Judge。

本项目的 `replay_evaluator.py` 还保存初始、实际、参考状态及重放错误，供字段级差分和归因使用。额外价值是解释**哪里不同、如何造成**，而非重复输出 0/1。

### 3.5 严格评测、Guard 与训练 reward 必须分开

严格评测采用固定能力组：评测完整性、最终状态、必要执行、无效动作规避、协议合规、意图一致、证据一致。原子结果为 `PASS / FAIL / REVIEW / NOT_APPLICABLE / ERROR`；缺少或重复必要证据会使评测无效，不直接记为模型失败。只有评测有效且所有必要组 PASS 才是 `strict_pass`。

`final_claim_matches_state` 消费已经结构化、绑定最终回答哈希的 claim-check，**不是自然语言万能理解器**。哈希证明绑定关系，不能证明 claim-check 本身正确。

另一个易混点：最新 RL 环境的 `_call_tool` 会记录 Guard finding，但仍调用上游工具；配置为 `policy_guard_used_as_reward=false`。因此本轮不是“Guard 阻断了错误动作后再宣称模型学会安全”。严格评测、Guard 与 staged scorer 是不同路径，不能把一条路径的测试通过转移为另一条路径的可信度。[C1—C3]

## 4. 教师数据：从上游任务到可训练轨迹

### 4.1 数据来源与实际规模

教师轨迹来自合成 Retail 世界中的模型—环境交互，不是真实商家生产对话，也不是直接把任务的 gold actions 展开成模型答案。

| 数据阶段 | 轨迹数 | TRAIN | VALIDATION | 任务覆盖 | 备注 |
|---|---:|---:|---:|---:|---|
| v3 合并池 | 80 | 58 | 22 | 55 个 task | 53 条既有教师候选 + 27 条 Wave A 所有者审阅修正候选 |
| 2026-09-05 refresh 增量 | 8 | 8 | 0 | Task44、73 | 每个 task 4 条；不是新增 8 个任务 |
| v4 最终池 | 88 | 66 | 22 | 55 个 task | 训练 40 个 task，验证 15 个 task；本轮读取 JSONL 重新计数 |

Wave A 另有 11 条 HOLDOUT 未发布。总共 88 条不是 88 条训练数据，更不能把 694 次业务工具调用称为 694 条独立轨迹。最新协议转换保留 694 次业务工具调用，包装 573 次 `respond_to_user`，追加 88 个终止目标。[E2]

### 4.2 教师生成可见什么

refresh 配置允许教师看到 Policy、工具 schema、已观察到的用户消息和工具结果；不允许看到隐藏用户情境、gold DB、预期 actions 和评价条件。用户模拟器持有隐藏场景，教师只能通过对话获知需求。

该批次教师配置为 `deepseek/deepseek-chat`，temperature 梯度 `0.2/0.4/0.6/0.8`；用户模拟器同一服务别名，temperature=0，seed=20260905。配置明确记录：`deepseek-chat` 是服务别名，不能从名称推断每次服务端实际返回的具体模型版本。本报告不把它擅自改写为“全部由 DeepSeek V4-Flash 生成”。

同系列教师与用户模型可能共享表达及行为偏差；确定性检查和所有者审阅降低部分风险，但不等于独立专家金标。

### 4.3 为什么不能只按 reward=1 筛数据

项目质量政策区分：可用正例 GOLD、需修正 SILVER、环境可疑或损坏轨迹、有效模型失败、混合案例和 benchmark 标签冲突。模型可能完成 DB 操作却缺少授权或错误宣称退款到账，这类轨迹不能不加审查地作为“完整正确行为”进行 SFT。

发布链路包括：

1. 保留原始候选及来源哈希，记录具体缺陷。
2. 修正模型行为/对话目标，工具结果通过真实环境重放获得，不手填工具成功。
3. 检查消息结构、call/result 唯一绑定、顺序、重放状态和修正文件哈希。
4. 经过相应审阅流程。开发数据允许所有者复核释放，但其标签仍不是独立专家 gold。
5. 校验 split、重复轨迹、实体交叉和 PII，再发布不可混淆的版本。

最新 8 条发布报告记录：`ready=true`，状态重放/哈希链通过，硬实体泄漏 0，最终 holdout task 交叉 0。这里是发布检查结论，不意味着每一句自然语言都已由完备 verifier 证明正确。

### 4.4 数据隔离的准确口径

最新数据以 `task_id / user_id / order_id` 为硬隔离键；共享商品目录的 `product_id` 重叠只报告、不作为硬阻断。否则共享商品会把大量任务连接起来，使可用划分失效。这意味着不能说“所有实体完全隔离”。

`configs/retail_final_task_holdout_v1.json` 标记 23 个保留 task，范围是 `TASK_UNSEEN_ENTITY_OVERLAP`，状态 `SEALED_NOT_RUN`。这是项目内保留集，不是官方榜单，更不是完全实体未见测试。本轮核对配置与最新发布报告，不把配置状态当成“已重新扫描全部历史产物并证明从未污染”的替代品。

Task44、73 已用于训练和修正，不能再用其后续结果证明未见任务泛化。Task113 不在这份 88 条 SFT 数据中，但此前已用于开发、奖励设计及之后 RL 训练，**不是最终干净测试任务**。

### 4.5 面试中如何回答“数据够不够、质量怎么样”

准确说法：这是单领域、预算受限的开发实验；当前 66 条训练轨迹，质量约束包括真实工具重放、所有者复核、实体划分和哈希。它足以验证工程可行性和局部行为适配，尚不足以支持工业规模或广泛泛化结论。

规模是否足够要看有效任务覆盖、行为覆盖、token 监督和留出表现，而不是仅看条数。重复训练相同任务不会创造新任务知识；添加同一 task 的更多措辞，也不等于扩大独立业务覆盖。

## 5. SFT：实现、训练现象与选择依据

### 5.1 模型沿革与真实起点

模型家族是 `Qwen3-4B-Instruct-2507`。v3 配置以该模型为起点；之后增加原生 tau2 对话到 Agentic 工具协议的桥接 SFT。最新 refresh SFT 从**已经训练过的协议桥 SFT 模型**继续适配，不是从随机初始化或原始基座开始。

最新 refresh 起点目录末尾为 `20260824-sft-v3-agentic-protocol-bridge-s20-v1/teacher_sft_merged`，起点目录哈希 `0A2E06C9…`。最新 GRPO 使用 refresh 100-step 运行选中的 checkpoint-30 合并模型，哈希 `2215F09D…`。目录叫 `sft100` 不意味着实际选择第 100 步。[E3—E5]

### 5.2 协议桥为何必要

tau2 原生对话中，assistant 普通文本直接发给用户；TRL Agentic 路径中，模型要调用 `respond_to_user(message=...)` 才会触发下一条用户回复。普通文本可能被当成模型完成输出，导致工具循环停止。

转换保留原始业务工具和结果，只改变表示：首条用户需求保留为 user；中途客户沟通包装成 `respond_to_user`；其真实用户回复包装成 tool result；观察到 STOP/TRANSFER 后增加 `Interaction complete.` 终止监督。它不是新增业务事实，也不表示重新生成了环境结果。

历史协议文档记录：桥接前出现“能生成自然语言但环境没有继续交互”；桥接后工具/客户交互覆盖增加，但仍存在未完成和长度预算问题。该材料支持协议错位是一个实际故障类别，不支持“桥接已解决全部长程任务”。[H1]

### 5.3 损失和 masking

实际训练由 `TRL SFTTrainer + PEFT LoRA + bitsandbytes NF4` 完成。项目 runner 负责构造消息、模板渲染、监督 mask、校验及产物。

设完整模板序列为 x，m 表示模型应学习的位置，则监督交叉熵为：

\[
L_{SFT}=-\frac{1}{\sum_t m_t}\sum_t m_t\log p_\theta(x_t\mid x_{<t}).
\]

system、用户输入、环境返回作为上下文，不应成为模型预测目标；assistant 文本、工具调用名和参数是目标。代码把非监督位置标签设为 `-100`，padding 同样屏蔽。

`tokenize_row` 优先读取模板的 assistant mask。Qwen3 模板缺少相应 generation 标记时，回退到**完整渲染—清空单条 assistant 输出后重渲染—字符差分—offset mapping**。同时验证所有删除片段重建后与全清空版本一致，不满足则报错。片段边界有 4 字符扩展，这是具体 tokenizer/template 下的工程实现，不是对任意模板都安全的通用定理，必须保留回归测试。

训练超过 `max_length` 不静默截断，直接拒绝；验证辅助函数支持尾部预算截取，但本轮 88 条最长 14,266 token，小于 16,384，最终 22 条验证记录均未截断。不能把历史 8,192 尾部验证损失与当前完整 16,384 上下文损失直接串成一条可比曲线。

### 5.4 最新 100-step 配置

| 项目 | 配置/实测 |
|---|---|
| 数据 | 66 TRAIN + 22 VALIDATION，全轨迹输入 |
| 精度 | 基底 NF4 双重量化，计算 BF16，QLoRA |
| LoRA | r=16，alpha=32，dropout=0.05 |
| 适配模块 | q/k/v/o_proj，gate/up/down_proj |
| 最大长度 | 训练与验证均 16,384 |
| 更新步数 | 100；独立从相同起点重跑，不是从 50 步续训 |
| 批量 | microbatch=1，累积=4；名义每次更新 4 条轨迹 |
| 学习率 | 初始 5e-5，100-step 线性日程 |
| 验证/保存 | 每 10 步；验证集 loss 最小的 checkpoint 用于最终合并 |
| seed | 20260824 |
| 硬件 | 1 张 RTX 4090；运行环境记录支持 BF16 |
| 训练耗时 | `train_runtime=2385.9752 s`，约 39 分 46 秒；不是 GPU 租赁账单 |

名义样本呈现数约为 100×4=400，但 epoch 末尾不足一个累积窗口可能影响精确计数；Trainer 实际记录 epoch=5.90909，不应武断声称每条样本恰好训练 6 遍。这也是后来 GRPO 专门增加实际累积校验的原因之一。

LoRA 的基本形式是 `W'=W+(alpha/r)BA`；本轮 alpha/r=2。选择 r=16、这些模块及学习率是现有受资源约束的工程配置，未有 rank 或模块消融证明其“最佳”。

### 5.5 真正用于判断过拟合的曲线

以下为同一个 100-step 运行、同一 QLoRA 精度、固定训练/验证集合的 eval-mode loss，避免把随机训练 batch loss 与固定验证 loss 混在一起。

| step | 固定 TRAIN66 loss | VALIDATION22 loss |
|---:|---:|---:|
| 10 | 0.243722 | 0.296094 |
| 20 | 0.230619 | 0.291058 |
| 30 | 0.219600 | **0.290027** |
| 40 | 0.208996 | 0.290038 |
| 50 | 0.200148 | 0.290462 |
| 60 | 0.191972 | 0.290566 |
| 70 | 0.186255 | 0.291860 |
| 80 | 0.182378 | 0.293420 |
| 90 | 0.180151 | 0.293622 |
| 100 | 0.179740 | 0.293710 |

观察：30→100，训练固定集 loss 下降，验证 loss 上升约 1.27%。这支持**该数据/训练日程下后期出现轻度过拟合趋势**；不支持“整个 SFT 无效”“模型已经死记所有答案”或“SFT 已达能力上限”。30 与 40 步差异很小，单 seed、22 条验证数据不能证明第 30 步在所有任务上严格最佳。

最终在合并 BF16 模型上另做验证：起点模型 assistant token 平均 NLL 为 0.306649，选中模型为 0.291242，相对下降约 5.02%；监督 token 数相同，为 41,853。这里 `evaluation_base.json` 的 base 指**本次继续训练的起点 SFT**，不是原始 Qwen 基座。

最终 BF16 评估与中途 NF4/BF16 验证不是同一数值口径，不能要求两者精确相等。低起始 loss 也有直接原因：起点已做过 SFT，且只统计目标 assistant token，并使用真实历史的 teacher forcing。

![SFT 同一运行固定训练集与验证集损失](D:/PolicyAgent-PostTrain/_local_private_runs/tr0905/s100/loss_comparison_v1/train_validation_loss.png)

### 5.6 loss 能与不能证明什么

loss、token accuracy 和 entropy 描述 teacher-forced token 拟合；Agent 自由 rollout 会遇到自己造成的分布偏移、状态变化及停止决策。验证 NLL 下降不保证业务成功。

最新 Task113 SFT 冻结采样：greedy 1/1 终局成功；8 组 n=4 随机采样合计 30/32 成功，各组都至少有一个成功。这证明该 checkpoint **能完成这个任务**，但只有一个任务，不能称为模型一般能力或总体 pass@k 结论。[E7]

因此不能把 GRPO 效果不明直接归咎于“Qwen3-4B 不够聪明”。也不能反过来说 SFT 已充分：其早停、错误声明和跨任务能力仍需独立行为指标回答。

## 6. 多步 rollout 如何进入训练

### 6.1 每条轨迹的输入与环境隔离

runner 构造 Policy/system prompt、工具 schema、冻结用户 opening；数据携带 task ID、split 与用户 seed。`reset` 为每条 rollout 新建环境状态及用户模拟器状态；同组共享任务/opening/用户 seed，模型采样不同。隐藏用户情境、gold actions 和 gold DB 不直接进入 actor prompt。

这既避免把答案泄漏给模型，也避免一条轨迹的写入污染另一条。确定性工具状态仍须与外部用户模拟器区分：temperature=0 和 seed 绑定提高可比较性，但不能保证外部服务跨版本、跨日期位级复现。

### 6.2 真实工具与用户交互

业务方法最终构造 tau2 ToolCall，通过 `environment.get_response` 执行并记录结果。`respond_to_user` 则调用动态用户模拟器，把返回消息写入环境对话，同时将其作为工具观察交给 TRL。

最新训练的用户服务预检记录 `deepseek/deepseek-chat`，全部 400 条保存的 `user_seed=20260902`；模型采样 seed 是另一个值 2026090601。**用户 seed、模型 RNG seed 和更新次数不是同一个概念。** 最新训练未启用 LLM reward judge；DeepSeek 的在线角色是用户模拟器。

### 6.3 模型视图、环境视图与 observation mask

同一交互有两种表示：

- 模型视图：assistant 生成 `respond_to_user` 工具调用，用户回复成为 tool observation。
- 环境视图：恢复为真实 assistant/customer 对话，供授权、停止和声明检查。

必须保存两者及其绑定，否则容易把“生成了普通最终回答”误解为“这句话已发给用户”。最新 wrapper 中，普通非工具 completion 不会自动调用用户模拟器；模型提前 EOS 可以是完整传输下的行为失败。

TRL 运行源码用 `tool_mask` 标记模型 token=1、工具返回 token=0；最终 loss mask 为 `completion_mask × tool_mask`。用户回复也属于外部 observation，不对它做策略梯度；模型生成的工具参数和 `respond_to_user` 中的消息则参与训练。

环境输出仍会作为后续 token 的条件上下文。**不参与 loss，不等于不占上下文、KV cache 或计算开销。**

### 6.4 长度与失败分类

最新预算为 completion=8192、客户交互最多 8 轮、业务工具最多 24 次、工具循环最多 32 轮。应分别记录输出预算、总上下文、工具次数和停止原因，而不是统称“长度”。

400 条实测均值：prompt=3828 token；保存的 completion=3738.73，其中模型 token=968.83、observation=2769.90。工具/用户返回约占该 completion 的 74.1%。不能只按模型文字长度估计显存和执行预算。

终止分类至少包括：正常用户 STOP 后模型 EOS、用户未 STOP 时模型 EOS、completion 预算耗尽、工具结果未闭合、上下文/工具循环上限、用户 API/运行系统失败。最新 400 条中，363 条记录 `USER_STOP_AND_MODEL_EOS`，37 条 `MODEL_EOS_BEFORE_USER_STOP`；收到 STOP 不代表 DB 一定正确。

特定配置允许“完整记录的输出预算耗尽”作为终局失败保留，并且 `mask_truncated_completions=false`。但未闭合 Tool Call、丢失 Tool Result 或系统异常不能一概伪装成普通 0 分。Task44 历史实验就因“预算耗尽且 unresolved_tool_call”被拒绝，不能为了跑完步数无条件吞掉这类错误。[E11]

## 7. GRPO：实际参数、数学和学习信号

### 7.1 最新 Task113 实际配置

| 参数 | 实际值 |
|---|---|
| 起点 | refresh SFT100 运行的 selected checkpoint-30，已合并 BF16 |
| 训练任务/opening | 仅 Task113，同一个冻结 opening |
| 模型加载 | BF16 基底 + 新 LoRA；非 NF4，不使用 vLLM |
| LoRA | r=16，alpha=32，dropout=0.05；同七类投影模块 |
| Dropout 开关 | 实际 GRPOConfig 的 `disable_dropout=false`；不能按其他版本默认值假定已关闭 |
| 最大更新次数 | 50 |
| `num_generations` | 4 |
| `per_device_train_batch_size` | 1 |
| `steps_per_generation` | 4 |
| `gradient_accumulation_steps` | 8 |
| 每次更新 | 8 条新轨迹，两个独立 n=4 组 |
| `num_iterations` | 1，每批采样不反复做多轮策略更新 |
| 学习率与日程 | 2e-6；warmup 2 步；linear |
| 解码 | temperature=0.8，top_p=1，top_k=0 |
| KL | beta=0.02，固定参考，不同步 reference |
| 损失 | `dr_grpo`，token-level ratio，`scale_rewards="group"`，epsilon=0.2 |
| 优化器 | AdamW；max_grad_norm=1；weight_decay=0 |
| 其他 | 梯度检查点，非重入；无 entropy bonus；未开启 Trackio/W&B 上报 |

运行依赖记录：Python 3.12.3、PyTorch 2.13.0+cu130、Transformers 5.14.1、TRL 1.9.0；SFT 环境另记录 PEFT 0.19.1、datasets 5.0.0、accelerate 1.14.0、bitsandbytes 0.49.2。这些是本项目保存的实际环境记录，不是建议无条件升级到的通用最新版。

### 7.2 两个 n=4 组，不等于一个 n=8 组

每条轨迹先在所属 4 条内求均值和标准差，再计算 advantage；累积的是两组的梯度。两个组之间不交换奖励基线。

例如一组 `[1,1,1,1]`、另一组 `[0,0,0,0]`，独立 n=4 标准化后全部 A=0；合成 n=8 则会产生正负 A。扩大 global batch 与扩大 group size 解决的是不同问题。

本次在优化器每次更新前检查：8 microsteps、8 fresh rollouts、2 generation batches，以及累积日志与 raw 行数一致。50 次检查均 PASS。为避免仅一个 opening 导致 epoch 在 4 微步处结束并提前更新，数据加载器对整个 prompt 池等比例平铺，使 epoch 对齐完整累积窗口；平铺没有创造新任务。

### 7.3 reward 到 advantage

对第 g 组的 G=4 条轨迹：

\[
\bar r_g=\frac1G\sum_i r_i,\quad
s_g=\sqrt{\frac{\sum_i(r_i-\bar r_g)^2}{G-1}},\quad
A_{g,i}=\frac{r_{g,i}-\bar r_g}{s_g+10^{-4}}.
\]

使用样本标准差（ddof=1）。当前分析按保存顺序每 4 条重建组，同时检查 task、计数和原生组统计；缺少显式持久化 group ID 是仍需披露的限制，不能把任意拼接的轨迹都这样分组。

同一组 A 的均值按定义接近 0，所以“平均 advantage 接近 0”不是 collapse 证据。应看零方差组比例、非零 A 数量和对应行为。

奖励全相等时策略奖励项无相对信号，但若 policy 已偏离 reference，KL 项仍可能产生梯度。反过来，reward std 很小也不必然使 standardized A 很小；标准化会消掉大部分共同尺度，不能把 raw std 叫“组内梯度差”。

### 7.4 DR-GRPO 的实际归一化

令 m 为有效模型 token mask，\(\rho_{i,t}=\exp(\log\pi_\theta-\log\pi_{old})\)。每条轨迹的 A 广播到其所有有效模型 token。KL 估计项为：

\[
d_{i,t}=\log\pi_{ref}-\log\pi_\theta,\qquad
k_{i,t}=\exp(d_{i,t})-d_{i,t}-1.
\]

本轮 Dr-GRPO 单个 microbatch 的实现可写为：

\[
L_{micro}=\frac{1}{KBL_{max}}\sum_{i,t}m_{i,t}
\left[-\min\{\rho_{i,t}A_i,\operatorname{clip}(\rho_{i,t},0.8,1.2)A_i\}+\beta k_{i,t}\right],
\]

其中 K=8 为实际累积数，B=1，\(L_{max}=8192\)。八个 microbatch 反向累积后才更新。它不是按每条轨迹实际长度分别除，也不是 DAPO 的全局有效 token 归一化。

因此把 Lmax 调大不仅可能改变生成预算，还会改变 Dr-GRPO 损失的归一化尺度。未经控制不能将不同长度预算下的 loss/grad_norm 直接横比。较短模型输出在 8192 分母下的数值较小属于需要考虑的尺度因素，**尚未证明是本次效果不佳的因果根因**。

### 7.5 reference、old policy 和新 adapter

该 runner 将已合并 SFT 作为底座，创建新的 LoRA。PEFT 路径不必额外驻留一整份 reference 模型：计算 reference log-prob 时禁用新 adapter，得到固定 SFT 起点。不是 `ref_model=None` 就没有 KL，也不是以原始 Qwen 基座为 reference。

本轮 num_iterations=1，生成频率与累积窗口对齐，不用 vLLM；运行 TRL 路径允许以当前 log-prob 的 detach 作为 old log-prob。ratio 数值可为 1，但其对新 log-prob 的导数非零，故不是“ratio=1 就不更新”。clip ratio 为零也不自动证明优化停滞。参考 SFT 与 old policy 的用途必须分别解释。

这些结论来自哈希与运行记录一致的 TRL 源码快照 [C6]。官方文档仅补充算法定义，不替代实际安装版本。[TRL 文档](https://huggingface.co/docs/trl/grpo_trainer)

### 7.6 n 的选择与局限

在独立同分布、二值成功概率为 p 的简化模型下，组内同时出现成败的概率为：

\[
P(\text{mixed})=1-p^n-(1-p)^n.
\]

增大 n 通常提高观察到成败差异的机会，也增加 rollout 成本；但真实样本可能相关，staged reward 也非二值，所以这个公式只能作直觉，不是当前项目收益预测。更多相近轨迹不能修复错误 verifier，增大累积也不增加同组排序范围。

本次 n=4、累积 8 已有 90/100 非零方差组，不能再把“始终没有组内信号”当作唯一解释。n=8、学习率或 beta 哪个更优，目前没有受控消融结论。

## 8. 分层 Reward：实际公式与判定语义

### 8.1 Terminal baseline 的精确定义

当前 terminal 路径不是只检查任意 reward 字段：

\[
R_{terminal}=\mathbf1[\text{用户已停止}]\cdot
\mathbf1[\text{EnvironmentEvaluator reward}=1].
\]

在 Task113 上主要对应 DB 匹配与用户停止，不包含完整退款自然语言正确性或全部授权规则。它是一个有边界的训练目标，不等于“整条轨迹完全正确”。

### 8.2 Task113 Staged v7.1 的六个分项

先计算：

\[
S=0.35E+0.10U+0.08I+0.12T+0.20A+0.15C.
\]

此处 A 表示授权分项，与第 7 节的 advantage 不是同一个量。

| 分项 | 实际检查 | 不能据此声称 |
|---|---|---|
| E：状态正确 | EnvironmentEvaluator reward 是否为 1 | 过程合规、自然语言全部正确 |
| U：交互结束 | 用户模拟器是否输出停止标记 | 用户停止就说明任务完成 |
| I：身份关联 | 成功 find 返回目标用户，之后 get_user_details 的订单集合覆盖所需订单 | 所有查询前都已认证；全部身份策略均满足 |
| T：目标证据 | 四个指定订单查询成功的比例，再乘身份门控 I | 模型真正比较、理解了这些结果 |
| A：写入授权 | 必要写动作全部有唯一后续非错误结果、I=1；授权 PASS 为 1，REVIEW 为 0.5，其他为 0 | REVIEW 是半个“已证明正确”；规则无误报 |
| C：写后沟通 | 必要写动作完整、必要沟通检查完整、最后已验证写结果后有非空 assistant 回复 | 回复的每个事实都正确 |

Task113 没有 `communicate_info` 项，沟通完整条件在这一支默认成立。因此 C 很大程度上检查写后是否有回复，退款事实正确性需要另一个 claim 规则封顶。这是实际覆盖缺口，不能把 C 命名为“完全真实的最终答复”。

必要写动作 ID 为 `114_0 / 114_1`，对应 task ID 113 的两次取消操作；action ID 与 task ID 不同并非必然是错配。读取的是上游 task 中实际字段，不按编号外观修改。

### 8.3 封顶、扣分与执行顺序

S 并不是最终奖励。先按实际条件施加 cap，随后减罚分，处理非预期写入，最后裁剪到 [0,1]。

| 条件 | cap |
|---|---:|
| 没有任何已绑定、非错误的必要写操作 | 0.25 |
| 已出现写操作，但身份门控失败或授权 FAIL | 0 |
| 授权 REVIEW | 0.75 |
| v7.1 claim-evidence 为 FAIL **或 ERROR** | 0.75 |
| claim-evidence 为 REVIEW | 0.90 |
| 终局成功、身份成立，但必要沟通不完整 | 0.90 |
| 存在非预期写入 | 0 |

令 Nerr 为工具错误数、Nrepeat 为重复同名同参数调用次数、Jlimit 为客户/业务工具上限是否触及：

\[
P=\min(0.03N_{err},0.09)+\min(0.02N_{repeat},0.06)+0.10J_{limit}.
\]

对本轮没有启用额外 premature-transfer 规则的 Task113，可概括为：

\[
R=\operatorname{clip}_{[0,1]}\left(\min\{S,\text{各适用前置 cap}\}-P\right),
\]

若有非预期写入，最终再硬封顶 0。不同 cap 取更严格者；不是把全部 cap 相加。输出预算例外失败走独立 gate，不能与这里的客户/工具上限扣分混同。

**一个需要明确披露的实现问题：**严格评测中的 ERROR 表示证据不完整，而训练 scorer 的 claim ERROR 被映射到 0.75 cap。这两条路径并非相同的错误语义。不能说“系统里所有 ERROR 都绝不影响模型奖励”。

### 8.4 为什么它不是逐步过程奖励

分项在轨迹结束时读取已保存证据，汇总成一个 R，再转为一个轨迹 A。没有给每个动作独立计算 reward-to-go，没有步骤价值网络或 learned PRM，也没有在相同前缀下对某一步做因果反事实评估。

因此更准确的名称是：**基于过程证据的分层轨迹奖励**。它增加了可解释性与稠密程度，但没有自动解决多步 credit assignment。

即使某条轨迹包含错误，只要总分高于该组均值，就可能获得正 advantage；所有有效模型 token，包括错误片段，都带同号的轨迹系数。另一方面，模型共享参数、组内正负样本和 KL 共同影响净更新，所以不能只凭一条正 A 就声称“该错误动作的概率一定上升”。

### 8.5 权重与封顶为什么这样设置

设计意图是：让最终状态占主要权重，给身份、证据、授权和沟通部分信用；用硬 cap 避免明显越权被其他得分抵消。实现和意图可以说明，**权重最优性不能说明**：没有完整权重消融，也没有证明这些分项覆盖了所有业务目标。

安全硬 cap 和 KL 也不能互相替代。cap 只约束能被 verifier 检出的行为；KL 限制相对 SFT 的分布漂移，并不理解哪一条业务政策被违反。

## 9. 实验结果：更新发生了，收益尚未证明

### 9.1 最新 Task113 实验的执行证据

本轮是 `staged-v7.1 / n=4 / accumulation=8 / 50 updates`，不是 320-step，也不是 n=8。训练结束状态为 `COMPLETED`，退出码为 0。[E5—E8]

| 核验项 | 结果 | 能证明什么 |
|---|---|---|
| 优化器更新 | 50 次 | 已完成计划训练更新 |
| rollout / group | 400 条 / 100 个独立 n=4 组 | 每次更新消费两组，共 8 条新轨迹 |
| 实际累积门禁 | 50 次全部 PASS | 每次 8 个 microbatch、8 条新轨迹、2 个生成组 |
| 可训练参数 | 504 个张量，33,030,144 个参数 | 更新对象是 LoRA 参数，不是整个基底 |
| 更新证据 | 参数指纹变化，优化证据 PASSED | 排除了“只采样没有更新”，不证明能力提升 |
| reward 组方差 | 90 组非零，10 组为零 | 存在相对排序信号，不保证排序正确 |
| completion 截断比例 | 本次训练日志各步为 0 | 本次没有观察到该类预算截断，不代表没有提前 EOS |
| 耗时 | 17,921.5804 秒，约 4 小时 58 分 42 秒 | 训练墙钟时间；未分解为纯 GPU、API 等待与其他开销 |

本次报告核验了 12 个关键原始文件/运行源码与回收的远端清单，大小及 SHA256 均一致。全部模型文件的最终下载验收不在这一结论内；`retrieval_state.json` 仍是未最终验收的状态，不能宣布完整模型已安全备份。

### 9.2 总量与前后窗口

| 指标 | 全程 | 前 25 次更新，200 条 | 后 25 次更新，200 条 |
|---|---:|---:|---:|
| 原训练 reward 均值 | 0.749475 | 0.769750 | 0.729200 |
| terminal_success | 360/400，90% | 180/200，90% | 180/200，90% |
| DB 成功率 / E 均值 | 92% | 92% | 92% |
| U 均值 | 0.9075 | 0.9050 | 0.9100 |
| I 均值 | 0.9550 | 0.9600 | 0.9500 |
| T 均值 | 0.95125 | 0.9550 | 0.9475 |
| A 授权分项均值 | 0.8200 | 0.8375 | 0.8025 |
| C 均值 | 0.9175 | 0.9200 | 0.9150 |
| 组内标准差均值 | 0.303306 | 0.298898 | 0.307714 |
| 原生 KL 均值 | 0.00038189 | 0.00034206 | 0.00042171 |
| 原生 entropy 均值 | 0.105865 | 0.105422 | 0.106308 |
| 原生 grad_norm 均值 | 0.015416 | 0.013511 | 0.017321 |

前后 reward 相对下降 5.27%，终局成功不变。最大 grad_norm 约 0.07097，最大 KL 约 0.000964；已检查字段均为有限值。未观察到数值爆炸、整体零方差或持续 entropy collapse，**但不能把这些稳定性检查当成“算法实现绝对没有问题”**。

这是同一次在线训练不同时间段的样本，不是两个固定 checkpoint 的独立评测。分布随训练变化，且 400 条共享一个 task/opening；把它们当成 400 个独立业务任务来做泛化推断是不成立的。

### 9.3 哪些行为拿到了正 advantage

按实际 `ddof=1`、epsilon=1e-4 重建 100 个组，共 217 条正 A、143 条负 A、40 条零 A。

| 保存的行为/规则标签 | 正 A | 负 A | 零 A |
|---|---:|---:|---:|
| 终局成功 | 212 | 109 | 39 |
| 终局失败 | 5 | 34 | 1 |
| 授权规则 FAIL | 0 | 37 | 0 |
| 退款声明规则 FAIL | 52 | 89 | 4 |
| 工具错误 | 1 | 2 | 0 |
| 非预期写入 | 0 | 2 | 0 |

各行标签可以重叠。37 条授权 FAIL 全部负 A，与 0 分硬 cap 一致，不能据此估算授权规则的准确率。5 条正 A 的终局失败为原始行 145、156、174、186、314：审计为 DB 正确但缺少用户 STOP，不能称为 5 次错误数据库操作被奖励。

按二值终局标签看，100 组中 66 组全成功、34 组成功失败混合、0 组全失败。它与“10 个零 reward 方差组”不矛盾：终局全成功的组仍可能因授权、声明等分项不同而有分数差。

### 9.4 长度与效率的实际口径

平均每条保存的 completion 序列长度为 3,738.73 token，其中模型生成约 968.83，工具/用户观察约 2,769.90；观察约占 74.1%。平均 prompt 长度约 3,828 token。

这说明不能把 `max_completion_length=8192` 简单理解为“允许模型独白 8192 token”；长工具结果也会占用上下文与生成循环预算。观察虽然不作为策略目标，却仍进入注意力上下文，产生计算和显存开销。模型 token 与观察 token 的 mask 错配会直接污染训练。

全程 363 条为 `USER_STOP_AND_MODEL_EOS`，37 条为 `MODEL_EOS_BEFORE_USER_STOP`。后者说明模型自行结束，不能只因完成长度较短就认为是截断、OOM 或 API 故障。

当前证据不足以给出 rollout 与 model update 的完整耗时占比、真实 API 金额或显存峰值的统一比较。不能拿墙钟时间乘一个未核实的租卡价格当成实际账单。

### 9.5 冻结后评测只完成了一组

SFT 基线已有 greedy 1 条和随机 32 条；GRPO 后评测只有 `n4_01` 完整结束，后续中断。可比较的是同一 `n4_01`，不能拿 SFT 33 条和 GRPO 4 条直接作最终对照。

| 相同冻结 n4_01 | SFT checkpoint-30 | GRPO-50 |
|---|---:|---:|
| terminal_success | 3/4 | 2/4 |
| 平均 reward | 0.4375 | 0.5500 |
| 授权规则 FAIL | 1/4 | 0/4 |
| 退款声明规则 FAIL | 1/4 | 0/4 |

两臂配置的任务、opening、采样、seed 和 reward 相同，模型不同。局部 reward 高了 25.7%，终局成功却少一条；这不是可宣布的提升，也不足以证明显著退化。规则 PASS 增多还可能来自没有执行后续动作，必须读取轨迹。

### 9.6 曲线应怎样解读

![Task113 50 次更新的原生 reward、reward_std、KL 与 loss](D:/PolicyAgent-PostTrain/_local_private_runs/task113_staged_acc8_20260906/final_retrieval_20260907_v1/analysis/training_four_curves.png)

![Task113 reward、终局、分项及 advantage 行为诊断](D:/PolicyAgent-PostTrain/_local_private_runs/task113_staged_acc8_20260906/final_retrieval_20260907_v1/analysis/reward_behavior_dashboard.png)

上述图来自已有真实日志，不补画理想曲线。50 个更新点只能产生一个完整的 50-step rolling mean，不能产生完整的 100-step rolling mean；图中保留这一缺失。loss 的符号和幅度受 advantage、clip、KL 与归一化共同影响，不能按 SFT 交叉熵的直觉要求 GRPO loss 单调下降。

### 9.7 历史 Task44 的 49 组到底是什么

另一个重要证据包是 Task44 的 49 个完整组、196 条轨迹。[E11] 其保存文件另有 2 条未成组记录，未混入 49 组 advantage 分析；复算结果与保存的训练数据最大绝对差约 2.17e-7。

196 条中有 111 条明确终局成功、84 条明确失败、1 条缺少该标签的预算失败。107 条正 A 中有 17 条终局失败，占正 A 的 15.89%；这些失败均记录为用户停止前模型 EOS，但其 DB 正误并不一致。

必须保留两个边界：第一，前 10 组与后续组的预算失败处理政策有变化，不是完全不变配置的单一受控实验；第二，所核验恢复运行有 `completion_token_budget_exhausted`、未闭合工具调用的失败记录，不能把 49 个完整组改称“完整 50 步训练成功”。历史工程中断和最新 Task113 的完整训练是不同事件。

## 10. 具体案例：从原文到奖励，再到学习系数

### 10.1 旧规则误伤正确退款说明：第 31 条

轨迹针对信用卡支付订单，向用户区分支付方式：礼品卡退款立即处理，信用卡退款需 5—7 个工作日。随后对两笔订单分别说明金额及信用卡退款时间，最后说明也保留 5—7 天。[E9—E10]

旧规则把多条 assistant 内容连接起来，用跨文本匹配识别“退款—立即”词汇；它没有可靠处理条件句作用域和当前订单支付方式。因此通用的礼品卡分支被当成对当前信用卡退款的承诺。

结果链：

```text
原始轨迹不变
→ 旧 claim 规则 FAIL → 总分 cap 到 0.75 → advantage 约 -0.4996
→ 候选修订识别条件作用域 → 总分恢复 1.00 → advantage 约 +0.8654
```

这是已保存样本上的**评分反事实**，不是重新训练实验。它证明 verifier 错误能够翻转优化系数，不能单独估计其对最终模型的净影响。

### 10.2 旧规则漏掉明确错误：第 392 条

这条轨迹旧评分为 1、终局成功、claim PASS，但最终说明把“信用卡支付”作为“退款立即到账”的原因。旧模式覆盖了部分 `immediately` 表述，却漏掉本例的 `immediate` 形式；候选 v2 将其判为 FAIL，分数封顶到 0.75。

它是“有错却满分”的具体反例，说明只审 FAIL 样本不够。它**不是**总体假阴性率：要估计漏检，必须审阅 PASS 样本并有可信的真实标签。

### 10.3 规则判错和相对排序是两个问题：第 21 组

原始行 81—84 的评分及标准化如下：

| 原始行 | raw reward | 原退款规则 | terminal_success | standardized advantage |
|---:|---:|---|---|---:|
| 81 | 0.75 | FAIL | 是 | +0.265976 |
| 82 | 1.00 | PASS | 是 | +0.899251 |
| 83 | 0.08 | PASS | 否 | -1.431203 |
| 84 | 0.75 | FAIL | 是 | +0.265976 |

组均值为 0.645。81、84 高于组均值，所以正 A；83 的严重失败降低了组基线。这不要求存在标准化代码错误。即使以后把所有自然语言误判修好，只要仍使用轨迹加权总分、允许违规轨迹高于组均值，就仍可能出现“含错误的相对赢家”。

不能根据表格直接断言所有 FAIL 都是真违规；也不能从单个正 A 推出错误动作的净概率一定上升。表格足以定位需要审计的信号，不替代逐动作因果结论。

### 10.4 后评测早停：不是所有失败都由长度造成

冻结 n4_01 的两臂第一条都只产生问候后 EOS：38 个 completion token，0 个工具调用。GRPO 第二条执行 6 次查询后 EOS，没有取消订单，保存的结束原因是 `MODEL_EOS_BEFORE_USER_STOP`。

其 completion 中模型 token 约 527、观察 token 约 2,124，没有触及 8,192 上限，也没有工具上限或框架异常证据。因此该例归为“交互提前结束”，不是把 token 上限调大就必然解决。具体为何选择 EOS，仍需区分模型协议偏好、上下文与更新影响；本次不编造进一步干预结果。

### 10.5 完整正确案例的验收边界

在已完成的后评测 n4_01 中，GRPO 第 3、4 条均记录终局成功、reward=1。读取第 3 条完整消息，可还原以下链路；消息位置从 0 开始计数，使用的是环境对话视图。

| 位置 | 实际内容 | 核验点 |
|---|---|---|
| 1—5 | 用户要取消全部 pending 订单，不能提供邮箱，改提供姓名和邮编 | 查询身份所需信息来自用户，不是隐藏 gold |
| 6—9 | `find_user_id_by_name_zip` → `get_user_details` | 得到目标用户及其订单集合 |
| 10—17 | 查询四笔订单 | 两笔 pending，一笔 delivered，一笔 cancelled |
| 18—23 | 汇总订单、询问取消原因、复述两笔金额和退款时效；用户明确同意取消 | 操作订单为 `#W5056519 / #W5995614`；原因为 `ordered by mistake` |
| 24—25 | 取消 `#W5056519` | 工具返回 `cancelled`；退款 243.62，原支付方式 `credit_card_3095586` |
| 26—27 | 取消 `#W5995614` | 工具返回 `cancelled`；退款 1,074.31，同一原支付方式 |
| 28—29 | 分别说明取消与退款，并说明 5—7 个工作日；用户 STOP | 有写后反馈与正常用户结束 |

但精读还发现：第 20 条 assistant 提供了“原支付方式还是另一张礼品卡”的退款选择。上游取消政策要求退回原支付方式，这个额外选择不受该政策支持；它更接近退货场景的选项。用户随后坚持原路退款，实际工具也原路退款，所以 DB 和当前 scorer 都成功，仍不能把这条称为毫无问题的 GOLD。

这是本报告核对完整轨迹后得到的**开发级过程问题判断**，没有据此改写原评分或新增正式标签。它进一步说明，只审“退款何时到账”也覆盖不了“退款去哪里”等其他业务声明。可打开 [后评测完整轨迹](D:/PolicyAgent-PostTrain/_local_private_runs/task113_staged_acc8_20260906/final_retrieval_20260907_v1/eval_grpo/results/n4_01/raw_rollouts.jsonl) 和 [上游取消政策](D:/tau2-bench/data/tau2/domains/retail/policy.md:86) 逐项核对。

### 10.6 从抽样审计到离线修复，证据是什么

早期审计针对前 320 条轨迹中的 118 条退款 FAIL：正 A 41 条抽 10 条，负 A 77 条抽 20 条，固定抽样 seed。30 条开发审阅中，20 条确实违规、9 条明确误判、1 条待确认；按分层权重估计明确误判占该 FAIL 集合约 30.2%。

这更接近“被判 FAIL 者中误判所占比例”，不是以所有真实负例为分母的 false positive rate。样本不是独立专家金标，而且只抽 FAIL，不能报告正式 precision/recall/F1，不能把 30.2% 无条件外推到后来全部 145 条 FAIL。

候选 v2 增加局部断言边界、有限条件作用域、申请处理与资金到账区分、即时同义词及支付对象绑定。对相同 400 条轨迹离线重评分：

| 指标 | 原规则 | 候选 v1 | 候选 v2 |
|---|---:|---:|---:|
| PASS | 255 | 298 | 302 |
| FAIL | 145 | 93 | 86 |
| REVIEW | 0 | 9 | 12 |
| reward 均值 | 0.749475 | 0.776975 | 0.779475 |

v2 与旧规则相比，有 14 条 advantage 严格正负翻转；不包含零值进出。终局成功始终为 360/400。v2 下仍有 27 条 FAIL 和 6 条 REVIEW 获得正 A，所以**修复误判没有同时解决相对奖励的信用分配问题**。

已有回归证据为退款模块 56 项通过、相关测试 171 项通过。18 条开发审阅和 8 个合成对抗例符合预期；更早 30 条标签中 29 条一致，第 79 条“processed immediately”保留 REVIEW，因为“立即处理申请”和“资金立即到账”不同。这些是修复和开发回归证据，不是未见分布准确率；本报告未重新运行这套测试。

## 11. 对 GRPO 效果不明的分层归因

### 11.1 哪些解释已有证据，哪些还只是可能性

| 假设 | 当前证据 | 判断 |
|---|---|---|
| 完全没有执行参数更新 | 50 次累积门禁、参数指纹变化 | 已有证据反驳 |
| 所有组都零方差 | 90/100 组非零 | 已有证据反驳；非零不等于好信号 |
| 本轮都是长度截断 | 训练截断记录为 0，后评测早停未触限 | 不能作为本轮主要解释；历史确有预算故障 |
| 模型基本不会 Task113 | SFT 随机终局成功 30/32 | 不支持“基本不会”；但不保证真实声明全部正确 |
| reward verifier 失真 | 条件句误伤、即时词漏检、14 条 A 翻转 | 已证实，属于明确可信度瓶颈 |
| 轨迹总分无法精确归因步骤 | 实际每轨迹一个 A；规则修订后仍有 FAIL 正 A | 已证实存在这种结构，不等于已量化净危害 |
| SFT 过拟合压缩探索空间 | loss 后期轻微回升；使用 checkpoint-30；GRPO entropy 未持续下降 | 有 SFT 后期过拟合趋势，尚不足以证明探索 collapse |
| 学习率过小 / KL 太强 | KL 小，但参数已更新；无单变量对照 | 可能，不能用绝对阈值直接定因 |
| 50 步太少 | 无更多同条件固定评测 | 可能，但不能推出继续训练必然有收益 |
| 泛化不足 | 单任务训练，最终留出未验收 | 泛化收益未证明，而非已经测得普遍泛化失败 |

### 11.2 当前最窄、可证伪的结论

**工程更新已发生；观察到的 reward 变化混有评分规则问题；完整可比后评测不足，因此尚不能确定策略净收益。**

退款候选 v2 对相同轨迹复算后，前后 25 步均值仍从 0.79650 降到 0.76245，下降约 4.27%。因此“全是旧退款规则导致横盘”也是过度归因。多种机制可以同时存在：终局已经较高、声明排序失真、提前 EOS、有限数据覆盖、轨迹级信用分配，以及尚未消融的优化参数。

### 11.3 用户手绘反例的正确解释

多个步骤加权求和确实可能把不同错误组合压缩成相似总分，甚至把含关键错误的轨迹排得更高。但“每个步骤都曾在某条轨迹里做错”不意味着奖励全部为假；关键是**在相同任务/状态下，奖励是否与有价值的动作差异一致，以及这种差异如何作用到策略**。

全轨迹 A 会对一条轨迹内正确和错误 token 使用同号系数，这是粗粒度信用分配。要证明某个错误动作真的被强化，需要进一步看共享上下文下该动作的概率或受控行为变化；本报告没有这类测量，不能把风险写成已证实的训练因果。

### 11.4 技术路线是否偏移

最初目标是提高多步 Tool Agent 的可靠性，而不是收集更多算法名称或把曲线画成上升。工程排错、数据修正、reward 审计都服务于这个目标；但如果长期只追“再多跑几步”，同时缺少稳定测量和完整复评，就会偏离目标。

现在停止追加训练、整理可审计技术证据是合理收束。它不抹去已经完成的 GRPO，也不把尚未获得的正向结果补写出来。本文不自动批准新训练、新模型、新规则上线或补跑评测。

## 12. 复现条件、产物与运行边界

### 12.1 不是仅凭 Git clone 就能复现

复现这次实验至少需要：

1. 本项目的运行载荷快照、文件哈希与对应配置；当前 dirty 工作区不能直接等价替代。
2. 固定 commit 的 tau2 源码和 Retail 数据，环境变量 `POLICYAGENT_TAU2_ROOT` 指向有效 checkout。
3. 起点 SFT 模型及 tokenizer，核对模型目录哈希、chat template 与特殊 token 配置。
4. 私有目录中的已发布数据、manifest、冻结 opening 及对应哈希。
5. 与保存的 `environment.json` 一致、支持该版 Agentic `environment_factory` 的训练环境。
6. 若运行在线 rollout，需要用户模拟器的服务凭据与本轮外发授权；凭据不能写入配置、报告或 Git。
7. 独立输出目录、足够磁盘空间及原始轨迹保存策略；不能覆盖旧实验。

`requirements-agentic-rl.txt` 引用 `requirements-posttrain-smoke.txt`。其中部分包精确锁定，Torch、accelerate 等仍使用范围约束，所以 requirements 不等于完整可复现 lockfile。以运行时版本记录与实际源码哈希补足，而不是盲目升级依赖。

本机可以做 JSONL 审计、规则测试和画图；SFT/GRPO 训练及依赖 GPU 的推理需要相应运行环境。**本报告没有保证本机现有 Python 环境能直接执行 GPU 命令。**

### 12.2 入口命令和历史执行命令要分清

以下仅展示已核验的 CLI 入口，未在本轮执行。需在数据/模型/上游绑定已恢复的项目目录中使用；预检也可能写独立诊断输出，不能据此绕开只读任务约束。

```powershell
# SFT 配置/绑定校验入口，不启动训练
python src/training/run_teacher_sft.py --config configs/retail_teacher_refresh_sft_v4_s100.json --validate-only

# Agentic GRPO 预检入口，不等于已经完成 GPU/API 实测
python src/training/run_retail_agentic_grpo.py --config configs/retail_agentic_qwen3_4b_task113_sft30_staged_v7_1_n4_acc8_s50_v1.json --preflight-only
```

最新 GRPO 的保存命令为：在独立部署目录 `/root/autodl-tmp/policyagent-deployments/20260906-task113-staged-acc8-s50-fixed-v1` 中运行以下命令。[E5]

```bash
/root/autodl-tmp/venvs/policyagent/bin/python \
  src/training/run_retail_agentic_grpo.py \
  --config configs/retail_agentic_qwen3_4b_task113_sft30_staged_v7_1_n4_acc8_s50_v1.json \
  --output-dir training \
  --allow-dirty
```

`--allow-dirty` 是这次隔离载荷的历史选择，不是以后忽略源码绑定的建议。没有验证载荷哈希，不应复用该参数宣称实验可追溯。以上不是新的运行授权。

命令输入配置与 `training/config.json` 是两份 JSON 序列化文件。本轮比较后内容完全相同，但字节哈希分别为 `2EBA9D82…` 和 `08B5ADAB…`。运行 manifest 绑定的是输入文件；不能拿重新序列化副本的哈希与其比较后直接断言配置被改动。

无更新采样使用同一个 runner 的 `--sample-only`；已完成的后评测还记录 `--completion-budget 8192 --groups-per-task 1`。SFT 的 `max_steps` 不用于这个采样分支，不能把评测条数说成“又训练了几步”。

### 12.3 每类产物用于回答什么

| 产物 | 用途 | 不足以单独证明 |
|---|---|---|
| `config.json`、`command.json` | 计划参数、实际入口 | 运行成功 |
| `environment.json`、源码哈希 | 依赖、精度、实现版本 | 驱动/服务端一切位级可重现 |
| `run_manifest.json` | 实际状态、模型/数据绑定、产物索引 | 每条语义标签正确 |
| `raw_rollouts.jsonl` | 逐条 prompt、对话、调用、结果、状态与 reward 分项 | 自动得到独立金标 |
| `log_history.json` | 每次更新的训练指标 | 泛化改善 |
| `optimization_evidence.json` | 参数变化、更新统计 | 行为一定变好 |
| `accumulation_events.jsonl` | 实际累积与新轨迹计数 | n=4 自动变 n=8 |
| 固定 SFT/GRPO 采样结果 | 相同协议下行为比较 | 小样本就有显著结论 |
| 离线 rescore 文件 | 换规则对同一轨迹分数/排序的影响 | 策略真的重新学习了 |

本轮运行关闭外部训练上报，主要依靠本地 JSON/JSONL 与 Matplotlib 静态图。没有据此声称仓库所有历史实验都从未接过其他可视化后端。缺少的指标应标为“未记录”，不从曲线外观推算补齐。

### 12.4 复现与安全的具体限制

- `STARTED / RUNNING` 的旧文件可能因人为停止未更新；当前是否运行必须另查进程，不能仅凭状态文件判断。本轮没有连接云端，不报告实时 GPU 状态。
- 当前回收数据足以支持本文所列局部分析，不等于全部模型文件已验收完成。续训还需要 optimizer、scheduler、RNG 和 checkpoint 绑定，只有 merged 权重不等于原进度可恢复。
- 外部用户模拟器有服务端版本和数值非确定性；temperature=0、seed 和冻结 opening 只能限定协议，不能承诺完全复现同一对话。
- 本地路径包含合成对话、审阅证据和可能较大的模型；发布 GitHub 前应区分可公开代码、可公开配置和需审查的私有产物。本文不执行上传、删除、提交或推送。

## 13. 技术债务

| 问题 | 已观察到的影响 | 风险边界 |
|---|---|---|
| dirty 部署 + 多个源码版本 | 单个 commit 无法描述实际运行；旧规则与候选规则容易混用 | 必须用载荷哈希和 as-run 源码解释历史结果 |
| 实验/证据主要位于私有目录 | GitHub 仓库不能独立提供完整复现材料 | 报告中的本地证据链接不等于公开可下载附件 |
| Reward scorer 集中且分支较多 | task-specific 逻辑、cap 和错误语义难整体审阅 | 有单元测试不代表所有分支可泛化 |
| 英文退款语义规则覆盖有限 | 条件作用域误伤、词形漏检、处理/到账歧义 | v2 是有限规则修订，不是自然语言完备验证器 |
| claim ERROR/REVIEW 被数值映射 | 不确定或损坏证据仍可能影响训练排序 | 不能宣传为全系统 fail-closed |
| 轨迹级加权和 + 同号 advantage | 错误片段与正确片段共享系数；部分 FAIL 正 A | 不是真正逐动作 credit assignment |
| C 等分项的名称比实际检查更强 | “有写后回复”容易被描述成“如实沟通” | 文档和结果必须使用窄口径 |
| 原生协议与 Agentic 协议双表示 | EOS、STOP、tool result、用户回复可能错位 | 需要同时保存两种视图与绑定证据 |
| TRL 对接依赖特定源码行为 | mask、生成缓存、epoch/累积可能随版本变 | 安装版本号仍需补源码哈希和运行门禁 |
| 分组依赖保存顺序 | 每 4 行复算要求没有缺行、混任务、重排 | 显式持久化 group/update/sample ID 更易审计；本次没有据此重写原数据 |
| 运行/下载/评测状态分散 | 训练完成容易被误读为评测与回收全部完成 | 不同阶段应分别验收，不能合并一句“闭环已完成” |
| 单任务训练与小验证集 | 训练成功率高、曲线平稳也不能回答跨任务泛化 | 不能把重复 rollout 数包装成独立任务规模 |

本节列的是基于实际文件发现的债务，不表示本轮已修复，也不自动授权扩展重构。

## 14. 完成度与可对外陈述的边界

| 能力或结果 | 当前证据状态 |
|---|---|
| 上游 Retail 版本固定、开发 Prompt 基线 | 已有证据；20-task Trial-1 为 16/20 业务成功，非正式榜单 [E1] |
| 轨迹审计、修正重放、开发数据发布 | 已实现并用于发布；标签主要为开发/所有者复核 |
| 多轮 SFT、损失监控、checkpoint 选择 | 已完成；最新 100-step 选择 checkpoint-30 |
| Agentic GRPO 真正训练 | 最新 Task113 50 次更新、400 条在线 rollout 已完成 |
| reward 与 advantage 的逐条追溯 | 已完成本报告所选证据包的复算与案例诊断 |
| Reward verifier 修订及回归 | 候选 v2 离线完成；未用于原训练 |
| 最新 GRPO 完整冻结后评测 | 未完成；仅一组 4 条 |
| 可信 GRPO 正向收益 | 未证明 |
| Terminal 优于/劣于 Staged 的最终公平结论 | 未证明；不能横比不同量纲 reward 均值 |
| 独立专家 verifier 指标 | 未建立；开发审阅不替代正式 gold |
| 未见任务泛化或工业生产收益 | 未证明 |
| PPO/DPO/GRPO/GSPO 全套正式 Retail 对照 | 不是本报告已完成成果，不因有计划或其他 smoke 就补写 |

因此可称：**完成了合成电商多步 Agent 的开发级数据—SFT—在线 GRPO—轨迹诊断链路，发现并复现奖励验证器和信用分配的可信度问题。**

不宜称：通过分层奖励显著提升了电商成功率、已证明 GRPO 泛化有效、拥有工业级专家数据、所有环节完全自动闭环。尤其不能把“修正了评分，所以均分升了”写成模型优化收益。

## 15. 面试追问与可证据化回答

以下回答面向技术追问，区分本项目事实和一般原理。仓库可以证明实现和运行，不替代对“哪些代码由本人亲自完成、哪些由 Agent 辅助”的如实说明。

### 15.1 你做的到底是什么，而不是调用了哪些库？

基于 tau2 Retail 的任务、Policy、DB 和工具，实现轨迹审计/修正发布、SFT 表示转换、Agentic 环境接入、分层轨迹奖励、训练完整性门禁及逐条失败诊断。底层 GRPO 来自 TRL，LoRA 来自 PEFT，业务环境来自 tau2；不是自研 benchmark 或优化器。

追问入口：为什么需要两种轨迹视图？查看第 6 节，而不是只列框架名。

### 15.2 数据从哪里来，多少，质量如何？

合成 Retail 任务驱动的教师—用户模拟器交互。最新发布 88 条轨迹，其中训练 66、验证 22，覆盖 55 个 task；不是 88 个训练任务。通过工具重放、结构/哈希/划分门禁和所有者复核，但不是独立专家金标，也不宣称工业数据量。

追问入口：Task44 已进训练；Task113 未进这批 SFT 数据，但用于开发和 RL，不能称最终留出。

### 15.3 为什么一开始 SFT loss 就很低？

起点已经过项目 SFT 和协议桥接；loss 只计算目标 assistant token，并在真实历史上 teacher forcing。低 NLL 可能反映格式、固定话术和已学协议，不能推出自由多步交互已经可靠。

### 15.4 你怎么判断过拟合？为什么选 30 步？

同一 100-step 运行每 10 步在固定 TRAIN66、VALIDATION22 上以 eval-mode 计算 loss。30→100，训练 loss 从 0.219600 降到 0.179740，验证从 0.290027 升到 0.293710，所以按已设定规则选验证最小的 checkpoint-30。趋势轻微、验证集小，不宣称证明了死记硬背或普遍最佳步数。

### 15.5 LoRA、QLoRA、TRL 分别负责什么？

LoRA 是低秩可训练增量；QLoRA 把冻结基底量化以降低存储开销；TRL 是使用这些组件的训练框架。最新 SFT 用 NF4+BF16 的 QLoRA，最新 GRPO 用 BF16 合并 SFT 基底加新 LoRA。不是“用了 LoRA 就没用 TRL”。

### 15.6 为什么普通 SFT 数据不能直接喂给 Agentic GRPO？

原生 tau2 的 assistant 文本就是发给用户；这里要调用 `respond_to_user` 才继续用户交互。缺少表示对齐时，普通自然语言可能让生成循环 EOS。协议桥保留业务事实和工具结果，改变交互表示及终止监督。

### 15.7 一个 optimizer step 到底消费多少数据？

本次单卡 microbatch=1、accumulation=8，每次更新 8 条新轨迹，来自两个独立 n=4 组。50 次更新是 400 条 rollout、100 个组。实际由生成/微步/优化前门禁校验，不只用配置相乘猜测。

### 15.8 n=8 和两个 n=4 梯度累积有什么不同？

前者 8 条共同计算均值和标准差，后者分别在 4 条内标准化，再累积梯度。它们的 advantage 可能完全不同。增大 global batch 能改变梯度估计汇总，不能自动解决组内无差异或错误排序。

### 15.9 reward std 小，是不是梯度就小？

不一定。A 按组标准差归一化，非零小差异也可能产生明显 standardized A。净梯度还取决于 token log-prob 的导数、mask、归一化分母、KL、样本抵消及裁剪。raw reward std 不是梯度范数；平均 A 近零则是组内中心化的正常性质。

### 15.10 KL 在哪里？reference 为什么没有单独一份完整模型？

本次 beta=0.02；固定参考是合并的 SFT 起点，通过禁用新 RL LoRA 计算 reference log-prob。PEFT 路径可以复用同一基底，不等于没有参考约束。reference 与生成样本的 old policy 不是一个概念。

### 15.11 你对哪些 token 做策略梯度？

只对模型生成 token，包括工具名、参数和发给用户的内容。工具返回及用户模拟器消息是观察，用 `tool_mask=0` 排除 loss，但仍作为后续上下文。不能训练模型去“预测并伪造”工具返回。

### 15.12 分层 reward 是逐步奖励吗？

不是。它在完成轨迹后提取 E/U/I/T/A/C，封顶扣分得到一个标量，然后得到一个轨迹 A。本项目没有步骤价值网络或逐动作 reward-to-go，所以称“基于过程证据的分层轨迹奖励”更准确。

### 15.13 为什么失败轨迹能获得正 advantage？

GRPO 比较的是组内相对总分，不是成功标签。第 21 组 `[0.75,1,0.08,0.75]` 中两条 0.75 高于组均值 0.645，因此正 A。需要进一步区分它们是否真实违规、哪部分值得学习、轨迹级系数会波及哪些 token；不能只靠增加 n 掩盖奖励语义问题。

### 15.14 你遇到过什么有证据的事故？

旧退款检查把正确礼品卡条件说明误当成当前信用卡立即退款，同时漏掉部分 `immediate` 表述。第 31 条在轨迹完全不变时从 0.75/负 A 变为 1/正 A；第 392 条反而从满分变成需封顶的明确错误。修订采用局部断言、支付绑定和 REVIEW，并在 400 条历史轨迹上回归。不是换一个更强 LLM 就已解决的问题；最新训练根本没有 LLM reward judge。

### 15.15 怎么保证修 verifier 不是把分数调好看？

保留原规则、原始 reward 和候选版本；对相同轨迹比较评分与排序变化；保留第 79 条标签分歧；单独报告终局成功始终 360/400。候选 v2 均分高约 4% 是测量口径变化，不记成能力收益。修复集已经成为开发回归材料，不用它报告未见准确率。

### 15.16 这次 RL 最终提升多少？

尚无可靠提升结论。训练完成且有参数更新，前后半段训练终局成功均为 90%；完整后评测没完成。唯一可比 n4_01 中 reward 从 0.4375 到 0.55，但终局从 3/4 到 2/4。样本太小且 verifier 存在已知问题，不能选一个好看的数字代表效果。

### 15.17 为什么不直接调大学习率、n 或训练步数？

这些是待验证的优化变量，不是必然解法。当前已存在非零组内信号，核心不再只是能否产生梯度；还需知道奖励排序是否正确、评测能否反映业务目标。尤其 Dr-GRPO 的 8192 归一化分母会影响数值尺度，不能只看 KL 小就断言 beta 锁死了模型。

### 15.18 这个项目的技术价值如何表述而不夸大？

价值是完成真实 Agent 后训练链路，并能用原始轨迹解释协议、数据、奖励、优化与测量各自的失效边界。已经有 SFT 拟合及 GRPO 更新证据，也有可复现的 reward 错误案例；没有证据的业务提升、最优超参数和生产泛化不写。面试时应能打开配置、具体轨迹和公式解释，而不是只背算法名称。

## 附录 A. 证据与源码索引

以下链接指向本机已存在文件。`_local_private_runs` 不是公开下载承诺；迁移报告时需随获准公开的材料制作对应副本，不能随意上传整个目录。

### A.1 数据与运行证据

| 编号 | 可打开的证据 | 支持内容 |
|---|---|---|
| E1 | [Prompt Trial-1 summary](D:/PolicyAgent-PostTrain/experiments/20260722_110504_retail_baseline20_trial1_deepseek/baseline_summary.json) | 20 个开发任务、16 个业务成功、0 系统失败 |
| E2 | [v3 数据 manifest](D:/PolicyAgent-PostTrain/_local_private_runs/merged_sft_v3_wave_a_owner_reviewed/manifest.json)；[v4 数据 manifest](D:/PolicyAgent-PostTrain/_local_private_runs/merged_sft_v4_teacher_refresh_20260905/manifest.json)；[v4 原始发布数据](D:/PolicyAgent-PostTrain/_local_private_runs/merged_sft_v4_teacher_refresh_20260905/release_v1/sft_dataset.jsonl)；[8 条发布报告](D:/PolicyAgent-PostTrain/_local_private_runs/tr0905/release_v2_owner_approved/release_report.json)；[协议桥 manifest](D:/PolicyAgent-PostTrain/_local_private_runs/merged_sft_v4_teacher_refresh_20260905/agentic_protocol_bridge_v1/manifest.json) | 来源、计数、split、发布检查及表示转换 |
| E3 | [SFT100 配置](D:/PolicyAgent-PostTrain/configs/retail_teacher_refresh_sft_v4_s100.json)；[部署 manifest](D:/PolicyAgent-PostTrain/_local_private_runs/tr0905/s100/deployment_manifest.json) | 起点模型、精度、超参数、部署源码绑定 |
| E4 | [SFT 运行 manifest](D:/PolicyAgent-PostTrain/_local_private_runs/tr0905/s100/final_metrics_v1/run_manifest.json)；[checkpoint 选择](D:/PolicyAgent-PostTrain/_local_private_runs/tr0905/s100/final_metrics_v1/checkpoint_selection.json)；[起点 NLL](D:/PolicyAgent-PostTrain/_local_private_runs/tr0905/s100/final_metrics_v1/evaluation_base.json)；[选中模型 NLL](D:/PolicyAgent-PostTrain/_local_private_runs/tr0905/s100/final_metrics_v1/evaluation_sft.json)；[固定集 loss 数据](D:/PolicyAgent-PostTrain/_local_private_runs/tr0905/s100/loss_comparison_v1/loss_data.csv) | SFT 训练、验证、checkpoint 与曲线口径 |
| E5 | [GRPO 命令输入配置](D:/PolicyAgent-PostTrain/_local_private_runs/task113_staged_acc8_20260906/final_retrieval_20260907_v1/configs/retail_agentic_qwen3_4b_task113_sft30_staged_v7_1_n4_acc8_s50_v1.json)；[重保存配置副本](D:/PolicyAgent-PostTrain/_local_private_runs/task113_staged_acc8_20260906/final_retrieval_20260907_v1/training/config.json)；[实际 GRPOConfig](D:/PolicyAgent-PostTrain/_local_private_runs/task113_staged_acc8_20260906/final_retrieval_20260907_v1/training/effective_grpo_config.json)；[运行 manifest](D:/PolicyAgent-PostTrain/_local_private_runs/task113_staged_acc8_20260906/final_retrieval_20260907_v1/training/run_manifest.json)；[实际命令](D:/PolicyAgent-PostTrain/_local_private_runs/task113_staged_acc8_20260906/final_retrieval_20260907_v1/training/command.json)；[环境记录](D:/PolicyAgent-PostTrain/_local_private_runs/task113_staged_acc8_20260906/final_retrieval_20260907_v1/training/environment.json) | 50-step GRPO 实际绑定与环境 |
| E6 | [400 条原始轨迹](D:/PolicyAgent-PostTrain/_local_private_runs/task113_staged_acc8_20260906/final_retrieval_20260907_v1/training/raw_rollouts.jsonl)；[训练日志](D:/PolicyAgent-PostTrain/_local_private_runs/task113_staged_acc8_20260906/final_retrieval_20260907_v1/training/log_history.json)；[逐条 advantage](D:/PolicyAgent-PostTrain/_local_private_runs/task113_staged_acc8_20260906/final_retrieval_20260907_v1/analysis/rollout_advantages.csv)；[逐更新指标](D:/PolicyAgent-PostTrain/_local_private_runs/task113_staged_acc8_20260906/final_retrieval_20260907_v1/analysis/step_metrics.csv)；[已有分析](D:/PolicyAgent-PostTrain/_local_private_runs/task113_staged_acc8_20260906/final_retrieval_20260907_v1/analysis/ANALYSIS.md) | 原始行为、曲线与聚合结果 |
| E7 | [SFT n4_01 原始轨迹](D:/PolicyAgent-PostTrain/_local_private_runs/task113_staged_acc8_20260906/final_retrieval_20260907_v1/baseline/eval_sft/results/n4_01/raw_rollouts.jsonl)；[GRPO n4_01 原始轨迹](D:/PolicyAgent-PostTrain/_local_private_runs/task113_staged_acc8_20260906/final_retrieval_20260907_v1/eval_grpo/results/n4_01/raw_rollouts.jsonl)；[后评测命令](D:/PolicyAgent-PostTrain/_local_private_runs/task113_staged_acc8_20260906/final_retrieval_20260907_v1/eval_grpo/results/n4_01/command.json) | 同一冻结组的局部配对；其余 SFT n4_02—08 在同级结果目录 |
| E8 | [优化证据](D:/PolicyAgent-PostTrain/_local_private_runs/task113_staged_acc8_20260906/final_retrieval_20260907_v1/training/optimization_evidence.json)；[累积事件](D:/PolicyAgent-PostTrain/_local_private_runs/task113_staged_acc8_20260906/final_retrieval_20260907_v1/training/accumulation_events.jsonl) | 真正更新与实际批量验证 |
| E9 | [30 条退款审计](D:/PolicyAgent-PostTrain/_local_private_runs/task113_staged_acc8_20260906/behavior_snapshot_20260907_0134/claim_audit30_v1/AUDIT.md) | 抽样范围、开发标签及原误判机制 |
| E10 | [候选 v2 修复报告](D:/PolicyAgent-PostTrain/_local_private_runs/task113_staged_acc8_20260906/refund_candidate_offline_v2_1/REPORT.md)；[复算 manifest](D:/PolicyAgent-PostTrain/_local_private_runs/task113_staged_acc8_20260906/refund_candidate_offline_v2_1/manifest.json)；[400 条新旧分数](D:/PolicyAgent-PostTrain/_local_private_runs/task113_staged_acc8_20260906/refund_candidate_offline_v2_1/rescore400.csv)；[局部声明证据](D:/PolicyAgent-PostTrain/_local_private_runs/task113_staged_acc8_20260906/refund_candidate_offline_v2_1/diagnostics400.json)；[保存的测试结果](D:/PolicyAgent-PostTrain/_local_private_runs/task113_staged_acc8_20260906/refund_candidate_offline_v2_1/tests.json) | 不改变策略的评分敏感性、回归及保留分歧 |
| E11 | [Task44 的 49 组校验](D:/PolicyAgent-PostTrain/_local_private_runs/task44_budget0_resume10/advantage_49groups_v1/verification_manifest.json)；[196 条逐条数据](D:/PolicyAgent-PostTrain/_local_private_runs/task44_budget0_resume10/advantage_49groups_v1/rollouts_196.csv)；[正 A 失败统计](D:/PolicyAgent-PostTrain/_local_private_runs/task44_budget0_resume10/positive_failure_components_v1/summary.json)；[历史失败 manifest](D:/PolicyAgent-PostTrain/_local_private_runs/task44_budget0_resume10/recovered_results_v1/training/failure_manifest.json) | 历史预算/终止/相对信用问题，不与最新训练混算 |
| H1 | [协议桥接设计及历史记录](D:/PolicyAgent-PostTrain/docs/04_数据治理与后训练/2026-08-24_SFTv3到Agentic-RL协议桥接设计.md) | 协议演进的带日期解释，不替代最新运行证据 |

### A.2 关键源码

| 编号 | 源码入口 | 阅读重点 |
|---|---|---|
| C1 | [重放 evaluator](D:/PolicyAgent-PostTrain/src/evaluation/replay_evaluator.py) | 初始/实际/参考 DB、重放错误与差分 |
| C2 | [严格任务评测](D:/PolicyAgent-PostTrain/src/evaluation/strict_task_evaluator.py)；[Retail 原子谓词](D:/PolicyAgent-PostTrain/src/evaluation/retail_predicates.py) | strict_pass、ERROR/REVIEW 与 claim 哈希绑定 |
| C3 | [执行前 Guard](D:/PolicyAgent-PostTrain/src/guards/retail_pre_action.py)；[运行时 RL 环境快照](D:/PolicyAgent-PostTrain/_local_private_runs/task113_staged_acc8_20260906/final_retrieval_20260907_v1/src/rl/retail_agentic_env.py) | Guard 的执行路径与 RL 诊断路径区别 |
| C4 | [训练时 staged scorer 快照](D:/PolicyAgent-PostTrain/_local_private_runs/task113_staged_acc8_20260906/final_retrieval_20260907_v1/src/evaluation/staged_reward_shadow.py) | 六分项、caps、penalties；不能用最新候选覆盖解释 |
| C5 | [训练时 GRPO runner 快照](D:/PolicyAgent-PostTrain/_local_private_runs/task113_staged_acc8_20260906/final_retrieval_20260907_v1/src/training/run_retail_agentic_grpo.py)；[累积契约快照](D:/PolicyAgent-PostTrain/_local_private_runs/task113_staged_acc8_20260906/final_retrieval_20260907_v1/src/training/accumulation_contract.py) | reference、预检、采样、优化及产物 |
| C6 | [运行对应的 TRL GRPOTrainer 源码](D:/PolicyAgent-PostTrain/_local_private_runs/task113_staged_acc8_20260906/behavior_snapshot_20260907_0134/grpo_trainer.py) | tool mask、group normalization、Dr-GRPO 分母和参考模型 |
| C7 | [SFT runner](D:/PolicyAgent-PostTrain/src/training/run_teacher_sft.py)；[数据发布门禁](D:/PolicyAgent-PostTrain/src/training/teacher_sft_release.py)；[所有者审核数据发布](D:/PolicyAgent-PostTrain/src/training/release_owner_reviewed_teacher_batch.py) | assistant mask、固定集验证、数据门禁 |
| C8 | [本地候选退款规则](D:/PolicyAgent-PostTrain/src/evaluation/refund_timing.py)；[候选 v2 测试](D:/PolicyAgent-PostTrain/tests/test_refund_timing_v2.py) | 未进入历史训练的规则修订 |
| U1 | [上游环境 evaluator](D:/tau2-bench/src/tau2/evaluator/evaluator_env.py)；[上游评测调度](D:/tau2-bench/src/tau2/evaluator/evaluator.py) | 实际 DB、断言与 evaluation type 行为 |

### A.3 关键哈希

模型哈希沿用对应 manifest 的目录哈希语义；数据/源码文件为 SHA256。它们验证内容绑定，不证明内容的语义正确性。

| 对象 | 记录值 |
|---|---|
| v4 发布原生 SFT JSONL | `ACD854FF2A01A9F49FAC499071FDCF308445F2D0CCA2FA87346C1E473384DBA0` |
| v4 协议桥 JSONL | `691DDB71C9C4A0013841D87BB1A5300DA51EFEF1B1558A40B7B2A51279A2E96F` |
| refresh SFT 起点模型 | `0A2E06C9BCA6082F3FE6723EC54A46D4CE1D37A8C6DD6B9BC116BBA4BAB16576` |
| GRPO 起点 selected SFT 模型 | `2215F09DF8A8FF03B1CED725E7F87F7F953B4D0B91F5DDE1618628C1B036963C` |
| GRPO 最终模型（manifest 记录，非全部本地备份验收） | `AB54FD38D455089F831DC2A581A53571CE739DF42AA87F6B8A29221E4724A449` |
| GRPO 命令输入配置 | `2EBA9D821E3EB645540235F069ABDC16D8293FA6A09372FD019DB654A1B5A98F` |
| GRPO 重保存配置副本（JSON 内容相同） | `08B5ADAB7E5D812D9B1953988EDFEC904F587128850E5181EECA000F58D04967` |
| 400 条 raw rollout 文件 | `73FA985839D0DA503D743BBEC448799BE799594650A3D6109CA08B0675FD92D6` |
| GRPO run manifest | `33CAB0E04826A06DE8AB8DF4B8BFDC59BAD8DC0E4656CC0DEA099968EB98C1EC` |
| 对应 TRL 源码快照 | `655D9AD98549290FD32381174BFCA5CAB9F849B839966EDC9A4D3ABE67DD97BC` |

## 附录 B. 报告核验与未覆盖事项

本报告读取了根工作协议、易错提醒、迁移/上游基线、执行边界、核心数据/训练/评分源码、相关配置、所选数据发布报告、原始训练/评测轨迹和离线审计产物。重新计数最新数据 split、Task113 的组/终局统计及局部冻结采样；重点核对运行源码和关键输入的哈希。

没有逐行审计所有历史脚本、所有模型权重张量或全部长期实验目录；没有新建独立专家标签，没有重跑 GPU/API 实验，没有补齐冻结后评测。历史测试通过只按已有记录引用，不冒充本轮重新执行。

交付前复核：60 个本地证据/源码/图表链接均可定位；已逐图检查三幅嵌入图。重新从 400 条 raw 记录计算的 reward 均值、终局计数、advantage 符号及成败组数与本文一致；输入配置与重保存副本的 JSON 内容一致、哈希分别标注。原有 13 个已跟踪修改文件的增删行统计保持不变。

交付验收口径：

- 新人能定位模型起点、数据、表示转换、训练公式、原始运行命令及关键产物。
- 已披露本地/私有数据、模型备份、依赖锁定和外部服务授权等复现缺口，不承诺仅靠公开仓库即可一键重跑。
- SFT、采样、GRPO 更新、后评测、规则复算分别报告，不混成同一类效果。
- 真实成功、规则标签、相对 advantage、参数更新与业务收益分开，不把缺失证据补成结论。
- 本轮只新增本报告，不修改原始证据、训练/评测代码或历史配置，不执行 Git 提交和远端操作。
