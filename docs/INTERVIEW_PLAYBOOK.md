# PolicyAgent-PostTrain 面试讲解手册

## 30 秒版本

我做了一个电商售后 Tool Agent 的可靠性项目，底层使用 tau2-bench Retail
环境。核心发现是：对话看起来合理、工具返回成功，甚至最终 reward 为 1，都
不一定代表 Agent 获得了正确授权、没有扩大操作范围或满足了业务政策。

所以我实现了确定性状态重放和结构化 diff，把失败拆成官方信号、因果根因和
业务影响；又在工具执行前增加了不读取 gold 的 Guard。冻结的 20 个开发任务
中，旧的轨迹 LLM pipeline 漏掉 4 个失败，状态重放与 20 个冻结结果一致，
Guard 离线拦截了 3 个非隔离失败。

我同时实现了 SFT/DPO/RL 的数据门禁。因为没有独立人工金标，我没有虚构训练
提升，而是把监督信号不足明确作为当前边界。

## 2 分钟版本

项目背景是电商售后 Agent 可以查询订单、退货、换货、取消订单和修改支付方式。
这种 Agent 的输出不是普通文本，而是会修改真实业务状态。

最初我发现 trajectory-only LLM Judge 容易被流畅对话迷惑。例如 Agent 最终
说只取消了某个商品，但工具可能已经取消整个多商品订单；或者工具调用成功，
但退款进入了错误支付方式。

我把系统拆成四层：

1. 冻结 Agent 轨迹和官方结果；
2. 重放 Agent 与目标工具操作，比较最终数据库；
3. 将差异归因到 variant、scope、payment、policy 等根因；
4. 把根因映射成错商品、错资金流向、请求未完成和政策风险。

然后我实现了运行前 Guard。它只读取用户请求、已观察状态和政策，在写工具
执行之前输出 ALLOW、确认、重试、阻止或转人工。Gold-only 诊断与运行时逻辑
严格隔离，避免评测信息泄漏。

最后，我没有把 reward 直接当训练标签。项目实现了轨迹质量、修正审批、数据
哈希、split 泄漏和 SFT/RL readiness 门禁。当前人工金标不足，所以训练门保持
关闭。这是一个刻意的工程决策：不可靠 reward 进入 RL 后只会被放大。

## 三个主讲案例

### Task 95：Schema 理解错误

- 现象：Agent 把 `available=true` 理解为只有一件库存。
- 后果：错误判断无法完成两个换货目标并转人工。
- 检测：状态重放发现 missing exchange、wrong status 和 missing mutation。
- Guard：观察到可行动变体后，阻止无必要的 transfer。
- 面试重点：模型不仅会选错参数，也会误解工具 schema 的业务语义。

### Task 98：Scope 与资金流

- 现象：商品级取消请求可能触发整单取消，换货支付方式也写错。
- 后果：越权影响其他商品，并可能导致资金进入错误账户。
- 检测：最终 DB diff 给出 wrong payment；轨迹检查发现 scope mismatch。
- Guard：写入前阻止 item-to-order scope expansion，并要求多写操作串行。
- 面试重点：Tool success 只表示函数执行成功，不表示副作用符合用户授权。

### Task 107：Policy-Tool gap

- 现象：Agent 选择错误变体，并使用相同 old/new item ID 请求换货。
- 后果：业务规则明确禁止，但后端工具仍接受调用。
- Guard：在工具执行前拦截 same-item exchange。
- 面试重点：Policy、Tool enforcement 和 Evaluator coverage 是三件不同的事。

## 高频追问

### 为什么不用 LLM Judge 直接判断？

LLM Judge 适合语义解释，但无法可靠还原数据库副作用。官方结果重建使用
确定性状态重放；LLM 只能作为补充解释，不能覆盖官方信号。

### V7 的 100% 是否说明泛化很好？

不说明。它只表示在冻结的 20 个开发任务上，重放结果与冻结官方结果一致。
真正的泛化需要新的 held-out 任务和不可回看调参的冻结协议。

### Guard 拦截成功是否等于 Agent 修复成功？

不等于。离线结果只证明危险调用能够被拦截。还需要在线 A/B 测量模型收到
结构化反馈后的恢复率、额外延迟、误拦截和最终业务成功率。

### 为什么没有直接做 SFT 或 GRPO？

因为 reward 不等于轨迹质量，且当前政策标签没有独立人工金标。直接训练可能
学习 benchmark 冲突、环境污染和政策违规轨迹。监督信号不可靠时，暂停训练
是比跑出一个无法解释的 checkpoint 更正确的决策。

### 什么条件下选择 DPO 或 GRPO？

- 有可靠优劣轨迹对、但没有稳定在线奖励：考虑 DPO；
- 有低噪声、难投机、可程序计算的奖励：才考虑 GRPO；
- 问题主要是格式、知识和基本工具行为：优先 SFT；
- 问题可以由确定性规则安全阻止：优先 Guard，而不是强行用 RL。

## 简历表达

### 中文

- 基于 tau2-bench Retail 构建 Tool Agent 确定性重放评测，将最终状态差异分解为
  官方信号、因果根因和业务影响；在冻结 20 任务开发集上复现全部官方结果，
  相比轨迹 LLM pipeline 消除 4 个失败漏报。
- 设计不读取 gold 的 Pre-action Guard，覆盖用户授权、操作范围、商品变体和
  一次性写操作；离线拦截 3 个非隔离失败，并将 gold-only 诊断与运行时逻辑
  隔离。
- 实现可追溯的 SFT/RL 数据治理门禁，包括轨迹质量裁决、修正哈希、实体级
  split 泄漏检测和 readiness gate，避免将 noisy reward 直接用作训练信号。

### English

- Built a deterministic replay evaluator for a tau2-bench Retail Tool Agent,
  separating official outcome signals, causal root causes, and business impact;
  reproduced all frozen outcomes on a 20-task development set and eliminated
  four failure false negatives from a trajectory-only LLM pipeline.
- Designed a gold-free runtime pre-action Guard for authorization, scope,
  product-variant, and one-shot mutation risks; intercepted three
  non-quarantined failures offline while isolating gold-only diagnostics from
  deployable logic.
- Implemented fail-closed SFT/RL data governance with trajectory-quality
  adjudication, correction hashes, entity-level split-leakage checks, and
  stage readiness gates to prevent noisy rewards from becoming training labels.
