# 教师 SFT 基准评估：Base vs SFT 对比与回退分析（30 任务开发级）

日期：2026-08-19
状态：双跑完成且可比；开发级证据，非正式业务门禁

## 0. 一句话结论

Qwen3-4B-Instruct 教师 SFT（`teacher_sft_merged`）在实体隔离的 30 个 Retail 任务上，
1 trial / temperature 0 开发级评估整体成功率 40.0%（12/30）→ 50.0%（15/30），净提升 3 个任务，
全部来自 test_clean（35.3% → 52.9%）；train_candidates 持平（46.2%）。6 升 3 降，其中
2 个回退（66、90）死于同一个字段——`cancel_pending_order` 的 reason 与 gold 精确不符导致
DB 状态不匹配；1 个（67）是 NL 信息错误（"最近订单"归属说错）。不构成业务改进声明。

## 1. 运行绑定

| 项目 | 值 |
| --- | --- |
| 评估 config | `configs/retail_teacher_eval_v1.json`，sha256 `83DC56A1…` |
| 运行 commit | `0462e71d`（fix: supply placeholder api_key for local vLLM） |
| 上游 | tau2 `58e5e1ac…`，source package sha256 `70561F84…` |
| 任务集 | 30 个（13 train_candidates + 17 test_clean），实体隔离 146 组，`must_be_disjoint=true` |
| 协议 | 1 trial / task，temperature 0，seed 20260818，max_steps 120，`ALL_WITH_NL_ASSERTIONS` |
| Base 模型 | `/root/autodl-tmp/models/Qwen3-4B-Instruct-2507`（sha256 `F34D14D6…`） |
| SFT 模型 | `/root/autodl-tmp/teacher_sft_v1/teacher_sft_merged`（sha256 `2A19D74C…`） |
| 服务 | 本地 vLLM :8000（`--enable-auto-tool-choice --tool-call-parser hermes`，max_model_len 20480） |
| 用户/评审 | DeepSeek `deepseek-chat`，temperature 0 |
| 远程产物 | `/root/autodl-tmp/teacher_eval_base_v1`、`/root/autodl-tmp/teacher_eval_sft_v1` |

坏现场 `/root/autodl-tmp/teacher_eval_base_v1_broken_apikey`（缺 API key 的全 0 无效运行）已保留，未参与对比。

## 2. 结果对比

| 指标 | Base | SFT | 变化 |
| --- | --- | --- | --- |
| 整体成功率 | 12/30 = 40.0% | 15/30 = 50.0% | +3 任务 |
| train_candidates（13） | 6 = 46.2% | 6 = 46.2% | 持平（成员有换） |
| test_clean（17） | 6 = 35.3% | 9 = 52.9% | +3 任务 |
| 系统故障 | 0 | 0 | — |

逐任务翻转（6 升 3 降）：

- 升 0→1：59、72（train_candidates）；40、74、79、108（test_clean）
- 降 1→0：66、67（train_candidates）；90（test_clean）
- 其余 21 个任务两遍结果一致

## 3. 回退根因（逐动作比对）

三个任务的 reward_basis 均为 `DB + NL_ASSERTION`。

### Task 66 — 流程回退（DB 不匹配）

- Base：取消前先确认，用户给出 reason "no longer needed"，agent 照用 → 与 gold 一致。
- SFT：**跳过确认**，直接 `cancel_pending_order(#W3361211, reason="ordered by mistake")`，
  自造 reason 与 gold（"no longer needed"）不符 → DB reward 0。
- 附加：SFT 在换货不可行后查了 4 个 `get_product_details`（Luggage Set / Smart Watch /
  Jigsaw Puzzle / Office Chair），偏离 gold 动作序列（gold 期望第二次 `get_order_details(#W3586556)`），
  该动作也判 mismatch。多绕路本身不扣分，但助长了对话偏移。

### Task 90 — 对话分歧 + 评价刚性（DB 不匹配）

- Base：用户主动说 reason 是 "ordered by mistake"，agent 照用 → 匹配。
- SFT：前面多问了"10x 变体的价格差"，对话分流；用户后来改口
  `use "no longer needed" as the reason`，agent 照做 → 与 gold（"ordered by mistake"）不符 → DB reward 0。
- 定性：SFT 在这条上是顺从用户指示，但 tau2 DB check 对 reason 是全等匹配，仍判失败。
  这是评估器的刚性设计，不是 bug；不据此调模型，仅记录。

### Task 67 — 信息准确性（NL 断言失败）

- 两遍 DB 均匹配（`db_match=true`）。
- Base：用户点名查 #W6729841，agent 报出总额 $829.43，断言满足。
- SFT：agent 给出订单摘要但把"最近订单"说成 #W3445693（$919.67），实际最近订单是
  #W6729841（$829.43）；用户信以为真结束对话。NL 断言 `met=false`。

## 4. 结论与局限

1. SFT 相对 Base 有净提升（+3 任务 / +10pp 整体），且提升集中在 test_clean；
   train_candidates 任务数持平，不能证明 SFT 在训练候选任务上更强。
2. 3 个回退中 2 个（66、90）死于 cancel reason 与 gold 精确不符：66 是真实流程回退
   （跳过确认、自造原因），90 是对话分流后用户改口、模型顺从但 gold 刚性。
   剩余 1 个（67）是订单归属信息错误，属真实能力回退。
3. 评估为开发级证据：1 trial / temp 0，user 与 judge 为 DeepSeek，实体隔离只覆盖
   task 定义文本层面（config known_limitations），按 config claims 不打开正式门禁、
   不允许业务改进声明。
4. 归档：`_local_private_runs/teacher_eval_base_v1/`、`_local_private_runs/teacher_eval_sft_v1/`
   （gitignore 不入库），summary 哈希与远程一致（base `5DEBC7A3…`，sft `3C54D762…`）。

## 5. 建议下一步（未执行）

1. 深挖 SFT 在 66/90 的 cancel reason 选择与 gold 的偏差是否来自训练数据中的 reason
   分布（合并池 47 条中 cancel reason 的多样性），决定是否值得在后续数据批次补充
   确认流程与 reason 多样性的轨迹。
2. 67 的订单归属错误检查 SFT 是否在摘要生成上过度自信（多订单场景），可纳入后续审计。
3. 若要更强结论，跑 4 trial 协议或扩大 test_clean 任务数后再定论。
