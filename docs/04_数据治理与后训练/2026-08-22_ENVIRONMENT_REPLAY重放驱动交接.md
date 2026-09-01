# ENVIRONMENT_REPLAY 重放驱动交接文档

日期：2026-08-22
作者：Codex 开发 Agent（交接给下一个继续开发/执行的 Agent 或人类）
范围：修正协议第 2 项（ENVIRONMENT_REPLAY 重放驱动）已完成部分与待办

## 1. 背景与目标

- 项目：`D:PolicyAgent-PostTrain`（tau2 Retail Agent 后训练，转码主项目）。
- 修正协议：`docs/04_数据治理与后训练/2026-07-28_修正轨迹协议.md`。
- 本阶段目标：为 7 条 CORRECTION_REQUIRED 轨迹（36/38/59/76/101/105/107）实现
  脚本化环境重放驱动，生成 corrected messages + replay manifest，供后续
  结构校验、独立复核（双 APPROVE）后进入 SFT 数据池候选。
- 铁律：工具结果必须环境真重放、不可手编；不调外部 API 的校验必须本地完成；
  规格/哈希绑定必须可复核；不迎合、不编造结果。

## 2. 已完成工作（均已提交到 main）

### 2.1 驱动实现 `src/training/run_scripted_replay.py`（1353 行）
- 提交：`f303dab`（含文档一并提交）。
- 功能：
  - `build_steps(spec, frozen)`：把 plan 线性展开为 ReplayStep 序列
    （EMIT_TEXT / EMIT_TOOL_CALL / EXPECT_USER / EXPECT_TOOL_RESULT / EXPECT_BRANCH）。
  - `validate_spec_dir(...)`：manifest schema、7 份规格 LF 哈希、source/policy/
    decisions raw 哈希、seed、工具名、plan 覆盖契约、交替合法性。
  - Agent：`ScriptedPolicyCompliantAgent(HalfDuplexAgent)`，工厂注册名
    `scripted_policy_compliant_agent_v1`；branch 支持 `ask_once_more_then_abort`。
  - `run_one`：每个 task 建 `task_<N>/`，产物含 `corrected_messages.json`、
    `agent_trace.json`、`replay_manifest.json`（spec/source 哈希、replay 绑定、
    result/reward/branch/prefix mismatch/tool result mismatch/protocol checks、
    原轨迹 vs 修正轨迹 DB 哈希对比）、`returned_results.json`。
  - CLI：`--spec-dir`、`--validate-only`、`--output-dir`、`--seed-source`
    （默认 20260818）、`--smoke-task`、`--llm-user`（默认 deepseek/deepseek-chat）、
    `--path-remap OLD=NEW`、`--upstream-commit`、`--allow-dirty`。

### 2.2 修复的 3 个关键 bug
1. **锚点插入丢失**：被 `remove` 的事件直接 `continue`，导致其后锚定的
   `assistant_text`/`tool_call`/`user_reply_expected`/`branch_on_user_reply`
   全部丢失（早期 validate 假性 7/7 通过的根因）。修复：移除事件不产生步骤，
   但锚点插入照常执行。
2. **branch 吸收**：同锚点组内 `user_reply_expected` 紧跟
   `branch_on_user_reply` 时，前者被 branch 吸收（59/76），记 warning。
3. **`is_stop` 缺失**：tau2 基类默认 `is_stop` 返回 False；若 agent 不重写，
   `###STOP###` 不会被识别，任务空转到 `max_steps=120`。已加类方法重写。

### 2.3 101 规格修订（方案 1，用户已确认）
- 原因：锚 26 文本「…let me verify which order that is…」后紧跟锚 28 工具调用，
  tau2 半双工协议不允许连续 assistant 消息，原计划不可执行。
- 修订：锚 26 后新增 `user_reply_expected`（用户模拟器重新生成确认回复）。
- 新哈希已同步至 manifest（见 `3）。

### 2.4 文档修订
- `docs/04_数据治理与后训练/2026-08-21_确认参数绑定修正目标规格.md`：
  - 7 份规格哈希表从陈旧值修正为 manifest 的 LF 规范化哈希；
  - 新增哈希语义说明（LF 规范化、磁盘 CRLF、source/policy/decisions 为 raw）；
  - 新增规格修订记录（101/59/76）与驱动语义补充；
  - 顶部状态行、第 5 节「下一步」同步更新（驱动已完成、远程先 38 冒烟）。

## 3. 关键绑定与哈希（均经实测核验）

- 规格目录：`_local_private_runs/correction_targets_20260821/`（gitignore，不入库）
- manifest LF SHA-256：`9F22B0EBB746EAFC4DC433F3B19915F6AC7ACD208A7D20B7C4DE1BE85639BE91`
- 7 份规格 LF SHA-256（与 manifest 一致）：

| 规格 | LF SHA-256 |
| --- | --- |
| 36 | `C2EAA933A9C77B66BF983F6769D0598DBF130AEDB28C8336335880CA3B97D26B` |
| 38 | `F93AE2AA549FB493EE1DF86868FE3503E1B6F591E9827036E8C09FF5F5350319` |
| 59 | `9271FA94C05F9805CF7CE47C0CAB2AEAF01BB9036D4C051119F71863F3B35E18` |
| 76 | `CBEB0902D1B533F0D21827E5A1D8EF943A89578E0EE137FCC501F1627BED371E` |
| 101 | `AE44941951D3980A2EE624B24FF7584402EAF0B9037B0D2850793AA02461AC48` |
| 105 | `58575D02F620546591C4665616906ECEA62A9BB6C1B96906764FF4C8108772D0` |
| 107 | `76BE56063A57707E35C23AD5FEFB76883E27CDD81DF43B0680E3E4B6DB6A2206` |

- decisions raw：`E87E8C3CE6472703A10ADDBB57E76B7FDE4E57BF9515CCE8CDA739F9FB3F0A5B`
- policy raw：`4313D3FEF8ACF919F555FA17FBCE929CC3ED1CEF2DD8F0D35FF5C8C3364DE176`
- upstream commit：`58e5e1ace69302e6982d27014569c03e0ffccdd2`
- trial seed：`seed_source=20260818` → `randint(0,1000000)=350291`（7 份规格一致）

## 4. 验证状态（2026-08-22 实测）

- `--validate-only`：`status=VALIDATED`，manifest LF 匹配，7/7 规格通过，
  `external_api_called=false`，`derived_trial_seed=350291`。
- 状态机干跑（无 LLM，用冻结用户消息/工具结果驱动 Agent）：
  7/7 全部正确终止；prefix/tool-result 零 mismatch；59/76 branch 匹配且
  `user_reason` 正确填充；101 插入查询与三笔写齐全；`is_stop` 对
  `###STOP###` True、普通消息 False。
- git 状态：驱动与文档均已提交（`f303dab`）；工作区仅剩一次性脚本
  （`_amend_spec.py`、`_dryrun_agent.py`、`_fix_agent.py`、`_fix_build_steps.py`、
  `_fix_doc.py`、`_inspect_doc.py`、`_inspect_spec.py`、`_update_doc.py`，
  删除被沙箱拒绝，可手动删）。

## 5. 下一步（未完成）

1. **远程执行重放**（需 AutoDL GPU + DeepSeek API key）：
   - 先把驱动 bundle + 规格目录 scp 到远程（规格目录不入库）；
   - 先 `--smoke-task 38` 单任务冒烟 → 通过后再全批 7 条；
   - 冒烟与全批输出目录必须不同（驱动拒绝覆盖）。
2. **生成修正产物**：corrected messages + change_log + replay manifest 绑定。
3. **协议流程**：`correction_validation.py` 结构校验 → replay 状态保持校验
   （DB 哈希）→ 双审阅（作者≠复核人，两个 APPROVE）→ 通过后进 SFT 数据池候选。
4. **文档收尾**：确认规格确认文档的状态行在重放完成后更新；
   若 101 修订需补审批记录，按协议补。

## 6. 已知注意事项

- reward 不作为门禁：修正后 DB 与 gold 可以不同（36 换件集、59/76 用户取消
  理由），`reward==0` 属预期，如实记录。
- 每条用户回复必须由冻结 `user_simulator`（temperature 0.0、seed 350291）
  生成，不可手编；前缀 keep 段软对比记录 mismatch。
- 每个 tool_call 后必须紧跟环境生成的真实结果；corrected_messages 里
  tool call/result ID 必须成对匹配（`_corrected_message_checks` 会校验）。
- 本地 Python：`D:	au2-bench.venvScriptspython.exe`，需设
  `PYTHONPATH=D:PolicyAgent-PostTrain;D:	au2-benchsrc`、
  `POLICYAGENT_TAU2_ROOT=D:	au2-bench`。
- 规格哈希是 **LF 规范化**（去 BOM + CRLF→LF），不是 raw；磁盘文件为 CRLF。
  若手工编辑规格/manifest，需保持 CRLF 并重算 LF 哈希，否则 validate 失败。
