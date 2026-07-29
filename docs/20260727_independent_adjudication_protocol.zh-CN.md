# 独立政策金标裁决协议

日期：2026-07-27

## 目的

在不接受分析员标签或 Verifier 输出作为事实的前提下，把 20 条 provisional
政策标签转换为独立裁决 gold。两位审阅者提交决定前，不得看到 Verifier 预测
或对方标签。

## 审阅文件

Reviewer A 和 Reviewer B 分别填写一份便携式审阅包。推荐使用可由 Excel
打开的 `review_sheet.csv`；`review_template.jsonl` 是机器可读的备选格式。

每一行包含：

```json
{
  "task_id": "1",
  "label": "PASS",
  "reviewer_id": "stable-independent-identity",
  "reviewed_at": "2026-07-27T14:00:00+08:00",
  "rationale": "基于轨迹和政策证据的判断。",
  "evidence_files": ["path/to/reviewed/evidence"]
}
```

规则：

- 标签只能是 `PASS`、`REVIEW` 或 `FAIL`；
- 每份提交必须准确覆盖全部 20 个任务，不能重复；
- 整份文件只能使用一个 reviewer identity；
- rationale、带时区的 reviewed_at 和 evidence_files 必填；
- 审阅者只能填写 `label`、`reviewer_id`、`reviewed_at` 和 `rationale`；
- 不得修改预填的 `task_id` 和 `evidence_files`；
- 两位审阅者必须独立工作。

审阅者可以查看冻结的 trajectory、tool result、task 和 policy，但不能查看：

- provisional 标签；
- 之前分析员的理由；
- Verifier 预测；
- 另一位审阅者的决定。

## 生成盲审包

```powershell
D:\tau2-bench\.venv\Scripts\python.exe -m src.verifiers.blind_review_packet `
  --annotations data\verifier_gold\policy_grounding_gold_v0.jsonl `
  --experiment experiments\20260722_110504_retail_baseline20_trial1_deepseek `
  --policy D:\tau2-bench\data\tau2\domains\retail\policy.md `
  --output experiments\YYYYMMDD_blind_review_packet `
  --bundle-evidence
```

## 提交预检

每一份返回文件必须分别预检：

```powershell
D:\tau2-bench\.venv\Scripts\python.exe -m src.verifiers.review_submission `
  --packet experiments\YYYYMMDD_blind_review_packet `
  --submission path\reviewer_a_completed.csv `
  --output experiments\YYYYMMDD_reviewer_a_preflight
```

预检检查：

- 完整且准确的任务覆盖；
- 单一 reviewer identity；
- 标签、理由和带时区时间；
- 证据路径没有被修改；
- 证据路径不能逃逸出盲审包；
- 所有证据文件的 SHA-256 与冻结值一致。

有效提交生成 `normalized_review.jsonl`。无效提交退出码为 2，只生成错误报告，
不会生成可供裁决的标准化文件。

## 冲突解决

Reviewer A 与 Reviewer B 标签不同的任务交给第三位独立审阅者。第三位身份
必须与前两位不同，而且只接收冲突任务。

流程 fail-closed：

- 未解决冲突只生成报告和 `conflicts.jsonl`；
- 任一冲突未解决时不生成 `adjudicated_annotations.jsonl`；
- 两人一致且全部冲突经第三人解决后，才生成完整 adjudicated 文件；
- provisional 源文件永远不会被覆盖。

## 裁决命令

```powershell
D:\tau2-bench\.venv\Scripts\python.exe -m src.verifiers.adjudication `
  --annotations data\verifier_gold\policy_grounding_gold_v0.jsonl `
  --reviewer-a path\reviewer_a.jsonl `
  --reviewer-b path\reviewer_b.jsonl `
  --output experiments\YYYYMMDD_policy_gold_adjudication
```

如有冲突，增加：

```powershell
  --resolver path\resolver.jsonl
```

## 发布边界

结构裁决通过只是 SFT 数据发布的必要条件，不是充分条件。环境污染和 `REVIEW`
轨迹仍需经过轨迹质量门禁。完成裁决也不自动证明应该进行 RL。

当前外部审阅无法完成，因此该门禁保持关闭。项目不会把模型标签或自填文件伪装
成独立人工 gold。

英文证据原文保存在
[20260727_independent_adjudication_protocol.md](20260727_independent_adjudication_protocol.md)。
