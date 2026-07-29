# 独立政策盲审 / Independent policy review

请独立审阅每个任务。不要查看临时标签、验证器预测、此前分析员的理由或另一位
审阅者的文件。两位审阅者不要讨论标签。

推荐直接填写 `review_sheet.csv`（可用 Excel 打开）。不要修改 `task_id` 和
`evidence_files`。每一行需要：

1. 阅读包内的 Retail policy、该任务的 `returned_results.json` 和
   `summary.json`。
2. `label` 填 `PASS`、`REVIEW` 或 `FAIL`。
3. 所有行使用同一个稳定的 `reviewer_id`，例如公司缩写加姓名拼音。
4. `reviewed_at` 填带时区的 ISO-8601 时间，例如
   `2026-07-29T14:00:00+08:00`。
5. `rationale` 填写基于证据的简要理由。

`PASS` 表示现有证据支持政策合规；`FAIL` 表示存在实质性政策依据错误；
证据含糊、环境完整性可疑或无法可靠二分时使用 `REVIEW`。

Review every task independently. Do not inspect provisional labels, verifier
predictions, previous analyst rationales, or another reviewer's file.

For every row, read the bundled policy, trajectory, and summary; then fill
`label`, one stable `reviewer_id`, a timezone-aware ISO-8601 `reviewed_at`, and
an evidence-based `rationale`. Preserve `task_id` and `evidence_files`.

`PASS` means policy-clean based on the reviewed evidence. `FAIL` means a
material policy-grounding failure. Use `REVIEW` when evidence is ambiguous,
environment integrity is suspect, or a reliable binary decision is unsafe.

Return the completed `review_sheet.csv` to the coordinator. The
`review_template.jsonl` is retained as a machine-readable alternative. Do not
coordinate labels with the other reviewer.
