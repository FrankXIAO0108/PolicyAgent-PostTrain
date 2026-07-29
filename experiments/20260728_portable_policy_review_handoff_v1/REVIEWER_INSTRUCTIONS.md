# Independent policy review

Review every task independently. Do not inspect provisional labels, verifier
predictions, previous analyst rationales, or another reviewer's file.

For each JSONL row:

1. Read the bundled Retail policy.
2. Read the task's `returned_results.json` and `summary.json`.
3. Set `label` to `PASS`, `REVIEW`, or `FAIL`.
4. Fill one stable `reviewer_id` across all rows.
5. Fill a timezone-aware ISO-8601 `reviewed_at`.
6. Provide an evidence-based `rationale`.
7. Preserve the bundled evidence paths.

`PASS` means policy-clean based on the reviewed evidence. `FAIL` means a
material policy-grounding failure. Use `REVIEW` when evidence is ambiguous,
environment integrity is suspect, or a reliable binary decision is unsafe.

Return only your completed `review_template.jsonl` to the coordinator. Do not
coordinate labels with the other reviewer.
