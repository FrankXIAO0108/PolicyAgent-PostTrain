# Trajectory-quality adjudication protocol v0

Date: 2026-07-28

## Purpose

Policy correctness and training eligibility are separate labels. A trajectory
may be policy `PASS` yet still contain low-quality unsupported communication,
and a policy `FAIL` may become useful only after an independently approved
correction. Policy adjudication must therefore finish before trajectory-quality
review begins.

## Labels

- `RAW_GOLD`: eligible for raw positive SFT. Requires adjudicated policy
  `PASS`.
- `CORRECTION_REQUIRED`: cannot enter training until a corrected target passes
  the corrected-trajectory protocol.
- `HOLDOUT`: excluded because of environment integrity, benchmark conflict,
  unresolved evidence, or other unsafe contamination.
- `SEGMENT_REQUIRED`: mixed trajectory that must be segmented and relabeled.
  The v0 whole-trajectory SFT release does not accept it.

These are eligibility decisions, not evaluator rewards and not verifier
predictions.

## Review sequence

1. Complete independent policy adjudication.
2. Generate the trajectory-quality review template.
3. Reviewer A and Reviewer B work independently.
4. A third independent reviewer resolves only disagreements.
5. `RAW_GOLD` is rejected unless the adjudicated policy label is `PASS`.
6. No output is promoted directly into SFT; correction approval and split/hash
   release gates still apply.

Each quality-review row includes `task_id`, `quality_label`, reviewer identity,
timezone-aware review time, rationale, and evidence files. One file must use
one stable reviewer identity and cover the complete task set exactly once.

## Command

Before reviewers submit files, this produces a blank template only when every
policy annotation is adjudicated:

```powershell
D:\tau2-bench\.venv\Scripts\python.exe -m src.training.quality_adjudication `
  --annotations path\adjudicated_policy_gold.jsonl `
  --experiment experiments\20260722_110504_retail_baseline20_trial1_deepseek `
  --output experiments\YYYYMMDD_quality_review
```

After two independent reviews:

```powershell
  --reviewer-a path\quality_reviewer_a.jsonl `
  --reviewer-b path\quality_reviewer_b.jsonl
```

Add `--resolver` for disagreements. Unresolved conflicts produce no
`adjudicated_quality.jsonl`. Non-empty output directories are never
overwritten.
