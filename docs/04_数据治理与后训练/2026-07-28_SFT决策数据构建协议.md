# SFT decision builder protocol v0

Date: 2026-07-28

## Purpose

Convert independently adjudicated trajectory-quality labels into the exact
decision schema consumed by the SFT release gate. This conversion is
deterministic; it does not infer quality, create corrections, or choose a split
from model output.

## Mapping

| Adjudicated quality | SFT disposition |
|---|---|
| `RAW_GOLD` | `RAW_POSITIVE` |
| `CORRECTION_REQUIRED` | `CORRECTED_POSITIVE` |
| `HOLDOUT` | `HOLDOUT` |
| `SEGMENT_REQUIRED` | `HOLDOUT` until a segment protocol exists |

`RAW_GOLD` is rechecked against adjudicated policy `PASS`.

## Required bindings

- Complete adjudicated policy annotations.
- Complete adjudicated quality labels with matching policy labels.
- Original source path and SHA-256.
- Exact correction and correction-validation paths/hashes for every
  `CORRECTION_REQUIRED` task.
- Explicit TRAIN/VALIDATION and source-split assignment for every released
  task.
- Automatically extracted user/order/product entity groups.

Correction-registry coverage must exactly equal the correction-required task
set. Split-plan coverage must exactly equal the released task set. Extra as
well as missing rows are rejected.

## Command

```powershell
D:\tau2-bench\.venv\Scripts\python.exe -m src.training.sft_decision_builder `
  --policy-annotations path\adjudicated_policy_gold.jsonl `
  --adjudicated-quality path\adjudicated_quality.jsonl `
  --corrections path\correction_registry.jsonl `
  --split-plan path\split_plan.jsonl `
  --output experiments\YYYYMMDD_sft_decisions
```

The command emits `sft_quality_decisions.jsonl` only when every binding is
complete. These decisions still pass through `src.training.sft_release`, which
revalidates hashes, correction approval, message structure, and group leakage.
