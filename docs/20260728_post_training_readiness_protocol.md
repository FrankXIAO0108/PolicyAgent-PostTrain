# Post-training readiness protocol v0

Date: 2026-07-28

## Purpose

Make SFT, DPO, and RLHF/GRPO stage transitions evidence-driven. A directory
named `sft`, `dpo`, or `grpo` is not completion evidence. Each stage is opened
only by readable manifests, exact hashes, comparable evaluation, and explicit
release gates.

## SFT start

Requires:

- adjudicated-only policy validation with its official release gate open;
- a released SFT dataset manifest;
- non-empty TRAIN and VALIDATION records;
- a sibling dataset whose SHA-256 matches the manifest.

## SFT evaluation complete

Additionally requires:

- completed SFT run manifest and successful local smoke;
- checkpoint path and verified hash;
- exact binding to the released dataset;
- completed Base-vs-SFT comparison;
- frozen and comparable task/runtime protocol;
- explicit no-post-hoc-tuning boundary;
- comparison bindings to the SFT run, checkpoint, task set, and runtime config.

## DPO ready

Additionally requires:

- residual systematic failures identified by the comparable SFT evaluation;
- a non-empty, fully adjudicated preference dataset;
- no entity-group leakage;
- exact binding of the preference data to the Base-vs-SFT comparison.

DPO is not opened merely because preference optimization is planned.

## RLHF/GRPO ready

Additionally requires:

- explicit evidence from comparable SFT evaluation that RL is justified;
- held-out reward validation with official release gate open;
- frozen reward-spec hash;
- exact binding to adjudicated policy validation;
- precision and recall at least 0.90;
- critical-risk recall at least 0.95;
- no unresolved FP or FN cases.

The default thresholds are initial safety gates, not universal production
standards. Any later threshold change must be frozen before evaluation.

## CLI behavior

```powershell
D:\tau2-bench\.venv\Scripts\python.exe -m src.training.readiness_gate `
  --policy-validation path\metrics.json `
  --sft-dataset-manifest path\dataset_manifest.json `
  --sft-run-manifest path\sft_run_manifest.json `
  --comparison-manifest path\base_vs_sft_manifest.json `
  --preference-manifest path\preference_manifest.json `
  --reward-validation path\reward_validation.json `
  --require-stage SFT_START `
  --output experiments\YYYYMMDD_post_training_readiness
```

Without `--require-stage`, the command is report-only. With it, the command
exits with status 2 only when that selected stage is closed. It never requires
DPO or RL to be ready before SFT can start.
