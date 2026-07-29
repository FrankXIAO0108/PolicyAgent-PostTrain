# SFT data release protocol v0

Date: 2026-07-28

## Purpose

Build a reproducible SFT dataset only after policy and trajectory quality are
independently adjudicated. The release tool is intentionally fail-closed:
diagnostic reward, provisional policy labels, directory names, and uncorrected
failure trajectories cannot release data.

The dated 2026-07-22 trajectory-quality document listed Tasks 16 and 28 as raw
positive candidates. Later policy audits found explicit process violations, so
that dated list is not a current release authorization. Current adjudicated
artifacts take precedence.

## Required inputs

1. A complete policy annotation file where every row is `ADJUDICATED`.
2. A complete trajectory-quality decision JSONL where every row is
   `ADJUDICATED`.
3. Exact source and correction SHA-256 hashes.
4. Explicit entity groups and a `TRAIN` or `VALIDATION` assignment.

One quality-decision row has this shape:

```json
{
  "task_id": "1",
  "status": "ADJUDICATED",
  "disposition": "HOLDOUT",
  "split": null,
  "source_split": "TRAIN",
  "source_path": "experiments/.../task_1/returned_results.json",
  "source_sha256": "UPPERCASE_SHA256",
  "correction_path": null,
  "correction_sha256": null,
  "correction_validation_path": null,
  "correction_validation_sha256": null,
  "group_ids": ["user_id:yusuf_rossi_9620", "order_id:#W2378156"],
  "rationale": "Independent trajectory-quality decision."
}
```

Allowed dispositions:

- `RAW_POSITIVE`: requires adjudicated policy `PASS` and a structurally
  policy-clean raw trajectory.
- `CORRECTED_POSITIVE`: requires a separately audited corrected trajectory and
  its hash, plus a ready correction-validation report and its hash. This is the
  only route for SILVER, REVIEW, or corrected failure cases.
- `HOLDOUT`: emits no SFT record and must not have a training split.

Official Tau2 `TEST` tasks cannot enter released SFT data.

## Correction format

A corrected trajectory is a JSON object containing:

```json
{
  "system_policy": "Frozen policy text",
  "messages": [
    {"role": "user", "content": "..."},
    {"role": "assistant", "content": "..."}
  ]
}
```

Corrected trajectories require independent evidence review before their
quality decision becomes `ADJUDICATED`. The tool does not manufacture or infer
corrections. See `docs/20260728_corrected_trajectory_protocol.md`.

## Structural and leakage checks

The release gate:

- rejects message turns that combine assistant text and tool calls;
- rejects more than one tool call in an assistant turn;
- assigns loss only to assistant messages;
- verifies every source and correction hash;
- extracts user, order, and product IDs from raw artifacts;
- combines extracted IDs with declared grouping IDs;
- rejects any entity group appearing across TRAIN and VALIDATION;
- refuses to overwrite a non-empty output directory.

## Command

First build the deterministic decision file described in
`docs/20260728_sft_decision_builder_protocol.md`. Then run:

```powershell
cd D:\PolicyAgent-PostTrain
D:\tau2-bench\.venv\Scripts\python.exe -m src.training.sft_release `
  --annotations path\adjudicated_annotations.jsonl `
  --decisions path\adjudicated_quality_decisions.jsonl `
  --output experiments\YYYYMMDD_sft_dataset_v1
```

When blocked, the command writes only `release_report.json` and exits with
status 2. A successful release additionally writes:

- `sft_dataset.jsonl`
- `dataset_manifest.json`

The resulting data still requires a local formatting/tokenization smoke before
any paid or GPU training run.
