# Corrected trajectory protocol v0

Date: 2026-07-28

## Boundary

A failed, mixed, REVIEW, or SILVER trajectory is not converted into positive
SFT merely by editing a JSON file. A corrected target must preserve frozen
evidence, identify every material change, and receive two independent
approvals. The author cannot approve their own correction.

## Generation modes

### `ASSISTANT_TEXT_EDIT`

Use only when tool calls and observed tool results remain valid and the defect
is confined to assistant communication. All user/tool observations and all
assistant tool calls must remain identical to the source trajectory.

### `ENVIRONMENT_REPLAY`

Use whenever a tool name, argument, ordering, write scope, or resulting state
must change. The correction must bind a separately frozen environment-replay
manifest and hash. Tool results must never be invented by hand.

Mixed trajectories requiring segmentation are not released by this v0
whole-trajectory format. They remain held out until a segment-level schema and
review protocol are approved.

## Correction artifact

Required fields include:

- `task_id`, `author_id`, and timezone-aware `authored_at`;
- `generation_mode`;
- source trajectory path and SHA-256;
- frozen policy path, SHA-256, and exact policy text;
- non-empty `change_log` with category and reason;
- corrected `messages`;
- replay-manifest path and hash for `ENVIRONMENT_REPLAY`.

Corrected messages must have matching tool call/result IDs, no assistant
text/tool mixing, and at most one tool call per assistant turn.

## Independent approval

Each approval JSONL row binds:

- task ID;
- exact correction SHA-256;
- reviewer identity;
- `APPROVE` or `REJECT`;
- timezone-aware review timestamp;
- rationale and evidence files.

Release requires two distinct matching `APPROVE` decisions, no matching
rejection, and reviewer identities different from the author.

## Command

```powershell
cd D:\PolicyAgent-PostTrain
D:\tau2-bench\.venv\Scripts\python.exe -m src.training.correction_validation `
  --correction path\corrected_task.json `
  --approvals path\correction_approvals.jsonl `
  --output experiments\YYYYMMDD_correction_validation_task_N
```

The tool exits with status 2 when blocked and refuses to overwrite non-empty
output directories. Its `correction_validation.json` path and hash must be
bound into the later SFT trajectory-quality decision.
