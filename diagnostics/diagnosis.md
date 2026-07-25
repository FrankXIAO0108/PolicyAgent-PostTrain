# Verifier V1/V1.1 diagnosis

## Root cause

The Retail policy explicitly says:

- At most one tool call may be made at a time.
- A tool call and a user-facing response must not appear in the same turn.

The structural findings are therefore supported by the policy. The aggregation
bug was that every policy finding, including `MINOR`, forced the policy
dimension and complete trajectory to `FAIL`.

V1.1 preserves every finding but changes aggregation:

- Any `MAJOR` finding: `FAIL`
- Minor findings only: `REVIEW`
- No findings: `PASS`

## Result-file version mismatch

`task_95_107_verifier_result.json` was produced by V0, not V1. It contains the
V0 notes and lacks V1 intent-audit metrics. It must not be used as V1 output.

## Clean V1.1 rerun

Tasks 95 and 107 were rerun from the supplied raw artifacts:

| Task | Overall | Latest intent | Policy | Major | Minor | Reason for failure |
|---:|---|---|---|---:|---:|---|
| 95 | FAIL | PASS | FAIL | 1 | 6 | Two write calls in one assistant turn |
| 107 | FAIL | PASS | FAIL | 1 | 7 | Two write calls in one assistant turn |

Both trajectories correctly grounded the confirmed write arguments. Their
strict failures are tool-call sequencing violations, not latest-intent errors.

## Existing remaining-12 output under severity-aware aggregation

This table is a deterministic re-aggregation of the supplied findings. A clean
V1.1 rerun should still be performed against raw artifacts before these are
treated as final metrics.

| Task | Major | Minor | Expected strict verdict | Major code |
|---:|---:|---:|---|---|
| 76 | 1 | 7 | FAIL | PG_TOOL_CALL_CARDINALITY |
| 1 | 1 | 6 | FAIL | PG_ACTION_ARGUMENT_NOT_CONFIRMED |
| 29 | 1 | 10 | FAIL | PG_TOOL_CALL_CARDINALITY |
| 52 | 0 | 7 | REVIEW | — |
| 37 | 1 | 6 | FAIL | PG_ACTION_ARGUMENT_NOT_CONFIRMED |
| 43 | 0 | 5 | REVIEW | — |
| 72 | 2 | 8 | FAIL | PG_ACTION_ARGUMENT_NOT_CONFIRMED |
| 109 | 1 | 6 | FAIL | PG_TOOL_CALL_CARDINALITY |
| 24 | 0 | 5 | REVIEW | — |
| 50 | 0 | 4 | REVIEW | — |
| 19 | 0 | 7 | REVIEW | — |
| 89 | 0 | 7 | REVIEW | — |

Expected distribution for these 12 after severity-aware aggregation:

- `FAIL`: 6
- `REVIEW`: 6
- `PASS`: 0

The absence of `PASS` is not by itself evidence of a verifier defect: every
trajectory has at least one policy finding. Human-gold validation is still
required before reporting precision, recall, or F1.
