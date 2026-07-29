# Verifier Targeted Audit Batch 3

## Scope and result

This offline analyst audit covers the six remaining Baseline20
`UNREVIEWED` trajectories. All six receive `PROVISIONAL REVIEW`; none is
promoted to PASS, FAIL, or ADJUDICATED.

| Task | Provisional label | Main correction need |
|---|---|---|
| 19 | REVIEW | Multiple reads and message/tool mixing |
| 24 | REVIEW | Over-answers with a third T-Shirt; call-format violations |
| 43 | REVIEW | Does not correct the user's 4GB/64GB misunderstanding |
| 50 | REVIEW | Message/tool mixing during reads and transfer |
| 52 | REVIEW | Call-format violations and subjective endorsement |
| 89 | REVIEW | Product information before authentication; call-format violations |

## Label maturity after this batch

- `ADJUDICATED`: 0
- `PROVISIONAL`: 20
- `UNREVIEWED`: 0

This completes provisional coverage, not human-gold validation. The release
gate remains closed and no training-data pool may treat these labels as
independently adjudicated gold.

## Updated diagnostic metrics

| Verifier | Three-class accuracy | FAIL precision | FAIL recall | FAIL F1 | Candidate FN |
|---|---:|---:|---:|---:|---|
| V1.2 | 0.800 | 1.000 | 0.875 | 0.933 | Task 95 |
| V2.0 | 0.850 | 1.000 | 1.000 | 1.000 | none |

V2.0 still over-escalates provisional REVIEW Tasks 1 and 37 to FAIL and
abstains on the sole provisional PASS, Task 28. The binary FAIL metrics exclude
REVIEW rows and remain diagnostic-only.
