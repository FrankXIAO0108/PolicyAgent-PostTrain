# Programmatic Verifier Gold Validation

- Mode: `provisional_included`
- Evaluated rows: 9
- Adjudicated: 0
- Provisional: 9
- Unreviewed: 11
- Official metric release allowed: `false`
- Gate reason: Human adjudication is incomplete; metrics are diagnostic only.

## FAIL detection

| TP | FP | FN | TN | Precision | Recall | F1 |
|---:|---:|---:|---:|---:|---:|---:|
| 3 | 0 | 1 | 1 | 1.000 | 0.750 | 0.857 |

- False positives: []
- False negatives: ['95']
- REVIEW prediction rate: 0.556

## Three-class confusion matrix

| Gold \ Pred | PASS | REVIEW | FAIL |
|---|---:|---:|---:|
| PASS | 0 | 1 | 0 |
| REVIEW | 0 | 3 | 1 |
| FAIL | 0 | 1 | 3 |

## Interpretation boundary

- `PROVISIONAL` labels are seeded from existing project audits and are not a substitute for independent human adjudication.
- `UNREVIEWED` rows do not contribute to metrics.
- A `REVIEW` prediction is an abstention. For FAIL detection it counts as a miss when the gold label is FAIL.
- These metrics validate policy-grounding rules; they do not replace Tau2 official reward reconstruction.
