# Programmatic Verifier Gold Validation

- Mode: `provisional_included`
- Evaluated rows: 14
- Adjudicated: 0
- Provisional: 14
- Unreviewed: 6
- Official metric release allowed: `false`
- Gate reason: Human adjudication is incomplete; metrics are diagnostic only.

## FAIL detection

| TP | FP | FN | TN | Precision | Recall | F1 |
|---:|---:|---:|---:|---:|---:|---:|
| 7 | 0 | 1 | 1 | 1.000 | 0.875 | 0.933 |

- False positives: []
- False negatives: ['95']
- REVIEW prediction rate: 0.357

## Three-class confusion matrix

| Gold \ Pred | PASS | REVIEW | FAIL |
|---|---:|---:|---:|
| PASS | 0 | 1 | 0 |
| REVIEW | 0 | 3 | 2 |
| FAIL | 0 | 1 | 7 |

## Interpretation boundary

- `PROVISIONAL` labels are seeded from existing project audits and are not a substitute for independent human adjudication.
- `UNREVIEWED` rows do not contribute to metrics.
- A `REVIEW` prediction is an abstention. For FAIL detection it counts as a miss when the gold label is FAIL.
- These metrics validate policy-grounding rules; they do not replace Tau2 official reward reconstruction.
