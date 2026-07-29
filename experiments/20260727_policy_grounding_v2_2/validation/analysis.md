# Programmatic Verifier Gold Validation

- Mode: `provisional_included`
- Evaluated rows: 20
- Adjudicated: 0
- Provisional: 20
- Unreviewed: 0
- Official metric release allowed: `false`
- Gate reason: Human adjudication is incomplete; metrics are diagnostic only.

## FAIL detection

| TP | FP | FN | TN | Precision | Recall | F1 |
|---:|---:|---:|---:|---:|---:|---:|
| 8 | 0 | 0 | 0 | 1.000 | 1.000 | 1.000 |

- False positives: []
- False negatives: []
- REVIEW prediction rate: 0.600

## Three-class confusion matrix

| Gold \ Pred | PASS | REVIEW | FAIL |
|---|---:|---:|---:|
| PASS | 0 | 0 | 0 |
| REVIEW | 0 | 12 | 0 |
| FAIL | 0 | 0 | 8 |

## Interpretation boundary

- `PROVISIONAL` labels are seeded from existing project audits and are not a substitute for independent human adjudication.
- `UNREVIEWED` rows do not contribute to metrics.
- A `REVIEW` prediction is an abstention. For FAIL detection it counts as a miss when the gold label is FAIL.
- These metrics validate policy-grounding rules; they do not replace Tau2 official reward reconstruction.
