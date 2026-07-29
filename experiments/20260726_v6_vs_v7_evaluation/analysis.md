# V6 vs V7 evaluation comparison

## Method

- Dataset: frozen 20-task Retail development baseline.
- Gold outcome: recorded Tau2 overall reward.
- V6 prediction: raw `prediction.has_failure` from the v6 JSONL.
- V7 prediction: reward reconstructed from DB replay and frozen NL results.
- Root-cause audit: four known failures; diagnostic set, not held-out.

## Outcome detection

| System | Accuracy | Failure recall | Precision | FP | FN | New LLM calls |
|---|---:|---:|---:|---:|---:|---:|
| V6 | 75.00% | 0.00% | 0.00% | 1 | 4 | 40 |
| V7 | 100.00% | 100.00% | 100.00% | 0 | 0 | 0 |

- V6 false positive tasks: ['24']
- V6 false negative tasks: ['59', '95', '98', '107']
- V7 replay inconsistencies: 0
- V7 measured replay time: 9.948605 seconds

## Root-cause audit

- Exact task-level taxonomy match: 4/4
- Micro precision: 100.00%
- Micro recall: 100.00%
- This is an audited development slice, not an unbiased generalization score.

## Conclusion

V6 misses all four official failures because trajectory-only semantic judgment cannot observe the gold state transition. V7 reconstructs the official outcome deterministically, then diagnoses cause and business impact in separate layers. The 100% V7 outcome score measures replay fidelity on frozen artifacts, not performance on unseen tasks.
