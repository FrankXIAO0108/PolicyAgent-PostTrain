# Policy Grounding V2.1 Diagnostic Evaluation

## Change

V2.1 fixes two root-cause errors found during targeted provisional review:

1. Modify confirmation checks now evaluate user-visible items, addresses, and
   payment references without requiring an internal order ID to be repeated in
   the final confirmation summary.
2. The shared runtime-safe Guard blocks a second exchange/modify write on the
   same order across assistant turns.

Task 37 is consequently downgraded from `FAIL` to `REVIEW`. Task 72 remains
`FAIL`, now with `policy.one_shot_order_mutation` as the relevant deterministic
finding.

## Diagnostic result

| Verifier | Three-class accuracy | FAIL precision | FAIL recall | FAIL F1 |
|---|---:|---:|---:|---:|
| V2.0 | 0.850 | 1.000 | 1.000 | 1.000 |
| V2.1 | 0.900 | 1.000 | 1.000 | 1.000 |

V2.1 matches 18 of 20 provisional labels. It still over-escalates provisional
REVIEW Task 1 to FAIL and abstains on provisional PASS Task 28.

## Boundary

All 20 labels are analyst-derived or migrated `PROVISIONAL` labels. Zero are
independently adjudicated. This result measures development-set integration
fidelity and does not open the official metric release gate or justify SFT/RL.
