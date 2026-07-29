# Verifier Targeted Audit Batch 2

## Scope

This offline batch reviews the two predicted-FAIL priority cases from the eight
previously unreviewed Baseline20 success trajectories. It makes no model calls
and does not modify frozen raw trajectories.

## Decisions

| Task | V1.2 | V2.0 | Provisional audit | Main reason |
|---|---|---|---|---|
| 37 | FAIL | FAIL | REVIEW | Confirmation finding is not accepted; observed tool state is corrupted and the final claim conflicts with it |
| 72 | FAIL | FAIL | FAIL | Two modify tools are called on the same order, violating the once-per-order policy |

Task 37 is a candidate three-class over-escalation from `REVIEW` to `FAIL`; it
does not enter the binary PASS/FAIL metric because REVIEW gold is excluded from
that scope. Task 72 is a true policy failure, but the current verifier's reported
`PG_ACTION_ARGUMENT_NOT_CONFIRMED` cause is incomplete: the stronger causal
evidence is the repeated modification of the same order.

## Updated diagnostic metrics

On the 14 provisional rows, V1.2 has FAIL precision/recall/F1 of
`1.000/0.875/0.933` and still misses Task 95. V2.0 has
`1.000/1.000/1.000`. In three-class evaluation, both versions over-escalate
Task 37 from provisional `REVIEW` to `FAIL`.

## Label maturity after this batch

- `ADJUDICATED`: 0
- `PROVISIONAL`: 14
- `UNREVIEWED`: 6

All reported metrics remain diagnostic-only. These two analyst audits are not
independent human adjudications.
