# Policy Grounding V2 Diagnostic Comparison

V2 composes the existing Policy Grounding V1.2 checks with runtime-safe
pre-action Guard rules. It does not load reference actions or gold DB state.

## Result

On 12 provisionally audited Baseline20 trajectories:

| Verifier | FAIL precision | FAIL recall | FAIL F1 | FN |
|---|---:|---:|---:|---|
| V1.2 | 1.000 | 0.857 | 0.923 | Task 95 |
| V2.0 | 1.000 | 1.000 | 1.000 | none |

Task 95 is recovered by
`goal.transfer_with_actionable_variant`: the observed product payload contains
an available variant satisfying the requested attributes, and boolean
availability does not imply a one-unit inventory limit.

## Additional targeted audits

Three reward-success trajectories were promoted from `UNREVIEWED` to
`PROVISIONAL FAIL` after raw-turn and frozen-policy inspection:

- Task 29: two exchange writes in one assistant turn.
- Task 76: two cancellation writes in one assistant turn.
- Task 109: two parallel address writes, followed by a second modify operation
  on the same pending order.

## Boundary

The 1.000 metrics are diagnostic only. There are zero adjudicated labels, eight
trajectories remain unreviewed, and the Guard was developed from these
development failures. No held-out generalization claim is permitted.
