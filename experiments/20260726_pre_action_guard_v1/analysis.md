# Deterministic pre-action guard V1 audit

## Scope

- Input: frozen 20-task Retail development trajectories.
- Runtime-safe guard: policy/state/user-scope checks without gold actions.
- Reference diagnostic: optional benchmark-only comparison to frozen gold.
- No LLM calls and no trajectory regeneration.

## Summary

- Tasks audited: 20
- Runtime guard blocking coverage on official failures: 3/4
- Runtime guard blocked successful trajectories: 4/16
- Reference diagnostic coverage on non-quarantined failures: 3/3

## Four failure cases

| Task | Runtime blocking categories | Reference-only categories | Interpretation |
|---|---|---|---|
| 59 | none | reference_mismatch | Gold/user conflict remains evaluator quarantine; runtime policy findings are secondary. |
| 95 | variant_error | none | Blocks premature transfer because boolean availability can satisfy both exchanges. |
| 98 | scope_error, policy_error | payment_error | Blocks item-scoped whole-order cancellation and serializes multiple writes; reference mode exposes payment mismatch. |
| 107 | policy_error | variant_error | Blocks same-item exchange; reference mode also exposes wrong replacement variant. |

## Interpretation boundary

A runtime guard cannot legitimately know Tau2 gold actions. Reference comparison is therefore isolated as a benchmark/training diagnostic and must not be described as a deployable safety rule. Blocking successful trajectories is not automatically a false positive: Tau2 reward does not score every policy requirement. Those cases require a live A/B rerun to measure utility impact.
