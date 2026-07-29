# Project route-alignment audit

Date: 2026-07-28

## Conclusion

There is no strategic route deviation:

- no SFT, DPO, RLHF, or GRPO run was started;
- no provisional label was promoted to training gold;
- the official Tau2 test split remains unused;
- the required order of audit, SFT, comparison, then conditional RL remains
  intact.

There is a mild tactical deviation:

- the label-blind policy packet was ready on 2026-07-27;
- independent policy review count remains zero;
- several downstream gates were implemented on 2026-07-28 while the top
  external dependency remained unresolved.

Those downstream gates are useful safeguards, but further DPO/RL or generic
workflow infrastructure would now displace the highest-priority work.

## Evidence

| Route stage | Current evidence | Status |
|---|---|---|
| Prompt baseline | Frozen 20-task Trial-1 | Complete development baseline |
| Failure audit | Taxonomy v2 and analyst audit batches | Complete provisional coverage |
| Programmatic verifier | Policy Grounding V2.2 | Development diagnostic only |
| Independent policy gold | Blind packet exists; 0 adjudicated rows | Blocking |
| Trajectory-quality gold | Gate/tooling exists; no eligible input | Blocking |
| Corrected SFT data | Release tooling exists; 0 released rows | Blocking |
| SFT | No dataset/checkpoint | Not started |
| Base vs SFT | No comparable artifact | Not started |
| DPO/RLHF/GRPO | Readiness gates closed | Not justified |

## Corrective action

1. Stop adding downstream optimization infrastructure.
2. Produce a portable, label-blind reviewer handoff with frozen raw evidence.
3. Obtain Reviewer A and Reviewer B decisions independently.
4. Resolve conflicts with a third reviewer.
5. Run adjudicated-only verifier validation and inspect every FP/FN.

Only after these steps should work resume on corrected targets and SFT data.

## Current external ask

The most useful contribution from a real business practitioner is not a broad
project opinion alone. It is one complete independent review file, produced
without seeing provisional labels, verifier predictions, or the other
reviewer's decisions.
