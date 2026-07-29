# Policy Grounding Targeted Review Queue V1

This offline queue contains only previously `UNREVIEWED` trajectories.
It does not assign labels or promote any row to `ADJUDICATED`.

- Review candidates: 0
- Predicted FAIL priority cases: 0
- Verifier disagreement cases: 0

## Queue

| Priority | Task | V1.2 | V2.0 | Tool calls | Packet |
|---|---:|---|---|---:|---|

## Required reviewer decision

For each packet, inspect the frozen task definition, complete parsed event sequence, raw source path and hashes. Record:

1. `PASS`, `REVIEW`, or `FAIL` for policy grounding.
2. A rationale tied to exact event indices and policy clauses.
3. Whether the trajectory is eligible for SFT, requires correction, or must be quarantined.
4. Reviewer identity and review date outside this generated artifact.

A generated packet is evidence routing, not independent human gold.
