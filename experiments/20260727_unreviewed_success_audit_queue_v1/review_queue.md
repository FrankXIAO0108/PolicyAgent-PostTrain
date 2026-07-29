# Policy Grounding Targeted Review Queue V1

This offline queue contains only previously `UNREVIEWED` trajectories.
It does not assign labels or promote any row to `ADJUDICATED`.

- Review candidates: 8
- Predicted FAIL priority cases: 2
- Verifier disagreement cases: 0

## Queue

| Priority | Task | V1.2 | V2.0 | Tool calls | Packet |
|---|---:|---|---|---:|---|
| P1_PREDICTED_FAIL | 37 | FAIL | FAIL | 11 | [JSON](packets/task_37_review_packet.json) |
| P1_PREDICTED_FAIL | 72 | FAIL | FAIL | 10 | [JSON](packets/task_72_review_packet.json) |
| P2_PREDICTED_REVIEW | 19 | REVIEW | REVIEW | 8 | [JSON](packets/task_19_review_packet.json) |
| P2_PREDICTED_REVIEW | 24 | REVIEW | REVIEW | 7 | [JSON](packets/task_24_review_packet.json) |
| P2_PREDICTED_REVIEW | 43 | REVIEW | REVIEW | 5 | [JSON](packets/task_43_review_packet.json) |
| P2_PREDICTED_REVIEW | 50 | REVIEW | REVIEW | 4 | [JSON](packets/task_50_review_packet.json) |
| P2_PREDICTED_REVIEW | 52 | REVIEW | REVIEW | 9 | [JSON](packets/task_52_review_packet.json) |
| P2_PREDICTED_REVIEW | 89 | REVIEW | REVIEW | 7 | [JSON](packets/task_89_review_packet.json) |

## Required reviewer decision

For each packet, inspect the frozen task definition, complete parsed event sequence, raw source path and hashes. Record:

1. `PASS`, `REVIEW`, or `FAIL` for policy grounding.
2. A rationale tied to exact event indices and policy clauses.
3. Whether the trajectory is eligible for SFT, requires correction, or must be quarantined.
4. Reviewer identity and review date outside this generated artifact.

A generated packet is evidence routing, not independent human gold.
