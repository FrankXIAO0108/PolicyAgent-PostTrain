# Trajectory Quality Taxonomy v1

## 1. Scope

Frozen experiment:

- Retail Prompt Base / Trial-1
- Raw Baseline: 16/20 = 80%
- Detailed Human Audit: 8/20
- Audited Reward=0: 4/4
- Audited Reward=1: 4/16
- Remaining Reward=1 trajectories unreviewed: 12

> Important: the 12 unaudited Reward=1 trajectories are UNREVIEWED,
> not automatically Gold.

---

## 2. Audited trajectory taxonomy

| Task | Raw Reward | Action | Bucket | Primary Label | Raw Positive SFT |
|---|---:|---|---|---|---|
| 16 | 1.0 | 8/9 | GOLD | OUTCOME_CORRECT_PROCESS_CORRECT | YES |
| 28 | 1.0 | 10/11 | GOLD | OUTCOME_CORRECT_PROCESS_CORRECT | YES |
| 46 | 1.0 | 4/7 | SILVER | OUTCOME_CORRECT_PROCESS_IMPERFECT | NO |
| 21 | 1.0 | 11/12 | SUSPECT | SOURCE_VERIFIED_TOOL_STATE_CORRUPTION | NO |
| 59 | 0.0 | 3/5 | EXCLUDED | USER_SIMULATOR_GOLD_MISMATCH | NO_RAW_BENCHMARK_LABEL |
| 98 | 0.0 | 1/3 | MIXED | CLAIM_ACTION_INCONSISTENCY | NO |
| 95 | 0.0 | 0/2 | NEGATIVE | ENVIRONMENT_STATE_SEMANTICS_MISUNDERSTANDING | NO |
| 107 | 0.0 | 1/2 | NEGATIVE | POLICY_GROUNDING_FAILURE | NO |

### Bucket counts among audited 8

- EXCLUDED: 1
- GOLD: 2
- MIXED: 1
- NEGATIVE: 2
- SILVER: 1
- SUSPECT: 1

---

## 3. Key audited examples

### GOLD — Task 16

Multi-goal completion, correct entity disambiguation,
safe destructive-action confirmation, and claim-state consistency.

The single action mismatch is only a missing explicit calculator path.

### GOLD — Task 28

Strong positive example for action-scope safety.

The user only authorizes cancelling one item, while the available
cancellation action would cancel the whole pending order.

The Agent correctly refuses to expand the destructive scope.

### SILVER — Task 46

Outcome and critical tool writes are correct, but the trajectory
contains entity-disambiguation weakness and user-context grounding loss.

Correct before positive imitation training.

### SUSPECT — Task 21

The Agent's tool arguments are correct, but the upstream
modify_pending_order_items implementation corrupts multi-item state.

The Agent then fails to verify the returned state before claiming success.

This is both:

- a source-verified Tool / Environment bug;
- a useful Post-Tool State Verification badcase.

Do not use the raw trajectory as Gold.

### EXCLUDED — Task 59

User Simulator final intent conflicts with Static Golden.

Do not convert Reward=0 directly into a negative training label.

### MIXED — Task 98

Contains both benchmark-alignment noise and real Agent scope/claim failures.

Must be segmented and relabeled.

### NEGATIVE — Task 95

Valid Agent failure:

Variant identity is incorrectly interpreted as physical inventory count,
causing false capability-boundary detection and premature escalation.

### NEGATIVE — Task 107

Valid Policy Grounding failure plus Tool Enforcement Gap.

---

## 4. SFT Data Eligibility Rules v1

### R01 — Raw Reward is not a training label.

Reward=0 may be benchmark/evaluator noise, and Reward=1 may contain hidden process, policy, or state-integrity defects.

### R02 — Action Match Rate is not a trajectory-quality score.

Equivalent reasoning paths can fail action matching, while high action match can coexist with hidden state corruption.

### R03 — Only human/verifier-audited GOLD trajectories enter raw SFT Gold.

Unaudited Reward=1 trajectories remain UNREVIEWED.

### R04 — SILVER trajectories require correction before positive SFT.

Outcome may be correct while process defects would be copied by imitation training.

### R05 — SUSPECT or environment-bug-contaminated trajectories are held out.

Do not train on corrupted observations/states before adjudication or environment repair.

### R06 — Valid Agent failures may be used only with explicit failure labels and corrected target trajectories.

Negative trajectory alone is insufficient for standard SFT.

### R07 — Mixed badcases must be segmented and relabeled.

A single trajectory can contain both correct behavior and benchmark noise or real failures.

### R08 — Benchmark-alignment cases are excluded from raw negative pools.

Training against an incorrect/static label can teach the Agent to ignore final explicit user intent.

### R09 — User Authorized Scope must cover Tool Effect Scope.

A technically executable write is unsafe when its actual effect exceeds what the user explicitly authorized.

### R10 — Tool success is not sufficient evidence of business success.

Tool output must be checked against policy, expected state, and user intent.

### R11 — Post-tool state must be verified before final success claims.

Task 21 demonstrates that a tool can return a corrupted state while Reward still equals 1.

### R12 — Final Claim must match Tool Result and Final State.

False or stale success claims are production-critical even when a tool call itself succeeded.

### R13 — Latest explicit user intent must be tracked subject to Policy and Tool constraints.

Dynamic conversations can legitimately diverge from a static pre-generated action path.

### R14 — Equivalent correct computation paths should not be penalized as failures.

Missing a specific calculator call is not a correctness failure when the verified result is mathematically correct.

---

## 5. Gold Gate v1

A trajectory may enter raw SFT Gold only if all required gates pass:

- Outcome correct
- Critical write arguments correct
- Policy compliant
- User authorized scope respected
- No unresolved environment/tool corruption
- Post-tool state consistent with intended state
- Final claim consistent with tool result/final state
- No high-risk hidden process failure

The following are NOT required for Gold:

- 100% exact Static Action Match
- Identical reasoning path to one Golden trajectory
- Mandatory calculator call when equivalent verified arithmetic is correct

---

## 6. Current training-pool decision

### Direct raw positive SFT

- Task 16
- Task 28

### Correct before positive SFT

- Task 46

### Hold / exclude raw because of environment integrity

- Task 21

### Valid corrected-negative / preference candidates

- Task 95
- Task 107

### Segment and relabel

- Task 98

### Exclude raw benchmark label

- Task 59

---

## 7. Critical conclusion

The current audited evidence rejects both naive rules:

```text
Reward=1 -> Gold
Reward=0 -> Negative
```

and:

```text
higher Action Match -> higher trajectory quality
```

The actual training-quality gate must evaluate:

```text
Outcome
  -> Policy Compliance
  -> User Authorized Scope
  -> Tool Arguments
  -> Tool Result / State Integrity
  -> Claim-State Consistency
  -> Process Quality
```

---

## 8. Boundary of this version

This taxonomy is based on only 8 deeply audited trajectories.

It must not be generalized to all 20 tasks yet.

The remaining 12 Reward=1 trajectories remain:

`UNREVIEWED_SUCCESS`

until automated verifier checks and/or targeted human audit are applied.
