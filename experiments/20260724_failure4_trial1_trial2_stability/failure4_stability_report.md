# Retail Failure-4 Trial-1 × Trial-2 Stability Report

Generated at: `2026-07-24T11:47:31.705543`

## Scope

This report compares the four tasks that received reward 0 in the frozen 20-task Retail train-split development Trial-1.

Trial-2 is a second sample of this failure-selected subset. It is not an independent 20-task baseline and is not an official tau2-bench leaderboard result.

## Raw result comparison

| Task | Trial-1 | Trial-2 | Transition | Existing audit label |
|---:|---:|---:|---|---|
| 59 | 0.0 | 0.0 | REWARD_STABLE_0 | BENCHMARK_ALIGNMENT_FAILURE |
| 98 | 0.0 | 0.0 | REWARD_STABLE_0 | MIXED_BADCASE |
| 95 | 0.0 | 1.0 | FAIL_TO_SUCCESS | VALID_AGENT_FAILURE_IN_TRIAL1 |
| 107 | 0.0 | 1.0 | FAIL_TO_SUCCESS | POLICY_GROUNDING_FAILURE_IN_TRIAL1 |

## Aggregate observations

- Repeated raw reward 0: `2/4` tasks.
- Trial-1 failure → Trial-2 success: `2/4` tasks.
- Successful samples across the eight task-runs: `2/8 = 25.0%`.
- Tasks with at least one success across two samples: `2/4 = 50.0%`.

These numbers describe only the failure-selected subset. They must not be reported as the model's general Retail success rate.

## Interpretation by task

### Task 59

- Transition: `REWARD_STABLE_0`
- Audit label: `BENCHMARK_ALIGNMENT_FAILURE`
- Trial-1 trajectory tier: `EXCLUDED`
- Interpretation: Repeated raw reward failure, but the existing audit found a dynamic user-intent versus static-gold mismatch. This must not be treated as a clean negative training example.

### Task 98

- Transition: `REWARD_STABLE_0`
- Audit label: `MIXED_BADCASE`
- Trial-1 trajectory tier: `MIXED`
- Interpretation: Repeated raw reward failure. Existing audit found both benchmark alignment issues and genuine trajectory defects, including action scope and final-claim consistency problems.

### Task 95

- Transition: `FAIL_TO_SUCCESS`
- Audit label: `VALID_AGENT_FAILURE_IN_TRIAL1`
- Trial-1 trajectory tier: `VALID_NEGATIVE`
- Interpretation: Trial-1 failed but Trial-2 succeeded. The task is sampling-sensitive: the Trial-1 failure remains a useful audited negative trajectory, while the Trial-2 success requires its own quality audit before it can be used as positive training data.

### Task 107

- Transition: `FAIL_TO_SUCCESS`
- Audit label: `POLICY_GROUNDING_FAILURE_IN_TRIAL1`
- Trial-1 trajectory tier: `VALID_NEGATIVE`
- Interpretation: Trial-1 failed but Trial-2 succeeded. The observed policy-grounding failure is not deterministic. The Trial-2 success must still pass policy, authorization, state, and final-claim verification.

## Main conclusions

1. Task 59 and Task 98 reproduced reward 0, but neither can be called a clean, stable agent failure from raw reward alone. Task 59 is benchmark-alignment-sensitive; Task 98 is a mixed badcase.

2. Task 95 and Task 107 changed from reward 0 to reward 1. Their Trial-1 failures are therefore not deterministic.

3. The successful Trial-2 trajectories for Task 95 and Task 107 are not automatically SFT Gold. They still require trajectory-quality verification.

4. The result supports the project route: Raw Reward → Human/Verifier Audit → Trajectory Quality → Training Eligibility.

## Next verifier focus

- Task 59: Latest Explicit Authorized Intent and benchmark alignment.
- Task 98: Authorization Scope and Claim–Action–State Consistency.
- Task 95: Multi-goal Completeness and premature escalation.
- Task 107: Policy Grounding and Policy–Tool Enforcement gap.
