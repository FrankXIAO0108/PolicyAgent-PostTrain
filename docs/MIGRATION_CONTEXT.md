# Migration Context

> Evidence snapshot created on 2026-07-23 while migrating the old ChatGPT
> project context into the local Codex workflow.

## 1. Scope

This document covers two closely related migration items:

1. `PolicyAgent-PostTrain`, the user-owned project in
   `D:\PolicyAgent-PostTrain`.
2. `sierra-research/tau2-bench`, the upstream Retail Tool Agent environment.

They are not two independent local project directories. This repository stores
the user-owned increment and experiment artifacts; the upstream source and
runtime version are recorded as dependencies and provenance.

## 2. Repository identity at migration

Local project:

```text
Path: D:\PolicyAgent-PostTrain
Branch: main
Commit: b5ce8006d93efac0678b618b597217901880a764
Describe: b5ce800
Remote: none reported by `git remote -v`
Working tree: clean before the migration files were added
```

Recent local commits:

```text
b5ce800 feat: establish retail prompt baseline and trajectory quality audit
08c5ee6 docs: map tau2 repository architecture
b6a2ac9 docs: record environment and upstream baseline
```

Upstream benchmark:

```text
Repository: https://github.com/sierra-research/tau2-bench
Official reference tag: v1.0.0
Official reference commit: 17e07b1da2bbc0cadfddeea36412686e0604127b
Actual runtime commit: 58e5e1ace69302e6982d27014569c03e0ffccdd2
Runtime CLI version recorded by the project: tau-bench v1.0.0
```

The old handoff proposed `v1.0.1`; current local evidence instead records the
full runtime commit above. Use the commit as the authoritative version.

The runtime commit was selected because the official `v1.0.0` checkout
prematurely imported optional Voice dependencies and failed without `scipy`.
Commit `58e5e1a` fixed optional Voice/Knowledge imports. The project recorded
successful `import tau2`, `tau2 intro`, and Retail environment creation on:

```text
Windows
Python 3.12.10
uv 0.11.29
Git 2.53.0.windows.2
```

See `environment_baseline.md` for the detailed upstream decision and
`repo_architecture.md` for the source map.

## 3. Project purpose and boundary

The project studies reliability and post-training for a policy-constrained
Retail Tool Agent. The central distinction is:

```text
correct final outcome
!=
policy-compliant trajectory
```

The intended verifier layers are:

1. Outcome: final DB state and task success.
2. Communication: required information and truthful final claims.
3. Action: tool choice and arguments.
4. Policy: authorization, ordering, permissions, and business constraints.

The upstream benchmark provides tasks, policies, tools, database, environment,
orchestration, evaluator, and Gym interface. This project owns the added
experiment selection, runners, manifests, analyses, audits, verifier work, and
future post-training work.

Current scope is Retail text interaction. Voice, multimodal, Banking
Knowledge/RAG, realtime providers, UI work, and unrelated deployment
complexity are out of scope.

## 4. Verified experiments

### 4.1 Five-task risk-stratified engineering smoke

Artifact:

```text
experiments/20260721_151851_retail_smoke5_deepseek/
```

Manifest:

```text
Upstream commit: 58e5e1ace69302e6982d27014569c03e0ffccdd2
Task IDs: 59, 29, 72, 50, 28
Agent/User/Judge: deepseek/deepseek-chat
Temperature: 0.0
Seed: 300
Max steps: 200
Trials: 1
Concurrency: 1
```

Observed report:

```text
Completed: 5/5
Successes: 3
Failures: 2
Mean reward: 0.6
Total reported agent + user + judge cost: $0.0108840144
```

This was deliberately risk-stratified. The observed `3/5` must never be
reported as the 20-task baseline rate.

### 4.2 Frozen 20-task development Prompt Baseline Trial-1

Artifact:

```text
experiments/20260722_110504_retail_baseline20_trial1_deepseek/
```

Verified summary:

```text
Role: Prompt Base / Trial-1
Scope: frozen 20-task Retail development baseline selected from train split
Completed: 20/20
Valid rewards: 20
System failures: 0
Business successes: 16
Business failures: 4
Success rate / mean reward: 0.8
Total model cost: $0.0447107304
Total simulation duration: 506.2374537 seconds
Failure task IDs: 59, 98, 95, 107
```

This is not an official leaderboard score and must not be presented as one.
The run manifest freezes task replacement and prompt tuning during Trial-1 and
keeps the official test split unused.

## 5. Audited findings

Failure Taxonomy v2 records:

- Task 95: genuine agent failure involving environment schema/variant
  understanding, capability judgment, premature escalation, and multi-goal
  completion.
- Task 107: genuine policy-grounding failure plus a Policy-Tool enforcement
  gap. The agent attempted a same-variant exchange, the tool allowed it, and
  the evaluator did not fully cover the policy violation.
- Task 98: mixed case and high-value verifier example connecting user
  authorization, tool effect scope, tool result, final DB state, and final
  claim.
- Task 59: simulator/static-golden conflict with the final user intent; exclude
  from the ordinary negative pool unless reconciled.

The eight-trajectory quality audit currently assigns:

```text
GOLD: 16, 28
SILVER / correct before positive SFT: 46
SUSPECT / environment integrity hold: 21
EXCLUDED benchmark-label case: 59
MIXED / segment and relabel: 98
NEGATIVE / corrected target candidates: 95, 107
```

Only these eight trajectories were deeply audited. The remaining twelve
`reward == 1` trajectories are `UNREVIEWED_SUCCESS`.

## 6. Current technical decisions

- Upstream and experiment versions are pinned by full commits and config
  hashes, not branding or remembered tags.
- Raw trajectories and evaluator output are preserved before analysis.
- Development, smoke, and frozen/final evaluation scopes remain separate.
- Official outcome evaluation is retained; the Policy Verifier supplements it.
- Tool success is insufficient when authorization, policy, state integrity, or
  final communication is wrong.
- Raw evaluator reward is not a direct SFT label or a complete RL reward.
- Training data requires audit, correction, strict split, data card, and hash.
- Verifier rules require human gold and P/R/F1 plus FP/FN review before they can
  influence reward.
- Local unit tests, dry runs, and small training smoke precede costly cloud
  training.
- RL is conditional future work, not a predetermined next command.

## 7. Current next work

The evidence-supported sequence is:

1. Audit the remaining successful trajectories or cover them with validated
   verifier checks plus targeted human review.
2. Finalize a trustworthy training-data pool and corrected target
   trajectories.
3. Implement Programmatic Verifier v0.1 with structured evidence across
   Outcome, Communication, Action, and Policy.
4. Build human gold and measure precision, recall, F1, FP, and FN.
5. Construct and validate strictly split SFT data, including tool-call and
   response loss masks.
6. Run SFT and compare it against the frozen Base protocol.
7. Decide whether DPO or Outcome/Policy-aware GRPO is justified by residual
   systematic failures and reward reliability.
8. Run final frozen comparisons without post-hoc prompt/rule/data tuning.

## 8. Not completed or not confirmed

As of this snapshot, the repository does not provide confirmed completion
evidence for:

- Programmatic Verifier v0.1 with human-gold P/R/F1.
- Full audited SFT dataset and checkpoint.
- DPO training.
- Outcome-GRPO training.
- Policy-aware GRPO training.
- Multi-trial/pass@k stability results.
- Final frozen Base/SFT/Verifier/RL comparison.
- Release tag and final resume metrics.

Do not convert these planned milestones into completion claims.

## 9. Historical conflicts resolved

- Old handoff: 20-task baseline was not confirmed.
  Current artifact: Trial-1 completed on 2026-07-22 with 20 valid results.
- Old handoff: planned upstream tag `v1.0.1`.
  Current artifact: actual runtime commit is
  `58e5e1ace69302e6982d27014569c03e0ffccdd2`, with `v1.0.0` as the formal
  reference tag.
- Old handoff: smoke values were unknown.
  Current artifacts: task IDs, results, model settings, seed, cost, and hashes
  are recoverable from the smoke manifest and report.
- `repo_architecture.md` says the project had not entered a formal baseline.
  That statement was accurate on 2026-07-21 but is superseded by the
  2026-07-22 baseline artifact.

## 10. Items still requiring human or future verification

- Where the upstream tau2 source checkout currently resides on disk.
- Whether this local project should be connected to a new Git remote; none is
  currently configured.
- The exact commands used for historical runs if they are not recoverable from
  runner arguments or shell history.
- Whether Runner/Simulation changes between `v1.0.0` and `58e5e1a` alter Retail
  execution semantics in a way that matters to the frozen protocol.
- Audit status of the twelve remaining successful Trial-1 trajectories.
- Local GPU, CUDA, proxy, and future cloud-training environment. These are not
  required to preserve the current API baseline evidence.

Secrets must not be copied into this document or any tracked file.

## 2026-07-24 Trial-2 and verifier update

A second sample was recovered for the four Trial-1 reward-0 tasks. No model or evaluator calls were made during recovery.

- Task 59: 0 -> 0
- Task 98: 0 -> 0
- Task 95: 0 -> 1
- Task 107: 0 -> 1
- Subset Trial-2 success rate: 2/4

This is a failure-selected subset, not an independent 20-task baseline or a general Retail success-rate estimate.

Interpretation:

- Task 59 remains benchmark-alignment-sensitive and is not a clean negative.
- Task 98 remains a mixed badcase.
- Trial-1 failures for Tasks 95 and 107 are valid audited negative trajectories, but are sampling-sensitive rather than deterministic.
- Trial-2 successes for Tasks 95 and 107 require trajectory-quality verification before positive-SFT use.
- The evidence supports: Raw Reward -> Human/Verifier Audit -> Trajectory Quality -> Training Eligibility.

Programmatic Verifier v1 has been implemented with configuration and tests. Diagnostic outputs now cover Tasks 95 and 107 plus the remaining twelve success trajectories. These outputs do not replace human gold; the twelve trajectories remain unreviewed until targeted human validation and FP/FN analysis are completed.
