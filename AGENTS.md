# PolicyAgent-PostTrain Working Agreement

## Project identity

This repository is the local `PolicyAgent-PostTrain` project. It contains the
user-owned experiment runners, configs, artifacts, audits, and documentation
built on top of the upstream `sierra-research/tau2-bench` Retail environment.
Do not describe the upstream benchmark as code implemented from scratch in this
repository.

The active objective is to build a reproducible Tool Agent reliability and
post-training project:

```text
Prompt baseline
-> trajectory audit and failure taxonomy
-> programmatic verifier
-> cleaned/corrected SFT data
-> SFT
-> re-evaluation
-> only then decide whether DPO or GRPO is justified
-> frozen evaluation
```

The latest evidence does not justify jumping directly to RL.

## Sources of truth

Use evidence in this order:

1. Current files and Git state.
2. Frozen experiment manifests, configs, raw results, and hashes.
3. Dated project documents.
4. Migrated historical context.
5. Plans and recollections.

Read [docs/MIGRATION_CONTEXT.md](docs/MIGRATION_CONTEXT.md) before making
project-level decisions. For upstream details, read:

- `docs/environment_baseline.md`
- `docs/repo_architecture.md`

The architecture document is a dated 2026-07-21 snapshot. Its statement that
the project had not entered the baseline stage is superseded by the completed
2026-07-22 Trial-1 artifacts.

## Current verified state

- Local project Git branch: `main`.
- Local project commit at migration: `b5ce8006d93efac0678b618b597217901880a764`.
- No Git remote was reported at migration time.
- Upstream runtime commit recorded by manifests:
  `58e5e1ace69302e6982d27014569c03e0ffccdd2`.
- The 5-task engineering smoke completed with 3 successes and 2 failures. It
  is not a representative baseline.
- Frozen 20-task Retail development Prompt Baseline Trial-1 completed with 20
  valid results, 16 business successes, 4 business failures, and no system
  failures. It is not an official leaderboard score.
- Failure Taxonomy v2 and an eight-trajectory quality audit exist.
- SFT, DPO, Outcome-GRPO, Policy-aware GRPO, and final frozen comparison are
  not confirmed as completed.

## Non-negotiable experiment discipline

- Never infer a result from a directory name or historical chat. Read the
  artifact.
- Bind every experiment to project commit, upstream commit, config hashes,
  task IDs, model, temperatures, seed, and exact command.
- Preserve raw outputs before analysis. Do not overwrite frozen artifacts.
- Do not tune prompts, rules, or data against a frozen evaluation result.
- Keep smoke, development baseline, and official/final evaluation scopes
  distinct.
- Treat `reward == 1` as task-level evaluator success, not proof of a
  policy-compliant or training-quality trajectory.
- Treat `reward == 0` as a review trigger, not automatically as a valid
  negative training example.
- A successful tool call does not prove correct authorization, policy
  compliance, state integrity, or truthful final communication.
- High-false-positive verifier rules remain diagnostic until validated against
  human gold with precision, recall, F1, FP, and FN.
- Do not expose or commit API keys. Record only where a secret is configured.
- Do not change agent model, user model, judge, prompt, task set, and runtime
  config simultaneously when the goal is causal diagnosis.

## Required workflow for changes

Before editing:

1. Run `git status` and identify the current branch/commit.
2. Read the nearest relevant config, runner, artifact, and dated decision doc.
3. State which result or behavior the change is intended to affect.

When editing:

1. Make the smallest scoped change.
2. Add or update focused tests where applicable.
3. Run focused tests before broader regression.
4. For experiment code, perform a local dry run or small smoke before any
   costly run.
5. Preserve a manifest and raw output for every new run.

After editing:

1. Report changed files and tests.
2. Separate confirmed facts from hypotheses.
3. Do not claim training or evaluation improvement without comparable frozen
   artifacts.

## Data-quality gates

Use `docs/20260722_trajectory_quality_taxonomy_v1.md` as the current training
data policy. In particular:

- Only audited `GOLD` trajectories enter raw positive SFT.
- `SILVER` trajectories require correction.
- Environment-corrupted or suspect trajectories are held out.
- Valid agent failures require explicit labels and corrected targets.
- Mixed cases must be segmented and relabeled.
- Benchmark-label conflicts must not be naively learned as negatives.

The current eight-trajectory audit must not be generalized to all 20 tasks.
The remaining twelve successful trajectories are still `UNREVIEWED_SUCCESS`
until audited by validated checks and/or humans.

## Priority order

1. Validate Programmatic Verifier v1 against human gold and review FP/FN cases.
2. Complete targeted human review of the twelve verifier-covered but still unreviewed success trajectories.
3. Establish a trustworthy data pool, then construct strictly split and hashed SFT data.
4. Re-evaluate Base vs SFT vs Verifier-assisted variants on a frozen protocol.
5. Proceed to preference optimization or RL only if residual systematic
   failures and a reliable reward signal justify it.

Do not let extra models, UI work, larger task sets, or additional complexity
displace these priorities.

