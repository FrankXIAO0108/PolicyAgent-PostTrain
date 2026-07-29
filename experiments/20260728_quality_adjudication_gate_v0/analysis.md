# Trajectory-quality adjudication gate v0

Current result: `BLOCKED_POLICY_ADJUDICATION`.

All 20 policy annotations remain provisional. As designed, the tool emitted:

- no trajectory-quality review template;
- no reviewer decisions;
- no adjudicated quality labels;
- no training data.

This prevents provisional policy judgments from being silently converted into
RAW_GOLD, correction, holdout, or segmentation decisions.
