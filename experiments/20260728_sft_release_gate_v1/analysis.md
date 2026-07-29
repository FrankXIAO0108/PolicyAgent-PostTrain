# SFT release gate v1 dry-run analysis

V1 adds a required binding from every `CORRECTED_POSITIVE` decision to a ready,
hashed correction-validation report. A correction path and hash alone can no
longer release training data.

The current 20-row pool remains blocked because all policy labels are
provisional and no adjudicated trajectory-quality decision file exists.
Observed release count is zero and no SFT dataset was emitted.

No training, model call, correction generation, or environment replay occurred.
