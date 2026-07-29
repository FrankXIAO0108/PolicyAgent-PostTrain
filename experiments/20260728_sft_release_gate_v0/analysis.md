# SFT release gate v0 dry-run analysis

The fail-closed release gate was run against the current 20-row policy
annotation pool without a trajectory-quality decision file.

Observed result:

- policy annotations: 20;
- adjudicated policy annotations: 0;
- released SFT records: 0;
- `sft_dataset.jsonl` emitted: no;
- release status: blocked as designed.

The two blocking reasons are complete lack of independent policy adjudication
and lack of adjudicated trajectory-quality decisions. This dry run validates
gate behavior only. It is not a dataset, training run, or model-improvement
result.
