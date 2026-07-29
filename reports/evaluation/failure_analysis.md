# Tau2-aligned Hybrid Evaluation v7

- Generated: `2026-07-26T13:00:51.661244+00:00`
- Experiment: `D:\PolicyAgent-PostTrain\experiments\20260722_110504_retail_baseline20_trial1_deepseek`
- Tasks: 20
- Official successes: 16
- Official failures: 4
- Replay inconsistencies: 0

## Failure analysis

### Task 59

- Official signal: DB=False, NL=False, reward=0.0
- Layer 1 / official signal: db_mismatch, nl_failure
- Layer 2 / root cause: dataset_alignment_error, missing_action, communication_error, policy_error
  - Primary causal: dataset_alignment_error, missing_action, communication_error
  - Secondary findings: policy_error
- Layer 3 / business impact: benchmark_data_risk, policy_risk, incomplete_customer_request, incomplete_customer_communication
- Quarantine recommended: True
- Agent hash: `b30b59edbe5d104f36e07977ecbd14fc4dd000cceee3e6bf30f783ce52522c1d`
- Gold hash: `8cf194e4eca8883c044792c4441d4b83b270934e22fce04f00569f5944c6d279`
- Causal evidence:
  - `golden_mismatch`: The executed cancellation follows the simulator's explicit confirmed order, while the static gold cancels a different order.
  - `dataset_issue`: Static gold and the latest simulated user intent are inconsistent; the trajectory should not be used as an ordinary negative.
  - `missing_action`: A gold cancellation was not reflected in final DB.
  - `communication_omission`: At least one frozen official NL assertion failed.
- Secondary findings:
  - `policy_violation`: Retail policy permits at most one tool call per assistant turn.
- DB diff flags: extra_cancel, missing_cancel, wrong_refund, wrong_address, wrong_status, missing_mutation, extra_mutation
- Improvement suggestions:
  - Reconcile latest user intent with static gold; quarantine case.
  - Run deterministic policy checks before write tool execution.
  - Track every requested goal until completed, denied, or transferred.
  - Generate final communication from verified tool results.

### Task 95

- Official signal: DB=False, NL=False, reward=0.0
- Layer 1 / official signal: db_mismatch, nl_failure
- Layer 2 / root cause: variant_error, missing_action, communication_error, policy_error
  - Primary causal: variant_error, missing_action, communication_error
  - Secondary findings: policy_error
- Layer 3 / business impact: wrong_product_selection, policy_risk, incomplete_customer_request, incomplete_customer_communication
- Quarantine recommended: False
- Agent hash: `b25c9cb211f5efcaee5dd646054a73c4a9f43f4f5acc32c10713cd9f9ac20e9c`
- Gold hash: `a0d1b0eab2e1e283dbf84607d02b637eb6bc61fc1cd4c8d3bed3d3377ecbba6b`
- Causal evidence:
  - `variant_understanding_failure`: The agent interpreted a variant's availability boolean as a single inventory unit, although the gold reuses that variant across multiple orders.
  - `missing_action`: One or more gold exchanges were not reflected in final DB.
  - `communication_omission`: At least one frozen official NL assertion failed.
- Secondary findings:
  - `policy_violation`: Retail policy permits at most one tool call per assistant turn.
- DB diff flags: missing_exchange, wrong_status, missing_mutation
- Improvement suggestions:
  - Resolve variants structurally from product options and availability.
  - Run deterministic policy checks before write tool execution.
  - Track every requested goal until completed, denied, or transferred.
  - Generate final communication from verified tool results.

### Task 98

- Official signal: DB=False, NL=True, reward=0.0
- Layer 1 / official signal: db_mismatch
- Layer 2 / root cause: payment_error, scope_error, policy_error
  - Primary causal: payment_error
  - Secondary findings: scope_error, policy_error
- Layer 3 / business impact: overbroad_or_incorrect_order_effect, wrong_refund_or_charge_destination, policy_risk
- Quarantine recommended: False
- Agent hash: `74f96064794e1864ead14309cd77d785e58c2ce9b751d2a934e4b314d78eee6a`
- Gold hash: `a38fddcc78ebe2ba2a41b1cffaff27fd678aa60e52c505a0cd0a43e3963ea641`
- Causal evidence:
  - `wrong_payment_method`: The exchange/return payment method persisted in DB differs from the gold action.
- Secondary findings:
  - `scope_confirmation_failure`: The tool cancelled a multi-item order, but the final claim described an item-only cancellation.
  - `policy_violation`: Retail policy permits at most one tool call per assistant turn.
- DB diff flags: wrong_payment
- Improvement suggestions:
  - Confirm and report the complete tool effect scope.
  - Bind the confirmed payment method to every exchange action.
  - Run deterministic policy checks before write tool execution.

### Task 107

- Official signal: DB=False, NL=True, reward=0.0
- Layer 1 / official signal: db_mismatch
- Layer 2 / root cause: variant_error, policy_error
  - Primary causal: variant_error, policy_error
  - Secondary findings: none
- Layer 3 / business impact: wrong_product_selection, policy_risk
- Quarantine recommended: False
- Agent hash: `5eb70a54dd4d8d92d9b5e540f608e0a8afe6bf35aeebeb38f8dd3b4b271a9744`
- Gold hash: `8111ebca39106dc47090d13df4491130add9e194d7d71697cf909a9a0be50040`
- Causal evidence:
  - `wrong_variant`: The reconstructed exchange_new_items differ from gold.
  - `policy_violation`: Retail policy requires exchange to a different product option, but the tool call used the same old and new item ID. The tool accepted a business-rule violation.
- Secondary findings:
  - `policy_violation`: Retail policy permits at most one tool call per assistant turn.
- DB diff flags: wrong_variant
- Improvement suggestions:
  - Resolve variants structurally from product options and availability.
  - Run deterministic policy checks before write tool execution.

## Interpretation boundary

Layer 1 is reconstructed from Tau2 state replay and frozen official NL results. Layers 2 and 3 are downstream deterministic diagnoses; they are not Tau2-native labels. Cases with `benchmark_data_risk` should be quarantined instead of used as ordinary optimization negatives.
