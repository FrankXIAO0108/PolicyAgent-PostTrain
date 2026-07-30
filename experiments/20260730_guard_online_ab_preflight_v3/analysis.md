# Guard Online Paired A/B V1 Preflight

- Status: **BLOCKED**
- Tasks: 95, 98, 107
- Paid calls executed: no
- API key value recorded: no

## Checks

| Check | Passed | Detail |
|---|---|---|
| tasks_in_frozen_parent | YES | All A/B tasks must be present in the frozen parent run config. |
| official_test_unused | YES | The official Tau2 test split must remain unused. |
| controlled_variables_complete | YES | Both arms may differ only by Guard implementation and retry behavior. |
| clean_git_tree | NO | Paid execution requires committed code and protocol hashes. |
| api_key_configured | NO | The required API key must exist in the environment; its value is never recorded. |
| explicit_paid_approval | NO | Paid model execution requires the explicit CLI approval flag. |
| failure_selected_scope_disclosed | YES | Results must be reported as a failure-selected subset. |

## Boundary

- This is a failure-selected three-task subset.
- It cannot produce a general Retail success-rate claim.
- Paid execution remains blocked until every preflight check passes.
- Raw arm outputs must pass V7 replay before a final recovery claim.
