# Verifier V1.2 diagnosis

## Executive result

V1.1 exposed two separate issues:

1. **Task 28 has a likely entity-grounding false negative.** The verifier
   requires internal `item_id` values to appear literally in the confirmed
   assistant summary. Users commonly confirm user-visible product names and
   variants instead. V1.2 resolves earlier tool-result entity records into
   aliases before checking write arguments.
2. **Task 16 is not proven to be a verifier false positive.** Its V1.1 result
   contains three write calls in one assistant turn. The frozen Retail policy
   permits at most one tool call at a time, so this remains a policy failure.
   The existing `GOLD / policy_compliant=YES` annotation should be re-audited
   against the raw trajectory rather than used to weaken the verifier.

## Why the earlier 2/7 accuracy claim is invalid

The human taxonomy and verifier output describe different targets:

- Human taxonomy: training eligibility (`GOLD`, `SILVER`, `NEGATIVE`,
  `EXCLUDED`).
- Verifier: trajectory dimensions (`latest_intent`, `explicit_confirmation`,
  `policy_compliance`, `action_result_truthfulness`) aggregated to
  `PASS / REVIEW / FAIL`.

A correct business outcome can still violate a trajectory-format policy, and a
negative trajectory can pass one verifier dimension. Mapping these labels
directly into a three-class confusion matrix produces uninterpretable metrics.

The next gold set must label each verifier dimension independently. Only then
should precision, recall, F1, FP, and FN be computed per finding type or
dimension.

## V1.1 baseline facts

- Overall: 10 FAIL, 10 REVIEW, 0 PASS.
- Policy compliance: 6 FAIL, 14 REVIEW.
- Latest intent: 4 FAIL, 5 REVIEW, 11 PASS.
- All 20 trajectories triggered message/tool exclusivity findings.
- 19/20 triggered multi-tool read findings.
- Six tasks contained multiple write calls in one turn: 16, 29, 76, 98, 107,
  and 109.
- Four tasks triggered unconfirmed internal-argument findings: 1, 28, 37, 72.

The absence of PASS does not by itself prove over-triggering. It may show that
the sampled agent systematically violated explicit tool-call formatting rules.

## V1.2 change

- Adds entity aliases from earlier tool results.
- Keeps payment-method aliases.
- Keeps multiple-write calls as a major policy failure.
- Does not manufacture classifier metrics from mismatched gold labels.

## Required validation

Re-run all 20 trajectories with V1.2. Then inspect the raw confirmation
summaries and entity records for tasks 1, 28, 37, and 72.

- If the summary names the correct item/product/variant, the internal ID is
  considered grounded.
- If neither the internal ID nor a resolved visible alias appears, the finding
  remains a failure.
- Task 16 remains a policy failure unless raw trajectory inspection shows the
  loader incorrectly merged separate assistant turns.
