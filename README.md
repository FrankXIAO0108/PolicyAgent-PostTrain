# Programmatic Verifier V1.2 patch

This patch preserves V1.1 severity aggregation and adds user-visible entity
alias grounding for internal item/product IDs.

## What changed

- A `MAJOR` finding produces `FAIL`.
- Minor-only findings produce `REVIEW`, not `FAIL`.
- Clean dimensions produce `PASS`.
- Metrics now include verifier version and major/minor finding counts.
- V1 metrics now include `latest_intent_failed_write_calls`.
- Internal IDs can be grounded by names and variants from earlier tool results.
- Regression tests cover clean, minor-only, major, and entity-alias cases.

The Retail policy still explicitly forbids multiple tool calls in one assistant
turn and forbids combining user-facing content with tool calls. The patch does
not suppress those findings; it preserves them with distinct severity.

## Install

Copy `src/verifiers` into the repository's existing `src/verifiers` directory
and add both verifier test files to the repository tests.

## Validate

From the repository root:

```powershell
python -m unittest discover -s tests -v
```

Then rerun V1 against the experiment directories:

```powershell
python -m src.verifiers.policy_grounding_v1 <experiment-path>
```

Do not compare new V1.2 output directly with the old
`task_95_107_verifier_result.json`: that file was produced by V0, as shown by
its V0 notes and missing V1 intent-audit metrics.
