# Policy Grounding V2.2 diagnostic analysis

V2.2 fixes the remaining Task 1 false escalation by preserving an immediately
preceding confirmed action summary when the user later performs a scope-only
reconfirmation. The rule does not treat arbitrary older requests as current
intent, and the existing Task 107 wrong-variant regression remains `FAIL`.

The Task 28 provisional policy label was separately corrected from `PASS` to
`REVIEW`: its three writes are correct, but its batched reads and message/tool
mixing are not policy-clean.

On the 20-row development pool, V2.2 matches all 20 provisional labels:
12 `REVIEW` and 8 `FAIL`. There are no provisional `PASS` rows, so the reported
FAIL precision/recall values do not test true negatives. These results are
integration diagnostics on rules refined against the same trajectories, not
held-out generalization.

All labels remain provisional. The official metric, data-release, SFT, and RL
gates remain closed pending independent adjudication.
