# Offer-ready portfolio packaging

Date: 2026-07-29

This change packages the verified Tool Agent reliability work as a recruiter-
and interviewer-facing project without changing the research evidence.

The package adds:

- a zero-API offline demo built only from frozen reports;
- an explicit upstream-versus-project ownership boundary;
- a project completion report;
- a 30-second and 2-minute interview narrative;
- three evidence-backed business cases;
- Chinese and English resume bullets.

The demo does not rerun tau2, call a model, modify frozen inputs, or promote
provisional labels. SFT, DPO, RLHF, and GRPO remain explicitly uncompleted.

Validation:

- `demo.ps1` completed successfully;
- the focused portfolio/evaluation/Guard Ruff scope is clean;
- 74 tests passed;
- the full Ruff scan still reports eight pre-existing findings under
  `src/trajectory`, which was outside this change and already modified in the
  working tree.
