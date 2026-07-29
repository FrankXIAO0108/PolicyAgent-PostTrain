# Reviewer submission workflow v1

Date: 2026-07-29

This change addresses the current P0 operational blocker without changing the
training route. The portable blind-review packet now contains a bilingual,
Excel-friendly CSV sheet in addition to the JSONL template.

One reviewer submission can be preflighted before adjudication. The preflight
checks:

- complete and exact task coverage;
- exactly one reviewer identity;
- supported labels, rationale, and timezone-aware timestamps;
- unchanged evidence paths;
- evidence paths confined to the packet;
- every bundled evidence file against its frozen SHA-256.

The workflow fails closed. A rejected submission emits a report but no
normalized adjudication input.

The generated v2 packet contains 20 empty CSV rows, 20 empty JSONL rows, 20
trajectories, 20 summaries, and one Retail policy. No reviewer decision,
training run, model call, or policy-label promotion occurred.
