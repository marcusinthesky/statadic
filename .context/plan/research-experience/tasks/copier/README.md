---
type: task
title: Migrate Copier to Exordio alpha
description: Migrate Copier to Exordio alpha
links: []
status: done
---

# Migrate Copier to Exordio alpha

## Goal

Point Copier at the published Exordio alpha without losing the saved selections
or STATADIC's authored changes. Reconcile template history before adopting its new
baseline.

## Acceptance

- [x] Verify the alpha tag and reconcile the old and new template renders.
- [x] Pin the canonical repository and `v0.1.0-alpha` baseline.
- [x] Link the published documentation site and explicit release update command.
- [x] Preserve STATADIC's README, CI and scientific implementation customizations.
- [x] Adopt and exercise transient TeX setup retries.
- [x] Check the migrated recipes, source records and final documentation.

[Migration review](review.md) · [Retry tests](retry-tests.md)

The old template baseline was absent from Exordio's new history. A reviewed
three-way migration reconciled all seven changed template paths before updating
the saved source and baseline. A same-release Copier update in the isolated
preview was clean. A subsequent direct Git query confirmed the canonical tag
`4c641448957366aae760e39e87b9c14640f62645`, and a permitted HTTP check returned
200 for the requested documentation site. Post-migration source/workflow/recipe,
documentation, Lean, LaTeX, AgentSync and browser checks passed. The explicit
networked TeX setup itself was not rerun; the retry behavior was tested with
controlled failure cases and cached builds passed.

<!-- insitu:begin sql
id = "planning-children"
model = "children"
template = "children.md.j2"
-->

_No open children._

`0 open; 0 terminal`

<!-- insitu:end -->

## Backlinks

<!-- insitu:begin sql
id = "reference-backlinks"
model = "backlinks"
template = "backlinks.md.j2"
-->

_No declared backlinks._

<!-- insitu:end -->
