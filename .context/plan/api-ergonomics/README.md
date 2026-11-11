---
type: plan
title: Explicit API choices and readable results
description: Explicit API choices and readable results
links: []
status: in-progress
---

# Explicit API choices and readable results

## Goal

Make model-defining choices explicit and fitted results readable, before the next
release target is frozen for retained campaigns.

## Acceptance

- [x] Require `metric` on angular-default energy functions.
- [x] Require the AFT family and `ML.process`; report the survival family.
- [x] Bound result reprs and add notebook display.
- [x] Add `valid`, `parameters`, `parameter_names` and `df_resid` properties.
- [x] Reject a missing `reference` in `distance_induced_gram` where it yields NaN.
- [x] Fix the rooted V convention on `MMD` and `Energy`, require `Wasserstein1D.p`,
  and explain prepared-state ownership in its error.
- [x] Consistency: GMM/GEL reports, `panel=` spelling, prediction `kind=`.
- [x] Discoverability: method links, Compose recipes, docs cleanup.
- [x] Retain logistic prediction artifacts with trials and offset roles.
- [x] Extension: ordinary/spatial AFT means and log-logistic AFT.
- [x] Extension: Gamma, NB2 and grouped probit with explicit dispersion/inference.
- [x] Extension: non-spatial IV with explicit GMM weighting and covariance.
- [ ] Resolve cold hosted layout-tool validation and qualify the frozen release.

## Verification

Slice 1 follow-ups and Phase 2 are implemented. The complete `just check` passed
on 2026-11-05; the [Phase 2 record](tasks/phase-2/README.md) contains the scope
and validation counts. [Phase 3](tasks/phase-3/README.md) adds discoverability and logistic artifacts.
The separate spatial-panel task records the acceptance-only SAC correction.
[Phase 4](tasks/phase-4/README.md) adds the bounded family extensions. The
[repair task](tasks/repairs/README.md) records numerical/evidence corrections and
the unresolved hosted layout-tool gate. Retained qualification campaigns remain
later work after source review and the exact release wheel are frozen.

<!-- insitu:begin sql
id = "planning-children"
model = "children"
template = "children.md.j2"
-->

| Status | Item | Description |
| --- | --- | --- |
| in-progress | [Apply the composition-review waves](tasks/composition-waves/README.md) | Apply the 2026-11-06 composition-review findings in waves: permutation designs and shared WX selection first, then warrants, model description and bootstrap plotting. |
| backlog | [Diagnostics for spatial, panel and GLM results](tasks/family-diagnostics/README.md) | Route existing spatial diagnostics and new family-specific tests through result.diagnose. |
| in-progress | [Repair numerical boundaries and evidence integrity](tasks/repairs/README.md) | Repair numerical boundaries and evidence integrity |
| in-progress | [Apply the assessed research-workflow decisions](tasks/research-workflow/README.md) | Implement, defer or decline each assessed research-workflow proposal, with publication on hold. |

`4 open; 6 terminal`

<!-- insitu:end -->

## Backlinks

<!-- insitu:begin sql
id = "reference-backlinks"
model = "backlinks"
template = "backlinks.md.j2"
-->

_No declared backlinks._

<!-- insitu:end -->
