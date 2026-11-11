---
type: task
title: Diagnostics for spatial, panel and GLM results
description: Route existing spatial diagnostics and new family-specific tests through result.diagnose.
links: []
status: backlog
---

# Diagnostics for spatial, panel and GLM results

## Goal

Extend `result.diagnose` beyond OLS without transferring OLS calibrations.

## Acceptance

- [ ] Expose residual Moran and LM lag/error tests through `diagnose` on fixed
  exogenous graphs; reject them for learned graphs.
- [ ] Add panel-aware serial correlation and GLM deviance/Pearson diagnostics
  with their own references and size studies.
- [ ] Add Ljung-Box, Goldfeld-Quandt and the D'Agostino omnibus test with
  statsmodels parity.

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
