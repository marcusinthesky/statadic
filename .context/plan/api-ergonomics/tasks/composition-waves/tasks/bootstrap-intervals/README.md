---
type: task
title: Make bootstrap intervals plottable
description: "Let the shared coefficient plot label and draw percentile-bootstrap intervals, not only Wald intervals."
links: []
status: backlog
---

# Make bootstrap intervals plottable

## Goal

`Report.coefficient_plot` (`api/reporting.py`) requires a numeric
`confidence` and builds its label from a covariance and reference law. The
dyadic node bootstrap (`api/dyadic.py`) reports its interval as the string
`"percentile [2.5, 97.5]"`, has `None` rows for dropped-collinear columns, and
has no covariance or reference law, so the plot raises or would mislabel it.
The 2.5 and 97.5 percentiles are fixed in `inference/dyadic/bootstrap.py`.

## Decisions to make

- A generic interval-construction key (method and level) in place of
  covariance/reference in report metadata.
- Whether unavailable rows are skipped with a note or the plot stays
  all-or-nothing.
- Whether the dyadic bootstrap gains a confidence parameter.

## Acceptance

- [ ] The same `coefficient_plot` accurately labels a Wald and a
  percentile-bootstrap interval.

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
