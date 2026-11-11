---
type: task
title: Add survival means, likelihood families and non-spatial IV
description: Add survival means, likelihood families and non-spatial IV
links: []
status: done
---

# Add survival means, likelihood families and non-spatial IV

## Goal

Add survival means and log-logistic AFT, Gamma/log-link, NB2 and binomial-probit,
plus conventional IV with 2SLS, two-step and iterated GMM. Preserve numerical
ownership and explicit inference choices. Start after local repair integration.

## Acceptance

- [x] Add ordinary/spatial survival means and log-logistic AFT including RMST.
- [x] Add fixed/estimated dispersion Gamma and NB2 plus grouped-binomial probit.
- [x] Add non-spatial IV with independent, HAC and one-way cluster inference.
- [x] Keep two-step and iterated GMM provenance/calibration distinct.
- [x] Register bounded studies and public capabilities without qualification claims.
- [x] Complete cross-review, focused verification and documentation.

## Verification

See the [integration evidence](tasks/verification/README.md). New studies remain
descriptive and unqualified. The separate repair task retains the hosted CI gate;
finishing these source extensions does not clear it.

No automatic Poisson fallback, universal distribution carrier, new persistence
codecs, fixed-effects IV or retained scientific campaigns are included.

<!-- insitu:begin sql
id = "planning-children"
model = "children"
template = "children.md.j2"
-->

_No open children._

`0 open; 1 terminal`

<!-- insitu:end -->

## Backlinks

<!-- insitu:begin sql
id = "reference-backlinks"
model = "backlinks"
template = "backlinks.md.j2"
-->

_No declared backlinks._

<!-- insitu:end -->
