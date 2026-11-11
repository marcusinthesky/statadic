---
type: task
title: Make the NB2 log likelihood accurate in float64
description: Replace the single-precision-grade log-beta in the NB2 row likelihood without losing small-dispersion or float32 accuracy.
links: []
status: backlog
---

# Make the NB2 log likelihood accurate in float64

## Goal

`statadic.model.likelihood.log_likelihood_rows` computes the NB2 coefficient with
`jax.scipy.special.betaln(size, y)`, where `size = 1 / dispersion`. Measured
against SciPy in float64, that function is accurate only to about 1e-7 absolute
for moderate arguments: 4e-7 at (10, 1), 1e-6 at (8, 3), 8e-8 at (5, 20). The
reference comparison saw the consequence: an estimated NB2 fit reports a log
likelihood about 1e-8 relative from statsmodels at the same parameters, while
coefficients and covariance agree.

A plain `gammaln(size) + gammaln(y) - gammaln(size + y)` is not the repair. It is
accurate to 1e-14 for moderate `size`, but cancels for large `size` (1e-8 at
`size = 1e7` in float64) and fails in float32 (1e-2 at `size = 1e5`, where
`betaln` holds 1e-6). Small dispersion is a supported regime, so the repair needs
a log-beta accurate in both precisions across the range.

Row R14 in the [dispositions](../../../../proposal-dispositions.md) owns this.
The two NB2 reference tests leave the log likelihood unasserted until then.

Owner: Backlog. See [deferred work](../../deferred.md).

## Acceptance

- [ ] A log-beta accurate to about 1e-12 relative in float64 for dispersion from 1e-12 to 10, and no worse than today in float32.
- [ ] The NB2 row likelihood stays transformable and its small-dispersion Poisson limit tests pass.
- [ ] Both NB2 reference tests assert the log likelihood against statsmodels at rtol 1e-10.

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
