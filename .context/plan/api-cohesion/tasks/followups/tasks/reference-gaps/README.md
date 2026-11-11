---
type: task
title: Compare shipped unreferenced variants with installed peers
description: Add matched comparisons for shipped variants that lack one, and record a reason where no peer applies.
links: []
status: backlog
---

# Compare shipped unreferenced variants with installed peers

## Goal

Read from the studies catalogue at `00e7d8f`: 96 of 309 variants map to an exact
reference node, and 26 of 51 procedures have no mapped variant. A mapping records
a comparison; nothing records why one is absent. These shipped variants are
unmapped although the locked reference group contains a possible peer:

| Catalogue record | Unmapped | Possible peer |
| --- | --- | --- |
| `survival.independent.loglogistic` | 1 | lifelines |
| `likelihood.regression` (gamma, NB2, probit) | 18 | statsmodels, glum |
| `likelihood.hurdle-poisson` | 2 | statsmodels |
| `iv.regression` 2SLS with HC0, HC1, HAC, clustered | 4 | linearmodels |
| `spatial.panel` | 39 of 40 | spreg |

Possible peers are leads, not checked: confirm each implements the same model and
convention before mapping. The three spatial panel walkthroughs compare
coefficients, likelihood and predictions, not standard errors. Pyfixest is not in
the reference group; the transcript wanted it as a second check on fixed-effect
absorption and clustered covariance, which are existing capabilities.

Rows R09–R12 in the [dispositions](../../../../proposal-dispositions.md) own
these. R08 keeps LIML and fixed-effects Poisson deferred as new capability.

A later hypothesis-test audit added four comparisons, listed below. PERMDISP
against R's vegan is not among them: the repository has no R dependency and the
owner declined to add one for now (row R13). The generic two-sided permutation
kernel counts `abs(null) >= abs(observed)`; a peer that doubles the smaller tail
is not the same definition.

Owner: Backlog. See [deferred work](../../deferred.md).

## Acceptance

- [x] For each listed variant, add a matched comparison with written conventions, or record why no peer applies.
- [x] Compare standard errors in the spatial panel walkthroughs, or record the convention that prevents it.
- [ ] Owner decides whether pyfixest joins the reference group.
- [ ] Compare a public joint restriction with nonzero targets, model-based and HC3, with statsmodels.
- [ ] Compare PERMANOVA null statistics and p-values with scikit-bio under one supplied design.
- [ ] Compare Holm and Bonferroni decisions with statsmodels `multipletests`, including ties.
- [ ] Compare the public OLS `diagnose` path with statsmodels, after reading what the existing walkthroughs assert.
- [ ] Run the reference lane with external references required and no skips; map only exact executed nodes.

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

- [Witness permutation and GMS p-values independently](../../../final-review/tasks/pvalue-witnesses/README.md)

<!-- insitu:end -->
