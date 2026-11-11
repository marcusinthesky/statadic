---
type: article-journal
title: Using Heteroscedasticity Consistent Standard Errors in the Linear Regression Model
description: Conventional HC3 sandwich and finite-sample simulation evidence.
author: [{family: Long, given: J. Scott}, {family: Ervin, given: Laurie H.}]
issued: {date-parts: [[2000, 8]]}
container-title: The American Statistician
volume: "54"
issue: "3"
page: "217-224"
DOI: "10.1080/00031305.2000.10474549"
URL: "https://doi.org/10.1080/00031305.2000.10474549"
links: []
---

# Using Heteroscedasticity Consistent Standard Errors in the Linear Regression Model

Publisher-deposited [Crossref metadata](https://api.crossref.org/works/10.1080/00031305.2000.10474549)
was checked on 2026-11-04. Section 2, manuscript pages 6–7 of the
[author-linked manuscript](https://nielsen.sites.oasis.unc.edu/soci709/odocs/00TAS.pdf)
defines the conventional uncentered HC3 sandwich. Its page 7 formula matches the
default full-rank unweighted STATADIC covariance; applying it to explicitly whitened
design and response also matches the fixed-positive-precision WLS adapter.

This formula approximates a more complicated jackknife estimator. It is not
identical to the original centered finite-sample MacKinnon–White expression.
Boundary leverage, optional multipliers, adaptive/survey weights, numerical
availability and reference-law calibration remain separate obligations. The
[correspondence review](../../plan/research-experience/tasks/studies/tasks/sources/hc3-followup.md)
records the checked whitening algebra and limitations.

## Backlinks

<!-- insitu:begin sql
id = "reference-backlinks"
model = "backlinks"
template = "backlinks.md.j2"
-->

- [Prove WLS whitening and the HC covariance algebra](../../plan/release-guarantees/tasks/proofs/tasks/linear-covariance/README.md)

<!-- insitu:end -->
