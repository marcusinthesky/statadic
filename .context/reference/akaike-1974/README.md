---
type: article-journal
title: A new look at the statistical model identification
description: Original Akaike information criterion, -2 log L + 2k.
author: [{family: Akaike, given: Hirotugu}]
issued: {date-parts: [[1974]]}
container-title: "IEEE Transactions on Automatic Control"
volume: "19"
issue: "6"
page: "716-723"
DOI: "10.1109/TAC.1974.1100705"
URL: "https://doi.org/10.1109/TAC.1974.1100705"
links: []
---

# A new look at the statistical model identification

Publication metadata was checked on 2026-11-11 against the Crossref record for
the DOI. Introduces the information criterion that STATADIC reports as `aic`: minus twice the maximised log likelihood plus twice the number of independently estimated parameters. STATADIC counts every estimated parameter of the fitted likelihood, including scale, dispersion, shape and spatial parameters, so that one rule holds across families. Packages that omit the Gaussian scale report a value smaller by two. The criterion compares models of the same response on the same rows; it does not establish that any compared model is correct.

## Backlinks

<!-- insitu:begin sql
id = "reference-backlinks"
model = "backlinks"
template = "backlinks.md.j2"
-->

- [Report fit statistics in fit and inference reports](../../plan/api-cohesion/tasks/report-review/tasks/table-content/README.md)

<!-- insitu:end -->
