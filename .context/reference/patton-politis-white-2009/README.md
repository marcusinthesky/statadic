---
type: "article-journal"
title: "Correction to “Automatic Block-Length Selection for the Dependent Bootstrap” by D. Politis and H. White"
description: "Primary correction supplying the stationary-bootstrap variance constant used by the selector."
author: [{"family": "Patton", "given": "Andrew"}, {"family": "Politis", "given": "Dimitris N."}, {"family": "White", "given": "Halbert"}]
issued: {"date-parts": [[2009]]}
container-title: "Econometric Reviews"
volume: "28"
issue: "4"
page: "372-375"
DOI: "10.1080/07474930802459016"
URL: "https://www.tandfonline.com/doi/abs/10.1080/07474930802459016"
links: []
---

# Correction to “Automatic Block-Length Selection for the Dependent Bootstrap” by D. Politis and H. White

Metadata and the correction were checked on 2026-11-04 against the [publisher](https://www.tandfonline.com/doi/abs/10.1080/07474930802459016) and [author-hosted paper](https://public.econ.duke.edu/~ap172/Patton_Politis_White_2009.pdf).

Page 373, corrections 1 and 4-5, replace the stationary-bootstrap variance constant by twice the squared zero-frequency spectral quantity and apply that correction to the plug-in block selector. STATADIC's `optimal_block_length_result` uses the matching factor `2.0 * spectral_density_zero**2` for the stationary scheme. This confirms that specific correction, not every bandwidth-search convention or numerical branch. Cite together with `politis-white-2004`; selector evaluation is not a universal confidence-interval coverage guarantee.

## Backlinks

<!-- insitu:begin sql
id = "reference-backlinks"
model = "backlinks"
template = "backlinks.md.j2"
-->

- [Primary-source citation audit](../../evidence/source-correspondence/citation-audit.md)

<!-- insitu:end -->
