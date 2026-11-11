---
type: "article-journal"
title: "Marginal maximum likelihood estimation of SAR models with missing data"
description: "Gaussian marginalization over unobserved spatial outcomes on the full roster."
author: [{"family": "Suesse", "given": "Thomas"}]
issued: {"date-parts": [[2018]]}
container-title: "Computational Statistics & Data Analysis"
volume: "120"
page: "98-110"
DOI: "10.1016/j.csda.2017.11.004"
URL: "https://doi.org/10.1016/j.csda.2017.11.004"
links: []
---

# Marginal maximum likelihood estimation of SAR models with missing data

The section on marginal log-likelihood applies the Gaussian linear transformation rule to the observed rows after equation (4). If $y$ has covariance $\sigma^2 V$ and $P$ selects those rows, $Py$ has covariance $\sigma^2 PVP^T$. The paper develops SAR and SEM marginal likelihood estimators. [Publisher's text](https://www.sciencedirect.com/science/article/abs/pii/S0167947317302396)

For STATADIC's period-specific composite filter $F=BA$, the innovation covariance $\sigma^2 I$ gives $V=F^{-1}F^{-T}$. STATADIC solves $F^T Z=P^T$ on the full roster. Transposition and right multiplication give $Z^T=PF^{-1}$, so $Z^TZ=PVP^T$. The registered theorem proves this matrix step. STATADIC's independent numerical example instead forms the full inverse, selects the covariance submatrix and scores it with SciPy. [Source mapping](source.json) · [Reference test](../../../src/python/packages/statadic/tests/api/test_spatial_graph_regression.py) · [SciPy Gaussian log density](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.multivariate_normal.html)

## Backlinks

<!-- insitu:begin sql
id = "reference-backlinks"
model = "backlinks"
template = "backlinks.md.j2"
-->

_No declared backlinks._

<!-- insitu:end -->
