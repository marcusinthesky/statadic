---
type: explanation
description: Mathematical statements, hypotheses and comparisons for STATADIC methods.
---

# From Python methods to mathematical results

Use the [obligation index](index.md) to see which guarantees remain open and
which have a checked statement. Each claim page gives the Lean statement, its
assumptions, a source locator and a downloadable LaTeX derivation. Compare
implementations on the same inputs, weights and sampling scheme for each method, since numerical
agreement and proof of a mathematical identity answer different questions. Each comparison in the tables below identifies an operation and a reference for it, with the conventions needed to interpret agreement between the two calculations. The linked claims state conclusions under their own assumptions, which must be assessed in the context of the particular data and model before applying them to an analysis. For a result in a report, use the matching entry to trace the mathematical statement and the independent calculation back to their sources.

```{raw} html
<a href="downloads/public-evidence.json">Download the certified symbol crosswalk and comparison definitions (JSON)</a>.
```

Each resolved target lists its public spellings, including re-exports and class members, while the claim IDs link it to statements in the [claim registry](index.md). The comparison IDs identify selected independent
checks and give their pytest node, reference, measured quantity, tolerances and
input domain. For example, `capped-simplex` compares weights and achieved cost
with SciPy SLSQP and checks the reported gap. The comparison record gives its exact test identifier.

Empty lists identify targets with no directly mapped claim or curated comparison, while the matrix reads a fresh proof certificate. Numerical entries describe test definitions, and a test report records which cases actually ran.

## Estimation and inference

| STATADIC operation | What the proof establishes | Independent implementation |
| --- | --- | --- |
| Fixed-design least squares | [Mean-zero integrable errors suffice for unbiasedness](claims/STATADIC-LINEAR-009.md). [OLS variance ordering](claims/STATADIC-LINEAR-010.md) adds spherical second moments. [Diagonal WLS](claims/STATADIC-LINEAR-011.md) compares all linear unbiased contrasts using fixed relative precisions. | [R weighted and ordinary linear models](https://stat.ethz.ch/R-manual/R-devel/library/stats/html/lm.html) |
| HC0 covariance | The [sandwich limit](claims/STATADIC-LINEAR-012.md) follows from explicit design and score-moment limits, with a residual replacement step that states the required error bound. | [statsmodels HC0](https://www.statsmodels.org/stable/generated/statsmodels.regression.linear_model.OLSResults.cov_HC0.html) |
| Bartlett HAC | [Raw Bartlett](claims/STATADIC-HAC-005.md) and [sample-centered contrasts](claims/STATADIC-HAC-007.md) have consistency results under lag-moment summability, bandwidth and sampling-variance conditions. | [statsmodels HAC sandwich](https://www.statsmodels.org/stable/generated/statsmodels.stats.sandwich_covariance.cov_hac.html) |

## Generalized models and panels

| STATADIC operation | What the proof establishes | Independent implementation |
| --- | --- | --- |
| GLM fitting | The [working-score identity](claims/STATADIC-GLM-001.md) explains one Fisher-scoring update under its derivative and variance assumptions. Exact row derivatives cover [grouped binomial logit](claims/STATADIC-GLM-002.md) and [Poisson log link](claims/STATADIC-GLM-003.md). The [fitted-root limit](claims/STATADIC-GLMCOV-003.md) assumes score convergence and a regular limiting sensitivity. | [R GLM](https://stat.ethz.ch/R-manual/R-devel/library/stats/html/glm.html) |
| Fixed-weight affine GMM | The [weighted residual norm gap](claims/STATADIC-GMM-003.md) proves global finite-sample minimization. Under the stated identification and convergence assumptions, the [affine coefficient](claims/STATADIC-GMM-006.md) is consistent, while its [sandwich limit](claims/STATADIC-GMM-005.md) uses the actual weight and score covariance. | [statsmodels GMM](https://www.statsmodels.org/stable/gmm.html) |
| Panel absorption | [Weighted group demeaning](claims/STATADIC-PANEL-001.md) extends to an [exact two-way projection certificate](claims/STATADIC-PANEL-002.md) on unbalanced panels. A [complete product-weight roster](claims/STATADIC-PANEL-003.md) permits either sweep order in one pass. For other rosters, check both margins, the [exact fixed-point condition](claims/STATADIC-PANEL-006.md) and the reported tolerance. | [R fixest demeaning](https://lrberge.github.io/fixest/reference/demean.html) |

## Distances and calibration

| STATADIC operation | What the proof establishes | Independent implementation |
| --- | --- | --- |
| Energy and kernel discrepancy | The [generating-kernel identity](claims/STATADIC-ENERGY-005.md) gives population cancellation. The [feature representation](claims/STATADIC-ENERGY-004.md) shows nonnegativity and equality of feature means at zero, while finite first moments imply population nonnegativity for [negative-type metric grounds](claims/STATADIC-ENERGY-007.md). The [separation theorem](claims/STATADIC-MMD-007.md) characterizes distributional separation by injectivity of the mean embedding on the chosen class. | [R energy](https://cran.r-project.org/package=energy) |
| Finite transport | [Uniform assignment](claims/STATADIC-TRANSPORT-002.md) and [weighted rectangular transport](claims/STATADIC-TRANSPORT-005.md) attain primal and dual optima. [Sorted matching](claims/STATADIC-TRANSPORT-003.md) covers equal-count quadratic real-line transport. | [SciPy linear programming](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.linprog.html) |

## Calibration and resampling

| STATADIC operation | What the proof establishes | Independent implementation |
| --- | --- | --- |
| Full-group permutation | The [rank test](claims/STATADIC-RANDOMIZATION-003.md) controls level under joint orbit-statistic invariance and counts ties inclusively. The [sampled result](claims/STATADIC-RANDOMIZATION-002.md) instead depends on its own sampling assumptions. | [SciPy permutation test](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.permutation_test.html) |
| Familywise adjustment | [Bonferroni](claims/STATADIC-MULTIPLICITY-001.md) and [Holm](claims/STATADIC-MULTIPLICITY-002.md) control the probability of rejecting any true null when its p-value is marginally valid. The result allows unrestricted dependence between tests. | [R p.adjust](https://stat.ethz.ch/R-manual/R-devel/library/stats/html/p.adjust.html) |
| IID bootstrap | The [conditional mean and variance](claims/STATADIC-BOOTSTRAP-001.md) use independent draws with replacement. For two rows, the [example](claims/STATADIC-BOOTSTRAP-002.md) gives an ideal bootstrap variance of the sample mean whose expectation is half its sampling variance. | [SciPy bootstrap](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.bootstrap.html) |

The equivalence helper counts bootstrap energies at or above one discrepancy margin, and under ordinary empirical resampling the [energy-margin example](claims/STATADIC-EQUIVALENCE-001.md) gives a true-null rejection probability of at least one quarter at a nominal level of one twentieth. A two-one-sided Student test uses different data, rejection rules and assumptions, while the [held-out selection bound](claims/STATADIC-SELECTION-001.md) requires independence of training and test evidence. Calibration of the same-sample selected-mixture procedure remains open.

## Spatial models

| STATADIC route | Result and assumptions | Reference calculation |
| --- | --- | --- |
| Spatial 2SLS | [Instrument projection and structural-residual orthogonality](claims/STATADIC-IV-001.md) require nonsingular instrument and projected-design Gram matrices. When exogeneity fails, a [full-rank instrument can identify the wrong target](claims/STATADIC-IV-002.md). | [PySAL GM_Lag](https://pysal.org/spreg/generated/spreg.GM_Lag.html) |
| Spatial-error GMM | [Sample moment polynomial](claims/STATADIC-GMM-001.md) for symmetric moment matrices and [zero population quadratic moment](claims/STATADIC-GMM-002.md) for zero-diagonal matrices with vanishing cross moments. [Global identification](claims/STATADIC-SPATIAL-004.md) requires excluding other admissible roots. [Two such roots](claims/STATADIC-SPATIAL-005.md) can coexist with local regularity. | [PySAL GM_Error_Het](https://pysal.org/spreg/generated/spreg.GM_Error_Het.html) |

## Spatial panels and impacts

| STATADIC route | Result and assumptions | Reference calculation |
| --- | --- | --- |
| KKP panel error components | [Temporal projection and spatial lag](claims/STATADIC-KKP-001.md) commute on every panel exactly when periods mixed with nonzero weight have the same graph. | [PySAL GM_KKP](https://pysal.org/spreg/generated/spreg.GM_KKP.html) |
| Spatial panel likelihood | For a full roster, use the [block log-Jacobian](claims/STATADIC-SPATIAL-001.md). For missing outcomes, a [rectangular solve](claims/STATADIC-SPATIAL-002.md) selects the observed covariance from the full model. | [SciPy multivariate normal](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.multivariate_normal.html) with the explicitly selected covariance |
| Direct, indirect and total effects | The [impact averages](claims/STATADIC-SPATIAL-003.md) use the trace and average row sum of the reduced-form multiplier. For a covariate perturbation, hold the graph fixed throughout the calculation. | [Spatialreg impact measures](https://r-spatial.github.io/spatialreg/reference/impacts.html) |

The spatial identities apply conditional on the supplied graph, including one returned by a learned-weight procedure. Identification, uncertainty from learning the graph and consistency of its coefficients each require their own assumptions and proofs.
