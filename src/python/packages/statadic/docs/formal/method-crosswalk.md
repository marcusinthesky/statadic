# From Python methods to mathematical results

Use the [obligation index](index.md) to see which guarantees remain open.
A claim page gives the Lean statement, assumptions, source locator and downloadable
LaTeX derivation. Compare implementations on the same inputs, weights and sampling
scheme. Agreement on a numerical example and proof of a mathematical identity
answer different questions.

## Look up a public Python name

```{raw} html
<a href="downloads/public-evidence.json">Download the certified symbol crosswalk and comparison definitions (JSON)</a>.
```

Run these commands from the repository root to reproduce the export or look up one name.

```bash
just lean::certify
python src/lean/tools/evidence_crosswalk.py > public-evidence.json
python src/lean/tools/evidence_crosswalk.py --symbol statadic.optimize.simplex.simplex_qp_result
```

Each resolved target lists its public spellings, including re-exports and class
members. The claim IDs link that exact target to statements in the
[claim registry](index.md). The comparison IDs identify selected independent
checks and give their pytest node, reference, measured quantity, tolerances and
input domain. For example, `capped-simplex` compares weights and achieved cost
with SciPy SLSQP and checks the reported gap. To repeat it, pass its `pytest`
value to `python -m pytest` from the repository root.

Empty lists identify targets with no directly mapped claim or curated comparison.
The matrix reads a fresh proof certificate. Numerical entries describe test
definitions. A test report records which cases actually ran.

## Estimation and inference

| STATADIC operation | What the proof establishes | Independent implementation |
| --- | --- | --- |
| Fixed-design least squares | [Mean-zero integrable errors suffice for unbiasedness](claims/STATADIC-LINEAR-009.md). [OLS variance ordering](claims/STATADIC-LINEAR-010.md) adds spherical second moments. [Diagonal WLS](claims/STATADIC-LINEAR-011.md) compares all linear unbiased contrasts using fixed relative precisions. | [R weighted and ordinary linear models](https://stat.ethz.ch/R-manual/R-devel/library/stats/html/lm.html) |
| HC0 covariance | The [sandwich limit](claims/STATADIC-LINEAR-012.md) follows from explicit design and score-moment limits. Its residual replacement step states the required error bound. | [statsmodels HC0](https://www.statsmodels.org/stable/generated/statsmodels.regression.linear_model.OLSResults.cov_HC0.html) |
| Bartlett HAC | [Raw Bartlett](claims/STATADIC-HAC-005.md) and [sample-centered contrasts](claims/STATADIC-HAC-007.md) have consistency results under lag-moment summability, bandwidth and sampling-variance conditions. | [statsmodels HAC sandwich](https://www.statsmodels.org/stable/generated/statsmodels.stats.sandwich_covariance.cov_hac.html) |
| GLM fitting | The [working-score identity](claims/STATADIC-GLM-001.md) explains one Fisher-scoring update under its derivative and variance assumptions. Exact row derivatives cover [grouped binomial logit](claims/STATADIC-GLM-002.md) and [Poisson log link](claims/STATADIC-GLM-003.md). The [fitted-root limit](claims/STATADIC-GLMCOV-003.md) assumes score convergence and a regular limiting sensitivity. | [R GLM](https://stat.ethz.ch/R-manual/R-devel/library/stats/html/glm.html) |
| Fixed-weight affine GMM | The [weighted residual norm gap](claims/STATADIC-GMM-003.md) proves global finite-sample minimization. The [affine coefficient](claims/STATADIC-GMM-006.md) is consistent under the stated identification and convergence assumptions. Its [sandwich limit](claims/STATADIC-GMM-005.md) uses the actual weight and score covariance. | [statsmodels GMM](https://www.statsmodels.org/stable/gmm.html) |
| Panel absorption | [Weighted group demeaning](claims/STATADIC-PANEL-001.md) extends to an [exact two-way projection certificate](claims/STATADIC-PANEL-002.md) on unbalanced panels. A [complete product-weight roster](claims/STATADIC-PANEL-003.md) permits either sweep order in one pass. Otherwise check both margins, the [exact fixed-point condition](claims/STATADIC-PANEL-006.md) and the reported tolerance. | [R fixest demeaning](https://lrberge.github.io/fixest/reference/demean.html) |
| Energy and kernel discrepancy | The [generating-kernel identity](claims/STATADIC-ENERGY-005.md) gives population cancellation. The [feature representation](claims/STATADIC-ENERGY-004.md) shows nonnegativity and equality of feature means at zero. For [negative-type metric grounds](claims/STATADIC-ENERGY-007.md), finite first moments imply population nonnegativity. The [separation theorem](claims/STATADIC-MMD-007.md) characterizes distributional separation by injectivity of the mean embedding on the chosen class. | [R energy](https://cran.r-project.org/package=energy) |
| Finite transport | [Uniform assignment](claims/STATADIC-TRANSPORT-002.md) and [weighted rectangular transport](claims/STATADIC-TRANSPORT-005.md) attain primal and dual optima. [Sorted matching](claims/STATADIC-TRANSPORT-003.md) covers equal-count quadratic real-line transport. | [SciPy linear programming](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.linprog.html) |
| Full-group permutation | The [rank test](claims/STATADIC-RANDOMIZATION-003.md) controls level under joint orbit-statistic invariance and counts ties inclusively. The [sampled result](claims/STATADIC-RANDOMIZATION-002.md) has separate sampling assumptions. | [SciPy permutation test](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.permutation_test.html) |
| Familywise adjustment | [Bonferroni](claims/STATADIC-MULTIPLICITY-001.md) and [Holm](claims/STATADIC-MULTIPLICITY-002.md) control the probability of rejecting any true null when its p-value is marginally valid. Dependence between tests is unrestricted. | [R p.adjust](https://stat.ethz.ch/R-manual/R-devel/library/stats/html/p.adjust.html) |
| IID bootstrap | The [conditional mean and variance](claims/STATADIC-BOOTSTRAP-001.md) use independent draws with replacement. The [two-row example](claims/STATADIC-BOOTSTRAP-002.md) gives an ideal bootstrap variance of the sample mean whose expectation is half its sampling variance. | [SciPy bootstrap](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.bootstrap.html) |

The equivalence helper accepts bootstrap replicates and counts energies at or
above a single discrepancy margin. Under ordinary empirical resampling the
[energy-margin example](claims/STATADIC-EQUIVALENCE-001.md) gives a true-null
rejection probability of at least one quarter at a nominal level of one
twentieth. A two-one-sided Student test has different data, rejection rules and
assumptions. The [held-out selection bound](claims/STATADIC-SELECTION-001.md) requires independence of training and test evidence. Calibration of the same-sample selected-mixture procedure remains open.

## Spatial models

| STATADIC route | Result and assumptions | Reference calculation |
| --- | --- | --- |
| Spatial 2SLS | [Instrument projection and structural-residual orthogonality](claims/STATADIC-IV-001.md) require nonsingular instrument and projected-design Gram matrices. A [full-rank instrument can identify the wrong target](claims/STATADIC-IV-002.md) when exogeneity fails. | [PySAL GM_Lag](https://pysal.org/spreg/generated/spreg.GM_Lag.html) |
| Spatial-error GMM | [Sample moment polynomial](claims/STATADIC-GMM-001.md) for symmetric moment matrices and [zero population quadratic moment](claims/STATADIC-GMM-002.md) for zero-diagonal matrices with vanishing cross moments. [Global identification](claims/STATADIC-SPATIAL-004.md) requires excluding other admissible roots. [Two such roots](claims/STATADIC-SPATIAL-005.md) can coexist with local regularity. | [PySAL GM_Error_Het](https://pysal.org/spreg/generated/spreg.GM_Error_Het.html) |
| KKP panel error components | [Temporal projection and spatial lag](claims/STATADIC-KKP-001.md) commute on every panel exactly when periods mixed with nonzero weight have the same graph. | [PySAL GM_KKP](https://pysal.org/spreg/generated/spreg.GM_KKP.html) |
| Spatial panel likelihood | For a full roster, use the [block log-Jacobian](claims/STATADIC-SPATIAL-001.md). For missing outcomes, a [rectangular solve](claims/STATADIC-SPATIAL-002.md) selects the observed covariance from the full model. | [SciPy multivariate normal](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.multivariate_normal.html) with the explicitly selected covariance |
| Direct, indirect and total effects | The [impact averages](claims/STATADIC-SPATIAL-003.md) use the trace and average row sum of the reduced-form multiplier. Hold the graph fixed for a covariate perturbation. | [Spatialreg impact measures](https://r-spatial.github.io/spatialreg/reference/impacts.html) |

The spatial identities apply conditional on the supplied graph, including on a
graph returned by a learned-weight procedure. Identification of that procedure,
uncertainty from learning the graph, and consistency of its coefficient estimates
require their own assumptions and proofs.
