# Spatial regression and static panels

Spatial fitting lives in `statadic.estimation.spatial` (`likelihood`, `iv`, `gmm`,
`panel`, `kkp`). Spatial uncertainty and diagnostics live in
`statadic.inference.spatial` (`covariance`, `diagnostics`). General static panels
remain in `statadic.estimation.panel` as estimator bricks over a `PanelProblem`
(`Pooled`, `Within`, `Between`, `FirstDifference`, `RandomEffects`,
`FamaMacBeth`), composed by `Regression(..., structure=PanelLayout(...),
estimator=...)`; their covariance
methods live in `statadic.inference.panel_covariance`. Graph and panel preparation belong to
`statadic.operators.spatial` and `statadic.operators.panel`. These operators supply
inputs to fitting without importing estimation or inference.

## Assumptions and evidence

<details><summary>Inspect assumptions, source review and validation</summary>

The [full-roster Jacobian](formal/claims/STATADIC-SPATIAL-001.md) and
[observed covariance](formal/claims/STATADIC-SPATIAL-002.md) results require an
invertible filter on the declared graph. [Two-way panel absorption](formal/claims/STATADIC-PANEL-002.md)
has a separate projection certificate. Claim pages expose exact assumptions,
source locators and human correspondence-review status. Conditioning on a graph
does not justify learning that graph from the same outcomes.

{ref}`Point fits <research-procedure-spatial.point-estimation>`,
{ref}`spatial panels <research-procedure-spatial.panel>` and
{ref}`covariance/impacts <research-procedure-spatial.covariance-impacts>` have
separate study mappings. The comparisons below establish matched numerical
behavior; [known limits](validation/known-limitations.md) retain adverse and
inconclusive calibration results. They are not overridden by solver convergence.

</details>

## Independent references

The tests exercise public fitting and inference operations. Parity always means
matched observations, graph, design, intercept, weight, normalization and
correction choices; matching an external implementation is not statistical
calibration. Optional references are selected explicitly:

```bash
uv sync --project src/python/packages/statadic --group reference
just python::statadic::test-fresh -m reference
```

Run the complete fast suite separately with `just python::statadic::test-fresh`.
Reference tests skip explicitly when an optional oracle is absent. A skipped
oracle is not evidence of parity. Package artifact checks install only the wheel's
runtime closure and verify that statsmodels, linearmodels, spreg, libpysal and
pandas are absent. JAX itself requires SciPy transitively; this does not make
the specialist econometrics libraries runtime dependencies.

| Public workflow | Independent comparison | Claim boundary |
| --- | --- | --- |
| OLS/WLS; classical, HC, HAC and cluster covariance; intervals/restrictions | statsmodels | Existing matched linear workflow |
| Affine GMM and actual-weight covariance | linearmodels | Existing IV/GMM conventions; weighting budget is explicit |
| Binomial-logit / Poisson-log | statsmodels / glum | Existing retained-response, likelihood and finite-optimum contracts |
| `SLXLeastSquares.estimate` | statsmodels OLS/HC1 and intervals | Selected exogenous WX columns |
| `ML.estimate` SAR/SDM, SEM/SDEM | spreg ML_Lag / ML_Error | Coefficients, scale, likelihood, expected information, predictions and shared inference |
| Combined lag/error and GNS likelihood | independently assembled NumPy/SciPy likelihood | Exact dense Gaussian model; deterministic multistart search is not a global proof |
| `SpatialIV.estimate` | spreg GM_Lag | Identified 2SLS and homoskedastic/White covariance; no coefficient clipping |
| `SpatialGMM.estimate` | spreg GM_Error_Hom/Het and GM_Combo_Hom/Het | Declared quadratic moments and iteration budget; retain actual coefficient-solve lambda |
| Residual Moran, LM/robust LM, AK, common-factor diagnostics | spreg public diagnostics | Each reference has model-specific null assumptions |
| General pooled/FE/RE/between/FD/Fama-MacBeth panels | linearmodels | Weighted and unbalanced complete samples; shared covariance, intervals and restrictions |
| General panel classical/HC/cluster/DK/AC/Fama-MacBeth HAC covariance | linearmodels | Explicit effects, debiasing, kernel and bandwidth conventions |
| Constant-W fixed-effects spatial panels | spreg Panel_FE_Lag / Panel_FE_Error | Coefficients and expected covariance; normalize the reported likelihood offset |
| Changing-W pooled/fixed/random spatial panels | independently assembled full joint GLS/marginal likelihood | Actual period filters and Jacobians; no constant-W shortcut |
| `KKP.estimate` | spreg GM_KKP | Constant-W component model; beta covariance is available from its actual transformed linear fit, not invented joint nuisance inference |

This table describes estimator families and matched scenarios. It is not a claim
that every option, graph, dtype, device or public STATADIC function has an external
equivalent. Existing distance, transport, resampling and hypothesis methods have
their own reference/property suites. Performance and statistical calibration
require evidence beyond software agreement.

### Fixed-graph Gaussian likelihood

For cross-sectional SAR and SEM, one spectrum of the supplied dense graph gives
the log-Jacobian at each scalar search point,
`sum(log(abs(1 - s * eigvals(W))))`. Complex eigenvalues retain their imaginary
parts. In SAR, a single QR projection residualizes both the outcome and its
spatial lag. The fit still evaluates the full likelihood at the selected point.

For a balanced panel with unit effects and a common graph, filtering and
time-demeaning commute. The scalar or joint search can therefore profile on the
demeaned arrays while retaining all T log-Jacobians and the NT variance divisor.
Demeaning the mean derivatives also gives the entity-effect Schur complement of
Gaussian expected information. Time-varying graphs retain their individual
period filters and nuisance-effect projection.

### Static-panel scientific corrections

Weighted random effects multiplies the entire quasi-demeaned response and design
by the square root of normalized relative precision:
`sqrt(w) * (data - theta * weighted_entity_mean)`. Earlier versions omitted the
root weight on the entity-mean term. Variance components use the actual ranks of
the auxiliary within and between projections, so time-invariant regressors and
rank-deficient entity means do not invalidate an otherwise identified RE model.
The optional Baltagi-Chang small-sample correction applies only to unbalanced
panels; its between regression weights entity means by observation counts and its
trace uses grouped `sqrt(w) * X`, with a pseudoinverse for redundant mean columns.
These changes intentionally alter fitted coefficients and their uncertainty.

`PanelFit.df_resid` exposes the transformed fit's residual degrees of freedom,
including absorbed effects. `FamaMacBethFit.df_resid` is the number of identified
cross sections minus one, for explicitly selected marginal Student-t inference.
Failed cross sections remain visible and do not contribute degrees of freedom.
Neither property chooses a reference distribution or a joint-test calibration.

RE `fitted_values` means `X beta`. Its `effects` are conditional-mean shrinkage
estimates under the retained variance-component convention; its original-row
`residuals` subtract those effects. These differ from linearmodels' effects
reconstructed from quasi-demeaned residuals. Reference comparisons therefore
match the transformed residuals separately and do not equate these two effects.

Autocorrelation covariance sums within-entity score products. It no longer gives
short entities the extra weight implied by averaging each entity's normalized
HAC estimate. The specified kernel bandwidth is retained for short entities.
STATADIC uses actual integer time distances, including gaps; linearmodels' AC path
uses observed within-entity row lags. External AC parity therefore uses panels
without interior gaps, while independent finite sums validate the actual-gap
contract. The reference environment requires linearmodels 7.1 or later.

## Data and inference contracts

- Include an intercept explicitly when the chosen model permits one. Host
  Regression with unit, time or two-way effects rejects `Intercept()`; the
  effects supply the location component. Select WX and instrumental columns
  explicitly; lagging a constant under row-stochastic W does not create an
  instrument. Full rank alone does not establish strong identification.
- Weight preparation preserves supplied edges and unit order. Normalization,
  isolates and diagonal policy are explicit. Dense graphs are never silently
  sparsified, and sparse likelihood input is never silently densified.
- Numeric unit keys and ordered period keys are retained as dynamic array leaves.
  Map external labels before preparation. A changing unit ordering must be
  explicitly reindexed before fitting a spatial panel.
- `PanelML` uses balanced, time-major design `(T,N,p)`, outcome `(T,N)`, and
  weights `(N,N)` or `(T,N,N)`. `ObservedPanelML` also supports changing
  populations and missing outcomes on retained graphs, with full node designs
  and observed-only outcomes. Graphs are conditioned on as exogenous; missing
  outcomes cannot be represented as observed zeros.
- Numerical problems own the process, effects and selected WX columns;
  estimator bricks own solver controls. Host graph-qualified names such as
  `rho[neighbors]`, `lambda[neighbors]` and `W[neighbors]:x` are the actual fitted
  parameter schema used by reports and named restrictions.
- The marginal spatial panel model is
  `A_t y_t = X_t beta + alpha + B_t^-1 epsilon_t`, with an unfiltered entity
  effect. Random effects use a Woodbury marginal likelihood. KKP instead
  spatially filters the entity component too; these are distinct specifications.
- General panels can be unbalanced. `prepare_panel_sample` makes missing-row,
  iterative singleton and absorbed-column removal explicit and retains original
  row/column indices. Fitters themselves keep complete input samples fixed.
- Native fixed-effects problems identify any retained intercept explicitly;
  that lower-level parameterization is separate from the host restriction. First
  differences require consecutive integer time keys and do not bridge gaps.
  Sort the oracle's global time axis explicitly when comparing unbalanced data.
- Fits retain solver, rank, support and availability separately. Failed or
  unidentified fits expose unavailable parameters. A random variance component
  on its boundary does not receive ordinary interior joint inference.
- Covariance methods consume actual fitted state; references and degrees of
  freedom remain explicit. Ordinary row HC is not a general spatial-error
  covariance method. Robust LM adjusts for a competing spatial specification,
  not arbitrary heteroskedasticity.

## Numerical and reference limits

The prespecified CPU float64 Gaussian SAR Wald study uses 10,000 independent
trials per fixed-design cell and counts unavailable fits as failures. At
`N=48, rho=0`, it observed 661 null rejections (6.61%) and 93.39% interval
coverage, with no unavailable fits; the one-sided exact size interval was
[6.2057%, 7.0331%]. This cell exceeds the declared 6% size/noncoverage limit.
At `N=96, rho=0.4`, it observed 611 rejections (6.11%) and 93.89% coverage,
again with no unavailable fits; size bounds [5.7205%, 6.5185%] make this cell
inconclusive. These normal/chi-square expected-information workflows are not
qualified for those finite-sample cells. Numerical agreement with spreg is a
separate result and does not establish finite-sample Wald calibration. The
complete executable scenarios and JSON/JUnit reports are produced by
`tests/qualification/test_panels.py`; no interval correction is silently applied.

Held-out numerical witnesses in `tests/qualification/test_sar_diagnostics.py`
check the declared directed graph and structural innovations, then compare eight
new trials per cell against a separate NumPy/SciPy Gaussian profile and spreg's
FULL likelihood. They also independently assemble the complete expected
information for beta, rho and sigma-squared, invert that joint matrix before
removing the scale nuisance, and check the public Wald statistic, p-value and
normal interval. Expected information is not the observed likelihood Hessian.
These witnesses agree with the retained implementation; optimizer error is
negligible relative to the coefficient's sampling standard error.

A separate descriptive diagnostic uses held-out trial IDs 10000–10255 in each
cell, with no unavailable, boundary or unconverged fits in this run:

| Cell | Bias in rho | Empirical SD | RMS reported SE | Mean fitted sigma-squared (truth 0.25) |
| --- | --- | --- | --- | --- |
| N=48, rho=0 | -0.01074 | 0.09098 | 0.08643 | 0.22784 |
| N=96, rho=0.4 | -0.00657 | 0.05093 | 0.04980 | 0.23920 |

Replacing only the fitted scale by its known DGP value in the independent
information calculation gives RMS standard errors 0.08992 and 0.05071. This
supports downward scale estimation as one contributor to the finite-sample
Wald discrepancy; it does not establish a unique cause or justify a new default.
For reproducibility, the diagnostic report also retains rejection counts for
the existing optional debiased covariance and an independent profile likelihood
ratio: default/debiased/LR counts are 13/11/13 and 15/13/12 out of 256. These are
mechanism controls, not qualification studies or adopted inference procedures.
All descriptive summaries explicitly condition on valid trials; requested,
valid and failed counts remain visible in the JUnit JSON property.

Run all numerical witnesses and the descriptive diagnostics with the reference
environment active:

```bash
UV_NO_SYNC=true just python::statadic::test-all tests/qualification/test_sar_diagnostics.py -n 0 --require-external-references
```

The JAX runtime is restricted to the qualified 0.11 series. Qualification of
0.12.0.1 failed existing multinomial-shape and byte-pinned random-stream gates;
upgrading needs an explicit scientific-stream migration. Neither sparse storage
nor the existence of a direct sparse solver supplies a scalable exact spatial
log-determinant or universal vmap support.

Native BCOO products and checked matrix-free GMRES support sparse SLX/IV and
spatial solves. Exact likelihoods, full trace diagnostics and quadratic-error
GMM currently require dense preparation. Sparse log-determinants and scalable
quadratic-moment trace preparation are separate work. Caller precision is retained;
no fitter changes global x64 configuration. Fixed-shape JIT/vmap and objective
derivatives are distinct from derivatives of a fitted solution.

spreg's fixed-effects panel `logll` retains a parameter-independent
`-NT/2 * log(NT)` offset because its profile uses log(SSE). STATADIC reports the full
Gaussian likelihood; reference tests normalize that offset and independently
check the joint likelihood. Its ORD path also classifies some reciprocal,
unequally weighted directed graphs by topology alone. Compare those graphs with
FULL/LU; the ORD complex-spectrum comparison uses nonreciprocal directed support.

Dynamic panels, regimes, SUR, automatic specification search and spatial GEL
estimators are outside this delivery. Static spatial panels explicitly include
time-varying exogenous W_t. See the [survival guide](survival-models.md) for
the distinct censored spatial log-time likelihood.

## Observed spatial panels and named lagged predictors

Use `Regression` from `statadic.api.regression` with `PanelData`,
`SpatialLag`, `SpatialError`, and `SpatialX` from `statadic.api.spatial.graphs`.
Pass `PanelData(nodes=nodes, outcomes=outcomes, panel=layout,
 graphs={"neighbors": graphs})` to retain full graph nodes and separately
observed outcomes. See [composition](reference/composition.md).

Live cross-sectional prediction accepts the named `SpatialData` input. Gaussian
spatial-panel and spatial-survival predictions retain period IDs, ordered node
IDs and one value array per period in `SpatialPanelPrediction`. Choose
`kind="structural"` or `"reduced-form"` explicitly when that distinction is
supported. Survival defaults to marginal medians and has its own prediction
targets. Frozen graph artifacts instead consume observations against the saved
graph realization; they do not replace it with a live graph registry.

Ordinary `PanelResult` fits, including SLX least squares, also accept `PanelData`.
They reuse training transformations, compute WX using the supplied full-node
covariates and named graph, then return values in the requested outcome-row order.
Omitted outcome-side predictor copies are allowed; conflicting copies are rejected.
`kind="conditional"` requires known fitted-effect keys. Supplying `Observations`
uses the retained graph realization. Native spatial-likelihood panel fits return
the graph-node carrier described above.

For graph construction from firm distances and article clouds, shared learned
parameters, and period-specific fields, see the
[learned panel graphs tutorial](tutorials/learned-panel-graphs.md).
