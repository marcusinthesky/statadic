# Residual diagnostics for OLS

`result.diagnose(*requests)` evaluates explicitly requested diagnostics against
an estimated OLS result. Estimation, diagnosis and inference stay separate: no
diagnostic refits the model, changes its estimates or selects a covariance.

```python
from statadic.api.diagnostics import (
    RESET,
    BreuschGodfrey,
    BreuschPagan,
    DurbinWatson,
    Influence,
    JarqueBera,
    White,
)
from statadic.api.inference import HC

from statadic.api.regression import Regression

fit = Regression("y", Intercept() + Linear("x")).estimate(data).require_valid()
diagnostics = fit.diagnose(JarqueBera(), BreuschPagan(), White(), RESET(), Influence())
print(diagnostics.report().to_text())
diagnostics["breusch_pagan"].pvalue  # one outcome by name
uncertainty = fit.infer(covariance=HC("HC3"))  # a separate, explicit choice
```

There is no "run everything" request. Each request names one procedure, its
options and its null hypothesis, and a result family accepts only the requests it
supports. Outcomes keep their order and unique names (pass `name=` to request
the same procedure twice). Unavailable diagnostics remain in the collection and
report with a reason.
Calibrated and statistic-only tests return `TestOutcome` with a structured
`Hypothesis`, reference, assumptions and construction metadata; influence remains
descriptive. See [hypotheses](reference/hypotheses.md).

## Procedures

| Request | Null hypothesis | Statistic and reference | Sources |
| --- | --- | --- | --- |
| `JarqueBera()` | Errors have skewness 0 and kurtosis 3 | `n/6 (S² + (K-3)²/4)` from biased residual moments; chi-square(2) | {cite:p}`jarque-bera-1987` |
| `BreuschPagan()` | Error variance does not depend on the variance design | Studentized `n R²` of `e²` on the design; chi-square(rank-1) | {cite:p}`breusch-pagan-1979`, {cite:p}`koenker-1981` |
| `BreuschPagan(studentized=False)` | As above, under Gaussian errors | Half the explained sum of squares of `e²/mean(e²)`; chi-square(rank-1) | {cite:p}`breusch-pagan-1979` |
| `BreuschPagan(variables=("z",))` | As above, with a constant and named training columns | Studentized `n R²`; chi-square(rank-1) | {cite:p}`koenker-1981` |
| `White()` | Variance does not depend on regressors, squares or cross-products | Studentized `n R²` on all products `x_j x_k`; chi-square(rank-1) | {cite:p}`white-1980` |
| `DurbinWatson(order_by="time")` | No first-order autocorrelation in the declared order | `Σ(e_t - e_{t-1})² / Σ e_t²`, in `[0, 4]`; no p-value | {cite:p}`durbin-watson-1950` |
| `BreuschGodfrey(lags=L, order_by="time")` | No autocorrelation at lags 1 to L in the declared order | `n R²` of `e` on the design and zero-filled lags; chi-square(L) | {cite:p}`breusch-1978`, {cite:p}`godfrey-1978` |
| `RESET(powers=(2, 3), reference="chi2")` | Added powers of the fitted values have zero coefficients | Classical Wald statistic of the augmented fit; chi-square(q) or `reference="F"` for F(q, n-k) | {cite:p}`ramsey-1969` |
| `Influence()` | None: a descriptive sensitivity | Leverage, internal and external studentized residuals, Cook's distance, DFFITS, DFBETAS | {cite:p}`cook-1977`, {cite:p}`belsley-kuh-welsch-1980` |

White and RESET auxiliary regressions count identified added columns, so
redundant columns can reduce their degrees of freedom. Breusch–Godfrey instead
requires all L requested lag restrictions to be identified; it remains
unavailable if fewer than L columns are added. Its pre-sample residuals are zero
and all n original rows enter the auxiliary regression. Trimming the first L
rows defines a different procedure. `statadic.inference.linear_diagnostics`
owns the JAX kernels; they take a `LinearFit` and support `jit` and `vmap`.

## When a diagnostic is unavailable

Residual tests require a valid, identified fit with positive residual degrees of
freedom and a non-zero residual sum of squares. Influence quantities use their
own masks, described below. Unavailable outcomes report one of these reasons:

| Reason | Meaning |
| --- | --- |
| `invalid-fit` | The point fit itself is unavailable |
| `weighted-fit` | Relative precisions differ, so the fit is WLS; these residual tests are defined for OLS |
| `no-residual-degrees-of-freedom` | The fit is saturated |
| `variance-design-lacks-constant-or-variation` | Breusch-Pagan needs a constant and one varying regressor |
| `design-lacks-constant-or-variation` | White and Breusch-Godfrey need a constant in the mean design |
| `auxiliary-lag-design-unavailable` | The requested residual-lag design is unavailable; inspect rank and available rows |
| `added-powers-not-identified` | The fitted-value powers add no identified column |
| `linear-influence-ineligible` | The full fit is ineligible for residual diagnostics; inspect each quantity's validity mask separately |

Influence quantities have separate masks: `leverage_valid`,
`studentized_internal_valid`, `studentized_external_valid`, `cooks_distance_valid`,
`dffits_valid` and `dfbetas_valid`. The first five have one entry per row; DFBETAS
has one entry per row and coefficient. A deleted fit with unresolved zero variance
makes that row's external studentization, DFFITS and DFBETAS unavailable. Other
rows and full-fit Cook's distance remain available when their conditions hold.
An exact fit can still report leverage. Unavailable numeric entries are `NaN`;
the maximum-leverage and Cook metrics use their own masks.

## Reading the results

- **Not evidence for the null.** Failing to reject is not evidence that an
  assumption holds; each test has power only against its stated alternatives.
- **Normality is not a Gauss-Markov condition.** OLS is BLUE without Gaussian
  errors; normality matters for exact finite-sample t and F references.
- **Robust covariance is not a test.** `HC("HC3")` or `HAC(...)` changes the
  inference; it does not test for heteroskedasticity or autocorrelation.
- **Residuals are not errors.** Exogeneity cannot be checked from OLS residuals.
- **Declare ordering.** BG, DW and `HAC` require `order_by="time"` or explicit
  `order_by=None` to acknowledge retained order. A named finite, unique numeric
  key sorts ascending without changing the fit or row identities. Ties raise;
  lags count adjacent ordered observations, with no calendar or grouped inference.
  Sorting does not establish equally spaced sampling or dependence assumptions.
- **Influence is not invalidity.** A large Cook's distance marks sensitivity of
  the estimate to one row, not a wrong observation.

## Evidence

**Reference parity.** `tests/inference/test_linear_diagnostics_reference.py`
compares statistics and p-values with statsmodels'
`het_breuschpagan`, `het_white`, `acorr_breusch_godfrey`, `linear_reset`,
`jarque_bera`, `durbin_watson` and `OLSInfluence`. The comparison includes a
design with an indicator, where White's auxiliary design is rank deficient.
Ordered BG/DW and OLS/WLS Bartlett HAC have separate shuffled-row controls; BG's
ordered control also asserts the lag degrees of freedom and full auxiliary
sample count against the peer. Auxiliary composition tests are internal
consistency checks, distinct from these independent peers.
Independent NumPy witnesses in `tests/inference/test_linear_diagnostics.py`
recompute each formula, and the influence measures are checked against
explicit leave-one-out refits.

**Retained historical size under a Gaussian null.** A prespecified study in
`tests/qualification/test_linear_diagnostics.py` uses 200 rows, an intercept and
two fixed covariates, a linear mean and standard Gaussian errors, with 10,000
trials. Failed trials count as rejections. Size at 5% is qualified when the
one-sided 95% upper bound is at most 6%. All 10,000 trials were valid in float32
and float64, with identical counts. These retained results do not qualify a
new wheel or an untested ordering/data regime:

| Procedure | Rejections | Rate | 95% upper bound | Disposition |
| --- | --- | --- | --- | --- |
| Jarque-Bera | 425 | 4.25% | 4.60% | qualified |
| Breusch-Pagan, studentized | 465 | 4.65% | 5.01% | qualified |
| Breusch-Pagan, original | 449 | 4.49% | 4.85% | qualified |
| White | 490 | 4.90% | 5.27% | qualified |
| Breusch-Godfrey, 1 lag | 482 | 4.82% | 5.19% | qualified |
| Breusch-Godfrey, 4 lags | 478 | 4.78% | 5.15% | qualified |
| RESET, chi-square | 484 | 4.84% | 5.21% | qualified |
| RESET, F | 468 | 4.68% | 5.04% | qualified |

The study does not qualify power, other sample sizes, non-Gaussian or
heteroskedastic nulls, or Durbin-Watson, which has no calibrated p-value.

**Mathematics.** Lean proves the Durbin-Watson range
({doc}`formal/claims/STATADIC-DW-001`), the Jarque-Bera zero set
({doc}`formal/claims/STATADIC-JB-001`) and the deleted-row studentization identity
used for external studentized residuals ({doc}`formal/claims/STATADIC-INFLUENCE-001`).
These are finite identities. The asymptotic chi-square calibration of the tests
remains an open obligation, `linear-diagnostics-002`.

## Not yet supported

Ljung-Box, Goldfeld-Quandt, the D'Agostino omnibus test and White's two-moment
specification test are not implemented. Panel, GLM and spatial results do not
accept these OLS requests. Spatial results have their own `diagnose` requests,
described with the [spatial models](spatial-models.md). A request a family does
not support raises a `TypeError` instead of applying an OLS calibration.
