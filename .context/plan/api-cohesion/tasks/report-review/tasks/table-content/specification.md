# Fit statistics specification

Normative for `table-content`, `fit-labels`, `likelihood-statistics`,
`spatial-statistics` and `fit-docs`. The owner decided on 2026-11-11: R² must be
reported and say whether it is adjusted; the log likelihood, AIC, BIC and HQIC
should all be shown, with the original citations
[Akaike (1974)](../../../../../../reference/akaike-1974/README.md),
[Schwarz (1978)](../../../../../../reference/schwarz-1978/README.md) and
[Hannan and Quinn (1979)](../../../../../../reference/hannan-quinn-1979/README.md).

## Metrics

Retained as `Metric` records with empty definitions. A report note states the
conventions once (see Notes).

| Name | Type | Reported for |
| --- | --- | --- |
| `n_observations` | `int` | Every fit (exists) |
| `n_parameters` | `int` | Fits with a log likelihood |
| `r_squared` | `float` | Linear fits whose column space contains a constant |
| `adjusted_r_squared` | `float` | The same fits |
| `uncentred_r_squared` | `float` | Linear fits whose column space lacks a constant |
| `log_likelihood` | `float` | Fits with a log likelihood |
| `aic`, `bic`, `hqic` | `float` | Fits with a log likelihood |

Order after `n_observations`: `n_parameters`, the R² metrics, `log_likelihood`,
`aic`, `bic`, `hqic`.

## Definitions

- `L` is the maximised likelihood with every normalising constant included: the
  Gaussian `2π` term, count factorials, and the spatial Jacobian. A likelihood
  profiled over a scale at its maximum equals the full maximum, and that scale is
  still counted in `k`.
- `k` counts every parameter the likelihood estimates: identified coefficients
  (the design rank), plus every estimated scale, dispersion, shape and spatial
  parameter. A parameter fixed by the caller is not counted. Fixed effects count
  whether they are explicit design columns or removed by a within transformation,
  because the within likelihood is the full likelihood profiled over them, as a
  concentrated scale is; one model then has one AIC on every route. Random
  effects count through their variance parameters. One rule across families
  keeps criteria comparable; packages that omit the Gaussian scale report AIC
  smaller by two.
- `n` counts the observations entering the likelihood.
- AIC `= -2 log L + 2k`; BIC `= -2 log L + k log n`; HQIC `= -2 log L + 2k log log n`.
  HQIC is unavailable below three observations.
- A nonfinite log likelihood makes `log_likelihood` and the criteria unavailable
  with the family's reason; `n_parameters` stays available.

Linear fits (OLS and WLS, `api/regression/_linear.py`):

- Gaussian log likelihood at the maximum-likelihood scale:
  `log L = -(n/2)(log 2π + log(SSE_w/n) + 1) + ½ Σ log w_i`, where `w_i` are the
  analytic precision weights (all 1 for OLS) and `SSE_w` the weighted residual sum
  of squares. `k = rank + 1`.
- The column space contains a constant when appending a column of ones does not
  raise the numerical rank of the retained design; statsmodels detects implicit
  constants the same way. Fixed-effect designs without an explicit intercept
  therefore count as centred.
- Centred: `r_squared = 1 - SSE_w / Σ w_i (y_i - ȳ_w)²` with the weighted mean `ȳ_w`,
  and `adjusted_r_squared = 1 - (1 - r_squared)(n - 1)/(n - rank)`.
- Uncentred: `uncentred_r_squared = 1 - SSE_w / Σ w_i y_i²`; no adjusted value.

Families without a likelihood (IV and GMM, GEL, non-spatial panel within
estimators, geometric and permutation reports) report no likelihood metrics.

## Notes

Two fixed sentences, added when the corresponding metrics are present:

- `Information criteria use the maximised log likelihood and k estimated parameters, counting scale, dispersion, shape and spatial parameters: AIC (Akaike 1974), BIC (Schwarz 1978), HQIC (Hannan and Quinn 1979).`
- Centred: `R² is centred and unadjusted; adjusted R² uses n - 1 and n - rank degrees of freedom.`
  Uncentred: `R² is uncentred because the design spans no constant; no adjusted R² is reported.`
  For WLS append `Both use the precision weights.`

## Flow into reports

- `statadic.api.base.FitStatistics` is a frozen record with `metrics: tuple[Metric, ...] = ()`
  and `notes: tuple[str, ...] = ()`.
- `Result.fit_statistics() -> FitStatistics` returns an empty record by default.
  A family overrides it, and its `report()` appends the metrics after its existing
  metrics and the notes after its existing notes.
- `InferenceResult` gains `fit_statistics: FitStatistics = field(default=FitStatistics(), kw_only=True)`.
  Every `infer` of a family with fit statistics passes `fit_statistics=self.fit_statistics()`.
  `InferenceResult.report` appends the metrics after `n_observations` and the notes
  after its own notes. The regression table then carries coefficients, standard
  errors, tests, N, R² and the criteria together.
- `information_metrics(log_likelihood, n_parameters, n_observations, *, reason)`
  in `api/base.py` builds `n_parameters`, `log_likelihood`, `aic`, `bic` and `hqic`
  for every family, so the formulas exist once.

## Reference evidence

Each family compares its log likelihood exactly with an installed peer where one
exists, and its criteria through the formulas. State each peer's parameter count
and the resulting offset in the test, for example statsmodels OLS AIC plus 2.
Peers: statsmodels OLS, WLS, GLM and discrete models; lifelines AFT fitters;
spreg ML lag and error models. No R dependency.
