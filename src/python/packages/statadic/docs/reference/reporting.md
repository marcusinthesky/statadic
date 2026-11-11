# Report: retain and render an analysis

{py:class}`~statadic.api.reporting.Report` contains already-computed parameter rows,
metrics, warnings, notes, plots and immutable JSON-compatible metadata. It does
not choose a model, covariance or reference law. Host report generation is eager.

| Producer | Report contents |
| --- | --- |
| `fit.report()` | Point-fit information and numerical availability |
| `fit.infer(...).report(reference=...)` | Supported uncertainty under the explicitly selected covariance and reference |
| `fit.diagnose(...).report()` | Requested diagnostics, hypotheses, assumptions and availability reasons |
| `outcome.report()` | One hypothesis-test statistic and its calibrated reference, when available |
| `inference.report(reference=..., null=...)` | The above plus a statistic and two-sided p-value per tested coefficient |

The [OLS chapter](../tutorials/ols/report.md) demonstrates the complete reporting
workflow. Use [save and share results](../guides/reporting.md) for a short recipe.

## Rows and availability

{py:class}`~statadic.api.reporting.ParameterRow` retains estimates and supported
uncertainty by parameter. {py:class}`~statadic.api.reporting.Metric` retains a named
quantity with its definition and availability. An absent value has an explicit
reason; a valid point estimate does not imply available intervals or tests.
`ParameterRow.interval_available` is true when both endpoints are present and the
interval status is absent or `"available"`. An explicitly available interval
requires both endpoints and no unavailable reason. An unavailable interval carries no
endpoints and a reason, and an unavailable metric carries no value. A row may keep
an estimate with `valid=False` and no standard error.

`report.require_available("interval")` explicitly requires intervals for the
selected parameter rows. It does not repair missing uncertainty or establish
coverage. Metadata records the procedure and assumptions, not their approval.

## Tests

{py:class}`~statadic.api.reporting.TestRow` retains a test name, statistic, reference
law, p-value and availability. `Report.tests` holds one row per test and renders
as a Test, Statistic, Reference and $p$ table. `TestOutcome.report()` and
`Diagnostics.report()` fill it, as do the GEL, PERMANOVA/PERMDISP, Mantel and
geometric permutation reports. `TestOutcome.test_row()` returns the row and
`TestOutcome.components()` returns the other values, such as ranks, as metrics.

`InferenceResult.report(..., null=)` is opt-in: a float applies to every
coefficient and a mapping tests the named ones. It needs a `reference`. A tested
row keeps `(estimate - null) / std_error` and the two-sided p-value of the
single-restriction Wald test of `InferenceResult.test`: chi-square(1) for the
normal reference, F(1, df) for the t reference. Without `null` no statistic or
p-value is reported. A p-value is the stated null under the stated reference law.
It does not establish that law, the model or coverage.

The null is converted to the fit's dtype before testing. In reduced precision
(for example float32) the effective null can differ from the requested one; the
test uses the effective value, the report metadata records it as `effective_null`,
and a warning names each affected row. The statistic is the signed square root of
the Wald quadratic, so its square matches the Wald statistic of
`InferenceResult.test`.

## Fit statistics

Each likelihood fit computes its statistics once. `fit.fit_statistics()` returns
a {py:class}`~statadic.api.fit_statistics.FitStatistics` holding the metrics and two
fixed notes that state the conventions and citations. `fit.report()` shows them,
and `fit.infer(...)` passes them to the inference report.

`statadic.api.fit_statistics` owns the names. A family supplies a frozen
{py:class}`~statadic.api.fit_statistics.LikelihoodSummary` (log likelihood, $k$, $n$
and the reason used when the log likelihood is not finite) through
`likelihood_summary()`, and `FitStatistics.from_likelihood` builds the metrics
once. The Protocols {py:class}`~statadic.api.capabilities.SummarizesLikelihood`
(`likelihood_summary()`) and {py:class}`~statadic.api.capabilities.ReportsFitStatistics`
(`fit_statistics()`) state these capabilities. A spatial fit exposes its maximised
value as `log_likelihood`.

| Metric name | Label | Meaning |
| --- | --- | --- |
| `n_observations` | Observations | Rows used |
| `n_parameters` | Parameters | $k$, every estimated parameter |
| `log_likelihood` | Log likelihood | $\log L$ at the estimate |
| `aic` | AIC | $-2\log L + 2k$ |
| `bic` | BIC | $-2\log L + k\log n$ |
| `hqic` | HQIC | $-2\log L + 2k\log\log n$; unavailable below three rows |
| `r_squared` | $R^2$ | OLS/WLS with a constant in the design column space; 1 at an exact fit |
| `adjusted_r_squared` | Adjusted $R^2$ | As `r_squared` |
| `uncentred_r_squared` | Uncentred $R^2$ | OLS/WLS without a constant; no adjusted value |

A nonfinite log likelihood makes it and the criteria unavailable, with a reason.
{py:func}`~statadic.api.fit_statistics.information_metrics` builds the four
likelihood metrics.

At exactly zero residual sum of squares, a least-squares fit's Gaussian log
likelihood and criteria are unavailable, while $R^2$ is 1. A tiny positive
residual sum of squares retains a finite likelihood; no near-zero cutoff is
applied. $R^2$ is unavailable
when the response has no variation about its weighted mean, Uncentred $R^2$ when
the response is zero in every row, and Adjusted $R^2$ when $n \le$ rank.
See [save and share results](../guides/reporting.md) for how $k$ is counted, the
statsmodels offset and the citations. Criteria compare only models of the same
response on the same rows. Known limit: the NB2 log likelihood differs from
statsmodels by about 3e-6, and its criteria inherit that.

| Family | Statistics |
| --- | --- |
| OLS, WLS, SLX least squares | Criteria and $R^2$ |
| GLM, gamma, NB2, probit, hurdle (gate plus count) | Criteria |
| Survival AFT, spatial survival | Criteria |
| Spatial ML cross-section, observed and graph regression | Criteria |
| Spatial panels | Criteria only |
| IV/GMM, GEL, non-spatial panel within, permutation reports | None |

## Value export

| Operation | Output |
| --- | --- |
| `report.to_values(prefix=...)` | TeX text of `\jcDeclareValue{key}{value}` lines |
| `report.save_values(path, prefix=...)` | Writes exactly `to_values(...)` as UTF-8 |

`prefix` matches `^[a-z][a-z0-9-]*$`; `digits` and `decimals` follow `to_latex`.
Keys use retained names, lowercased with runs of other characters turned into
`-`. A parameter gives `{prefix}-{name}-{q}` for `estimate`, `se`, `statistic`,
`p`, `lower` and `upper`; a metric gives `{prefix}-{name}`; a test row gives
`{prefix}-{name}-statistic` and `-p`, plus `-df` (chi-square) or `-df1` and
`-df2` (F). A duplicate key raises `ValueError`. P-values carry their relation
(`= 0.497`, `< 0.001`). The file defines `\jcDeclareValue`, `\jcvalue` and
`\jcnum` with `\providecommand`; input it after `siunitx`. An unavailable value
is not declared and is listed as a comment, so citing it fails the compile with
the key named. Quantities never requested, such as a p-value without a null, are
omitted silently.

## Rendering and plots

| Operation | Output |
| --- | --- |
| `report.to_text()` | Complete plain-text snapshot |
| `report.to_html()` | Complete HTML representation |
| `report.to_latex()` | booktabs table with siunitx `S` columns |
| `report.save_latex(path)` | Writes exactly `to_latex(...)` as UTF-8 |
| `report.coefficient_plot()` | A retained interval plot when all selected rows have estimates and bounds |
| `plot.to_pgfplots()` | PGFplots source for a retained {py:class}`~statadic.api.reporting.PlotSpec` |
| `plot.save_pgfplots(path)` | Writes the PGFplots source to a file; `width_cm` and `height_cm` set the size |

A table needs `booktabs` and `siunitx` in the manuscript preamble. A figure needs
`pgfplots` with `\pgfplotsset{compat=1.18}`. `to_latex` and `save_latex` take
`digits` (significant digits, default 6), `decimals` (fixed decimal places),
`labels` (display labels for parameter and metric names), `label_markup`
(`"plain"` or `"tex"`), `notes` (`False` omits the trailing notes) and `metrics`
(the metric names shown in the footer, in the order given; the default shows
every retained metric). A Normal reference with parameters renders as
$N(\mu, \sigma^2)$ with each parameter rounded to `digits` significant digits,
for example `$N(\num{-0.0001}, \num{0.001}^2)$`; `(0, 1)` renders as `$N(0, 1)$`. Counts are
kept as integers and print exactly; an inference report carries an
`n_observations` metric as a footer row. Formatting never changes retained
values. Include a saved table with `\input` inside a `table` float whose caption
and label the manuscript owns.

Notebook display uses a bounded preview. Rendering formats retained values; it
does not rerun estimation or infer missing quantities. Coefficient intervals are
not prediction intervals for future observations.

## Persistence and evidence

`save_analysis(path, report)` and `load_analysis(path).report` preserve a report
for rendering. They restore neither a fit nor executable inference. Predictor
artifacts have separate, family-specific capabilities; see [saved artifacts](artifacts.md).

A source key or assumption in report metadata is not a source-fidelity review,
proof of implementation or statistical qualification. Consult the exact
[independent reference mappings](../validation/references.md),
[study dispositions](../validation/research.md) and
[formal claim assumptions](../formal/index.md) for the method being reported.
