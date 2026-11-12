---
type: reference
description: Render, export and check the Report object that carries computed parameters, tests, metrics and plots.
---

# Report

## Usage

```python
from pathlib import Path

from statadic.api.design import Intercept, Linear
from statadic.api.inference import HC
from statadic.api.regression import Regression

data = {"x": [0.0, 1.0, 2.0, 3.0, 4.0, 5.0], "y": [1.1, 1.8, 3.2, 3.7, 5.3, 5.8]}
fit = Regression.ols("y", Intercept() + Linear("x")).estimate(data).require_valid()
report = fit.infer(covariance=HC("HC3")).report(reference="normal", null=0)
report.require_available("interval")

output = Path("results")
output.mkdir(exist_ok=True)
report.save_latex(output / "table.tex", decimals=3, labels={"x": "Slope"})
report.save_values(output / "values.tex", prefix="ols")
report.coefficient_plot().save_pgfplots(output / "figure.tex", width_cm=12, height_cm=7)
print(report.to_text())
```

## Description

{py:class}`~statadic.api.reporting.Report` holds quantities that are already
computed: parameter rows, test rows, metrics, warnings, notes, plots and
immutable JSON-compatible metadata. It never chooses a model, a covariance or a
reference law, so every renderer formats the same stored values without
rerunning estimation. The result that produces a report builds it eagerly, in one step.

| Producer | Report contents |
| --- | --- |
| `fit.report()` | The point fit, its fit statistics and numerical availability |
| `fit.infer(...).report(reference=...)` | Uncertainty under the covariance and reference you selected |
| `inference.report(reference=..., null=...)` | The above plus a statistic and two-sided p-value per tested coefficient |
| `fit.diagnose(...).report()` and `outcome.report()` | One test row per diagnostic or hypothesis test, with its reference law |

{py:class}`~statadic.api.reporting.ParameterRow` stores the estimate and its
uncertainty for one parameter, and {py:class}`~statadic.api.reporting.Metric`
stores a named quantity with its definition. Both carry an availability status.
A missing value comes with a reason, so a valid point estimate never implies
that an interval or a test exists for it. For a parameter row,
`ParameterRow.interval_available` is true when both endpoints are present and
the status is absent or `"available"`.
{py:class}`~statadic.api.reporting.TestRow` holds a test name, statistic,
reference law, p-value and status. For a report, `Report.metadata` records the
procedure, covariance, reference and confidence level used in its calculation.

## Options

| Operation | Output |
| --- | --- |
| `report.to_text()` | Plain-text snapshot of every section |
| `report.to_html()` | HTML with escaped labels, and a bounded preview in notebooks |
| `report.to_latex(...)` | A booktabs table with siunitx `S` columns |
| `report.save_latex(path, ...)` | Writes exactly `to_latex(...)` as UTF-8 |
| `report.coefficient_plot()` | A {py:class}`~statadic.api.reporting.PlotSpec` of intervals, when every row has an estimate and both bounds |
| `plot.to_pgfplots()`, `plot.save_pgfplots(path)` | PGFplots source, sized by `width_cm` and `height_cm` |
| `report.to_values(prefix=...)`, `report.save_values(path, prefix=...)` | TeX lines of the form `\jcDeclareValue{key}{value}` |
| `report.require_available(quantity, parameters=None)` | Returns the report, or raises `ValueError` naming the first missing value |

`to_latex` and `save_latex` accept `digits` (significant digits, default 6),
`decimals` (fixed places), `labels` (display names for parameters and metrics),
`label_markup` (`"plain"` or `"tex"`), `notes` (`False` drops the trailing
notes) and `metrics` (footer rows, in the order given). Counts print exactly.
A table needs `booktabs` and `siunitx` in the preamble of your manuscript, and a
figure needs `pgfplots` with `\pgfplotsset{compat=1.18}`. For a manuscript, include a saved table
with `\input` inside a `table` float, where your manuscript owns the caption and label.

`quantity` is one of `"estimate"`, `"std_error"`, `"interval"`, `"statistic"` or
`"pvalue"`. Value keys are lowercased names joined by `-`. A parameter gives
`{prefix}-{name}-{q}` for `estimate`, `se`, `statistic`, `p`, `lower` and
`upper`, and a metric gives `{prefix}-{name}`. A test row gives `-statistic` and
`-p`, plus `-df` or `-df1` and `-df2`. The `prefix` must match
`^[a-z][a-z0-9-]*$`, and a duplicate key raises `ValueError`.

## Results

With `null`, each tested row holds `(estimate - null) / std_error` and the
two-sided p-value of the single-restriction Wald test in
`InferenceResult.test`. The `"normal"` reference gives chi-square(1) and the
`"t"` reference gives F(1, df). A float `null` applies to every coefficient and a
mapping tests the named ones. Without `null`, a report shows no statistic or p-value.
In reduced precision the effective null can differ from the requested one, so
the metadata records it as `effective_null` and a warning names each affected row.

Likelihood fits add fit statistics to the report, and `fit.fit_statistics()`
returns them as {py:class}`~statadic.api.fit_statistics.FitStatistics`.

| Metric name | Meaning |
| --- | --- |
| `n_observations`, `n_parameters` | Rows used and $k$ |
| `log_likelihood` | $\log L$ at the estimate |
| `aic`, `bic`, `hqic` | $-2\log L + 2k$, $-2\log L + k\log n$ and $-2\log L + 2k\log\log n$ |
| `r_squared`, `adjusted_r_squared` | OLS and WLS with a constant in the design |
| `uncentred_r_squared` | OLS and WLS without a constant |

Criteria appear for OLS, WLS, GLM, gamma, NB2, probit, hurdle, survival, spatial
likelihood and spatial panel fits, and $R^2$ appears only for least squares. IV/GMM,
GEL, non-spatial panel within fits and permutation reports have no likelihood
and report none. `report.save_values` exports these as keys such as `ols-aic`.

## Methods and formulas

$R^2$ is centred when the design spans a constant, either an intercept or
fixed-effect indicators. Without a constant, only the uncentred value is
reported, and the two are not comparable. At exactly zero residual sum of
squares $R^2$ is 1, but the Gaussian likelihood and criteria are unavailable.

$k$ counts every estimated parameter: the identified coefficients (the design
rank), each estimated scale, dispersion, shape or spatial parameter, learned
graph coordinates, and fixed effects whether explicit or removed by a within
transformation. A random effect counts its variance parameters. Fixed
parameters are not counted, so one model has one AIC on every route.

statsmodels OLS and WLS count the rank and no scale. Relative to statsmodels,
STATADIC AIC is larger by 2, BIC by $\log n$ and HQIC by $2\log\log n$.
statsmodels GLM, discrete NB2 and Probit, and lifelines `AIC_` agree with
STATADIC. AIC {cite:p}`akaike-1974`, BIC {cite:p}`schwarz-1978` and HQIC
{cite:p}`hannan-quinn-1979` compare only models of one response on the same rows.

## Assumptions and limits

- A p-value measures the data against the stated null under the stated
  reference law. It does not establish that law, the model or the coverage of intervals.
- Coefficient intervals are not prediction intervals for future observations.
- `require_available` checks that values exist. It does not repair missing
  uncertainty or establish coverage.
- Metadata names the procedure and its assumptions without approving them.
- `HQIC` needs at least three rows. A nonfinite log likelihood makes it and the
  criteria unavailable. $R^2$ needs variation in the response about its weighted
  mean, and adjusted $R^2$ needs $n$ above the rank.
- The NB2 log likelihood differs from statsmodels by about 3e-6, and its
  criteria inherit that difference.
- `save_values` omits quantities never requested, such as a p-value without a
  null. It lists unavailable values as comments, so citing one fails the TeX
  compile with its key named.

## Evidence

- Criteria and $R^2$ are matched against statsmodels and lifelines in the
  {ref}`reference index <reference-api.regression>`.
- Evidence for the method behind each report lives on its reference page, such as
  [regression](regression.md).

## See also

- [Saved artifacts](artifacts.md) for `save_analysis` and `load_analysis`
- [OLS reporting chapter](../tutorials/ols/report.py) and
  [save and share results](../guides/save-and-share-results.py)
- [Regression](regression.md) and [hypotheses](hypotheses.md)
