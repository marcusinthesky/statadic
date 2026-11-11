# Save and share results

Start with an already-computed `report` from the analysis you intend to share.
Choose its covariance and reference before this step; see
[inference choices](inference.md).

## Export a retained report

```python
from pathlib import Path
from statadic.api.artifacts import save_analysis

output = Path("results")
output.mkdir(exist_ok=True)
report.save_latex(output / "table.tex")
(output / "report.html").write_text(report.to_html(), encoding="utf-8")
save_analysis(output / "analysis", report)
```

These are continuation fragments. The [OLS chapter](../tutorials/ols/report.md)
provides a complete executable example, including plots and output checks.

## Use the table in a manuscript

A saved table needs `booktabs` and `siunitx` in the preamble. A saved figure
needs `pgfplots` with `\pgfplotsset{compat=1.18}`.

```latex
\begin{table}
  \centering
  \caption{Slope estimate with HC3 intervals.}
  \label{tab:slope}
  \input{results/table}
\end{table}
```

The manuscript owns the caption and label. Save a figure with
`plot.save_pgfplots(output / "figure.tex", width_cm=12, height_cm=7)` and
include it the same way.

Options for `to_latex` and `save_latex`: `digits` sets significant digits
(default 6), `decimals` fixes decimal places instead, `labels` maps parameter
and metric names to display labels, `label_markup="tex"` passes labels as TeX
instead of escaping them, and `notes=False` omits the trailing notes. Counts
print exactly. Formatting never changes retained values.

```python
report.save_latex(output / "table.tex", decimals=3, labels={"x": "Slope"})
```

## Test coefficients

A coefficient test is opt-in. Pass a null value and a reference:

```python
tested = inference.report(reference="normal", null=0)
tested.save_latex(output / "table-tests.tex")
```

`null` is a float for every coefficient, or a mapping that names the tested
ones, such as `null={"x": 1.0}`. It needs a `reference`; the `"t"` reference
also needs `degrees_of_freedom`. Each tested row shows
`(estimate - null) / std_error` under a `$z$` or `$t$` header, with a two-sided
p-value equal to `InferenceResult.test` for that single restriction. Without
`null`, the table has no statistic or p-value column.

The p-value states the null under the stated reference law. It does not
establish the law, the model or interval coverage.

## Tabulate tests

`diagnostics.report()` and `outcome.report()` retain one `TestRow` per test:
statistic, reference law and p-value. They render as a Test, Statistic,
Reference and $p$ table.

```python
diagnostics.report().save_latex(output / "diagnostics-table.tex")
```

A test with no reference law, such as Durbin-Watson, prints `---` for the
reference and p-value.

## Cite values in text

`save_values` writes a TeX file of named values for the prose.

```python
tested.save_values(output / "values.tex", prefix="ols")
```

```latex
\usepackage{siunitx}
\input{results/values}
...
The slope is \jcnum{ols-x-estimate} (SE \jcnum{ols-x-se}),
$p \jcvalue{ols-x-p}$, from \jcnum{ols-n-observations} observations.
```

Input the file after `siunitx`. `\jcnum` prints a number with a true minus and
digit grouping. A p-value keeps its relation, so `$p \jcvalue{ols-x-p}$` reads
"p = 0.497" or "p < 0.001". A value that is unavailable is not declared and is
listed in a comment. Citing it fails the compile and names the key. Call
`report.require_available(...)` before export to require values.

## Report a regression table

`fit.infer(covariance=...).report(reference=..., null=0)` gives a full table:
coefficients, standard errors, $z$ or $t$, p-values, intervals, then the fit
statistics.

```python
table = fit.infer(covariance=HC(3)).report(reference="normal", null=0)
table.save_latex(output / "table-fit.tex")
```

The footer shows Observations, Parameters, $R^2$, Adjusted $R^2$, Log
likelihood, AIC, BIC and HQIC. Pass `labels=` to rename them, and
`metrics=["n_observations", "aic"]` to choose which footer rows appear and in
what order.

Conventions differ across packages and disciplines, so read these four points.

- **Which $R^2$.** OLS and WLS report $R^2$ and Adjusted $R^2$ when the design
  contains a constant, either an explicit intercept or fixed-effect indicators.
  Without one they report Uncentred $R^2$ only, with no adjusted value. The
  two are not comparable. An exact fit has $R^2 = 1$ but no log likelihood or
  criteria, because the residual sum of squares is zero.
- **How $k$ is counted.** $k$ is every estimated parameter: the coefficients
  that are identified (the design rank), plus each estimated scale, dispersion,
  shape or spatial parameter, learned graph coordinates, and fixed effects
  whether explicit or removed by a within transformation. Parameters you fix
  are not counted. A random effect counts its variance parameters. One model
  therefore has one AIC on every route.
- **Offset from statsmodels.** statsmodels OLS and WLS count the rank and no
  scale. statadic AIC equals the statsmodels value plus 2, BIC plus $\log n$, and
  HQIC plus $2 \log\log n$. statsmodels GLM, discrete NB2 and Probit, and
  lifelines `AIC_` agree with statadic.
- **What the criteria compare.** AIC {cite:p}`akaike-1974`, BIC
  {cite:p}`schwarz-1978` and HQIC {cite:p}`hannan-quinn-1979` compare models of
  the same response fitted to the same rows. A Gaussian density and a Poisson
  probability are not on one scale, so do not compare their criteria.

IV/GMM, GEL, non-spatial panel within estimators and permutation reports have
no likelihood and report no criteria. `save_values` exports the footer as keys
such as `ols-aic` and `ols-r-squared`. The [reporting reference](../reference/reporting.md)
lists every name and family.

## Fit the page width

A coefficient table with tests and intervals has seven columns. At four digits
it overflows a single-column `article` by about 29pt. Inside the float, put
`\small` and `\setlength{\tabcolsep}{4pt}` before `\input`, or lower `digits`.
The manuscript owns this layout.

## Reload for rendering

```python
from statadic.api.artifacts import load_analysis

snapshot = load_analysis(output / "analysis")
print(snapshot.report.to_text())
```

The snapshot restores computed values and interpretation, without restoring a
model or uncertainty calculation. Loading saved TeX markup requires explicit
trust as described in the [artifact contract](../reference/artifacts.md).

## Preserve prediction separately

For a fitted result with that capability, `fit.save_predictor(output / "predictor")`
saves its supported frozen prediction state. Load it with the family-specific
artifact loader. Prediction artifacts do not restore covariance or refitting.
Keep the analysis script, environment and data identity when reproducibility
requires repeating the computation.
