# Your first regression

After [installation](../start/installation.md), estimate a line from synthetic
observations. The complete script includes diagnostics and saved reports.

```{literalinclude} ../../examples/first_regression.py
:language: python
:start-after: "# tutorial-start"
:end-before: "def analyze"
```

```{literalinclude} ../../examples/first_regression.py
:language: python
:start-after: "output.mkdir(parents=True, exist_ok=True)"
:end-before: "    diagnostics ="
:dedent: 4
```

The result retains the training design. Request covariance and a reference law
explicitly before interpreting uncertainty:

```{literalinclude} ../../examples/first_regression.py
:language: python
:start-after: "print(diagnostics.report().to_text())"
:end-before: '    report.save_latex(output / "table.tex")'
:dedent: 4
```

{download}`Complete executable script <../../examples/first_regression.py>`

<!-- statadic:regression-result -->

The fit report describes numerical validity. HC3/normal intervals are asymptotic;
they do not establish causal identification. Continue through the
[complete OLS chapter](ols/index.md) for diagnostics, hypotheses, prediction and
reporting, or experiment in the [native marimo lab](../labs/regression.md).

<details><summary>Assumptions and evidence</summary>

{ref}`OLS mathematics and assumptions <formal-STATADIC-LINEAR-010>` ·
{ref}`Matched comparisons <reference-api.regression>` ·
{ref}`Monte Carlo scope <research-procedure-linear.covariance>`.
Each concerns its stated configuration; registration is not release qualification.

</details>
