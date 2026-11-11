# 4. Predict and retain the analysis

Prediction evaluates the retained conditional-mean line at new rows. It does
not create prediction intervals, model a new error distribution or refit the
design.

```{literalinclude} ../../../examples/ols_analysis.py
:start-after: "# ols-predict-start"
:end-before: "# ols-predict-end"
:language: python
:dedent: 4
```

<!-- statadic:ols-result report -->

The callable returns actual fits, diagnostics, inference objects and reports.
`Report` is an immutable presentation snapshot: rendering text or LaTeX does
not recalculate inference. The CLI explicitly saves the HC3 normal-reference
report as the main analysis and retains separate model-based, HAC and diagnostic
snapshots. Plots show residuals and the selected coefficient intervals.

```{literalinclude} ../../../examples/ols_analysis.py
:start-after: "# ols-report-start"
:end-before: "# ols-report-end"
:language: python
```

A saved predictor applies training preparation to new inputs. A saved analysis
holds reports and plot specifications; it does **not** restore a live fit,
permit new covariance calculations or certify that assumptions hold. The example
checks reloaded predictions and report bytes. Keep the original data and analysis
choices when a reproducible scientific record requires them.

The lab's LaTeX download follows its selected covariance and reference. Changing
the dropdown changes that output; it does not silently replace the model.

**Workflow check:** the docs application's `test_reporting_workflow.py` checks
the shared callable, independent least-squares/HC3 quantities and artifact
round trips. That is integration and numerical evidence, not a Lean proof or
Monte Carlo coverage claim. Performance depends on workload and environment;
this teaching run makes no speed or memory superiority claim.

Return to the [OLS chapter](index.md) or explore
[other explicit compositions](../../compose/index.md).
