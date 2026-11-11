# Choose covariance and reference separately

A fitted model answers an estimation problem. Uncertainty additionally needs a covariance construction and a reference distribution. STATADIC keeps these choices visible instead of inferring them from a shared method name.

## Linear-regression choices

| Choice | Interpretation and required care |
|---|---|
| `ModelBased()` | Classical covariance under the model's variance assumptions; WLS uses relative precision, not survey/frequency weights |
| `HC("HC0")` / `HC("HC1")` | Heteroskedasticity-robust row-influence covariance; HC1 applies its documented count correction |
| `HC("HC2")` / `HC("HC3")` | Leverage-adjusted covariance; HC3 uses a squared leverage denominator and requires the supported leverage domain |
| `HAC(bandwidth, order_by="time")` | Bartlett covariance in ascending unique numeric-key order; explicit `order_by=None` acknowledges retained order |
| `Clustered(column, ...)` | One-way cluster aggregation; label alignment, occupied groups and finite-sample correction matter |

This is a linear-family table, not a list of options every model accepts. Poisson does not acquire HC2/HC3 merely because its result also has `.infer()`.

## Select the reference deliberately

For a retained host linear result, a classical Student interval can be requested with the fit's residual degrees of freedom:

```python
from statadic.api.inference import ModelBased

# Continuation: result is a valid fitted OLS/WLS host result.
inference = result.infer(covariance=ModelBased())
intervals = inference.intervals(
    reference="t", degrees_of_freedom=result.df_resid
)
```

This fragment is not an independent example. Normal and chi-square references do not accept a degrees-of-freedom override. A chosen Student/F reference with robust covariance is an explicit approximation, not automatically exact finite-sample inference.

Host restriction tests take `Hypothesis(LinearRestriction.zero("exposure"))`.
See [hypotheses](../reference/hypotheses.md) for named restrictions, diagnostics
and retained construction metadata.

## Inspect availability and evidence

Point estimates, covariance, intervals and restriction tests can fail at different stages. Check the status of the output you intend to report. A valid point fit may be saturated, leaving no residual degrees of freedom for classical uncertainty. Joint restrictions need their own rank checks. Finite computed tail probabilities can still fall outside a qualified numerical range.

The native linear guide records a finite-reference range diagnostic, including degrees of freedom above 10,000. Do not silently switch references or precision to hide it. Report the choice and evidence relevant to the analysis.

## Minimum reporting record

State the estimator and sample, intercept/design specification, weight meaning, covariance estimator and corrections, reference law and degrees of freedom, numerical availability and applicable qualification limits. For panels or spatial models, also state the relevant dependence, graph and ordering assumptions.
