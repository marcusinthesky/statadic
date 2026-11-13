---
type: reference
description: State a null hypothesis about a fitted model and read the outcome of the procedure that tests it.
---

# Hypothesis

## Usage

```python
import random

from statadic.api.design import Intercept, Linear
from statadic.api.hypotheses import Hypothesis, LinearRestriction
from statadic.api.inference import HC
from statadic.api.regression import Regression

rng = random.Random(3)
exposure = [i / 10 for i in range(60)]
data = {"exposure": exposure, "y": [0.5 + 0.4 * v + rng.gauss(0, 1) for v in exposure]}

fit = Regression.ols("y", Intercept() + Linear("exposure")).estimate(data).require_valid()
uncertainty = fit.infer(covariance=HC("HC3"))
tested = uncertainty.test(
    Hypothesis(LinearRestriction.zero("exposure")),
    reference="chi2",
)
print(tested.report().to_text())
```

## Description

{py:class}`~statadic.api.hypotheses.Hypothesis` states a question about a model
as a `null` and an optional `alternative`. A fitted result supplies the sample
and the parameter names, and the procedure you call supplies the statistic, the
reference distribution and the assumptions. Every supported procedure currently
takes `alternative=None`, which means the complement of the null.

Two kinds of procedure return a `TestOutcome`. Inference procedures such as
`uncertainty.test(...)` and resampling take a hypothesis from you. Diagnostics
such as those in [linear diagnostics](linear-diagnostics.md) build their own
nulls, because each diagnostic is defined by what it tests.

## Options

The null is a proposition, and the table lists those you can write yourself.

| Null | States | Used with |
| --- | --- | --- |
| `LinearRestriction.zero("a", "b")` | Each named coefficient equals zero | `uncertainty.test(..., reference=...)` |
| `LinearRestriction(rows, values)` | $R\theta = r$, with rows given by parameter name | `uncertainty.test(..., reference=...)` |
| `LinearRestriction.from_contrasts(...)` | A restriction built from contrasts of one fitted design | `uncertainty.test(..., reference=...)` |
| `Reduced(("intercept", "age"))` | No effect beyond the named columns of the fitted design | `fit.infer(resampling=...).test(...)` |
| `HansenJ(assume_correct_specification=True)` | The overidentifying restrictions hold | `result.diagnose(...)` on IV and GMM results |

Names resolve against the actual fit, including named ancillary parameters, and
`from_contrasts` binds the restriction to the exact design that produced the
contrasts. For linear restrictions, `reference="chi2"` gives a Wald test, and
`reference="f"` also needs positive `degrees_of_freedom` for the denominator.
Resampling procedures run at `.test(...)`, once the null is known, and they are
described in the [composition reference](catalogue-fitting.md).

## Results

A `TestOutcome` holds the `hypothesis`, the `statistic`, the `pvalue` and the
`reference` distribution that calibrates it. It also records `available` with an
`unavailable_reason`, the `assumptions` of the procedure, citation `sources` and
the `construction` details. Calling `report()` turns the outcome into a
[report](reporting.md), and its metadata survives `save_analysis`, as described
for [saved artifacts](artifacts.md).

Reports never carry a list of row ids as long as the sample. The construction
metadata records `rows` as a count and the SHA-256 digest of the compact JSON
list of row ids. Ordered calculations add `calculation_order`, the same digest of the ids in
calculation order. Hashing the ids of your data, sorted by `order_by` where one
was named, then verifies a report against its data. The `order_by` rule is
described for `HAC` in the [regression reference](regression.md).

A resampling outcome keeps the native run of its procedure in
`TestOutcome.evidence`, including the schedule and any failed draws. Reports
serialize the outcome metadata and leave that live object out.

## Methods and formulas

A linear restriction $R\theta = r$ is tested with the Wald statistic
$(R\hat\theta - r)^\top (R\hat V R^\top)^{-1} (R\hat\theta - r)$, where
$\hat V$ is the covariance chosen in `infer`. The statistic is compared with
$\chi^2_q$ for $q = \operatorname{rank}(R)$ or, divided by $q$, with
$F(q, d)$ for the denominator degrees of freedom $d$ you supply. Hansen's $J$ is
the minimised efficient-weight GMM criterion, compared with a chi-square law
whose degrees of freedom equal the instruments minus the parameters.

## Assumptions and limits

- A fit that succeeds numerically does not establish the assumptions behind a
  reference law. You select the reference, and covariance construction alone does
  not make an F law exact.
- Combining a robust covariance with an F law is a declared approximation.
- Unknown names and unnamed positional fits raise, and redundant restrictions or
  a singular restriction covariance make the test unavailable.
- Resampling that adjusts for nuisance predictors is approximate, even when
  every permitted schedule is enumerated. A geometric test does not turn into a coefficient Wald
  test or inherit the calibration of another procedure.
- The Hansen J acknowledgement is substantive. Numerical checks do not
  establish instrument validity, strong identification or correct specification,
  and [IV and GMM results](moment-estimators.md) differ in when an efficient
  weight is available.
- Durbin-Watson has a statistic without a calibrated p-value.

## Evidence

- Mathematical results: the {ref}`Durbin-Watson range <formal-STATADIC-DW-001>`
  proves a finite bound, and the
  {ref}`efficient GMM reference result <formal-STATADIC-JTEST-003>` states its
  limit, rank and efficient-weight assumptions. Neither is a theorem about
  `Hypothesis` itself.
- Reference comparisons: the diagnostic nulls are matched against statsmodels in
  the {ref}`reference index <reference-inference.linear_diagnostics>`.
- Studies: the {ref}`residual diagnostics <research-procedure-linear.residual-diagnostics>`
  study covers the diagnostic nulls, and each procedure keeps its own evidence.

## See also

- [Linear diagnostics](linear-diagnostics.md) and
  [regression](regression.md)
- [Reports](reporting.md) and [composition](composition.md)
