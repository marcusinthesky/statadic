# State a hypothesis, then choose its procedure

{py:class}`~statadic.api.hypotheses.Hypothesis` accepts `null=...` and
`alternative=...`. A fitted
result supplies the sample and parameter schema; an inference or diagnostic
procedure supplies its statistic, reference and assumptions. The first supported
procedures require `alternative=None`, meaning the complement of the null.
The generic `core.statistics.Null` axis remains separate from this host value.

## Test named coefficients

```python
from statadic.api.hypotheses import Hypothesis, LinearRestriction
from statadic.api.inference import HC

# Continuation: result retains a fitted design with a column named "exposure".
uncertainty = result.infer(covariance=HC("HC3"))
tested = uncertainty.test(
    Hypothesis(LinearRestriction.zero("exposure")),
    reference="chi2",
)
print(tested.report().to_text())
```

`LinearRestriction` represents `R @ theta = r` using parameter names. Names are
resolved against the actual fit, including named ancillary parameters. Unknown
names and unnamed positional fits raise. `from_contrasts(...)` also binds the
restriction to the exact fitted design that created those contrasts. Redundant
restrictions and singular restriction covariance remain unavailable.

`reference="f"` additionally requires explicit positive denominator degrees of
freedom. Selecting robust covariance and an F law is permitted as a declared
approximation; covariance construction alone does not establish exact F calibration.

## Diagnose a retained fit

```python
from statadic.api.diagnostics import BreuschGodfrey, DurbinWatson
from statadic.api.inference import HAC

diagnostics = result.diagnose(
    BreuschGodfrey(lags=4, order_by="time"),
    DurbinWatson(order_by="time"),
)
uncertainty = result.infer(covariance=HAC(bandwidth=4, order_by="time"))
```

These host requests require an explicit ordering choice. A named finite, unique
numeric column sorts ascending; `order_by=None` acknowledges retained row order.
Ties are rejected. Lag units are adjacent ordered observations. Sorting neither
checks calendar spacing nor establishes dependence assumptions. The fit and its
row identities remain unchanged.

Reports never carry sample-sized row-id tuples. Construction metadata records
`rows` as `{"count", "sha256"}`, the SHA-256 digest of the compact JSON list of
retained row ids. Ordered calculations add `calculation_order`, the same digest
of the ids in calculation order. Re-hashing the ids, sorted by `order_by` where
named, verifies a report against its data.

Diagnostics construct their own nulls and return `TestOutcome` values alongside
descriptive influence results. Outcomes retain assumptions, citation keys,
construction details and availability reasons. Durbin–Watson has a statistic but
no calibrated p-value. `report()` metadata survives `save_analysis`; no new
predictor artifact or scientific approval is created.
See the [Report reference](reporting.md) for rendering and persistence.

For dedicated IV or affine-GMM results, request
`result.diagnose(HansenJ(assume_correct_specification=True))` after importing
`HansenJ` from `statadic.api.diagnostics`. The acknowledgement is substantive:
numerical checks do not establish instrument validity, strong identification or
correct specification. [IV](../iv-fitting.md) and [GMM](../gmm-fitting.md) retain
different efficient-weight availability conditions.

## Select a nested geometric null

Bind a resampling procedure with `fit.infer(resampling=...)`, then call
`.test(Hypothesis(Reduced(("intercept", "age"))))`. `Reduced` selects columns
of the already-fitted full design; it
does not learn another representation. [Freedman–Lane](composition.md#adjusting-for-nuisance-predictors)
remains approximate, including exhaustive enumeration. This wrapper does not
convert geometric inference into a coefficient Wald test or transfer its proof
and calibration to another procedure.

Resampling executes at `.test(...)`, after the null is known. The returned
`TestOutcome.evidence` retains the procedure's native run, including its schedule
and failed draws. Reports serialize the outcome metadata, not that live carrier.

<details><summary>Evidence belongs to the selected procedure</summary>

The [Durbin–Watson range](../formal/claims/STATADIC-DW-001.md) proves a finite bound,
not a universal p-value. The
[efficient GMM reference result](../formal/claims/STATADIC-JTEST-003.md) has explicit
limit, rank and efficient-weight assumptions. Neither is a generic theorem about
`Hypothesis` or `TestOutcome`; the claim pages retain exact source-review status.

[Residual diagnostics](../linear-diagnostics.md) describe their independent
comparisons and historical study boundaries. Inspect exact
[reference mappings](../validation/references.md),
[study dispositions](../validation/research.md) and
[known limits](../validation/known-limitations.md) for the actual procedure.
The {download}`native restriction controls <../../tests/inference/test_parameters.py>`
and {download}`independent diagnostic comparisons <../../tests/inference/test_linear_diagnostics_reference.py>`
retain their tested configurations; they do not establish universal calibration.

</details>
