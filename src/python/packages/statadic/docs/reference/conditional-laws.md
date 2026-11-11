# Response laws

Regression owns the predictor terms. A law selects the distribution, link and
dispersion policy. Observation descriptors declare trials or censoring.

```python
from statadic.api.design import Intercept, Linear
from statadic.api.regression import Binomial, Observed, Regression

model = Regression(
    Observed("successes", trials="n"),
    Intercept() + Linear("dose"),
    law=Binomial(link="probit"),
)
fit = model.estimate({
    "dose": [-1., 0., 1.], "successes": [2, 3, 6], "n": [10, 10, 10],
})
probability = fit.predict({"dose": [0.25]}, kind="probability")
expected = fit.predict({"dose": [0.25], "n": [10]}, kind="mean")
```

Trials are positive integers; successes must be integers between zero and their
trial totals. Omitted trials mean one trial per row. Logit and probit retain their
native estimators, existence diagnostics and covariance contracts.

## Counts and positive outcomes

```python
from statadic.api.regression import Gamma, NegativeBinomial2, Poisson
from statadic.estimation.likelihood import FixedDispersion

terms = Intercept() + Linear("x")
poisson = Regression("count", terms, law=Poisson(), exposure="hours")
gamma = Regression("y", terms, law=Gamma(dispersion=FixedDispersion(0.5)))
nb2 = Regression(
    "count", terms,
    law=NegativeBinomial2(dispersion=FixedDispersion(0.5)),
    exposure="hours",
)
```

Offsets are additive on the link scale. Exposure is positive and multiplies a
rate. Predicting a rate does not require exposure; predicting the corresponding
count mean does. Inspect `fit.prediction_options()` for exact input roles,
targets and unavailable reasons.

The law owns fixed versus estimated dispersion. Solver options belong to the
actual compatible estimator. See [likelihood details](extended-regression.md)
for parameter, variance and boundary conventions.

## A hurdle at zero

Two named term blocks specify genuinely different predictors:

```python
from statadic.api.regression import HurdlePoisson

model = Regression(
    "events",
    terms={
        "zero_probability": Intercept() + Linear("z"),
        "count": Intercept() + Linear("x"),
    },
    law=HurdlePoisson(zero_link="logit"),
    exposure="hours",
)
```

The gate trains on every row; the positive-count design trains on positive rows.
This distinction matters for learned category levels, splines and ordination.
Named offsets use the same two block names.

| Prediction | Quantity | Required blocks |
| --- | --- | --- |
| `zero_probability` | Probability of zero | Gate |
| `rate` | Untruncated intensity per unit exposure | Count |
| `positive_mean` | Mean conditional on a positive count | Count and exposure |
| `mean` | Marginal mean | Both and exposure |

All-zero samples cannot train the positive block. A failed block retains its
diagnostics and withholds dependent predictions. The other block may remain
usable. Reporting does not fabricate a joint covariance; joint hurdle inference
and predictor serialization remain unsupported.

See [survival models](../survival-models.md) for censoring and delayed entry.
Native estimators remain directly usable for qualified
[JAX transformations](../capabilities.md).
