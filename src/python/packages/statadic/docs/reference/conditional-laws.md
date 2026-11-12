---
type: reference
description: Choose a response law for regression, from binomial and Poisson to Gamma, negative binomial and hurdle models.
---

# Response laws

## Usage

```python
from statadic.api.design import Intercept, Linear
from statadic.api.regression import Binomial, Observed, Regression

model = Regression(
    Observed("successes", trials="n"),
    Intercept() + Linear("dose"),
    law=Binomial(link="probit"),
)
fit = model.estimate({
    "dose": [-1.0, 0.0, 1.0], "successes": [2, 3, 6], "n": [10, 10, 10],
}).require_valid()
probability = fit.predict({"dose": [0.25]}, kind="probability")
expected = fit.predict({"dose": [0.25], "n": [10]}, kind="mean")
```

## Description

A response law tells {py:class}`~statadic.api.regression.Regression` which
distribution, link and dispersion policy describe the outcome. The predictor
terms stay on `Regression`, and observation descriptors such as
`Observed("successes", trials="n")` declare the trials. Each fitted result keeps
the design built from the training rows, so prediction, inference and reports
are separate requests, as in the [regression reference](regression.md).

Every law uses exactly the terms you list and keeps every row. Offsets are
additive on the link scale, and exposure is a positive multiplier of a rate.
Prediction rows supply their own offset, exposure and trials, because the
values from training are never reused. Inspect `fit.prediction_options()` for
the input roles, the targets and the reasons a target is unavailable.

## Options

| Law | Mean | Variance | Outcome |
| --- | --- | --- | --- |
| `Binomial(link="logit")` or `"probit"` | `trials * p` | `trials * p * (1 - p)` | Integer successes, positive integer trials |
| `Poisson()` | `exp(X beta + offset) * exposure` | mean | Nonnegative integer counts |
| `Gamma(dispersion=...)` | `exp(X beta + offset)` | `phi * mean**2` | Positive continuous outcomes |
| `NegativeBinomial2(dispersion=...)` | `exp(X beta + offset) * exposure` | `mean + alpha * mean**2` | Nonnegative integer counts |
| `HurdlePoisson(zero_link="logit")` | Gate and truncated count | Two blocks | Counts with excess zeros |

Omitted trials mean one trial per row. Gamma and NB2 take a dispersion policy
from `statadic.estimation.likelihood`, which is either `FixedDispersion(value)`
or `EstimatedDispersion()`. Fixed dispersion is a setting of the model rather than a
coefficient, while estimated dispersion joins the score and Hessian of the fit.

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

A hurdle model takes two named term blocks, because the gate and the count
often depend on different predictors. Named offsets use the same block names.

```python
from statadic.api.regression import HurdlePoisson

hurdle = Regression(
    "events",
    terms={
        "zero_probability": Intercept() + Linear("z"),
        "count": Intercept() + Linear("x"),
    },
    law=HurdlePoisson(zero_link="logit"),
    exposure="hours",
)
```

The gate trains on every row, and the positive-count design trains on positive
rows. This matters for learned category levels, splines and ordination.

## Results

Binomial and Poisson fits return `LogisticResult` and `PoissonResult`. Gamma and
NB2 return a likelihood result. All offer `predict(kind=...)`, `infer`, `report`
and `require_valid()`.

| Law | Default `kind` | Other kinds |
| --- | --- | --- |
| Binomial | `probability` | `mean` (expected successes, needs new trials), `link` |
| Poisson | `mean` | `rate` (no exposure needed), `link` |
| Gamma | `mean` | `link` |
| NB2 | `mean` | `rate`, `link` |

Hurdle predictions depend on the blocks they need:

| Prediction | Quantity | Required blocks |
| --- | --- | --- |
| `zero_probability` | Probability of zero | Gate |
| `rate` | Untruncated intensity per unit exposure | Count |
| `positive_mean` | Mean given a positive count | Count and exposure |
| `mean` | Marginal mean | Both and exposure |

A failed block of a hurdle model keeps its diagnostics and withholds the predictions that
depend on it, while the other block may remain usable. Binomial and Poisson
fits also expose `fit.existence`, which records whether a finite maximum
likelihood estimate exists for the data.

`infer(covariance=ModelBased())` uses the joint observed information, and
`HC("HC0")` or `HC("HC1")` use the full joint score contributions of each row.
Intervals and tests then choose a reference distribution in a separate step of the report.

For array workflows inside `jax.jit` or `jax.vmap`, use the native estimators
`BinomialLogitMLE` and `PoissonMLE` from `statadic.estimation.glm` with
`BinomialProblem` and `PoissonProblem`. Inputs are complete finite arrays, and
the columns of each array include any intercept. Covariance comes from
`statadic.inference.glm_covariance.glm_covariance`, and intervals come from
`statadic.inference.parameters.coefficient_intervals`.

```python
import jax.numpy as jnp
from statadic.estimation.glm import BinomialLogitMLE, BinomialProblem
from statadic.inference.glm_covariance import glm_covariance
from statadic.inference.parameters import coefficient_intervals

x = jnp.column_stack((jnp.ones(6), jnp.arange(6.0)))
native = BinomialLogitMLE().estimate(
    BinomialProblem(x, jnp.array([2, 4, 3, 6, 7, 8]), trials=10)
)
uncertainty = glm_covariance(native, method="model")
interval = coefficient_intervals(uncertainty, reference="normal")
```

## Methods and formulas

For a canonical logit or Poisson model, each row contributes one score,
$s_i = x_i (y_i - \mu_i)$, and the aggregate mean Jacobian is
$-X^\top \operatorname{diag}(v_i) X / n$. The variance $v_i$ is
$m_i p_i (1 - p_i)$ for binomial rows with $m_i$ trials and $\mu_i$ for Poisson
rows. Model-based covariance uses the information equality at a dispersion of
one, HC0 uses uncentred outer products of the scores divided by $n$, and HC1
multiplies HC0 by $n/(n-p)$.

The log of the expected count is $x_i^\top\beta + o_i + \log e_i$ for offset
$o_i$ and exposure $e_i$, and the deviance uses the saturated likelihood. For
Gamma and NB2, the solver works with the log of the dispersion, while the
reported parameters and covariance describe the dispersion itself.

A finite maximum exists unless a recession direction $d$ satisfies
$x_i^\top d = 0$ on interior success counts, $x_i^\top d \ge 0$ on all-success
rows and $x_i^\top d \le 0$ on zero-success rows, with at least one strict margin among the rows.
For Poisson models, the directions vanish on positive counts and are
nonpositive on zero counts.

<!-- statadic:math STATADIC-GLM-002 -->

## Assumptions and limits

- Binomial and Poisson models fix the dispersion at one, so frequency weights,
  survey weights, quasi-likelihood and penalised fitting need other contracts.
- Full column rank and a finite optimum are separate requirements. Complete
  and quasi separation (Albert and Anderson, 1984) leave no finite optimum,
  and a Poisson fit with an intercept and all-zero counts is separated.
- Existence rests on a floating-point certificate with a residual resolution of
  `max(1e-7, 2048 * epsilon)`. Samples near that resolution record `UNKNOWN`,
  and an unresolved status never authorises a fit.
- Invalid inputs, separated or unresolved fits and nonconvergence yield NaN
  scientific outputs, while the original inputs and diagnostics remain available.
- A robust covariance gives no exact finite-sample reference distribution and
  does not justify a model for the dispersion.
- An estimated NB2 dispersion at the Poisson boundary stays an NB2 fit, and it
  does not authorise ordinary interior inference.
- All-zero samples cannot train the count block of a hurdle model, and joint
  hurdle inference with predictor saving is unsupported.
- Fixed-shape fits, covariance and prediction support JIT and batching, but no
  derivative of the fitted solution is promised.
- Distribution objects and new saved-predictor codecs are unsupported for
  Gamma and NB2.

## Evidence

- Mathematical results: the {ref}`binomial score <formal-STATADIC-GLM-002>`,
  the {ref}`Poisson score <formal-STATADIC-GLM-003>` and the
  {ref}`fitted-root limit <formal-STATADIC-GLMCOV-003>`.
- Comparisons with statsmodels and glum appear in the
  {ref}`canonical GLM procedure <research-procedure-glm.canonical>`, and the
  {ref}`GLM covariance procedure <research-procedure-glm.covariance>` covers the
  covariance mappings.
- Gamma, NB2 and probit fits appear in the
  {ref}`regression likelihood procedure <research-procedure-likelihood.regression>`.
- Hurdle models appear in the {ref}`hurdle Poisson procedure <research-procedure-likelihood.hurdle-poisson>`
  and the {ref}`positive Poisson procedure <research-procedure-likelihood.positive-poisson>`.

## See also

- [Regression](regression.md) and [survival models](survival.md)
- [Transformation scope](transformation-support.md) and [known limits](../validation/known-limitations.md)
- [Counts with exposure and Gamma regression](../guides/count-models-with-exposure.py)
