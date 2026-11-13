---
type: glossary
description: The terms used for data, model fitting, uncertainty and evidence.
---

<!-- rumdl-disable MD040 -->

# Glossary

```{glossary}
Observation
    One aligned unit of analysis. An observation can contain many atoms when its value is a distribution.

Atom
    A point carrying probability mass within an empirical distribution.

Design
    The predictor columns prepared for a model, together with the state needed to prepare new rows.

Estimator
    The computational rule used to estimate parameters from a specified model and data.

Fitted result
    The estimated parameters, prepared design and numerical diagnostics returned by estimation.

Inference
    A calculation of uncertainty or a test under explicit sampling assumptions.

Covariance estimator
    A rule for estimating how parameter estimates vary together across repeated samples.

Reference distribution
    The distribution used to turn a statistic and its scale into an interval or p-value.

Relative precision
    An inverse relative error variance used by weighted least squares.

Cluster
    A group within which observations may be dependent under a clustered covariance model.

Fixed effect
    A coefficient or absorbed effect for a named group or period.

Ground distance
    The comparison between individual points used to construct a comparison between distributions.

V-statistic
    An empirical average that includes within-sample diagonal terms.

U-statistic
    An empirical average that excludes the designated within-sample diagonal terms.

Exchangeability
    Invariance of the joint sampling distribution under the transformations used by a randomization test.

Identification
    The ability to distinguish the parameters targeted by a model under its stated assumptions.

Numerical validity
    Availability of a requested calculation under its numerical checks.

Calibration
    Agreement between a procedure's stated uncertainty and its behavior over repeated experiments.

Right censoring
    An observation that the event time exceeds the recorded elapsed duration.

Delayed entry
    Observation beginning after a spell has started, with likelihood conditional on survival to entry.
```
