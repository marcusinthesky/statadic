---
type: reference
description: Declare a covariance or a set of supplied moment conditions and estimate it from aligned observations.
---

# Estimand

## Usage

```python
from statadic.api.data import Observations
from statadic.api.estimand import Covariance, Estimand
from statadic.estimation.covariance import SampleCovariance

observations = Observations(
    {"a": [1.0, 4.0, 2.0, 8.0], "b": [0.0, 3.0, 7.0, 2.0]},
    row_ids=("u", "v", "w", "x"),
)
model = Estimand(Covariance(columns=("a", "b")), estimator=SampleCovariance(ddof=1))
result = model.estimate(observations)
print(result.sigma)
print(result.report().to_text())
```

## Description

{py:class}`~statadic.api.estimand.Estimand` pairs a declared quantity with a numerical
estimator and fits the pair to aligned data. It supports two quantities, the covariance of
named columns and a set of supplied fixed or affine moment conditions. Preparation takes a
snapshot of the arrays from your input without sorting rows, dropping rows or inventing a response, and the
result keeps the estimator, the labels and the row identities.

`Estimand` is a host workflow for fitting, and it is separate from
`statadic.core.statistics.PopulationEstimand`, the root of the axis that names what a
statistic targets. Naming a target on that axis does not run an estimator for it.

## Options

| Quantity | Data | Estimators |
| --- | --- | --- |
| `Covariance(columns=("a", "b"))` | `Observations` | `SampleCovariance(ddof=1)`, `LedoitWolf()` |
| `MomentConditions()` | `MomentData` | `IteratedGMM`, `FixedWeightGMM`, `FixedGEL`, `AffineGEL` |

Columns are explicit and keep their order. For a rectangular array you can use
`Covariance()` instead, and its generated labels assert no identity outside the array.
`SampleCovariance` divides by `n - ddof`, while `LedoitWolf` shrinks toward a
constant-correlation target with `ddof=0`.

The rows, derivatives and optional weight of moment conditions live in `MomentData.problem`,
an `AffineMomentProblem` or `FixedMomentProblem` already aligned to its row identities. The
affine convention is $g_i(\theta) = g_{0i} + D_i \theta$, and `MomentConditions()` owns no
sample of its own. Call `prepare` once and reuse the result with `estimate_prepared` on the
same model instance.

## Results

Covariance results expose the matrix as `result.sigma` together with a report. A singular
covariance is allowed, since a finite matrix does not promise an inverse or standard errors.
Moment fits return the GMM or GEL result types of the
[moment estimators reference](moment-estimators.md), each with its own form of inference:

```python
import jax.numpy as jnp

from statadic.api.estimand import Estimand, MomentConditions
from statadic.api.moments import GMMSandwich, MomentData
from statadic.core.equations import PlainLongRun
from statadic.estimation.gmm import FixedWeightGMM
from statadic.estimation.moments import AffineMomentProblem

problem = AffineMomentProblem(
    jnp.array([[-1.0, 0.2], [-2.0, -0.4], [-0.5, 0.1], [-3.0, 0.3]]),
    jnp.array([[1.0], [0.5]]),
    weight=jnp.eye(2),
)
data = MomentData.from_arrays(problem, row_ids=(10, 20, 30, 40))
fitted = Estimand(MomentConditions(), estimator=FixedWeightGMM(PlainLongRun)).estimate(data)
print(fitted.infer(covariance=GMMSandwich()).report().to_text())
```

GEL results use `fitted.infer(calibration=GELCalibration(...))` followed by
`.test(Hypothesis(OveridentifyingRestrictions(...)))`. The calibration chooses the LR, LM
or Wald construction and acknowledges the iid reference for the test. It builds no sampling
covariance, which is why its keyword differs from `covariance`.

## Assumptions and limits

- A covariance request does not establish correct moments or justify a reference
  distribution for overidentification.
- Pairs of estimator and result without a defined workflow have no generic fallback.
- Affine GEL supports the profiled LR test only. Without the iid acknowledgement, the
  finite test statistics are descriptive.
- Alignment, preparation and reporting run eagerly, and only the numerical problems
  themselves are compatible with `jax.jit`.

## Evidence

- Reference comparisons: the [reference index](../validation/references.md) lists the
  comparisons made against other software.
- Studies: the [statistical studies](../validation/research.md) and the
  {ref}`affine GMM estimation <research-procedure-gmm.affine-estimation>` procedure give
  their exact scope.
- Known gaps appear in the [known limitations](../validation/known-limitations.md).

## See also

- [Moment estimators](moment-estimators.md) and [hypotheses](hypotheses.md)
- [Reports](reporting.md) and [transformation support](transformation-support.md)
