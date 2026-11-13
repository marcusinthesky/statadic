---
type: reference
description: Fit models from moment conditions with GMM, generalized empirical likelihood or instrumental variables, and test the overidentifying restrictions.
---

# Moment estimators

## Usage

```python
import jax.numpy as jnp

from statadic.api.diagnostics import HansenJ
from statadic.api.estimand import Estimand, MomentConditions
from statadic.api.moments import GMMSandwich, MomentData
from statadic.core.equations import MeanCenteredLongRun
from statadic.estimation.gmm import IteratedGMM
from statadic.estimation.moments import AffineMomentProblem

time = jnp.arange(40.0)
derivative = jnp.asarray([[1.0], [0.5], [-0.25]])
g0 = jnp.stack((jnp.sin(time), jnp.cos(0.7 * time), jnp.sin(1.3 * time)), axis=1)
g0 = g0 - (derivative @ jnp.asarray([0.4]))[None, :]

estimator = IteratedGMM(
    variability=MeanCenteredLongRun, bandwidth=2, tolerance=1e-6, parameter_names=("effect",)
)
data = MomentData.from_arrays(AffineMomentProblem(g0, derivative), row_ids=tuple(range(40)))
result = Estimand(MomentConditions(), estimator=estimator).estimate(data)
print(result.infer(covariance=GMMSandwich()).report().to_text())
print(result.diagnose(HansenJ(assume_correct_specification=True)).report().to_text())
```

## Description

A moment estimator chooses the parameters of a model so that the sample averages of its
moment functions sit close to zero. STATADIC fits affine moments of the form
$g_i(\theta) = g_{0i} + D_i \theta$ with GMM or generalized empirical likelihood (GEL),
and fits linear instrumental-variable (IV) regressions with 2SLS or IV GMM. Rows,
moments and parameters keep the order you supply, and both exactly identified
(`m = p`) and overidentified (`m > p`) problems are supported.

Fitting, covariance and testing are separate requests. A fit records the actual weight from
the end of its last solve, so the covariance and the Hansen J test of that fit rely on this
weight instead of an efficient-weight replacement. The [`Estimand` reference](estimand.md) explains how
`MomentData` aligns the arrays with row identities in the host workflow, while the
numerical classes below take arrays directly and support `jax.jit` and `jax.vmap`.

## Options

| Estimator | Module | Notes |
| --- | --- | --- |
| `IteratedGMM(variability, bandwidth, max_iter=25, tolerance=1e-10)` | `statadic.estimation.gmm` | Starts from the identity weight, with up to `max_iter` updates. `max_iter=1` is two-step GMM |
| `FixedWeightGMM(variability, bandwidth)` | `statadic.estimation.gmm` | One solve with the weight `W` from the problem |
| `FixedGEL(criterion)` | `statadic.estimation.gel` | Multiplier for moment rows |
| `AffineGEL(criterion)` | `statadic.estimation.gel` | Parameters and multiplier, with criterion `"cue"`, `"el"` or `"et"` |
| `TwoStageLeastSquares()` | `statadic.estimation.iv` | Starts from a projection onto the instruments |
| `TwoStepIVGMM(variability, bandwidth)` | `statadic.estimation.iv` | One weight update from 2SLS residuals |
| `IteratedIVGMM(variability, bandwidth, max_iter=256)` | `statadic.estimation.iv` | Updates the weight until a stopping rule |

The `variability` argument is `PlainLongRun` for raw cross-products, or
`MeanCenteredLongRun` to subtract the sample mean of the moments first. Both apply
Bartlett weights up to `bandwidth`, where zero suits independent rows, and IV
estimators also accept `OneWayCluster`.

For IV, `Regression` takes every structural term with the scalar endogenous terms last.
The other exogenous terms enter both the structural and the instrument matrices, and
`instruments` names the excluded instruments:

```python
from statadic.api.design import Intercept, Linear
from statadic.api.regression import Regression
from statadic.core.equations import PlainLongRun
from statadic.estimation.iv import TwoStageLeastSquares, TwoStepIVGMM

terms = Intercept() + Linear("control") + Linear("treatment")
roles = {"endogenous": ("treatment",), "instruments": ("assignment", "encouragement")}
tsls = Regression("y", terms, estimator=TwoStageLeastSquares(), **roles)
two_step = Regression("y", terms, estimator=TwoStepIVGMM(PlainLongRun, bandwidth=0), **roles)
```

Choose the covariance of an IV fit with `ModelBased()`,
`HC("HC0")`, `HAC(bandwidth=b, order_by="time")` or
`Clustered(column=...)`. GMM takes `GMMSandwich(finite_sample_correction=1.0)`.
GEL takes `GELCalibration(kind="lr" | "lm" | "wald", assume_iid=False)`.

## Results

GMM fits return an `AffineGMMFit` holding the final coefficient, the mean Jacobian, the
actual `weight` and `effective_weight` of the last solve, and the mean criterion. Three
flags in the result answer separate questions about the fit. `fit.input_valid` checks the inputs, `fit.identified`
checks the rank, and `fit.valid` requires a finite full-rank solve. The flag
`fit.solver.converged` reports whether the iteration of the weight settled, and a finite
budget of updates can leave it false while the point solve stays valid.

GEL fits return a `FixedGelFit` or an `AffineGelFit` with the multiplier, the objective
and the numerical rank. Affine fits add the parameters and separate inner and outer
convergence. The host workflow wraps these fits in a `Result` with `infer` and `diagnose` methods for
the same requests.

IV reports include `first_stage_partial_r2.<name>` for each endogenous predictor.
This measures the contribution of the excluded instruments after accounting for
the exogenous regressors. A high value alone establishes neither instrument
validity nor strong population identification.

Hansen J is requested once, the same way for GMM and IV results:

```python
from statadic.api.diagnostics import HansenJ

diagnostics = result.diagnose(HansenJ(assume_correct_specification=True))
print(diagnostics.report().to_text())
```

The outcome records its null, its construction and its assumptions, along with the reason
when the test is unavailable. Exactly identified fits, 2SLS fits, arbitrary fixed weights
and a missing acknowledgement all leave the p-value unavailable. For GEL, call
`result.infer(calibration=GELCalibration(...)).test(Hypothesis(OveridentifyingRestrictions(moment_count=m, parameter_count=p)))`.
Fixed fits support LR, LM and Wald statistics, while affine fits support the profiled LR
statistic. The [hypotheses reference](hypotheses.md) describes the outcome object in detail.

## Methods and formulas

With a weight $W$ and a mean Jacobian $\bar D$, the fixed-weight solution minimises
$\bar g' W \bar g$. The actual-weight sandwich is $V = A \Omega A' / n$ with
$A = (\bar D' W \bar D)^{-1} \bar D' W$, where $\Omega$ is the long-run variability of
the moments. Hansen J is the criterion at the final coefficient, scaled by $n$ and compared
with $\chi^2_{m-p}$. GEL minimises a dual objective over the values of a multiplier with a CUE, EL or ET
criterion, and its affine degrees of freedom subtract the structural dimension from the
rank of the moments.

## Assumptions and limits

- Sandwich and J inference assume correct moments, strong identification, an interior
  parameter and a consistent variability estimator. Centering the moments does not make
  the procedure robust to a misspecification of the model.
- Numerical rank cannot diagnose weak instruments, and full rank of the instruments does
  not establish their exogeneity.
- Hansen J needs overidentification, convergence of the coefficients, full-rank variability
  and a final weight that inverts that variability to within the square root of the machine
  epsilon.
- In 32-bit precision, iterated GMM needs a looser `tolerance`, such as `1e-6`, for the
  weight iteration to converge. With the default `1e-10` the J test is unavailable.
- Naming parameters with `parameter_names` gives readable reports. Without names, rows are
  labelled `parameter_0`, `parameter_1` and so on.
- GEL p-values need `assume_iid=True` and are NaN without it. The frozen numerical subspace
  does not establish rank stability during optimisation.
- IV prediction evaluates the structural index at the supplied endogenous values, so it is
  not a causal effect.
- Tests that are robust to weak instruments, IV with fixed effects and constrained estimates
  lie outside this surface.

## Evidence

- Mathematical results: the {ref}`fixed-weight minimum <formal-STATADIC-GMM-003>`,
  {ref}`consistency <formal-STATADIC-GMM-006>`, the
  {ref}`actual-weight sandwich <formal-STATADIC-GMM-005>`,
  {ref}`GEL stationarity <formal-STATADIC-GEL-001>`, the
  {ref}`wrong-target counterexample <formal-STATADIC-IV-002>` and the
  {ref}`actual-fit J calibration <formal-STATADIC-JTEST-003>`.
- Reference comparisons: coefficient sequences and covariance match linearmodels
  (`IVGMM`), and GEL matches SciPy.
- Studies: the {ref}`affine estimation <research-procedure-gmm.affine-estimation>`,
  {ref}`covariance <research-procedure-gmm.actual-fit-covariance>`,
  {ref}`test <research-procedure-gmm.tests>`, {ref}`GEL <research-procedure-gel.estimation-tests>`
  and {ref}`IV <research-procedure-iv.regression>` procedures.

## See also

- [Estimand](estimand.md) and [hypotheses](hypotheses.md)
- [Regression](regression.md) and [transformation support](transformation-support.md)
- [Known limitations](../validation/known-limitations.md)
