# Generalized empirical likelihood: fit, then test

`statadic.estimation.gel` owns the criterion and its fitted numerical state.
`statadic.inference.gel` consumes that state to select a statistic and a reference
law. The CUE, EL and ET criteria retain their established multiplier caps,
frozen numerical subspace and optimizer behavior; this organization does not
replace their numerical algorithms.

## Assumptions and evidence

<details><summary>Inspect assumptions, source review and validation</summary>

The [positive balancing-weights proof](formal/claims/STATADIC-GEL-001.md) concerns
exact EL/ET stationarity in the retained moment space. It does not cover signed
CUE weights, optimizer convergence or a chi-square reference law. Its claim
page records the source locator and human correspondence-review status.

The {download}`independent SciPy comparisons <../tests/estimation/test_gel_reference.py>`
and {ref}`exact GEL study mappings <research-procedure-gel.estimation-tests>`
have their own numerical domains and operating-characteristic requirements.
Check retained dispositions, [capabilities](capabilities.md) and
[known limits](validation/known-limitations.md); iid acknowledgement is an
assumption, not a diagnostic result.

</details>

## Retain the actual fitting state

Use `FixedGEL(criterion="cue").estimate(FixedMomentProblem(g))` for pre-formed moment rows without fitted
structural parameters. Use `AffineGEL(criterion="cue").estimate(AffineMomentProblem(g0, jacobian))` for
`g_i(theta) = g0_i + D_i @ theta`; the derivative can be common `(m,p)` or
rowwise `(n,m,p)`. The affine door retains the original overidentification and
mean-Jacobian identification requirements.

```python
from statadic.estimation.gel import AffineGEL, FixedGEL
from statadic.estimation.moments import AffineMomentProblem, FixedMomentProblem
from statadic.inference.gel import gel_test

fixed_fit = FixedGEL(criterion="cue").estimate(FixedMomentProblem(moment_rows))
fixed_test = gel_test(fixed_fit, kind="lm", assume_iid=True)

affine_fit = AffineGEL(criterion="el").estimate(AffineMomentProblem(moments0, derivative))
profile_test = gel_test(affine_fit, kind="lr", assume_iid=True)
```

A `FixedGelFit` retains the moment panel, multiplier, objective and numerical
rank basis. An `AffineGelFit` additionally retains original moments and
derivatives, structural parameters, final moments, separate inner/outer solve
evidence and final-rank comparison. The profile subspace is frozen at its
least-squares start. Equality of final and frozen ranks does not establish
equality of eigenspaces or rank stability throughout optimization.

For the host workflow, estimate an aligned `MomentData` through
`Estimand(MomentConditions(), estimator=AffineGEL(...))`, then bind the
calibration before stating the null:

```python
from statadic.api.estimand import Estimand, MomentConditions
from statadic.api.hypotheses import Hypothesis, OveridentifyingRestrictions
from statadic.api.moments import GELCalibration, MomentData

result = Estimand(
    MomentConditions(), estimator=AffineGEL(criterion="el"),
).estimate(MomentData.from_arrays(AffineMomentProblem(moments0, derivative)))
tested = result.infer(
    calibration=GELCalibration(kind="lr", assume_iid=True),
).test(Hypothesis(OveridentifyingRestrictions(
    moment_count=moments0.shape[1], parameter_count=derivative.shape[-1],
)))
print(tested.report().to_text())
```

The original moment and parameter counts must match the retained fit. The
actual test result is available as `tested.evidence`; reporting does not
replace its multiplier, frozen basis, solver or reference diagnostics.

## Select a statistic and its reference

`gel_test` accepts an actual fit, without refitting or an externally supplied
parameter count. Fixed fits support likelihood-ratio, LM and Wald statistics;
affine fits currently support profiled LR only. Affine reference degrees of
freedom subtract the fitted structural dimension from the retained stochastic
moment rank.

Successful numerical statistics remain descriptive unless the caller explicitly
acknowledges iid or serially uncorrelated contributions with `assume_iid=True`.
Without that acknowledgement, p-values are NaN. Failed solves or an incompatible
retained/final rank withhold inferential statistics. Check convergence and
`reference_valid`; neither a named criterion nor numerical rank supplies the
statistical assumptions.

The fitting and testing doors are eager host boundaries. Caller precision is
retained, and the private fixed-shape dual kernels remain separate from host
validation and report assembly. No blanket JIT/vmap, dependence correction,
parameter covariance or fitted-solution derivative promise is made by these
public doors. See [numerical boundaries](numerical-host-boundaries.md) and the
[separate affine GMM workflow](gmm-fitting.md).
