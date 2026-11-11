# Preview affine GMM fitting and inference

## Assumptions and evidence

<details><summary>Inspect assumptions, source review and validation</summary>

The [fixed-weight norm-gap proof](formal/claims/STATADIC-GMM-003.md) requires a
nonsingular weighted Gram matrix. [Consistency](formal/claims/STATADIC-GMM-006.md)
and the [actual-weight sandwich limit](formal/claims/STATADIC-GMM-005.md) require
their stated population identification and convergence assumptions. Claim pages
expose source locators and human correspondence-review status; none proves
convergence of the weight iteration or validity under arbitrary misspecification.

The {download}`linearmodels comparisons <../tests/estimation/test_gmm_reference.py>`
match the update budget and covariance conventions.
{ref}`Estimation variants <research-procedure-gmm.affine-estimation>`,
{ref}`covariance variants <research-procedure-gmm.actual-fit-covariance>` and
{ref}`test variants <research-procedure-gmm.tests>` have distinct study mappings.
Consult retained dispositions and [known limits](validation/known-limitations.md);
registration and numerical agreement do not establish calibration.

</details>

## Estimate affine moments

`statadic.estimation.gmm.IteratedGMM` and `FixedWeightGMM` separate point fitting from covariance and
Hansen-J calibration. It accepts complete finite moment rows and either a common
derivative `(m,p)` or a rowwise derivative `(n,m,p)` for
`g_i(theta) = g0_i + D_i @ theta`. Every row, moment and parameter stays in its
supplied order. Exactly identified `m=p` and overidentified `m>p` fits are
supported. No intercept, instrument partition, dropped row or parameter
constraint is inferred.

```python
import jax.numpy as jnp
from statadic.core.equations import MeanCenteredLongRun
from statadic.estimation.gmm import FixedWeightGMM, IteratedGMM
from statadic.estimation.moments import AffineMomentProblem
from statadic.inference.gmm.covariance import gmm_covariance
from statadic.inference.gmm.hypothesis import hansen_j_from_fit
from statadic.inference.parameters import coefficient_intervals

time = jnp.arange(40.0)
derivative = jnp.asarray([[1.0], [0.5], [-0.25]])
g0 = (
    jnp.stack((jnp.sin(time), jnp.cos(time * 0.7), jnp.sin(time * 1.3)), axis=1)
    - (derivative @ jnp.asarray([0.4]))[None, :]
)
fit = IteratedGMM(
    variability=MeanCenteredLongRun,
    bandwidth=2,
    tolerance=1e-6,
    parameter_names=("effect",),
).estimate(AffineMomentProblem(g0, derivative))
uncertainty = gmm_covariance(fit)
intervals = coefficient_intervals(uncertainty, reference="normal")
j_test = hansen_j_from_fit(fit, assume_correct_specification=True)
```

The scheme choice is explicit: `PlainLongRun` uses raw moment cross-products;
`MeanCenteredLongRun` subtracts the sample moment mean before computing Bartlett
HAC. Both divide every lag product by the same retained row count `n`. Scheme
types remain distinct through the fit and covariance. Neither choice certifies
an appropriate bandwidth, weak identification, or inference under
misspecification.

Without an explicit weight, fitting preserves the existing affine-GMM
specification: solve once with identity weight, then use at most `max_iter`
updates. Each update estimates variability at the previous coefficient,
inverts it by the established pseudoinverse rule, and solves the corresponding
weighted least-squares problem. Once the relative coefficient change is below
`tolerance`, the coefficient and its actual generating weight both freeze.
`max_iter=0` returns the initial identity-weight solve without claiming
iteration convergence.

`FixedWeightGMM(variability=MeanCenteredLongRun).estimate(
AffineMomentProblem(g0, derivative, weight=W))` performs one fixed-weight solve. `W` must be a
finite symmetric positive-definite `(m,m)` matrix. Floating supplied weights retain
their values and dtype; integer weights promote to represent invalid-state NaNs.
There is no silent efficient-weight replacement. Symmetry is checked at caller
precision; for accepted roundoff-scale asymmetry, `effective_weight` is the
symmetric part used in the criterion and all its diagnostics.

The default is an iterated estimator with up to 25 weight updates. A two-step
fit uses `IteratedGMM(..., max_iter=1)` explicitly.
`Estimand(MomentConditions(), estimator=...)` adds host preparation and reports
around aligned `MomentData`;
numerical problems and bricks remain the interface for transformed computations.

The immutable fit retains original `moments0`, the complete derivative panel,
the final coefficient and moments, the mean Jacobian, the actual last solve's
`weight`, its `effective_weight`, and variability recomputed at the final
coefficient. It also retains
the mean criterion `gbar' W gbar` and first-order residual `Dbar' W gbar` at
that same coefficient and effective weight. A newly recomputed inverse variability need
not equal that weight at a finite iteration budget.

`fit.input_valid`, `fit.identified` and `fit.valid` answer separate questions.
Validity requires a finite full-rank stationary solve for the retained weight.
The mean derivative and the whitened design are separately column-equilibrated
for their numerical ranks. `fit.rank` records the former; `fit.weighted_rank`
records the latter and uses the same singular-value cutoff as the solve,
`max(m, p) * eps` at working precision. A rank check does not establish
population or strong identification. An invalid intermediate weight or solve
fails the lane instead of publishing a preceding iterate.
`fit.solver.converged` additionally records convergence of the iterated
coefficient sequence, with iteration count and stopping diagnostics. A valid
finite-budget point solve can have an unconverged weight sequence. Invalid
numeric lanes expose NaN scientific state while retaining original inputs and
raw solver diagnostics. Numerical rank checks do not diagnose weak instruments.

`gmm_covariance(fit)` evaluates the full actual-weight sandwich using the
fit's retained `moment_map`, formed by the same factorization as its coefficient:

```text
A = (Dbar' W Dbar)^-1 Dbar' W
V = A Omega A' / n * finite_sample_correction
```

It uses the fit's actual rows, parameter order, evaluation coefficient and
weight. Omitting a weight from generic covariance composition would select an
efficient shortcut and can produce the wrong covariance for this fit. The
default finite-sample correction is one; reference distribution and degrees of
freedom are separate choices. The result owns its fit and composes with
`coefficient_intervals` and `linear_restriction_test`. Positive marginal
variances and a valid covariance are required; joint restrictions additionally
check projected rank. A finite-budget fit may supply this actual-weight
covariance even when iteration convergence is false.

Conventional sandwich inference assumes correctly specified moments, strong
identification, an interior parameter, appropriate dependence/moment conditions
and a consistent selected variability estimator. Centering variability alone
does not establish a universally misspecification-robust GMM procedure.

`hansen_j_from_fit` requires explicit acknowledgement of those conventional
assumptions, overidentification, coefficient convergence, full-rank positive
variability and final-weight efficiency evidence. It checks
`||W Omega - I||_F / ||I||_F <= sqrt(machine epsilon)` at the final coefficient.
Supplied arbitrary fixed weights remain unavailable for this reference, even
if numerically close to the final inverse variability. Exactly identified fits
succeed but have no Hansen-J p-value. Unavailable tests return a false validity
flag, a reason status and NaN statistic/p-value, rather than a fabricated
chi-square result.

For a retained host result, use the same diagnostic spelling as dedicated IV:

```python
from statadic.api.diagnostics import HansenJ

diagnostics = result.diagnose(HansenJ(assume_correct_specification=True))
print(diagnostics.report().to_text())
```

Its `TestOutcome` records the overidentifying null, actual construction,
assumptions and reason when unavailable. The wrapper preserves this GMM
kernel's checks; it does not transfer dedicated IV source review or statistical
qualification. The [hypothesis guide](reference/hypotheses.md) describes metadata
and named coefficient restrictions through
`result.infer(covariance=GMMSandwich()).test(...)`, with `GMMSandwich` imported
from `statadic.api.moments`. See the [Estimand reference](reference/estimand.md) for
the complete host composition.

Fixed-shape fitting, covariance and fit-derived J support JIT and `vmap`.
Bandwidth, iteration budget, scheme and parameter names are static; sample
arrays and lane statuses are dynamic. Precision remains caller-owned. This
surface does not certify derivatives of the fitted solution, constrained
estimates or weak-identification inference.

The former `linear_hansen_j` and `linear_hansen_j_kernel` entry points have been
removed. Fit affine moments with `IteratedGMM.estimate`, then pass that retained
fit to `hansen_j_from_fit`; it preserves the actual generating weight and applies
the availability checks above. Fixed-moment tests live in
`inference.gmm.hypothesis`; fit-derived covariance lives in
`inference.gmm.covariance`.

GEL uses the separate [fit-then-test workflow](gel-fitting.md):
`estimation.gel` owns multiplier and structural-parameter fitting, while
`inference.gel` consumes that retained fit to select a statistic and reference.

The independent references match the actual
[linearmodels IVGMM coefficient sequence](https://bashtage.github.io/linearmodels/iv/iv/linearmodels.iv.model.IVGMM.fit.html)
using identity initialization and a fixed update budget, and the
[actual-weight GMM covariance](https://bashtage.github.io/linearmodels/iv/iv/linearmodels.iv.gmm.IVGMMCovariance.html)
with Bartlett bandwidth, centering and finite-sample correction matched
explicitly. These references qualify the tested complete-array affine slice,
including deliberately non-efficient weights; they do not certify every GMM
specification.
