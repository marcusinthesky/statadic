# Non-spatial instrumental-variable regression

The complete example generates an endogenous exposure, fits the model and
requests explicit HC0 inference:

:::{literalinclude} ../examples/iv_regression.py
:language: python
:::

`Regression` prepares explicit structural and instrument roles. Include all
structural predictor terms, placing scalar endogenous terms last in their
declared order. The remaining exogenous terms enter both the structural and
instrument matrices; excluded instruments enter only the instrument matrix.
The intercept is included only when requested as a design term.

```python
from statadic.api.design import Intercept, Linear
from statadic.api.regression import Regression
from statadic.estimation.iv import TwoStageLeastSquares, TwoStepIVGMM
from statadic.core.equations import PlainLongRun

model = Regression(
    response="y",
    terms=Intercept() + Linear("control") + Linear("treatment"),
    endogenous=("treatment",),
    instruments=("assignment", "encouragement"),
    estimator=TwoStageLeastSquares(),
)
two_step = Regression(
    response="y",
    terms=Intercept() + Linear("control") + Linear("treatment"),
    endogenous=("treatment",),
    instruments=("assignment", "encouragement"),
    estimator=TwoStepIVGMM(variability=PlainLongRun, bandwidth=0),
)
```

## Assumptions and evidence

<details><summary>Inspect assumptions, source review and validation</summary>

Full instrument rank does not establish exogeneity: the
[wrong-target counterexample](formal/claims/STATADIC-IV-002.md) has a vanishing
fitted moment residual. Read its assumptions, source and human review status,
then the [actual-weight covariance conditions](formal/claims/STATADIC-GMM-005.md).

The {download}`dedicated IV Hansen-J comparisons <../tests/inference/test_iv_reference.py>`
match two-step and iterated producers, plain/centered independent-row weights,
coefficients, actual W, J, degrees of freedom and tails. The separate
{download}`affine-GMM comparisons <../tests/estimation/test_gmm_reference.py>`
exercise the affine-GMM producer. {ref}`IV study mappings <research-procedure-iv.regression>`
retain their own gaps; neither comparison grants inherited study coverage.
Neither a small J statistic nor peer agreement supplies
instrument validity, weak-instrument robustness or finite-sample coverage.
See [known limits](validation/known-limitations.md) before interpreting inference.

</details>

## Point estimation and covariance are separate choices

`TwoStageLeastSquares` starts from the instrument-space projection.
`TwoStepIVGMM` estimates a weight from the preliminary 2SLS residuals and updates
the coefficient once. `IteratedIVGMM` updates within an explicit budget and
reports coefficient/weight convergence separately from availability of its last
stationary coefficient solve. Native bricks accept aligned `IVProblem` arrays.

GMM weighting requires an explicit variability scheme. Plain or mean-centered
long-run schemes use an explicit Bartlett bandwidth, including zero for
independent rows. `OneWayCluster` uses declared cluster labels and no HAC lags;
the host `clusters` role is aligned to retained rows before numerical fitting.
HAC depends on supplied row order. Centering does not establish exclusion or
misspecification-robust inference.

Choose inference with `ModelBased()`, `HC("HC0")`, `HC("HC1")`,
`HAC(bandwidth=..., order_by="time")` or `Clustered(column=...)`. Covariance uses the actual fitting
weight and the requested variability, rather than assuming an efficient-weight
inverse-bread shortcut. Cluster inference needs independent clusters and
sufficient cluster information; ordinary row count does not certify it.
Reference laws, degrees of freedom and finite-sample multipliers stay explicit.
`order_by=None` explicitly acknowledges retained order. Named ordering sorts a
finite, unique numeric key, applying the permutation to complete moment-score
rows. It does not refit the estimator, reorder its already-retained fitting
weight or change Hansen J. Long-run fitting weights therefore still require
the intended input order at estimation time.

## Interpretation and diagnostics

Prediction evaluates the structural linear index using the supplied endogenous
values. It is not automatically the observational conditional mean or a causal
effect of an intervention. Reports expose first-stage partial R² conditional on
included exogenous terms, sample ranks and weighting convergence. None proves
instrument validity or strong population identification.

Hansen-J inference {cite:p}`hansen-1982` requires explicit acknowledgement of the conventional
assumptions. Two-step weight provenance differs from iterated fixed-point
provenance; exactly identified and unsupported fits retain unavailable tests.

```python
from statadic.api.diagnostics import HansenJ

diagnostics = result.diagnose(HansenJ(assume_correct_specification=True))
print(diagnostics.report().to_text())
```

The two-step reference requires a valid prescribed estimated-weight update.
The iterated reference additionally checks convergence and that its actual
weight approximately inverts variability at the final coefficient. A small
sample fixed-point error cannot establish population efficiency. Missing
acknowledgement, 2SLS, unknown producer provenance and exact identification do
not supply a conventional J p-value. The [hypothesis outcome](reference/hypotheses.md)
retains its null, assumptions, construction and unavailability reason.

Weak-instrument-robust tests, fixed-effects IV and nonlinear/survival IV remain
separate work. Bounded descriptive studies retain unavailable procedures as
failures and do not constitute qualification.
