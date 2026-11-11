# Estimand implementation

Implemented 2026-11-10 against the API-cohesion working tree. Numerical execution
and independent integration review remain coordinator-owned and pending here.

`api.estimand.Estimand` has the closed initial quantity grammar `Covariance` and
`MomentConditions`. Four typed constructor paths preserve the existing covariance,
GMM, fixed-GEL and affine-GEL result families. Explicit estimators consume typed
numeric problems. Qualified `core.statistics.Estimand` is unchanged.

Covariance estimator bricks and `CovarianceProblem` now live in
`estimation.covariance`; existing kernels remain the single numerical producers.
`api.covariance` retains aligned preparation, context and `CovarianceEstimate`.
Named observations require explicit columns; preparation snapshots arrays and
preserves column/row order. Singular finite covariance remains allowed, without
promising inversion or parameter confidence intervals.

Moment sample arrays remain solely in `MomentData.problem`. `PreparedMoments`
owns copied arrays and the producing model token. The free host `moments.estimate`
door is retired; a narrow family factory wraps actual native fits without
recomputation. The inference agent owns the coordinated GMM/GEL request migration.
Unavailable numerical results remain reportable.

Updated `docs/reference/estimand.md` shows explicit quantity/estimator composition
and separates numerical availability from inference assumptions and qualification.
The supplied fixed-weight example and regression fixture explicitly supply an
identity weight; the native fixed-weight contract is unchanged.

The documentation application's frozen covariance consumer adapter now composes
`Estimand(Covariance(), estimator=LedoitWolf())`. It retains the original
innovation/total-risk arithmetic and source snapshot, changes no downstream
production file, and continues deriving the adapted code during verification.
A focused adapter control preserves the innovation expression and exact
estimated sigma/shrinkage. The worker's independent dense-Gaussian and
least-squares risk checks remain present. Execution is coordinator-owned.

The authored README/site/reference/concept pages now orient readers around five
API entry points. Survival examples use one `SpatialSurvivalData` input,
explicit prediction kinds/grids and prepared refinement; geometric and GEL
examples bind requests before `.test(Hypothesis(...))`. Numerical-family and
proof/source/calibration limits remain explicit. The onboarding budget and
generated documentation regions are unchanged.

Controls in `tests/api/test_estimand.py` cover independent named covariance,
row/column identities, preparation ownership, cross-model rejection, structural
estimator identity, unavailable reporting and concrete return typing. Existing
covariance and moment tests retain their independent calculations and assertions.
Native covariance tests add typed-problem JIT/batching and an analytic covariance
trace derivative witness. These are proposed checks, not executed evidence.

Locked scoped Ruff format/check passed for owned source/tests. No numerical
tests, build or aggregate checks were run by this agent. Coordinator selection:

```bash
just python::statadic::test-fresh tests/api/test_estimand.py tests/api/test_covariance.py tests/api/test_moments.py tests/api/test_moments_reporting.py tests/estimation/test_covariance.py
```

Additional migration assistance moved native panel flags into `PanelProblem` in
`tests/estimation/test_panel_reference.py`, `test_panel_validation.py`,
`tests/inference/test_panel_covariance_reference.py` and
`test_panel_covariance_validation.py`. Native FE-intercept controls and all
reference assertions were preserved. Scoped Ruff passed; execution is pending.
