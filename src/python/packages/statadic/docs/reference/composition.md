# Composable analysis workflows

**Align → Specify → Estimate → Inspect → Predict or infer → Report.**

Use `Regression` for an outcome relationship, `Ordination` for coordinates and
`Estimand` for covariance or supplied moments. Ordination composes with Regression
through the fitted design. Mathematical operations and native estimators remain
directly available.

## Lifecycle and retained state

`model.estimate(data)` performs `prepare` followed by `estimate_prepared`.
Preparation belongs to that exact model instance; another instance, including
one created by `dataclasses.replace`, rejects its state. Predictor preparation
learns training transformations once. It uses the internal `Term._prepare_term`
extension hook so Ordination retains its ordinary model lifecycle.

Results hold their native fit and context. `report()` inspects
that evidence. Prediction, covariance and testing belong to the families that
implement them. A geometric decomposition has no fabricated coefficient vector
or outcome prediction.

## Observations and measures

Each outer row is one statistical observation. `Observations` aligns scalar,
categorical and empirical-measure columns by exact identities. Atom masses,
regression precisions, graph weights and weights between measures have different
meanings.

Regression accepts named mappings and supported eager frames. Only declared
roles are converted; later inference can name an additional fit-time column,
such as `Clustered("firm")`. Fit-time mappings are copied. Prediction mappings
broadcast scalars over vector rows, while measures never broadcast.

Missing required values and misaligned identities raise. Select rows jointly.
Caller precision governs computation; explicit wide arrays require application-
enabled JAX x64 or an explicit source cast. See
[table preparation](../concepts/observations-weights.md) and [precision capabilities](../capabilities.md).

## Compare, embed and test

```python
from statadic.api.comparison import PermutationPlan
from statadic.api.design import Categorical, Intercept
from statadic.api.hypotheses import Hypothesis
from statadic.api.ordination import DistanceGram, Ordination, PrincipalCoordinates
from statadic.api.regression import GeometricResponse, Reduced, Regression, ResponsePermutation
from statadic.api.representation import MMD
from statadic.geometry.ground.similarities import RBFKernel

geometry = MMD(RBFKernel(bandwidth=1.0)).prepare(
    training.measure("distribution"), row_ids=training.row_ids,
)
coordinates = Ordination(DistanceGram(), PrincipalCoordinates(dim=2)).estimate(geometry)
comparison = Regression(
    GeometricResponse(geometry, distance_power=2),
    Intercept() + Categorical("treatment"),
).estimate(training)
test = comparison.infer(
    resampling=ResponsePermutation(PermutationPlan(n=999, seed=42)),
).test(Hypothesis(Reduced(("intercept",))))
```

Raw-distance DISCO uses `distance_power=1`; squared-distance PERMANOVA uses
`distance_power=2`. Neither runs or truncates an ordination first. Geometry
construction evidence and the observed signed spectrum are distinct: an
observed PSD matrix cannot establish a population law.

### Adjusting for nuisance predictors

Fit the full design, then select a nested reduced design from its retained
column names:

```python
from statadic.api.design import Linear
from statadic.api.hypotheses import Hypothesis
from statadic.api.regression import FreedmanLane, Reduced

full = Regression(
    GeometricResponse(geometry, distance_power=2),
    Intercept() + Linear("age") + Categorical("treatment"),
).estimate(training)
test = full.infer(
    resampling=FreedmanLane(PermutationPlan(n=999, seed=42)),
).test(Hypothesis(Reduced(("intercept", "age"))))
```

The reduced design reuses fitted columns and never learns another encoding or
representation. Reports retain actual ranks and degrees of freedom.
Freedman–Lane calibration is approximate with general nuisance adjustment,
including when every permitted schedule is enumerated. Its interpretation
requires the qualified geometry, mean structure and exchangeability assumptions.
Decomposition availability alone is not evidence of calibrated inference.
The [hypothesis guide](hypotheses.md) also covers named coefficient restrictions
and the hypotheses constructed by residual and Hansen-J diagnostics.

### Assignment units and exchangeability blocks

`PermutationPlan(unit="subject", blocks="site")` declares assignment units
and exchangeability restrictions. Existing one-factor exact enumeration,
supplied arrangements, tail conventions and failed-draw accounting are retained.
Adjusted response permutations must preserve the nuisance design. Blocks alone
do not establish robustness to arbitrary heteroskedasticity or dependence.

PERMDISP remains a separate dispersion operation with its own approximate
residual-permutation assumptions. Its normal lifecycle accepts either constructed
geometry or `ObservedDissimilarity` inside
`PERMDISP().estimate(ComparisonInput(geometry, groups, group_ids))`.
Observed matrices retain their signed spectrum and withhold dispersion when
material negative inertia prevents the required Euclidean representation; they
gain no population-law warrant. Mantel retains its object-permutation contract.

## Predict from distributions

`DistanceToLandmarks` retains training measures, masses, identities and geometry.
An Ordination term instead retains the fitted coordinate map. Both produce
ordinary scalar predictors and compose with linear or supported count laws.
Prepare learned terms inside each training fold.

```python
from statadic.api.design import DistanceToLandmarks, Linear
from statadic.api.inference import HC
from statadic.api.regression import Poisson

features = Intercept() + Linear("price") + DistanceToLandmarks(
    "distribution", geometry=MMD(RBFKernel(1.0)),
    landmarks=("training-a", "training-b"),
)
fit = Regression("outcome", features).estimate(training)
uncertainty = fit.infer(covariance=HC("HC3"))
counts = Regression("count", features, law=Poisson(), exposure="hours").estimate(training)
```

Inspect `prediction_options()` before choosing a target. Covariance and reference
laws remain separate choices. Learned coordinates require refitting within a
resampling procedure before it can claim representation-learning uncertainty.

## Reports, persistence and model scope

`fit.save_predictor(path)` stores supported frozen regression predictors.
Ordination uses the same method and
`load_ordination_predictor(path)` restores a transform-only artifact.
Nested ordination terms persist their training maps inside regression designs.

Schema 6 uses closed JSON records and checked little-endian numeric payloads.
Earlier prerelease schemas are rejected. Arbitrary executable classes, closures
and pickle are unsupported. Restored predictors provide no fabricated fitted
inference. See [artifact contracts](artifacts.md).

Supplied moments do not need an outcome model:

```python
from statadic.api.estimand import Estimand, MomentConditions
from statadic.api.moments import MomentData
from statadic.core.equations import PlainLongRun
from statadic.estimation.gmm import IteratedGMM

result = Estimand(
    MomentConditions(), estimator=IteratedGMM(PlainLongRun),
).estimate(
    MomentData.from_arrays(problem, row_ids=observation_ids),
)
```

GMM/GEL retain their native fits, original weights and explicit inference
assumptions. The [Estimand reference](estimand.md) also covers covariance of
observed vectors. Report generation does not rerun inference.

(replace-an-estimator)=

## Replace an estimator

`Regression("y", terms, structure=PanelLayout(...), estimator=Pooled())`
and the corresponding `Within(...)` specification reuse the same aligned
observations. Effects and identification assumptions change; covariance is still
a separate decision. Unit, time and two-way fixed effects reject an explicit
`Intercept()` term; their retained effects supply the location component.

```{literalinclude} ../../examples/change_estimator.py
:language: python
```

(change-geometry)=

## Change geometry

Ground geometry changes the scientific question. Angular geometry ignores radial
scale; Euclidean geometry retains it.

```{literalinclude} ../../examples/change_geometry.py
:language: python
```

## Spatial and panel paths

(specify-what-spatial-weights-act-on)=

`structure=PanelLayout(...)` selects observation identities.
Effects, named exposure lags and stochastic dependence remain separate.
`SpatialX`, `SpatialLag` and `SpatialError` live in `statadic.api.spatial.graphs`;
Regression lowers supported combinations to their existing numerical owners.

Full graph nodes remain distinct from observed outcome rows. Missing responses
on active nodes are marginalized by the supported observed likelihood.
Dyadic structures preserve endpoint identities and node-count bootstrap weights.

See [spatial models](../spatial-models.md),
[survival models](../survival-models.md) and
[workflow-specific limits](../validation/known-limitations.md).

## Extension and migration

The API depends on scientific owners, never the reverse. Compatible estimators
implement the narrow problem-to-fit capability. Extend an actual family and
qualify its supported combinations; no universal model compiler is required.

The prerelease consolidation removes Equation, Conditional and the competing
family model constructors without aliases. Scientific labels such as OLS, PCA
and PERMANOVA remain useful names for the resulting procedures.

[Mathematical foundations](../formal/index.md), independent numerical comparisons
and [statistical studies](../validation/index.md) have different scopes.
Numerical success does not establish identification or calibration.
