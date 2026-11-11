# Estimand: estimate a declared quantity

`Estimand` specifies a quantity and its numerical estimator. This initial host
workflow supports covariance and supplied fixed or affine moments. It does not
turn every numerical component into a model or attach a sampling law by default.

`Estimand` is the host workflow that declares a quantity and estimates it. It is
distinct from `statadic.core.statistics.PopulationEstimand`, the root of the
population-estimand axis that names what a statistic targets (a discrepancy, a
dispersion, a transformed estimand or a `LinearPredictorContrast`). Naming a
target there does not run an estimator.

## Covariance of named observations

```python
from statadic.api.data import Observations
from statadic.api.estimand import Covariance, Estimand
from statadic.estimation.covariance import LedoitWolf, SampleCovariance

observations = Observations(
    {"a": [1.0, 4.0, 2.0, 8.0], "b": [0.0, 3.0, 7.0, 2.0]},
    row_ids=("u", "v", "w", "x"),
)
model = Estimand(Covariance(columns=("a", "b")), estimator=SampleCovariance(ddof=1))
result = model.estimate(observations)
covariance = result.sigma
print(result.report().to_text())

shrunk = Estimand(Covariance(("a", "b")), estimator=LedoitWolf()).estimate(observations)
```

Columns are explicit and retain their order. A rectangular array can instead
use `Covariance()`; generated variable labels do not assert external identities.
`SampleCovariance` uses the declared `n - ddof` divisor. `LedoitWolf` uses the
existing constant-correlation target and its `ddof=0` sample convention. These
estimators are numerical bricks, not competing host model constructors.

Preparation owns array snapshots. `prepared = model.prepare(observations)` can
be reused by `model.estimate_prepared(prepared)` on that same model instance.
The covariance result retains estimator, variable labels and row identities.
Singular covariance is allowed; a finite matrix does not promise an inverse,
parameter standard errors or confidence intervals.

## Supplied moment conditions

```python
import jax.numpy as jnp

from statadic.api.estimand import MomentConditions
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
quantity = Estimand(MomentConditions(), estimator=FixedWeightGMM(PlainLongRun))
fitted = quantity.estimate(data)
uncertainty = fitted.infer(covariance=GMMSandwich())
print(uncertainty.report().to_text())
```

The affine convention is `g_i(theta) = g0_i + D_i @ theta`. Rows, derivatives
and an optional supplied weight live only in `MomentData.problem`, already
aligned to its observation identities. `MomentConditions()` owns no sample.
Preparation snapshots these arrays without sorting, dropping rows or inventing
a response. GMM and fixed/affine GEL retain their existing result types and
method-specific inference capabilities. Unsupported fitting/result combinations
do not acquire a generic fallback.

Inference consumes actual retained weights, moment maps and solver evidence.
Numerical availability, convergence, identification and reference assumptions
remain separate. In particular, a covariance request does not establish correct
moments or justify an overidentification reference.

GEL uses `fitted.infer(calibration=GELCalibration(...))` followed by
`.test(Hypothesis(OveridentifyingRestrictions(...)))`. This explicit request
chooses LR, LM or Wald construction and acknowledges the iid reference
assumption. It constructs neither a sampling covariance nor a resampling
schedule, so its keyword retains that distinct meaning. Affine GEL currently
supports the profiled LR test only; omitting the iid acknowledgement leaves
finite test statistics descriptive.

Inspect the [reference comparisons](../validation/references.md),
[statistical studies](../validation/research.md) and
[known limitations](../validation/known-limitations.md) for their exact scope.
This host composition grants no additional proof, source approval or Monte Carlo
qualification. Native numerical problems remain the qualified JAX boundary;
host alignment, preparation and reporting stay eager.
