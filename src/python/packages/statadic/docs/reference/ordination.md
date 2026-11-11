# Ordination: represent geometry with coordinates

{py:class}`~statadic.api.ordination.Ordination` composes a geometric construction
with a coordinate criterion. Its result retains the supported training query
map. See the [core concepts](../concepts/index.md) for its role in a larger analysis.

## Start with feature geometry

```python
from statadic.api.data import Observations
from statadic.api.ordination import LinearGram, Ordination, PrincipalCoordinates

training = Observations(
    {
        "x": [0.0, 1.0, 2.0, 3.0, 4.0, 5.0],
        "z": [0.0, 0.2, 0.8, 1.8, 3.2, 5.0],
        "y": [1.0, 2.0, 1.5, 3.2, 4.1, 5.7],
    },
    row_ids=("a", "b", "c", "d", "e", "f"),
)
components = Ordination.pca(("x", "z"), dim=1, scale=True, name="components")
result = components.estimate(training)
query = Observations({"x": [1.5, 3.5], "z": [0.4, 2.5]})
coordinates = result.transform(query)
```

This is centered PCA with training sample-standard-deviation scaling. The native
SVD operates on the feature factor; it does not allocate an observation Gram
matrix merely to resemble the other constructions. `result.loadings` contains
the retained feature directions. Scaling defaults to `False`.
`Ordination.pca(...)` returns the same `LinearGram`/`PrincipalCoordinates`
composition; it introduces no separate model type or statistical assumptions.

## Change the geometric construction

The same principal-coordinate criterion serves these compositions:

| Construction | Mathematical problem | Retained query source |
| --- | --- | --- |
| `LinearGram(features=...)` | Centered feature factor, solved by SVD | Feature order, means, scales and loadings |
| `KernelGram(kernel=..., features=...)` | Centered declared PSD kernel | Frozen kernel, training features and kernel-centering state |
| `DistanceGram()` | Double-centered squared supplied distances | Ordered training references and cross-distance centering |
| `DistanceGram(ShortestPathDistances(...))` | Union-neighbor graph paths followed by classical scaling | Training graph paths, neighbor rule and original query-distance source |

```python
from statadic.api.ordination import DistanceGram, KernelGram, ShortestPathDistances
from statadic.geometry.ground.similarities import RBFKernel

kernel_components = Ordination(
    KernelGram(RBFKernel(bandwidth=1.2), features=("x", "z")),
    PrincipalCoordinates(dim=2),
)
graph_components = Ordination(
    DistanceGram(ShortestPathDistances(neighbors=2, features=("x", "z"))),
    PrincipalCoordinates(dim=2, negative_inertia="positive-subspace"),
)
```

The first is kernel PCA; the second is Isomap. The neighborhood size and kernel
bandwidth are explicit scientific choices. Isomap rejects disconnected graphs;
it does not bridge components, drop rows or select a larger neighborhood.
Edges retain original lengths, including zero-length edges between distinct
objects. Dense graph paths require quadratic storage and cubic path computation.

For empirical-measure distances, bind an existing producer, for example
`DistanceGram(Energy(ground="euclidean"), column="distribution")`. A named
measure column can also feed `ShortestPathDistances(distances=Energy(),
column="distribution", neighbors=...)`. Atom counts and masses remain inside
each observation.

## Inspect signed inertia and identities

Use `ObservedDissimilarity.from_array(distances, row_ids=ids)` from
`statadic.api.representation` for an observed square matrix, then estimate
`Ordination.pcoa(dim=...)`. The explicit
`Ordination(DistanceGram(), PrincipalCoordinates(...))` construction is equivalent.
Validation checks
observable matrix facts; it supplies no population law.

The default rejects material negative inertia. Explicit
`negative_inertia="positive-subspace"` retains the full signed spectrum and
reports that coordinates approximate its positive subspace. With `dim=None`,
all supported positive axes are retained, including an empty set for zero
inertia. Fixed dimensions cannot exceed the supported positive rank.

`result.spectrum` records signed eigenvalues, the numerical cutoff and positive
and negative axis counts. Eigenvalues use **centered-Gram units**. Linear PCA's
`explained_variance` separately divides by the training sample divisor. Feature
variance and variance-ratio arrays contain the **selected k axes**, matching
coordinate columns and loadings. Ratios divide by the complete available feature
variance, so a truncated selection need not sum to one. Full signed eigenvalues
remain available separately, including numerical zero axes.
Feature
loadings and variance conveniences are unavailable for other constructions.
Neither an observed PSD spectrum nor graph paths acquire a population
Hilbertian warrant.

Inspect `result.capabilities`. A matrix-only result requires
`transform_distances(cross_distances, reference_ids=training_ids)` with columns
in the exact retained order. A retained feature or measure producer can evaluate
its own query distances. Kernel centering uses training statistics. Isomap
attaches each query independently to frozen training paths; other queries never
change its coordinates. No inverse/preimage map is implied.

## Compose directly into regression

```python
from statadic.api.design import Intercept
from statadic.api.regression import Regression

fit = Regression("y", Intercept() + components).estimate(training)
predictions = fit.predict(query)
retained = fit.design.ordination("components")
loadings = retained.loadings
signed_spectrum = retained.spectrum
fit.save_predictor("component-regression")
```

There is no separate PCR constructor or scores wrapper. The term learns only
from training predictors and retains its transformation for prediction and
persistence. Fit the entire composition within each training fold. Selecting
dimensions or geometry using outcomes requires its own evaluation and inference
argument. A predictor requires a fixed dimension and a named transformable
source. See the {download}`complete PCR example <../../examples/principal_component_regression.py>`.

Named inspection reads the actual training decomposition; it never estimates
the ordination again. Give independent terms distinct `name=` values, including
terms inside interactions. Extension terms remain opaque to this lookup.
Frozen predictor artifacts retain their transformation maps and do not restore
live spectrum or inference inspection; retain an analysis report separately.

<details><summary>Mathematics, assumptions and evidence</summary>

Classical scaling uses the centered distance Gram {cite:p}`gower-1966`;
kernel PCA uses a centered kernel eigenproblem
{cite:p}`schoelkopf-smola-mueller-1998`. Isomap adds neighborhood paths
{cite:p}`tenenbaum-desilva-langford-2000`; frozen query extension follows
{cite:p}`bengio-etal-2003`. These sources do not establish manifold recovery,
statistical calibration or source-fidelity approval for this implementation.

The exact-real [full-factor reconstruction](../formal/claims/STATADIC-EMBEDDING-001.md)
and [Euclidean realization](../formal/claims/STATADIC-EMBEDDING-002.md) certificates
assume symmetric hollow squared distances and consistent double centering.
The [spectral cutoff bound](../formal/claims/STATADIC-EMBEDDING-003.md) excludes
eigensolver rounding and additional dimension truncation. Those certificates
do not prove the neighborhood construction recovers a manifold.

The {download}`ordination reference controls <../../tests/api/test_ordination.py>`
compare covariance, centered kernels, independent shortest paths and held-out
projections. The {download}`artifact controls <../../tests/api/test_ordination_artifacts.py>`
exercise standalone and nested frozen maps. Inspect current
[selected studies](../validation/research.md), [limitations](../validation/known-limitations.md)
and [transformation scope](../capabilities.md); registration is not retained
Monte Carlo evidence. Native fixed-shape kernels support qualified JIT/batching;
host preparation is eager. Repeated eigenvalues, changing neighborhoods, rank
changes and selected axes have no general smooth-coordinate derivative promise.

</details>
