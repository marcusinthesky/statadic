---
type: reference
description: Represent observations by coordinates from feature, kernel, distance or graph geometry, and carry the fitted map into regression.
---

# Ordination

## Usage

```python
from statadic.observation.data import Observations
from statadic.ordination.fit import Ordination

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
print(coordinates, result.loadings)
```

## Description

{py:class}`~statadic.ordination.fit.Ordination` pairs a geometric construction
with a coordinate criterion. The construction says how similarity between
observations is defined, and `PrincipalCoordinates` turns the centred geometry
into a small number of axes. The fitted result keeps everything needed for the placement of
new observations, so the same map applies to held-out rows and to the
predictors in a regression.

`Ordination.pca(...)` is shorthand for a linear Gram construction with
principal coordinates. It centres the features and optionally scales them by
the training sample standard deviation, with `scale=False` as the default, and
its SVD works on the feature matrix so that no Gram matrix of observations is
formed.

## Options

| Construction | Problem solved | What the result keeps for queries |
| --- | --- | --- |
| `LinearGram(features=...)` | Centred feature factor, by SVD | Feature order, means, scales and loadings |
| `KernelGram(kernel, features=...)` | Centred positive-semidefinite kernel | Kernel, training features and centring state |
| `DistanceGram()` | Double-centred squared distances you supply | Ordered training references |
| `DistanceGram(ShortestPathDistances(...))` | Graph paths, then classical scaling | Training paths and neighbour rule |

```python
from statadic.ordination.fit import (
    DistanceGram,
    KernelGram,
    PrincipalCoordinates,
    ShortestPathDistances,
)
from statadic.geometry.ground.similarities import RBFKernel

kernel_components = Ordination(
    KernelGram(RBFKernel(bandwidth=1.2), features=("x", "z")),
    PrincipalCoordinates(dim=2),
)
graph_components = Ordination(
    DistanceGram(ShortestPathDistances(neighbors=2, features=("x", "z"))),
    PrincipalCoordinates(dim=2, negative_inertia="positive-subspace"),
)
kernel_fit = kernel_components.estimate(training)
graph_fit = graph_components.estimate(training)
```

The first composition is kernel PCA and the second is Isomap. For distances
between empirical distributions, bind a producer such as
`DistanceGram(Energy(), column="distribution")`, where [Energy](energy.md)
supplies the rooted distances between the measures. The column is a
`MeasureColumn` passed to `Observations(..., measures={"distribution": column})`.
For an observed square matrix of distances, wrap it with
`ObservedDissimilarity.from_array(distances, row_ids=ids)` from
`statadic.observation.geometry` and fit `Ordination.pcoa(dim=...)`.

By default, material negative inertia raises an error, and setting
`negative_inertia="positive-subspace"` keeps the full signed spectrum and
reports that coordinates approximate its positive part. With `dim=None`, every
positive axis is kept.

### Empirical distributions and frozen rotations

The same measure column can supply centroid or full-distribution geometry.
`MMD(LINEAR_KERNEL)` equals Euclidean distance between probability-weighted
centroids. `Wasserstein(p=2, ground=EUCLIDEAN)` compares the empirical
distributions themselves, including information their centroids discard.
It accepts nonuniform probabilities and unequal support sizes; omitted masses
are uniform within each measure. Atom coordinates are used as supplied, so
normalization or a topic-probability transformation belongs in the declared
preparation applied before either geometry.

```python
from pathlib import Path

import jax.numpy as jnp

from statadic.geometry.ground.metrics import EUCLIDEAN
from statadic.geometry.ground.similarities import LINEAR_KERNEL
from statadic.measure.clouds import PackedClouds
from statadic.observation.data import MeasureColumn, Observations
from statadic.observation.design import Intercept
from statadic.observation.geometry import MMD, Wasserstein
from statadic.ordination.fit import Ordination
from statadic.prediction.artifacts import load_ordination_predictor, load_predictor
from statadic.regression import Regression

ids = ("a", "b", "c", "d", "e", "f")
clouds = PackedClouds.from_clouds(
    [
        [[0.0, 0.0]],
        [[1.0, 0.0], [1.0, 1.0]],
        [[0.0, 2.0]],
        [[2.0, 1.0], [3.0, 1.0]],
        [[3.0, -1.0]],
        [[-1.0, -1.0], [-1.0, 0.0]],
    ],
    masses=[[1.0], [0.25, 0.75], [1.0], [0.5, 0.5], [1.0], [0.75, 0.25]],
)
firms = Observations(
    {"y": [1.0, 2.0, 1.5, 3.2, 4.1, 0.7]},
    row_ids=ids,
    measures={"articles": MeasureColumn(ids, clouds)},
)
geometry = Wasserstein(p=2, ground=EUCLIDEAN)
coordinates = Ordination.pcoa(
    geometry,
    column="articles",
    dim=2,
    negative_inertia="positive-subspace",
    name="news",
).estimate(firms)
centroid_geometry = MMD(LINEAR_KERNEL)  # Alternative geometry on the same clouds.
centroid_coordinates = Ordination.pcoa(
    centroid_geometry, column="articles", dim=2, name="centroid"
).estimate(firms)

# Rotate the retained basis, or align to reference coordinates by observation ID.
rotated = coordinates.rotate(jnp.asarray([[0.0, -1.0], [1.0, 0.0]]))
aligned = coordinates.align_to(rotated.coordinates[::-1], row_ids=ids[::-1])
query_ids = ("new-firm",)
query = Observations(
    {},
    row_ids=query_ids,
    measures={
        "articles": MeasureColumn(
            query_ids, PackedClouds.from_clouds([[[0.5, 1.0], [1.0, 1.5]]])
        )
    },
)
new_coordinates = aligned.transform(query)

# Keep the already learned map when estimating the outcome model.
fit = Regression("y", Intercept() + aligned.as_term()).estimate(firms)
aligned.save_predictor(Path("news-coordinates"))
fit.save_predictor(Path("news-regression"))
saved_map = load_ordination_predictor(Path("news-coordinates"))
saved_model = load_predictor(Path("news-regression"))
saved_coordinates = saved_map.transform(query)
predictions = saved_model.predict(query)
```

Both artifacts retain the training measure references, probabilities, geometry
and rotation. Held-out queries use those references independently of other
query rows. Explicit cross-distance input must retain the exact training
reference order: use
`aligned.transform_distances(cross, reference_ids=aligned.row_ids)`.

`rotate` accepts orthogonal changes of basis, including reflections.
`align_to` learns an orthogonal Procrustes transform using the same unique
observation identities, reorders reference rows by identity and rejects a
rank-deficient alignment. Neither operation estimates translation, scaling or
semantic labels. Rotated columns generally cease to be individual principal
axes; the original signed spectrum remains available on the live result.
`as_term()` freezes a learned map explicitly. Passing the unfitted `Ordination`
specification into `Regression` instead learns its map on that regression's
training sample. Choose the learning sample inside the intended validation
design; freezing a map does not remove information leakage from its construction.

General multivariate Wasserstein has no Hilbertian construction warrant.
The strict policy therefore rejects this producer before estimating coordinates;
an observed finite-sample spectrum cannot supply its missing construction law.
The positive-subspace policy is an explicit approximation when the fitted
distance geometry has material negative inertia; it is not a statement that
the distance preserves all information in the retained coordinates.
Exact unregularized transport, finite-projection sliced transport and an
entropically regularized objective are different statistical declarations.

The native weighted solver currently serves small-support comparisons. A
synthetic float64 CPU benchmark with 16-dimensional atoms measured median warm
times of 4.4 ms at 16×16, 222 ms at 32×32 and 8.8 s at 64×64; runs containing
one first execution and three warm executions at 128×128 and 128×96 exceeded
60 seconds. The equal-uniform 128×128 assignment path measured 2.2 ms warm.
These measurements exclude corpus preparation and establish no GPU throughput.
Measure the chosen support sizes before scheduling all-pairs weighted analyses.
The scalar workflow returns an unavailable distance when it detects cost-power
underflow; host preparation rejects it. Lower result/inspection doors retain
the candidate and its evidence for diagnosis.

Detailed pair explanations stay with the transport family:

- `statadic.geometry.discrepancy.transport.decomposition.euclidean_wasserstein2_decomposition`
  reports total squared W2, squared centroid separation, centered-distribution
  squared W2 and reconstruction error. This identity requires Euclidean ground
  geometry.
- `statadic.geometry.discrepancy.transport.exact.exact_balanced_wasserstein_inspect`
  retains a sparse assignment and numerical evidence for one supported
  uniform equal-count comparison. `balanced_wasserstein_from_cost_inspect`
  accepts supplied unsquared ground costs. One optimal assignment need not
  provide a unique explanation.
- `statadic.geometry.discrepancy.transport.empirical.wasserstein_distance_result`
  retains the selected weighted pair's coupling, scaled costs and solver
  evidence. It does not promise a derivative through the optimum.

Distance-matrix production retains scalar distances; request detailed
couplings only for selected pairs rather than storing every all-pairs plan.

## Results

| Operation | What it does |
| --- | --- |
| `result.transform(query)` | Coordinates for new rows from the frozen training map |
| `result.loadings` | Feature directions, for linear constructions only |
| `result.spectrum` | Signed eigenvalues, the numerical cutoff and axis counts |
| `result.capabilities` | Which query sources the result can use |
| `result.transform_distances(cross, reference_ids=ids)` | Coordinates from query-to-training distances, in the stored order |
| `result.rotate(rotation)` | A frozen orthogonal change of retained coordinate basis |
| `result.align_to(reference, row_ids=ids)` | Identity-matched orthogonal Procrustes alignment |
| `result.as_term()` | An already learned map bound as a frozen regression term |

Eigenvalues are in centred-Gram units, whereas for linear PCA the
`explained_variance` divides by the training sample divisor. Variance and ratio
arrays cover the selected axes, and ratios divide by the full feature variance,
so a truncated selection sums to less than one. Kernel centring uses training
statistics, and Isomap attaches each query to the frozen training paths on its
own, which keeps the coordinates of every other query fixed.

An ordination term composes directly into a regression:

```python
from statadic.observation.design import Intercept
from statadic.regression import Regression

fit = Regression("y", Intercept() + components).estimate(training)
predictions = fit.predict(query)
retained = fit.design.ordination("components")
print(retained.loadings, retained.spectrum)
```

The term learns from the training predictors and carries its map into
prediction and persistence. Give independent terms distinct `name=` values,
including inside interactions, and inspect them by name to read the training
decomposition. A frozen predictor saved with `fit.save_predictor(path)` keeps
the map but drops live spectrum inspection.
The [principal component regression guide](../guides/principal-component-regression.py)
shows the full workflow.

## Methods and formulas

Classical scaling eigendecomposes the double-centred squared-distance matrix
$B = -\tfrac12 J D^{(2)} J$ with $J = I - n^{-1}\mathbf 1\mathbf 1^\top$
{cite:p}`gower-1966`. Kernel PCA decomposes the centred kernel matrix
{cite:p}`schoelkopf-smola-mueller-1998`, and Isomap replaces $D$ with shortest
paths on a neighbour graph {cite:p}`tenenbaum-desilva-langford-2000`. Queries
are placed by the out-of-sample extension {cite:p}`bengio-etal-2003`.

## Assumptions and limits

- The size of the neighbourhood and the kernel bandwidth are scientific
  choices, and Isomap rejects a disconnected graph instead of bridging its
  components or dropping rows.
- Dense graph paths need quadratic storage and cubic computation in the number
  of observations.
- A fixed dimension cannot exceed the supported positive rank of the geometry.
- The construction provides no inverse or preimage map and establishes neither
  manifold recovery nor any population law.
- Selecting dimensions or geometry with the help of the outcome needs its own
  evaluation and inference argument, so fit the whole composition inside each
  training fold.
- Host preparation is eager, and repeated eigenvalues, changing neighbourhoods
  and rank changes carry no guarantee of a smooth derivative.

## Evidence

- Mathematical results: {ref}`full-factor reconstruction <formal-STATADIC-EMBEDDING-001>`,
  {ref}`Euclidean realisation <formal-STATADIC-EMBEDDING-002>` and the
  {ref}`spectral cutoff bound <formal-STATADIC-EMBEDDING-003>`.
- Reference comparisons: the {download}`ordination controls <../../tests/ordination/test_fit.py>`
  compare covariance, centred kernels, independent shortest paths and held-out
  projections.
- Artifacts: the {download}`artifact controls <../../tests/ordination/test_fit_artifacts.py>`
  exercise standalone and nested frozen maps.
- Distribution workflow: the {download}`Wasserstein composition controls <../../tests/integration/test_wasserstein_workflow.py>`
  exercise uniform and weighted unequal supports, signed geometry, frozen
  rotations and standalone/nested predictor persistence.

## See also

- [Core concepts](../concepts/index.md) and [geometry](../concepts/geometry.md)
- [Regression](regression.md), which accepts ordination terms
- [Energy](energy.md) for distances between distributions
