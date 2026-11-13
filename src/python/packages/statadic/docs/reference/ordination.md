---
type: reference
description: Represent observations by coordinates from feature, kernel, distance or graph geometry, and carry the fitted map into regression.
---

# Ordination

## Usage

```python
from statadic.api.data import Observations
from statadic.api.ordination import Ordination

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

{py:class}`~statadic.api.ordination.Ordination` pairs a geometric construction
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
from statadic.api.ordination import (
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
`statadic.api.representation` and fit `Ordination.pcoa(dim=...)`.

By default, material negative inertia raises an error, and setting
`negative_inertia="positive-subspace"` keeps the full signed spectrum and
reports that coordinates approximate its positive part. With `dim=None`, every
positive axis is kept.

## Results

| Operation | What it does |
| --- | --- |
| `result.transform(query)` | Coordinates for new rows from the frozen training map |
| `result.loadings` | Feature directions, for linear constructions only |
| `result.spectrum` | Signed eigenvalues, the numerical cutoff and axis counts |
| `result.capabilities` | Which query sources the result can use |
| `result.transform_distances(cross, reference_ids=ids)` | Coordinates from query-to-training distances, in the stored order |

Eigenvalues are in centred-Gram units, whereas for linear PCA the
`explained_variance` divides by the training sample divisor. Variance and ratio
arrays cover the selected axes, and ratios divide by the full feature variance,
so a truncated selection sums to less than one. Kernel centring uses training
statistics, and Isomap attaches each query to the frozen training paths on its
own, which keeps the coordinates of every other query fixed.

An ordination term composes directly into a regression:

```python
from statadic.api.design import Intercept
from statadic.api.regression import Regression

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
- Reference comparisons: the {download}`ordination controls <../../tests/api/test_ordination.py>`
  compare covariance, centred kernels, independent shortest paths and held-out
  projections.
- Artifacts: the {download}`artifact controls <../../tests/api/test_ordination_artifacts.py>`
  exercise standalone and nested frozen maps.

## See also

- [Core concepts](../concepts/index.md) and [geometry](../concepts/geometry.md)
- [Regression](regression.md), which accepts ordination terms
- [Energy](energy.md) for distances between distributions
