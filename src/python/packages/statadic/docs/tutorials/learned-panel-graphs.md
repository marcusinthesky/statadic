# Fixed and changing graphs in a return panel

The observation is a firm-date return. The graph acts on firms within a date.
A measure-valued input can have hundreds of articles per firm and only one
graph row per firm. Keep the ordered integer firm IDs alongside the graph.
Encode string tickers with one stable codebook across all dates.

## Fit with one graph per date

The example constructs two spatial fields from observed firm coordinates. The
outcome lag uses an evolving field. The news field supplies a lag of a
firm-varying characteristic. Both fields operate on the same four firms each
period. The fixed distance matrix would also work as a repeated outcome graph
by setting `graphs["info"] = fixed`.

```python
import numpy as np
from statadic.api.spatial.graphs import PanelData, SpatialLag, SpatialX
from statadic.api.regression import Regression
from statadic.api.data import Observations
from statadic.api.design import Intercept, Linear
from statadic.api.spatial.graphs import DistanceGraph, RBFWeights
from statadic.api.data import PanelLayout

rng = np.random.default_rng(17)
ids = (0, 1, 2, 3)
years = tuple(range(2010, 2026))
nodes = np.array([0.1, 0.5, 1.2, 2.0])


def field(coords):
    d = np.abs(coords[:, None] - coords[None, :])
    return DistanceGraph.from_distances(d, unit_ids=ids, weighting=RBFWeights(0.6))


fixed = field(nodes)
by_year = tuple((t, field(nodes + rng.normal(0, 0.15, 4))) for t in years)
x = rng.normal(size=(len(years), 4))
y = []
for k, (_, source) in enumerate(by_year):
    w = np.asarray(source.realize().values)
    z = np.asarray(fixed.realize().values) @ x[k]
    y.append(
        np.linalg.solve(
            np.eye(4) - 0.35 * w, 1.0 + 1.5 * x[k] + 0.6 * z + rng.normal(size=4)
        )
    )
rows = Observations(
    {
        "firm": np.tile(ids, len(years)),
        "year": np.repeat(years, 4),
        "size": x.ravel(),
        "return": np.asarray(y).ravel(),
    }
)
data = PanelData(
    rows, PanelLayout("firm", "year"), graphs={"info": by_year, "news": fixed}
)
model = Regression(
    "return",
    Intercept() + Linear("size") + SpatialX("size", "news"),
    dependence=SpatialLag("info"),
)
fit = model.estimate(data)
report = fit.report()
```

Use `from_costs` for squared Wasserstein costs that already enter the
exponential as `exp(-cost / h)`. `from_distances` squares its input once.
A supplied `SpatialWeights` works alongside either graph recipe. A common
market factor repeated across firms is unchanged by row-normalized weights.
Including both its original and lagged columns makes the design rank deficient.

For two simultaneous outcome fields pass
`dependence=SpatialLag("info") + SpatialLag("news")`. For different outcome
and disturbance fields use `SpatialLag("info") + SpatialError("news")` with
`SpatialError` imported from `statadic.api.spatial.graphs`. The likelihood forms one
outcome filter and one disturbance filter per date. Each filter combines its
own coefficients and operators before taking its log determinant. Fixed or
time-varying firm rosters must match across roles within each period. The
observed-outcome path builds weights on the full roster and marginalizes over
missing returns. Truncating the graph to firms with observed returns changes
the model. Generalized graph ML supports pooled and fixed effects.

## Learn from the article coordinates

A `MeasureColumn` binds one `PackedClouds` collection to its firms. Map every
active article coordinate before comparing the empirical distributions. This
preserves the atom masses and avoids assigning a firm-level target to each
article. For a frozen external metric-learn transformation, pass
`projection=jnp.asarray(fitted_metric.components_.T)` with the original
preprocessing. A precomputed firm-by-firm distance matrix cannot reconstruct
the input coordinates.

```python
import jax.numpy as jnp
from statadic.api.data import MeasureColumn
from statadic.api.spatial.learning import GraphML, LearnLinear
from statadic.api.spatial.graphs import ProjectionRef
from statadic.api.representation import Energy
from statadic.sample.packed import PackedClouds

articles = PackedClouds.from_clouds([jnp.asarray(rng.normal(size=(5, 3))) for _ in ids])
semantic = ProjectionRef("semantic")
info = DistanceGraph.from_measures(
    MeasureColumn(ids, articles),
    projection=semantic,
    geometry=Energy(),
    weighting=RBFWeights(0.6),
)
data = PanelData(rows, data.panel, graphs={"info": info, "news": fixed})
model = Regression(
    "return",
    Intercept() + Linear("size") + SpatialX("size", "news"),
    dependence=SpatialLag("info"),
    estimator=GraphML(projections={semantic: LearnLinear(rank=1)}),
)
joint = model.estimate(data)
```

Inspect `joint.valid` and `joint.report()` before interpreting the example.
The synthetic returns above were generated using the observed firm coordinates,
so fitting the unrelated article clouds need not recover the generating graph.
A named projection shared across recipes or periods has one fitted array.
With period-specific article clouds use the same reference in each dated recipe
to learn a common map. Use different references if the research design
identifies separate period-specific maps.

`Energy()` supplies the squared rooted V-energy directly.
`SlicedWasserstein(ProjectionDesign(...))` reuses directions in the projected
coordinate space. Slices are piecewise differentiable at order changes.
`SinkhornDivergence(epsilon=..., iterations=..., tolerance=...)` uses a
debiased squared-Euclidean entropic cost and a fixed number of log-domain
updates. Failed marginal checks return an unavailable numerical value. The OT
regularizer epsilon and the graph bandwidth have distinct meanings.
Barycentric graphs can enter as supplied weights. Their assignments and
reconstruction require a separately qualified differentiated solver.

## Choose a bandwidth and evaluate uncertainty

Use `RBFWeights(BandwidthRef("h"))` and
`GraphML(bandwidths={BandwidthRef("h"): LearnBandwidth(initial=0.6)})` to
estimate a positive bandwidth by likelihood. The joint estimator requires a
fixed projection norm when both the map and its bandwidth are fitted. For
squared Euclidean/Wasserstein costs, multiplying the map by c and h by c²
preserves the weights. Different rotations can also represent the same geometry.

For held-out selection construct separate `training` and `validation`
`PanelData` objects, both using the named bandwidth reference, then call

```python
from statadic.api.spatial.graphs import BandwidthRef
from statadic.api.spatial.selection import select_bandwidth

selection = select_bandwidth(
    Regression("return", Intercept() + Linear("size"), dependence=SpatialLag("info")),
    training,
    validation,
    parameter=BandwidthRef("h"),
    candidates=(0.2, 0.6, 1.8),
)
```

The selection result retains every candidate score and both period ranges.
Scores are summed conditional joint Gaussian log densities with fitted
coefficients and scale frozen. Supplying an explicit `refit` panel fits the
chosen bandwidth again on the union of the two windows. Reserve another later
window for final performance. Fit factor residuals, pair labels, initialization
and the corpus cutoff using only information allowed by the design. A
retrospectively measured corpus answers a different question from a live
point-in-time forecast.

Fixed-graph inference conditions on the graph. A learned map can disappear
from the likelihood at zero spatial dependence if no other active role uses it.
Published estimates should label that conditioning and the inference window.
Jointly trained geometry has no automatic Wald covariance. For full-procedure
uncertainty, a justified resampling design needs to repeat the supervised
learning and any selection on the relevant training periods.

## Save and rebuild

`save_analysis(path, fit.report())` stores a rendering snapshot.
`save_graph_predictor(path, fit)` stores a restricted frozen spatial predictor.
`save_graphs(path, {"info": info}, parameters=joint.graph_parameters)` stores
the upstream recipe, masks, masses, checked identities and shared parameter
values. `load_graphs(path)` returns recipes and parameters that can reconstruct
the weights. Use distinct artifact directories. A new firm changes row
normalization and the simultaneous system. Rebuild an explicitly aligned
universe before comparing predictions.

## Train a neural projection through SAR

The executable example below trains a Flax NNX **6 → 8 → 2** MLP on eight
units, each with three vectors, and 24 independent simulated periods. Each
unit's projected vectors remain an empirical distribution. Euclidean V-energy
compares those distributions; zero-diagonal RBF weights enter the Gaussian SAR
profile likelihood, including its log determinant. The MLP and a bounded rho
are updated together, while beta and innovation variance are profiled out.
Bandwidth is fixed at one. The teacher uses a separate NumPy implementation
with rho = 0.35, intercept = 0.4, slope = 0.7 and innovation SD = 0.15.

Run from a prepared repository checkout:

```bash
devenv shell -- uv run --locked --project src/python/apps/docs --package statadic-docs python src/python/packages/statadic/examples/neural_sar.py
```

Flax is a dependency of the documentation environment, rather than the STATADIC
wheel. For an installed STATADIC environment, install `flax>=0.12,<0.13` as well
before running the example.

```{literalinclude} ../../examples/neural_sar.py
:language: python
```

{download}`Download the complete script <../../examples/neural_sar.py>`

The example checks a composition and improves a training objective. Rotations
and other neural parameter symmetries prevent interpreting it as recovery of a
unique neural network. The final `Regression` refits SAR with the learned graph
frozen; that report conditions on the graph and excludes training uncertainty.
There is no new built-in neural estimator or automatic joint Wald covariance.

For return-supervised encoder experiments, a JAX encoder can supply
`clouds.with_coordinates(encoder(tokens, prefix))` to a pure cost and
`rbf_weights` evaluation inside a scalar `graph_profile` objective. Cache
tokenization and rosters, and recompute graph values when the prefix changes.
An encoder checkpoint, tokenizer and input text are external dependencies of
that experiment. The built-in optimizer fits linear maps and bandwidths.

## Sources and reproducible evidence

- Elhorst (2003), [Specification and Estimation of Spatial Panel Data Models](https://doi.org/10.1177/0160017603253791).
- Székely and Rizzo (2013), [Energy statistics](https://doi.org/10.1016/j.jspi.2013.03.018).
- Bonneel et al. (2015), [Sliced and Radon Wasserstein Barycenters of Measures](https://doi.org/10.1007/s10851-014-0506-3).
- Feydy et al. (2019), [Interpolating between Optimal Transport and MMD using Sinkhorn Divergences](https://proceedings.mlr.press/v89/feydy19a.html).
- [metric-learn Mahalanobis factor convention](https://contrib.scikit-learn.org/metric-learn/generated/metric_learn.base_metric.MahalanobisMixin.html) and [PySAL fixed-effects panel lag](https://pysal.org/spreg/generated/spreg.Panel_FE_Lag.html).
- [Benchmarks, external comparisons and Monte Carlo size and power](https://github.com/marcusinthesky/statadic/blob/main/src/python/apps/studies/spatial-graph-evidence.md).

The [time-block graph refit guide](../guides/graph-resampling.md) shows the
implemented bootstrap for balanced pooled and unit-fixed-effects panels,
including failed-refit handling and which parameter intervals it supports.
