# Write a downstream extension

An application can introduce a predictor term, a geometry, or an estimator
using public imports. Place the {download}`extension module <../../examples/external_extensions.py>`
beside your analysis script and install statadic.

## Learn a term once

The declared term specifies its required column and learns the training mean.
Its fitted term names the coefficient and applies that same mean to each future
row. With a training mean of 30, age 50 encodes as 20 in a single row or a batch.
Implement the internal `Term._prepare_term(training)` extension hook; public
model `prepare()` retains its own lifecycle. Extend the mathematical component
or supply a compatible native estimator rather than subclassing Regression.

```{literalinclude} ../../examples/external_extensions.py
:start-at: @dataclass(frozen=True)
:end-at: return FittedCentered(self.column, jnp.mean(training.numeric(self.column)))
:language: python
```

```python
from external_extensions import Centered
from statadic.api.design import Intercept
from statadic.api.regression import Regression

training = {"age": [20.0, 30.0, 40.0], "y": [1.0, 2.0, 3.0]}
fit = Regression("y", Intercept() + Centered("age")).estimate(training)
fit.predict({"age": [50.0]})  # [4.]
```

The same term can supply a predictor to binomial regression. Live prediction
uses the fitted transformation. A `RowDifference` can compare two complete
rows in that learned basis. The term-specific contrast helper recognizes
selected built-in bases, and saving a predictor requires an allowlisted codec.

## Compare distributions

`MeanDistance` compares mass-weighted means. Its rectangular output has query
rows and reference columns. The reference IDs specify the column order. Its
`matrix` method declines to supply a Hilbertian construction warrant, so strict
distance Ordination rejects this constructed geometry even for a sample with a
well-behaved matrix. Explicit observed-matrix analysis remains descriptive.

```python
import jax.numpy as jnp
from external_extensions import MeanDistance
from statadic.sample.packed import PackedClouds

reference_clouds = PackedClouds.from_clouds(
    [jnp.array([[2.0], [4.0]]), jnp.array([[10.0]])]
)
query_clouds = PackedClouds.from_clouds([jnp.array([[1.0]])])
prepared = MeanDistance().prepare(reference_clouds, row_ids=("left", "right"))
distances = prepared.cross(query_clouds)  # [[2., 9.]]
```

## Inject an estimator

An application can require an overdetermined design before invoking ordinary
least squares. Passing the original typed problem through to the native solver
preserves the coefficient names, weights, scores and fitted state used by
inference. A different estimating equation requires corresponding score and
covariance contracts or its own result type.

```python
from external_extensions import OverdeterminedLeastSquares

fit = Regression(
    "y", Intercept() + Centered("age"), estimator=OverdeterminedLeastSquares()
).estimate(training)
```
