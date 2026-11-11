# Can a distance miss a change in scale?

**Learning objective:** distinguish the choice of ground geometry from the
choice of raw or rooted energy output.

## Predict first

Suppose Y is exactly three times X. Does a distribution comparison necessarily
notice? First decide whether your geometry retains magnitude.

## Try it

Choose **scaled-copy**, set the shift to zero, and increase the scale to three.
Compare Euclidean and angular ground. Then return to **independent** clouds and
set shift=0 and scale=1. Independent samples from the same population can still
have nonzero empirical discrepancy.

There are 64 points per cloud. A fixed seed and fixed underlying draws prevent
slider movements from introducing unrelated simulation changes. Changing the seed
creates a different experiment.

## Know which quantity is being reported

`energy_distance(..., metric="euclidean", exponent=1.0)` returns raw empirical
V-energy. `Energy(ground="euclidean", exponent=1.0)` is a rooted V-energy measure
comparison. The notebook calls both native interfaces; it does not manufacture a
root by clipping a signed estimate.

For exponent one, the raw V quantity is twice the cross-sample mean distance,
minus the two within-sample mean distances, including self-pairs. It is not the
signed U-statistic convention and it is not a p-value.

```python
import jax.numpy as jnp
from statadic.discrepancy.mmd.energy import energy_distance
from statadic.sample.packed import PackedClouds

x = jnp.array([[0.0], [1.0]])
y = jnp.array([[2.0], [3.0]])
raw = energy_distance(
    PackedClouds.from_points(x), PackedClouds.from_points(y),
    metric="euclidean", exponent=1.0,
)
# Analytic expected raw value: 3.0.
```

**Explicit ground.** The scalar interface requires `metric`; the host `Energy` interface defaults to Euclidean. The notebook passes the ground explicitly to both. A transformed copy of X is a paired geometry demonstration, not two independent samples suitable for an unrestricted label-permutation test.

[Next: evidence from permutations](permutation-evidence.md)

```{raw} html
<a href="snapshots/energy-distance.html">View the executed static snapshot</a> · <a href="downloads/energy-distance.py">Download the notebook</a> · <a href="snapshots/energy-distance.provenance.json">Snapshot provenance</a>
```

[Run this lab locally](index.md)
