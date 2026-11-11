# Energy: two interfaces, two output conventions

The [finite probability energy proof](../formal/claims/STATADIC-ENERGY-002.md) requires a compatible symmetric positive-semidefinite generating kernel. This theorem does not verify the numerical implementation or every ground named by an API.

## Compare before substituting

| Property | Scalar `energy_distance` | Host `Energy` representation |
|---|---|---|
| Public owner | `statadic.discrepancy.mmd.energy` | `statadic.api.representation` |
| Default ground | None; `metric` is required | Euclidean |
| Exponent default | `1.0` | `1.0` |
| Quantity | Raw/unrooted V-energy functional | Rooted V-energy geometry |
| Primary input | `PackedClouds` distributions with atom masses | Collections of the same distributions plus row identities |
| Supported convention | Scalar V-statistic; explicit ground/exponent checks | Always rooted V-statistics; not a constructor choice |
| Scale interpretation | Angular ground cannot detect positive radial scale changes; `metric` has no default | Euclidean default retains scale information in its supported construction |
| Zero vectors | Angular ground violates its nonzero domain | Depends on the chosen ground; do not inherit angular rules by name alone |

<!-- statadic:math STATADIC-ENERGY-002 -->

## Scalar signature

```python
energy_distance(x, y, exponent=1.0, *, metric)
```

For two single distributions, returns a zero-dimensional JAX array; leading cloud
axes broadcast. Pair expectations use atom probabilities, including self-pairs.
Uniform masses recover the denominators `n*n` and `m*m`. Construct distributions
with `PackedClouds.from_points` or collections with `PackedClouds.from_clouds`.
The numerical core validates the supported distribution-exponent regime and
ground/exponent compatibility. Mathematical nonnegativity does not remove
finite-precision considerations.

The sibling `energy_distance_with_geometry` consumes a prevalidated declared `EnergyGeometry`. Import from the canonical family, and keep ground/exponent construction explicit in advanced numerical examples.

## Host specification

```python
Energy(ground="euclidean", exponent=1.0)
```

Preparation retains measure identities and the concrete geometry. The host supports pairwise and cross evaluation through retained references. Rooted measure geometry is not a generic wrapper around the scalar function.

## Related interfaces with their own contracts

`MMD` requires a declared PSD kernel strategy and rooted V-statistics in this host layer. `Wasserstein1D(p=2)` currently qualifies the real-line W2 construction only. Broader low-level transport functions do not expand this high-level constructor's accepted orders.

See the [complete energy lesson](../tutorials/energy-and-scale.md) and [geometry concepts](../concepts/geometry-contracts.md).
