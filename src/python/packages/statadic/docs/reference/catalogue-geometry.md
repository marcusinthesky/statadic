---
type: reference
description: Prepare samples and find geometric structure in distance matrices.
---

# Geometry and samples

## Usage

```python
import jax.numpy as jnp
from statadic.sample.packed import pack

clouds = pack([jnp.array([[0.0], [1.0]]), jnp.array([[2.0], [3.0]])])
```

## Description

A sample carrier keeps clouds with different numbers of points in a fixed-size batch. Geometry operations use those samples or a matrix of distances to construct coordinates and representative objects.

The same prepared observations can support several analyses, provided that each operation receives the geometry that its contract requires.

## Options

| you want | call | notes |
|---|---|---|
| coordinates from features, kernels or distances | `statadic.api.ordination.Ordination` | Compose linear, kernel, distance or shortest-path geometry with `PrincipalCoordinates`. Inspect signed inertia and stored query capabilities in the [ordination guide](ordination.md). `statadic.geometry.embedding.pcoa_kernel` remains the pure fixed-width distance kernel |
| a geometric median per group | `statadic.geometry.median.spatial_medians` | converged, with a per-group flag. `weiszfeld_medians` is the fixed-budget approximation |
| barycentre weights over candidate clouds | `statadic.geometry.barycentre.mmd.energy.energy_barycentre_weights` | Exact simplex QP with convergence diagnostics. `_batched` / `_pairwise` variants |
| barycentre weights from a declared PSD kernel | `statadic.geometry.barycentre.mmd.kernel_barycentre_weights` | Target-directed MMD objective. Energy is a specialized reduction over the same solver |
| measure or free-support transport barycentres | `statadic.geometry.barycentre.wasserstein` | Objective-specific assignment and convergence evidence |
| neighbour-set stability across encoders | `statadic.geometry.neighbour` | |
| a metrizing square root of an empirical energy matrix | `statadic.core.matrices.metrize_energy` | Roots the actual stored empirical V-energy construction when its ground/exponent branch separates measures. Persisted values alone carry no such evidence |

| you want | call | notes |
|---|---|---|
| ragged clouds as one **rectangular batch** | `statadic.sample.packed.pack` | `PackedClouds` + `masked_sum`/`masked_mean`/`masked_row_mask`. `pair_chunks` bounds peak memory in the O(n²) kernels |
| **reduce dimension without leaking** the test | `statadic.sample.reduction.fit_projection` | fit on split A, apply to both. This is correction #5 — fitting on the pooled sample invalidates the two-sample test that follows |
| **reference alternatives** to size a test | `statadic.sample.alternatives.draw` | `AltSpec` covers normal, Student-t, mean-shift, scale and mixture departures — power curves without writing a DGP |

## Results

Results retain the geometry or sample structure needed by subsequent operations. Inspect convergence information when an operation solves an optimization problem.

## Assumptions and limits

Fit a projection on training data before applying it to test data. A metric construction requires the declared properties of its input. Its finite-sample evidence does not establish a population law.

## Evidence

[Independent comparisons and studies](../validation/index.md) describe the scope of evidence for each method.

## See also

- [Ordination](ordination.md)
- [Distance properties](catalogue-metrics.md)
