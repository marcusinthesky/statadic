---
type: reference
description: Operations for testing association between distance matrices and for comparing two samples or distributions.
---

# Comparison and association

## Usage

```python
import jax
from statadic.discrepancy.mmd.energy import UNIT_EXPONENT, energy_geometry
from statadic.geometry.ground.metrics import EUCLIDEAN
from statadic.inference.resampling.permutation import energy_permutation_test

x = jax.random.normal(jax.random.key(0), (30, 2))
y = jax.random.normal(jax.random.key(1), (30, 2)) + 1.0
test = energy_permutation_test(x, y, energy_geometry(EUCLIDEAN, UNIT_EXPONENT), seed=7)
print(test.statistic, test.pvalue)
```

## Description

These operations answer two questions about data measured on the same units.
The first is whether two objects are associated, and the second is whether two
samples come from the same distribution. Energy distance, maximum mean
discrepancy (MMD) and transport distances measure how far apart two samples are.
The permutation functions then calibrate that distance into a test.

For a comparison to answer the intended question, select a statistic and a calibration that match the way the observations were collected.

For example, a permutation test compares the observed statistic with a distribution built by rearranging labels under its stated null.

## Options

| You want | Call | Notes |
| --- | --- | --- |
| Correlation between two distance matrices | `statadic.api.association.Mantel` | Pearson or Spearman, with a permutation null and node bootstrap intervals |
| Variance decomposition by group | `statadic.association.decomposition.disco_components` | Components only, tested with `statadic.inference.disco.disco_permutation_null` |
| A difference in dispersion | `statadic.association.dispersion.dispersion_f_ratio` | Null from `statadic.inference.dispersion.dispersion_permutation_null` |
| Energy distance | `statadic.discrepancy.mmd.energy.energy_distance` | Requires `metric`, and angular ground cannot see scale |
| Mean ground distances between clouds | `statadic.discrepancy.mmd.energy.empirical_mean_ground_distances` | Allows unequal sizes and defaults to Euclidean |
| MMD squared | `statadic.discrepancy.mmd.mmd_squared` | V-statistic or U-statistic, and the U-statistic can be negative |
| A matrix of MMD distances | `statadic.discrepancy.mmd.mmd_matrix` | RBF bandwidth must be positive |
| A kernel Gram matrix | `statadic.geometry.ground.similarities.gram` | Linear, cosine, RBF or distance-induced |
| A permutation test for any statistic | `statadic.inference.resampling.permutation.permutation_test_result` | Takes a statistic and a PRNG key, and works under `jit` and `vmap` |
| The traceable half of that test | `statadic.inference.resampling.permutation.permutation_test_kernel` | Pure JAX, for Monte Carlo grids |
| An energy permutation test | `statadic.inference.resampling.permutation.energy_permutation_test_result` | Records ground, exponent and null |
| One-dimensional transport | `statadic.discrepancy.transport.sliced` | `w1_1d` and `w2_1d`, plus multivariate `sliced_wasserstein` |
| Exact balanced Wasserstein-1 | `statadic.discrepancy.transport.exact` | Hungarian assignment on a chord ground cost |

## Results

Distances and statistics return arrays. The permutation tests return a result
holding the observed statistic, the p-value and the null draws, so you can plot
the null or recompute the p-value at another tail. P-values use the add-one
Monte Carlo rule, which keeps them above zero.

## Assumptions and limits

- A permutation test assumes exchangeability under the null. Dependent data
  need the block schemes in [resampling and performance](catalogue-resampling.md).
- The unbiased MMD estimate has no distance law, so only the V-statistic can be
  rooted into a distance.
- Whether a dissimilarity separates distributions depends on its ground
  distance, as listed in [distance properties](catalogue-metrics.md).
- Exact transport is a combinatorial assignment. It is not differentiable
  through the assignment itself.

## Evidence

- Energy statistics follow Székely and Rizzo, as cited in the
  [references](../validation/references.md).
- Study results are in the [study catalogue](../validation/research.md).

## See also

- [Energy distance](energy.md) and [equivalence tests](catalogue-equivalence.md)
- [Geometry and samples](catalogue-geometry.md)
