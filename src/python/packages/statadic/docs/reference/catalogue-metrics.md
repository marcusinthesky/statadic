---
type: reference
description: Check which dissimilarities are metrics before choosing a default, with an example of scale blindness.
---

# Distance properties

## Usage

```python
import jax
from statadic.discrepancy.mmd.energy import energy_distance
from statadic.sample.packed import pack

sample = jax.random.normal(jax.random.key(0), (30, 2))
x = pack([sample])
scaled = pack([3 * sample])
print(energy_distance(x, scaled, metric="angular"))
print(energy_distance(x, scaled, metric="euclidean"))
```

## Description

A dissimilarity name identifies the objects being compared, while its declared
law identifies the distance axioms that hold on the specified domain.

Normalising vectors before comparing their directions removes their magnitude.
For example, tripling a sample leaves its angular energy distance at zero,
while the Euclidean comparison in the example above remains positive.

Pick the Euclidean ground when scale matters to your question. Random draws
almost never produce parallel vectors, so this equality appears on specially
constructed inputs such as the sample and its rescaling above.

The laws in the table describe each mathematical construction on its stated
domain, which can differ from the properties of a finite numerical result.
When choosing a distance for an analysis, compare its domain and equality
relation with the distinctions that matter for the observations.

For example, two vectors on the same positive ray share a direction even when their lengths differ. The declared ground therefore matters when a matrix is reused for an ordination or a test.
For a comparison of distributions, this choice determines which changes in the
underlying observations remain visible to the resulting distance.

## Options

| Dissimilarity | What holds |
| --- | --- |
| `euclidean_distance` | Metric of negative type |
| `angular_distance`, `cosine_distance`, `normalized_euclidean_distance` | Premetric only, since `d(x, 2x) = 0` |
| `energy_distance` with a Euclidean ground | Semimetric |
| `energy_distance` with an angular ground | Premetric only |
| `mmd` with RBF or Euclidean distance-induced kernels | Metric of negative type on the V-statistic domain |
| `mmd` with linear, cosine or angular distance-induced kernels | Premetric only |
| Unbiased `mmd_squared` | Signed estimate with no distance law |
| `w1_1d`, `w2_1d` | Metric |
| Finite-design `sliced_wasserstein` | Pseudometric |
| `metrize_energy` | Metric of negative type |
| `exact_balanced_wasserstein` for finite `p >= 1` | Premetric on rows, and a metric on equal-size empirical measures over the unit sphere |

## Results

Each distance carries its declared law with the matrix it returns, so downstream
code can check what it is allowed to assume. Ordination and `metrize_energy`
read that law before accepting a matrix.

## Assumptions and limits

- A declared law describes the construction. It does not prove that your data
  satisfy the moment conditions behind a population result.
- The `metrize_energy` square root is a metric only when the energy construction
  separates measures, which angular energy does not.
- Oriented rays and projective classes use different equality relations, and
  ordinary angular distance does not identify antipodes.

## Evidence

- Laws and carriers are defined in [contracts](../concepts/geometry.md).
- Property checks and adversarial inputs run in the package test suite, and
  [known limitations](../validation/known-limitations.md) lists what remains open.

## See also

- [Comparison and association](catalogue-comparison.md)
- [Geometry and samples](catalogue-geometry.md)
