---
type: reference
description: Test whether a sample is close enough to a target to count as equivalent, including mixtures of candidate clouds.
---

# Equivalence and mixtures

## Usage

```python
import jax
import jax.numpy as jnp
from statadic.inference.equivalence import equivalence_test

keys = jax.random.split(jax.random.key(0), 3)
x = jax.random.normal(keys[0], (40, 2))
candidates = [jax.random.normal(keys[1], (30, 2)), jax.random.normal(keys[2], (30, 2)) + 0.2]
result = equivalence_test(
    x,
    candidates,
    delta=0.5,
    selection_split=None,
    reoptimize_in_permutation=True,
    delta_grid=jnp.array([0.25, 0.5, 1.0]),
    num_permutations=99,
    seed=0,
)
print(result.energy_point_estimate, result.energy_ci)
print(result.equivalence, result.equivalence_curve)
```

## Description

An ordinary test asks whether two samples differ, and a high p-value only says
the data cannot tell. Equivalence testing reverses the question. The null states
that the energy distance between the target and a weighted mixture of candidate
clouds is at least a margin `delta`, and rejecting it supports closeness.

`equivalence_test` finds the mixture weights, reports the distance with a
bootstrap interval and traces an equivalence curve across margins. You choose
`delta`, since no standard margin exists for energy distance, and should justify
it against a reference from your field.

The interpretation of a margin depends on the units and geometry of the chosen distance, so state the choice alongside the result.

For the same samples, the bootstrap interval describes uncertainty in the distance, while the test for a difference addresses a separate null.

## Options

| You want | Call | Notes |
| --- | --- | --- |
| A relevance-margin test | `statadic.inference.equivalence.equivalence_test` | Reports the estimate, a bootstrap interval and the curve |
| A p-value from bootstrap draws | `statadic.inference.equivalence.equivalence_pvalue_from_bootstrap` | Small values favour equivalence |
| A selection-corrected mixture test | `statadic.inference.mixture.corrected_mixture_test` | Reoptimises weights in each permutation, or selects on split A and tests on B |
| A studentized mixture statistic | `statadic.inference.studentized.studentized_mixture_statistic` | Chung and Romano form |
| The effective size of a mixture | `statadic.discrepancy.mmd.energy.balancing.inverse_variance_m_eff` | Kish form, `1/m_eff = sum w_k^2 / m_k` |
| Holm or Bonferroni across claims | `statadic.inference.multiplicity.equivalence_multiplicity_control` | Holm is the default |
| Dimension reduction fitted on split A | `statadic.sample.reduction.fit_projection` | Apply to both splits |

## Results

`equivalence_test` returns a result with `energy_point_estimate`, `energy_ci`,
`equivalence` (a flag) and `equivalence_curve`. It also keeps `equivalence_pvalue`,
the mixture `weights`, `m_eff` and the solver status. The ordinary `pvalue` field
tests for a difference instead, so a small value there argues against equivalence.

## Assumptions and limits

- The selected-mixture bootstrap procedure is approximate, and its p-values have
  no proved size guarantee. Holm or Bonferroni adjustment does not supply one.
- `m_eff` normalises the studentized statistic. It is not an exact null
  quantity, because the null statistic is degenerate.
- Fit dimension reduction on one split and test on the other. Fitting on pooled
  data invalidates the two-sample test that follows.

## Evidence

- The approximation is recorded in [known limitations](../validation/known-limitations.md).
- Background is in Székely and Rizzo (2013), listed in the
  [references](../validation/references.md).

## See also

- [Comparison and association](catalogue-comparison.md)
- [Energy distance](energy.md)
