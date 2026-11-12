---
type: reference
description: Find the numerical operation that answers a statistical question, with pages for each family of methods.
---

# Method catalogue

## Usage

```python
import jax.numpy as jnp
from statadic.discrepancy.mmd.energy import energy_distance
from statadic.sample.packed import pack

x = pack([jnp.array([[0.0], [1.0], [2.0]])])
y = pack([jnp.array([[1.0], [2.0], [3.0]])])
print(energy_distance(x, y, metric="euclidean"))
```

## Description

The workflow classes in {py:class}`~statadic.api.regression.Regression`,
`Ordination` and `Estimand` cover the common analyses. This catalogue lists the
array-level operations beneath them, organised by the question you are asking
rather than by module. Each row names the function or class to import and says
what to expect from it.

Import from the module that owns an operation, as in the example above. The
root `statadic` package and compound namespaces such as `statadic.inference` only
point to their children. Each family page documents the arguments and return
values of its operations. A test result can include a reference distribution,
while the availability of simulated null draws depends on the procedure.

## Options

| Question | Page |
| --- | --- |
| Are two things associated, or do two samples differ? | [Comparison and association](catalogue-comparison.md) |
| Are two samples close enough to treat as equivalent? | [Equivalence and mixtures](catalogue-equivalence.md) |
| Which dissimilarities are metrics? | [Distance properties](catalogue-metrics.md) |
| What is the structure in a distance matrix, and how do I prepare samples? | [Geometry and samples](catalogue-geometry.md) |
| How do I fit a linear, GLM, moment, panel or spatial model? | [Fitting models](catalogue-fitting.md) |
| How much sampling error is in an estimate? | [Covariance and error](catalogue-covariance.md) |
| Does a moment model satisfy its restrictions? | [Moment restrictions](catalogue-restrictions.md) |
| How do I resample dependent data or assess a track record? | [Resampling and performance](catalogue-resampling.md) |

:::{toctree}
:hidden:

catalogue-comparison
catalogue-equivalence
catalogue-metrics
catalogue-geometry
catalogue-fitting
catalogue-covariance
catalogue-restrictions
catalogue-resampling
:::

## Results

Return values vary by operation and include scalars, arrays and result records.
Each family page describes its result type and the uncertainty, reference
distribution or status it carries. Inspect numerical status and availability
before interpreting estimates or p-values, because a solver can finish without
converging.

## Assumptions and limits

- Listing a function here says where to find it. Whether a method is valid for
  your data is described on the reference page for its family.
- Operations that mention `jit`, `vmap` or `grad` follow the rules in
  [transformation support](transformation-support.md).
- Names in these pages are stable import paths. Earlier paths are mapped in
  [upgrading](upgrading.md).

## Evidence

- Independent comparisons, studies and formal results are indexed in the
  [validation section](../validation/index.md).
- Known gaps in coverage are collected in [known limitations](../validation/known-limitations.md).

## See also

- [Regression](regression.md), [Ordination](ordination.md) and
  [Estimand](estimand.md) for the workflow classes
- [Contracts](../concepts/geometry.md) for the meaning of laws and carriers
