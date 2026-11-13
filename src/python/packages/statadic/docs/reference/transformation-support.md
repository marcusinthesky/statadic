---
type: reference
description: Check JAX transformation, precision and backend support for numerical operations.
---

# Transformation support

## Usage

```python
import jax
import jax.numpy as jnp
from statadic.estimation.linear import LeastSquares, LinearProblem

x = jnp.array([[1., 0.], [1., 1.], [1., 2.]])
y = jnp.array([1., 2., 3.])
fit = jax.jit(lambda a, b: LeastSquares().estimate(LinearProblem(a, b)))(x, y)
```

## Description

Preparation of named data, report rendering and artifact loading run eagerly.
Fixed-shape numerical operations can support JAX transformations independently
of the surrounding host workflow, and the caller selects their precision.

The scope of a transformation depends on the operation and its input shape, as well as the numerical conditions encountered during execution.

For example, a coefficient derivative is checked in the interior of the OLS domain, while a change in rank can alter the problem being solved.

## Options

| Operation | Transformation scope |
| --- | --- |
| Ordination | Fixed-shape coordinate kernels, frozen maps and shortest paths support JIT and vmap. Selection and named preparation are eager, with no general derivative promise at rank, axis or neighborhood changes |
| OLS | Fixed-shape JIT and vmap, including invalid inputs. Coefficient derivatives are checked away from rank and validity changes |
| WLS and prediction | Fixed-shape JIT and vmap. Fitted WLS derivatives lack a general guarantee |
| Affine GMM | Fixed-shape JIT and vmap. Derivatives of the iterated fit lack a general guarantee |
| Fixed and affine GEL | Eager solve, even when the problem is a pytree |
| GLM | Fixed-shape JIT and vmap. Interior score and information are checked, but derivatives of the converged fit are not promised |
| Balanced spatial panels | Fixed-shape JIT and vmap with static process, effects and iteration controls. KKP requires constant weights |
| Observed panels and survival | Host alignment and integration refinement are eager. Numerical operations have their own contracts |
| Fit covariance, restrictions and OLS diagnostics | Fixed-shape JIT and vmap in checked domains. Reference choices and availability remain explicit |
| Resampling schedules and summaries | Fixed structure, JIT and outer vmap. Discrete schedules and empirical quantiles have no general smoothness promise |
| Packed clouds and ground distances | Fixed capacities support JIT and vmap. Floating coordinates and logits support automatic differentiation with counts fixed |
| Exact transport | Fixed-shape assignment kernels support JIT and vmap. Discrete assignment does not imply a smooth solution |
| One-dimensional and sliced transport | Fixed atom capacities and batching. Powered objectives have piecewise derivatives away from ordering and quantile ties |
| Dynamic kernels, projections and energy geometry | Numerical parameters remain dynamic through checked MMD and energy consumers |
| Simplex optimization and PSD repair | Fixed-shape numerical contracts apply. Inspect convergence and feasibility diagnostics |

## Results

The numerical result keeps estimates and status as arrays. Static options and
shapes define the traced computation, while batching applies that computation
to compatible inputs.

## Assumptions and limits

The checked CPU configurations use caller-selected float32 or float64. A JAX pytree does not establish statistical validity, smoothness at a rank change, or GPU performance. Packing ragged data and choosing dimensions are eager. Finite projections give sliced transport a pseudometric law.

## Evidence

The [evidence index](../validation/index.md) links to tests and comparisons for
individual methods. The [performance workloads](performance-workloads.md)
describe the CPU configurations used for measurements.

## See also

- [Installation](../start/installation.md)
- [Method catalogue](method-catalogue.md)
