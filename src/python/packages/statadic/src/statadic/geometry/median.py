# jaxtyping shape strings are runtime contracts.
"""Per-group spatial medians for coordinate clouds.

:func:`weiszfeld_medians` runs a fixed softened Weiszfeld budget;
:func:`spatial_medians` iterates to stationarity and reports convergence.
This geometry module returns per-group locations. PERMDISP association and its
residual-permutation null live in :mod:`statadic.association.dispersion` and
:mod:`statadic.inference.dispersion` respectively.
"""

from __future__ import annotations

from functools import partial
from typing import Final

import jax
import jax.numpy as jnp

__all__ = [
    "SPATIAL_MEDIAN_MAX_ITERATIONS",
    "WEISZFELD_ITERATIONS",
    "spatial_medians",
    "weiszfeld_medians",
]

#: Fixed Weiszfeld iteration count. A fixed budget rather than a convergence
#: test keeps the routine cheap under ``jit``/``vmap``; it is an approximation
#: whose stationarity is not checked.
WEISZFELD_ITERATIONS: Final = 30

#: Softening added under every square root so the gradient and the reciprocal
#: weight stay finite when a point coincides with the running median.
_DISTANCE_EPSILON: Final = 1e-12


@partial(jax.jit, static_argnames=("n_groups", "iters"))
def weiszfeld_medians(
    coords: jnp.ndarray,
    group_codes: jnp.ndarray,
    n_groups: int,
    iters: int = WEISZFELD_ITERATIONS,
) -> jnp.ndarray:
    """Compute per-group spatial medians with vectorized Weiszfeld iterations.

    Args:
        coords: Coordinate cloud, shape ``(n, k)``.
        group_codes: Contiguous integer group code per row, shape ``(n,)``.
        n_groups: Number of distinct groups; static under ``jit``.
        iters: Fixed Weiszfeld iteration budget.

    Returns:
        Per-group spatial medians, shape ``(n_groups, k)``, initialised at the
        group centroid.

    """

    def median_for_group(g: jnp.ndarray) -> jnp.ndarray:
        mask = (group_codes == g).astype(coords.dtype)
        w_sum = jnp.sum(mask)
        init = jnp.sum(mask[:, None] * coords, axis=0) / w_sum

        def body(m: jnp.ndarray, _: None) -> tuple[jnp.ndarray, None]:
            diffs = coords - m[None, :]
            dists = jnp.sqrt(jnp.sum(diffs**2, axis=1) + _DISTANCE_EPSILON)
            w = mask / dists
            new_m = jnp.sum(w[:, None] * coords, axis=0) / jnp.sum(w)
            return new_m, None

        m_final, _ = jax.lax.scan(body, init, xs=None, length=iters)
        return m_final

    return jax.vmap(median_for_group)(jnp.arange(n_groups))


#: Iteration ceiling for :func:`spatial_medians`. Convergence, not this budget,
#: ends a regular fit; reaching it reports nonconvergence.
SPATIAL_MEDIAN_MAX_ITERATIONS: Final = 20_000

#: Radius, in machine epsilons of the scaled cloud, below which a point
#: coincides with the iterate.
_COINCIDENT_EPS: Final = 32

#: Per-point stationarity tolerance, in machine epsilons.
_STATIONARITY_EPS: Final = 1000

type _LoopState = tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]


def _gradient(
    cloud: jnp.ndarray, member: jnp.ndarray, at: jnp.ndarray
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Return unit-vector force, coincident multiplicity, weights and radii."""
    differences = cloud - at
    radii = jnp.linalg.norm(differences, axis=1)
    coincident = member & (radii <= _COINCIDENT_EPS * jnp.finfo(cloud.dtype).eps)
    moving = member & ~coincident
    weights = jnp.where(moving, 1.0 / jnp.where(moving, radii, 1.0), 0.0)
    force = jnp.linalg.norm(jnp.sum(weights[:, None] * differences, axis=0))
    multiplicity = jnp.sum(coincident.astype(cloud.dtype))
    return force, multiplicity, weights, radii


def _objective(cloud: jnp.ndarray, member: jnp.ndarray, at: jnp.ndarray) -> jnp.ndarray:
    squared = jnp.sum((cloud - at) ** 2, axis=1)
    return jnp.sum(jnp.where(member, jnp.sqrt(jnp.where(member, squared, 1)), 0))


def _newton_step(
    cloud: jnp.ndarray, median: jnp.ndarray, weights: jnp.ndarray
) -> jnp.ndarray:
    """Take a Newton step on the smooth objective away from data points."""
    objective = partial(_objective, cloud, weights > 0)
    return median - jnp.linalg.solve(
        jax.hessian(objective)(median), jax.grad(objective)(median)
    )


def _vardi_zhang(
    cloud: jnp.ndarray, member: jnp.ndarray, median: jnp.ndarray
) -> jnp.ndarray:
    """Update by Vardi-Zhang, or by Newton when that lowers the objective.

    Weiszfeld is linear and stalls near a data point; the Newton candidate is
    kept only when it is finite and strictly better.
    """
    force, multiplicity, weights, _ = _gradient(cloud, member, median)
    weight_sum = jnp.sum(weights)
    proposal = jnp.sum(weights[:, None] * cloud, axis=0) / jnp.where(
        weight_sum > 0, weight_sum, 1.0
    )
    fraction = jnp.minimum(1.0, multiplicity / jnp.where(force > 0, force, 1.0))
    weiszfeld = (1 - fraction) * proposal + fraction * median
    newton = jax.lax.cond(
        multiplicity == 0,
        lambda: _newton_step(cloud, median, weights),
        lambda: weiszfeld,
    )
    use_newton = (
        (multiplicity == 0)
        & jnp.all(jnp.isfinite(newton))
        & (_objective(cloud, member, newton) < _objective(cloud, member, weiszfeld))
    )
    return jnp.where(use_newton, newton, weiszfeld)


def _median_step(
    cloud: jnp.ndarray, member: jnp.ndarray, median: jnp.ndarray
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Return the next iterate and whether the current one already solves."""
    tolerance = _STATIONARITY_EPS * jnp.finfo(cloud.dtype).eps * jnp.sum(member)
    force, multiplicity, _, radii = _gradient(cloud, member, median)
    stationary = force <= multiplicity + tolerance
    # Weiszfeld only approaches a median that is a data point, so test
    # Kuhn's vertex condition at the nearest member point exactly.
    vertex = cloud[jnp.argmin(jnp.where(member, radii, jnp.inf))]
    vertex_force, vertex_multiplicity, _, _ = _gradient(cloud, member, vertex)
    at_vertex = vertex_force <= vertex_multiplicity + tolerance
    updated = jnp.where(
        stationary,
        median,
        jnp.where(at_vertex, vertex, _vardi_zhang(cloud, member, median)),
    )
    return updated, stationary | at_vertex


def _continue(max_iterations: int, state: _LoopState) -> jnp.ndarray:
    _, iteration, done = state
    return ~done & (iteration < max_iterations)


def _advance(cloud: jnp.ndarray, member: jnp.ndarray, state: _LoopState) -> _LoopState:
    median, iteration, _ = state
    updated, done = _median_step(cloud, member, median)
    return updated, iteration + 1, done


def _group_median(
    coords: jnp.ndarray, member: jnp.ndarray, max_iterations: int
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Iterate one group's median in centroid-centred, radius-scaled units."""
    mask = member.astype(coords.dtype)
    origin = jnp.sum(mask[:, None] * coords, axis=0) / jnp.sum(mask)
    centred = jnp.where(member[:, None], coords - origin, 0.0)
    scale = jnp.max(jnp.linalg.norm(centred, axis=1))
    cloud = centred / jnp.where(scale > 0, scale, 1.0)
    start = jnp.zeros(coords.shape[1], dtype=coords.dtype)
    median, _, converged = jax.lax.while_loop(
        partial(_continue, max_iterations),
        partial(_advance, cloud, member),
        (start, jnp.asarray(0), scale <= 0),
    )
    return origin + scale * median, converged


@partial(jax.jit, static_argnames=("n_groups", "max_iterations"))
def spatial_medians(
    coords: jnp.ndarray,
    group_codes: jnp.ndarray,
    n_groups: int,
    max_iterations: int = SPATIAL_MEDIAN_MAX_ITERATIONS,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Compute per-group spatial medians to a stationarity tolerance.

    Each group is centred at its centroid and scaled by its largest radius
    before Vardi-Zhang iteration, which moves correctly through data points
    instead of softening their distances. A Newton step replaces the
    Vardi-Zhang update whenever it attains a lower objective. Iteration stops
    once the unit-vector gradient norm, at the iterate or at its nearest data
    point (Kuhn's vertex condition), is within the coincident-point
    multiplicity plus ``1000 * eps * n_g``.

    Args:
        coords: Coordinate cloud, shape ``(n, k)``.
        group_codes: Contiguous integer group code per row, shape ``(n,)``.
        n_groups: Number of distinct groups; static under ``jit``.
        max_iterations: Static iteration ceiling.

    Returns:
        ``(medians, converged)`` with shapes ``(n_groups, k)`` and
        ``(n_groups,)``. A group whose points coincide converges immediately.

    """

    def _median(g: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
        return _group_median(coords, group_codes == g, max_iterations)

    return jax.vmap(_median)(jnp.arange(n_groups))
