"""Unregularized Wasserstein distances between finite probability measures.

Uniform equal-count measures use assignment; all other measures use the native
balanced transportation solver. Both minimize the same powered ground cost.
The numerical result door retains a coupling and convergence evidence for
selected comparisons. It does not promise fitted-solution derivatives.
"""

from dataclasses import dataclass

import jax
import jax.numpy as jnp

from statadic.core.axioms import Law  # noqa: TC001
from statadic.core.typing import Array  # noqa: TC001
from statadic.geometry.discrepancy.transport.exact import (
    exact_balanced_wasserstein_result,
)
from statadic.geometry.discrepancy.transport.weighted import (
    TransportationResult,  # noqa: TC001
    balanced_transport_from_cost_result,
)
from statadic.geometry.ground.metrics import (
    EUCLIDEAN,
    GroundDistance,  # noqa: TC001
    cdist,
)
from statadic.measure.clouds import PackedClouds

__all__ = [
    "WassersteinResult",
    "wasserstein_distance_jax",
    "wasserstein_distance_result",
]


def _relative_costs(costs: Array, scale: Array) -> Array:
    """Divide on mantissas before restoring exponents, including float max."""
    mantissas, exponents = jnp.frexp(costs)
    scale_mantissa, scale_exponent = jnp.frexp(scale)
    return jnp.ldexp(mantissas / scale_mantissa, exponents - scale_exponent)


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class WassersteinResult:
    """Rooted distance and the scaled LP that produced its coupling.

    ``transport.plan * scaled_costs`` sums to the LP objective. The distance
    is recomputed from the selected costs with its own scale, so irrelevant
    large costs do not erase a small optimum. ``scaled_objective_contributions``
    sum to ``(value / selected_cost_scale)**p``. Power underflow warns that
    the finite-precision LP may have obscured assignment differences.
    When it is true, ``value`` is a candidate rather than a qualified optimum;
    the scalar workflow rejects it as unavailable.
    """

    value: Array
    transport: TransportationResult
    ground_costs: Array
    scaled_costs: Array
    cost_scale: Array
    selected_cost_scale: Array
    scaled_objective_contributions: Array
    power_underflow: Array


def wasserstein_distance_result(
    left: PackedClouds,
    right: PackedClouds,
    p: float | Array = 2.0,
    *,
    ground: GroundDistance[Law] = EUCLIDEAN,
    max_iterations: int = 10_000,
    tolerance: float = 1e-6,
) -> WassersteinResult:
    """Solve one weighted, possibly unequal-support pair and retain evidence.

    Ground costs are unsquared distances. Only positive-mass coordinates are
    evaluated; unused storage may contain nonfinite values. Invalid domains or
    an exhausted solver budget produce NaN rather than an approximate distance.
    """
    if left.counts.ndim or right.counts.ndim:
        message = "A transport solve requires two individual distributions."
        raise ValueError(message)
    if left.data.shape[-1] != right.data.shape[-1]:
        message = "Measure feature widths must agree."
        raise ValueError(message)
    active = (left.masses[:, None] > 0) & (right.masses[None, :] > 0)
    distances = jnp.where(active, cdist(left.coordinates, right.coordinates, ground), 0)
    valid = (
        left.valid
        & right.valid
        & jnp.isfinite(p)
        & (p >= 1)
        & jnp.all(jnp.isfinite(distances) & (distances >= 0))
    )
    scale = jnp.max(jnp.where(jnp.isfinite(distances), distances, 0))
    safe_scale = jnp.where(scale > 0, scale, 1)
    safe_p = jnp.where(jnp.isfinite(p) & (p >= 1), p, 1)
    normalized = _relative_costs(distances, safe_scale)
    powered = normalized**safe_p
    transport = balanced_transport_from_cost_result(
        jnp.where(valid, powered, jnp.nan),
        left.masses,
        right.masses,
        max_iterations=max_iterations,
        tolerance=tolerance,
    )
    selected_scale = jnp.max(jnp.where(transport.plan > 0, distances, 0))
    selected_scale = jnp.where(selected_scale > 0, selected_scale, 1)
    selected_costs = _relative_costs(
        jnp.where(transport.plan > 0, distances, 0), selected_scale
    )
    contributions = transport.plan * selected_costs**safe_p
    value = selected_scale * jnp.sum(contributions) ** (1 / safe_p)
    return WassersteinResult(
        jnp.where(valid & transport.converged, value, jnp.nan),
        transport,
        distances,
        powered,
        safe_scale,
        selected_scale,
        contributions,
        jnp.any((distances > 0) & (powered == 0)),
    )


def wasserstein_distance_jax(
    left: PackedClouds,
    right: PackedClouds,
    p: float | Array = 2.0,
    *,
    ground: GroundDistance[Law] = EUCLIDEAN,
    max_iterations: int = 10_000,
    tolerance: float = 1e-6,
) -> Array:
    """Evaluate the same unregularized target using the applicable exact solver.

    This pair kernel supports JIT and vmap with fixed support capacities. Use
    explicit outer maps for collections so all-pairs couplings need not remain
    live in memory. Detected power underflow makes the scalar unavailable;
    inspect the result-bearing doors to examine the candidate and evidence.
    No entropic or sliced approximation is substituted.
    """

    def pair(a: PackedClouds, b: PackedClouds) -> Array:
        dtype = jnp.result_type(a.data, b.data, a.masses, b.masses, p, 1.0)
        a = PackedClouds(a.data.astype(dtype), a.counts, a.masses)
        b = PackedClouds(b.data.astype(dtype), b.counts, b.masses)
        uniform = a.is_uniform & b.is_uniform & (a.counts == b.counts)

        def assignment() -> Array:
            result = exact_balanced_wasserstein_result(a, b, p, ground=ground)
            return jnp.where(result.power_underflow, jnp.nan, result.value)

        def transportation() -> Array:
            result = wasserstein_distance_result(
                a,
                b,
                p,
                ground=ground,
                max_iterations=max_iterations,
                tolerance=tolerance,
            )
            return jnp.where(result.power_underflow, jnp.nan, result.value)

        return jax.lax.cond(uniform, assignment, transportation)

    return left.map_aligned(right, pair)
