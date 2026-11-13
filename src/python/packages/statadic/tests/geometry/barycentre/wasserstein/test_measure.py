# jaxtyping shape strings are runtime contracts.
"""Independent transport objectives and Wasserstein barycentre certificates."""

from __future__ import annotations

import itertools
from typing import TYPE_CHECKING

import jax.numpy as jnp
import pytest
from jax import config
from statadic.geometry.barycentre.wasserstein import (
    WassersteinBarycentreResult,
    solve_wasserstein_free_support_barycentre,
    solve_wasserstein_measure_barycentre,
)
from statadic.geometry.barycentre.wasserstein.measure import _objective_and_assignments
from statadic.geometry.ground.config import (
    resolve_ground_distance,
)
from statadic.geometry.ground.metrics import (
    cdist,
)
from statadic.sample.packed import pack

config.update("jax_enable_x64", val=True)

if TYPE_CHECKING:
    from statadic.geometry.ground.config import (
        GroundDistanceName,
    )

_ROOT_HALF = 0.7071067811865476

# Three two-point clouds on the unit circle. Small enough that every assignment
# can be enumerated (2! per source), so the reference objectives below never
# depend on the solver's own matching routine, and on the sphere so that the
# cosine and angular ground metrics are meaningful rather than degenerate.
_UNIT_SOURCES = jnp.asarray(
    [
        [[1.0, 0.0], [0.0, 1.0]],
        [[0.0, -1.0], [-1.0, 0.0]],
        [[_ROOT_HALF, _ROOT_HALF], [-_ROOT_HALF, _ROOT_HALF]],
    ]
)

# Five-point clouds along the x-axis, the y-axis, and the anti-diagonal. Chosen
# because the alternating iteration needs four sweeps here: the two-point
# fixtures above have only 2! assignments per source and stabilize immediately,
# which cannot tell a budget stop apart from a fixed point.
_CROSSING_SOURCES = jnp.asarray(
    [
        [[float(index), 0.0] for index in range(5)],
        [[0.0, float(index)] for index in range(5)],
        [[float(4 - index), float(index)] for index in range(5)],
    ]
)
_EQUAL_THIRDS = jnp.asarray([1 / 3, 1 / 3, 1 / 3])


def _exhaustive_objective(
    support: jnp.ndarray,
    sources: jnp.ndarray,
    weights: jnp.ndarray,
    *,
    p: float,
    metric: GroundDistanceName,
) -> float:
    """Score a support by brute force over every per-source assignment."""
    ground = resolve_ground_distance(metric)
    orders = tuple(itertools.permutations(range(support.shape[0])))
    total = 0.0
    for weight, source in zip(weights, sources, strict=True):
        cost = cdist(support, source, metric=ground)
        matched = (
            jnp.asarray([cost[row, column] for row, column in enumerate(order)])
            for order in orders
        )
        total += float(weight) * min(float(jnp.mean(c**p)) for c in matched)
    return total


@pytest.mark.component
@pytest.mark.numerical
def test_two_source_solver_uses_a_powered_hungarian_assignment() -> None:
    """The exact two-source support follows the optimal W2 matching."""
    sources = jnp.asarray([[[0.0], [2.0]], [[3.0], [1.0]]])

    result = solve_wasserstein_measure_barycentre(
        pack(sources, require_positive_counts=True), jnp.asarray([0.5, 0.5]), p=2.0
    )

    assert isinstance(result, WassersteinBarycentreResult)
    assert jnp.allclose(result.support_values, jnp.asarray([[0.5], [2.5]]))
    assert result.objective == pytest.approx(0.25)
    assert result.converged
    assert result.iterations == 1


@pytest.mark.component
@pytest.mark.reference
def test_free_support_solver_matches_the_two_source_geodesic() -> None:
    """Alternating Hungarian/mean updates recover the exact K=2 W2 result."""
    sources = jnp.asarray([[[0.0], [2.0]], [[3.0], [1.0]]])
    weights = jnp.asarray([0.25, 0.75])

    exact = solve_wasserstein_measure_barycentre(
        pack(sources, require_positive_counts=True), weights, p=2.0
    )
    iterative = solve_wasserstein_free_support_barycentre(
        pack(sources, require_positive_counts=True), weights, p=2.0
    )

    assert jnp.allclose(iterative.support_values, exact.support_values)
    assert iterative.objective == pytest.approx(exact.objective, abs=1e-6)
    assert iterative.converged


@pytest.mark.unit
@pytest.mark.contract
def test_free_support_order_one_is_not_certified_by_a_mean_update() -> None:
    """The documented mean heuristic for W1 does not claim optimality."""
    sources = jnp.asarray([[[0.0], [2.0]], [[3.0], [1.0]]])

    result = solve_wasserstein_free_support_barycentre(
        pack(sources, require_positive_counts=True), jnp.asarray([0.5, 0.5]), p=1.0
    )

    assert not result.converged


@pytest.mark.component
@pytest.mark.reference
@pytest.mark.parametrize("metric", ["euclidean", "cosine", "angular"])
def test_objective_is_reported_in_the_declared_ground_metric(
    metric: GroundDistanceName,
) -> None:
    """The objective must measure the cost the assignment actually minimized.

    Evaluating matched pairs with a Euclidean norm while assigning on a
    non-Euclidean ground cost silently reports a different quantity: for unit
    rows under ``cosine`` the two differ by an order of magnitude, which reads
    as a plausible number rather than as an error.
    """
    support = _UNIT_SOURCES[0]
    objective, _ = _objective_and_assignments(
        support,
        _UNIT_SOURCES,
        _EQUAL_THIRDS,
        p=2.0,
        metric=resolve_ground_distance(metric),
    )
    expected = _exhaustive_objective(
        support, _UNIT_SOURCES, _EQUAL_THIRDS, p=2.0, metric=metric
    )
    assert float(objective) == pytest.approx(expected, abs=1e-6)


@pytest.mark.unit
@pytest.mark.contract
@pytest.mark.parametrize("metric", ["cosine", "angular", "normalized_euclidean"])
def test_free_support_rejects_unjustified_non_euclidean_mean_updates(
    metric: GroundDistanceName,
) -> None:
    """Stable assignments under another ground do not certify support stationarity."""
    with pytest.raises(ValueError, match="weighted-mean support update requires"):
        solve_wasserstein_free_support_barycentre(
            pack(_UNIT_SOURCES, require_positive_counts=True),
            _EQUAL_THIRDS,
            metric=metric,
        )


@pytest.mark.unit
@pytest.mark.contract
def test_two_source_solver_rejects_a_non_euclidean_ground_metric() -> None:
    """Linear interpolation is the geodesic only under the Euclidean metric."""
    with pytest.raises(ValueError, match="geodesic only under the Euclidean"):
        solve_wasserstein_measure_barycentre(
            pack(_UNIT_SOURCES[:2], require_positive_counts=True),
            jnp.asarray([0.5, 0.5]),
            p=2.0,
            metric="cosine",
        )


@pytest.mark.component
@pytest.mark.reference
def test_two_source_order_one_is_exact_at_equal_weights() -> None:
    """At equal weights the whole order-1 geodesic is optimal, midpoint included."""
    sources = jnp.asarray([[[0.0], [2.0]], [[3.0], [1.0]]])

    result = solve_wasserstein_measure_barycentre(
        pack(sources, require_positive_counts=True), jnp.asarray([0.5, 0.5]), p=1.0
    )

    assert result.converged
    assert result.objective == pytest.approx(
        _exhaustive_objective(
            result.support_values,
            sources,
            jnp.asarray([0.5, 0.5]),
            p=1.0,
            metric="euclidean",
        ),
        abs=1e-6,
    )


@pytest.mark.unit
@pytest.mark.contract
@pytest.mark.parametrize("order", [1.0, 3.0])
def test_two_source_solver_rejects_orders_it_cannot_solve_exactly(
    order: float,
) -> None:
    """Interpolating at ``w[1]`` minimizes only at ``p == 2`` or equal weights.

    The alternative is returning a support that is not a minimizer under
    ``converged=True``: at order 1 with weights ``(0.9, 0.1)`` the interpolant
    scores roughly 80% above the true minimum, which sits at a source.
    """
    with pytest.raises(ValueError, match="minimized elsewhere on the geodesic"):
        solve_wasserstein_measure_barycentre(
            pack(
                jnp.asarray([[[0.0], [2.0]], [[3.0], [1.0]]]),
                require_positive_counts=True,
            ),
            jnp.asarray([0.9, 0.1]),
            p=order,
        )


@pytest.mark.component
@pytest.mark.property
def test_convergence_certifies_a_discrete_fixed_point() -> None:
    """``converged`` means the assignments repeat, not that the support settled.

    A support that has drifted below a tolerance while the assignments are still
    moving is not a fixed point; a repeated assignment set is, because the next
    weighted mean reproduces the support exactly. That is checkable, so it is
    checked here rather than trusted.
    """
    result = solve_wasserstein_free_support_barycentre(
        pack(_CROSSING_SOURCES, require_positive_counts=True), _EQUAL_THIRDS, p=2.0
    )

    assert result.converged
    # Exact, not approximate: reaching the fixed point assigns the literal 0.0
    # rather than a small residual, and that distinction is the contract.
    assert result.support_shift == 0.0  # noqa: RUF069

    _, permutations = _objective_and_assignments(
        result.support_values,
        _CROSSING_SOURCES,
        _EQUAL_THIRDS,
        p=2.0,
        metric=resolve_ground_distance("euclidean"),
    )
    matched = jnp.take_along_axis(
        _CROSSING_SOURCES,
        jnp.broadcast_to(permutations[..., None], _CROSSING_SOURCES.shape),
        axis=1,
    )
    next_support = jnp.sum(_EQUAL_THIRDS[:, None, None] * matched, axis=0)
    assert jnp.allclose(next_support, result.support_values, atol=1e-6)


@pytest.mark.unit
@pytest.mark.contract
def test_budget_exhaustion_reports_an_uncertified_residual() -> None:
    """Stopping on ``max_iterations`` must not be dressed up as convergence."""
    result = solve_wasserstein_free_support_barycentre(
        pack(_CROSSING_SOURCES, require_positive_counts=True),
        _EQUAL_THIRDS,
        p=2.0,
        max_iterations=1,
    )

    assert not result.converged
    assert result.support_shift > 0.0
    assert result.iterations == 1
