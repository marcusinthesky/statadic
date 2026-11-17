"""One-solve assignment witnesses and declared-ground exact transport."""

from __future__ import annotations

from itertools import permutations

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from statadic.geometry.discrepancy.transport.exact import (
    balanced_wasserstein_from_cost_inspect,
    balanced_wasserstein_from_cost_result,
    exact_balanced_wasserstein,
    exact_balanced_wasserstein_inspect,
    exact_balanced_wasserstein_jax,
    exact_balanced_wasserstein_result,
)
from statadic.geometry.ground.metrics import EUCLIDEAN
from statadic.measure.clouds import PackedClouds


@pytest.mark.unit
@pytest.mark.reference
@pytest.mark.parametrize("order", [1.0, 2.0, 3.0])
def test_inspection_attains_assignment_and_reconstructs_distance(order: float) -> None:
    """A reported witness attains the independently enumerated powered optimum."""
    cost = np.asarray([[0.0, 2.0, 8.0], [2.0, 3.0, 8.0], [9.0, 9.0, 1.0]])
    observed = balanced_wasserstein_from_cost_inspect(jnp.asarray(cost), order)
    rows, columns = (
        np.asarray(observed.row_indices),
        np.asarray(observed.column_indices),
    )
    np.testing.assert_array_equal(np.sort(rows), np.arange(3))
    np.testing.assert_array_equal(np.sort(columns), np.arange(3))
    np.testing.assert_allclose(observed.matched_costs, cost[rows, columns])
    best = min(
        np.mean(cost[np.arange(3), perm] ** order) for perm in permutations(range(3))
    )
    np.testing.assert_allclose(float(observed.result.value) ** order, best, rtol=2e-6)
    rebuilt = observed.cost_scale * jnp.sum(
        observed.scaled_objective_contributions
    ) ** (1.0 / order)
    np.testing.assert_allclose(rebuilt, observed.result.value, rtol=2e-6)
    projected = balanced_wasserstein_from_cost_result(jnp.asarray(cost), order)
    np.testing.assert_allclose(projected.value, observed.result.value, rtol=2e-6)


@pytest.mark.unit
@pytest.mark.contract
def test_tied_inspection_retains_a_feasible_witness() -> None:
    """Ties guarantee an optimum rather than a unique matching or stable label."""
    inspected = jax.jit(balanced_wasserstein_from_cost_inspect)(jnp.ones((3, 3)), 2.0)
    np.testing.assert_array_equal(np.sort(inspected.row_indices), np.arange(3))
    np.testing.assert_array_equal(np.sort(inspected.column_indices), np.arange(3))
    np.testing.assert_allclose(inspected.matched_costs, 1.0)
    assert float(inspected.result.value) == pytest.approx(1.0)


@pytest.mark.unit
@pytest.mark.numerical
def test_large_cost_inspection_avoids_original_unit_power_overflow() -> None:
    """Inspectable power contributions stay finite at float32's distance scale."""
    maximum = np.finfo(np.float32).max
    cost = jnp.asarray([[maximum, maximum], [maximum, maximum]], dtype=jnp.float32)
    inspected = jax.jit(balanced_wasserstein_from_cost_inspect)(cost, 20.0)
    assert bool(inspected.result.input_valid & inspected.result.finite)
    assert not bool(inspected.result.power_underflow)
    np.testing.assert_allclose(inspected.scaled_objective_contributions, [0.5, 0.5])
    assert float(inspected.result.value) == pytest.approx(float(maximum), rel=2e-6)


@pytest.mark.component
@pytest.mark.contract
def test_padded_euclidean_inspection_matches_jit_vmap_and_ignores_padding() -> None:
    """Different capacities retain only active article indices and raw geometry."""
    left = PackedClouds(jnp.asarray([[0.0], [2.0], [jnp.nan]]), jnp.asarray(2))
    right = PackedClouds(jnp.asarray([[3.0], [1.0], [99.0], [jnp.nan]]), jnp.asarray(2))
    inspect = jax.jit(exact_balanced_wasserstein_inspect)
    observed = inspect(left, right, 2.0, ground=EUCLIDEAN)
    assert bool(observed.result.input_valid & observed.result.finite)
    assert float(observed.result.value) == pytest.approx(1.0)
    active = np.asarray(observed.active)
    np.testing.assert_equal(np.count_nonzero(active), int(left.counts))
    np.testing.assert_array_equal(np.sort(observed.row_indices[active]), [0, 1])
    np.testing.assert_array_equal(np.sort(observed.column_indices[active]), [0, 1])
    np.testing.assert_array_equal(observed.row_indices[~active], -1)
    np.testing.assert_array_equal(observed.column_indices[~active], -1)
    np.testing.assert_array_equal(observed.matched_costs[~active], 0)
    np.testing.assert_array_equal(observed.scaled_objective_contributions[~active], 0)
    clouds = PackedClouds(jnp.stack([left.data, left.data + 2]), jnp.asarray([2, 2]))
    batch = inspect(clouds, right, 2.0, ground=EUCLIDEAN)
    mapped = jax.jit(
        jax.vmap(exact_balanced_wasserstein_inspect, in_axes=(0, None, None))
    )(clouds, right, 2.0, ground=EUCLIDEAN)
    np.testing.assert_allclose(batch.result.value, mapped.result.value)
    np.testing.assert_allclose(batch.result.value, [1.0, 1.0])
    np.testing.assert_array_equal(batch.row_indices, mapped.row_indices)


@pytest.mark.component
@pytest.mark.contract
def test_euclidean_eager_door_preserves_raw_coordinates() -> None:
    """Explicit Euclidean requests admit zero and nonunit rows without projection."""
    left = PackedClouds.from_array(jnp.asarray([[0.0, 0.0], [4.0, 0.0]]))
    right = PackedClouds.from_array(jnp.asarray([[2.0, 0.0], [6.0, 0.0]]))
    assert exact_balanced_wasserstein(
        left, right, 2.0, ground=EUCLIDEAN
    ) == pytest.approx(2.0)
    scalar = jax.jit(exact_balanced_wasserstein_jax)(left, right, 2.0, ground=EUCLIDEAN)
    result = jax.jit(exact_balanced_wasserstein_result)(
        left, right, 2.0, ground=EUCLIDEAN
    )
    np.testing.assert_allclose(scalar, result.value)
    assert bool(result.input_valid)
    with pytest.raises(ValueError, match="zero-norm"):
        exact_balanced_wasserstein(left, right, 2.0)


@pytest.mark.component
@pytest.mark.contract
def test_invalid_batched_inspection_reports_evidence() -> None:
    """Nonuniform masses and unequal active counts cannot masquerade as assignments."""
    coordinates = jnp.asarray([[[0.0], [2.0]], [[0.0], [2.0]], [[0.0], [2.0]]])
    clouds = PackedClouds(
        coordinates,
        jnp.asarray([2, 2, 1]),
        jnp.asarray([[0.5, 0.5], [0.75, 0.25], [1.0, 0.0]]),
    )
    reference = PackedClouds.from_array(jnp.asarray([[1.0], [3.0]]))
    observed = jax.jit(exact_balanced_wasserstein_inspect)(
        clouds, reference, 2.0, ground=EUCLIDEAN
    )
    np.testing.assert_array_equal(observed.result.input_valid, [True, False, False])
    np.testing.assert_array_equal(observed.result.finite, [True, False, False])
    assert np.all(np.isnan(observed.result.value[1:]))
    assert np.all(np.isnan(observed.matched_costs[1:]))
