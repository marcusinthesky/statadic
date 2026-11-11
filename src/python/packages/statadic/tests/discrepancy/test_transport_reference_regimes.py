"""Independent transport objectives across ties, masses and fixed projections."""

from __future__ import annotations

from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st
from hypothesis.extra import numpy as hnp
from statadic.discrepancy.transport.exact import balanced_wasserstein_from_cost_jax
from statadic.discrepancy.transport.sliced import (
    ProjectionDesign,
    sliced_wasserstein_under_design,
)
from statadic.discrepancy.transport.univariate import wasserstein_1d, wasserstein_1d_power
from statadic.sample.packed import PackedClouds
from scipy.optimize import linprog

from .._budgets import examples

if TYPE_CHECKING:
    from collections.abc import Callable
    from types import ModuleType

    from jax import Array


def _transport_lp(
    x: np.ndarray, y: np.ndarray, a: np.ndarray, b: np.ndarray, p: float
) -> float:
    """Solve a finite transportation LP with independently normalized marginals."""
    constraints = np.concatenate(
        [
            np.kron(np.eye(len(x)), np.ones((1, len(y)))),
            np.tile(np.eye(len(y)), (1, len(x))),
        ],
        axis=0,
    )
    result = linprog(
        (np.abs(x[:, None] - y[None, :]) ** p).ravel(),
        A_eq=constraints,
        b_eq=np.concatenate([a / a.sum(), b / b.sum()]),
        bounds=(0.0, None),
        method="highs",
    )
    assert result.success, result.message
    return float(result.fun)


@pytest.mark.unit
@pytest.mark.reference
@pytest.mark.external_reference
@pytest.mark.requires_optional("scipy")
@pytest.mark.parametrize("order", [1.0, 2.0, 3.0])
@examples(10)
@given(
    positions=hnp.arrays(np.float64, (5,), elements=st.integers(-3, 3).map(float)),
    masses=hnp.arrays(np.float64, (5,), elements=st.integers(1, 5).map(float)),
)
def test_generated_weighted_ties_match_lp_and_coordinate_scale_law(
    order: float,
    positions: np.ndarray,
    masses: np.ndarray,
) -> None:
    """Repeated sites retain their own masses under paired reordering and scaling."""
    x, y, a, b = positions[:2], positions[2:], masses[:2], masses[2:]
    expected = _transport_lp(x, y, a, b, order)
    permutation = np.asarray([2, 0, 1])

    powered = wasserstein_1d_power(
        PackedClouds.from_array(
            jnp.asarray(jnp.asarray(x))[..., None],
            jnp.asarray(jnp.asarray(a))
            / jnp.sum(jnp.asarray(jnp.asarray(a)), axis=-1, keepdims=True),
        ),
        PackedClouds.from_array(
            jnp.asarray(jnp.asarray(y))[..., None],
            jnp.asarray(jnp.asarray(b))
            / jnp.sum(jnp.asarray(jnp.asarray(b)), axis=-1, keepdims=True),
        ),
        p=order,
    )
    rooted = wasserstein_1d(
        PackedClouds.from_array(
            jnp.asarray(jnp.asarray(x))[..., None],
            jnp.asarray(jnp.asarray(a))
            / jnp.sum(jnp.asarray(jnp.asarray(a)), axis=-1, keepdims=True),
        ),
        PackedClouds.from_array(
            jnp.asarray(jnp.asarray(y))[..., None],
            jnp.asarray(jnp.asarray(b))
            / jnp.sum(jnp.asarray(jnp.asarray(b)), axis=-1, keepdims=True),
        ),
        p=order,
    )
    transformed = wasserstein_1d_power(
        PackedClouds.from_array(
            jnp.asarray(jnp.asarray(3 * x[::-1] - 2))[..., None],
            jnp.asarray(jnp.asarray(7 * a[::-1]))
            / jnp.sum(jnp.asarray(jnp.asarray(7 * a[::-1])), axis=-1, keepdims=True),
        ),
        PackedClouds.from_array(
            jnp.asarray(jnp.asarray(3 * y[permutation] - 2))[..., None],
            jnp.asarray(jnp.asarray(11 * b[permutation]))
            / jnp.sum(
                jnp.asarray(jnp.asarray(11 * b[permutation])), axis=-1, keepdims=True
            ),
        ),
        p=order,
    )

    assert float(powered) == pytest.approx(expected, rel=1e-11, abs=1e-11)
    assert float(rooted) == pytest.approx(expected ** (1 / order), rel=1e-11, abs=1e-11)
    assert float(transformed) == pytest.approx(
        3**order * expected, rel=1e-11, abs=1e-11
    )


@pytest.mark.component
@pytest.mark.reference
@pytest.mark.external_reference
@pytest.mark.requires_optional("POT")
@pytest.mark.requires_optional("scipy")
@pytest.mark.parametrize("order", [1.0, 2.0, 3.0])
def test_sliced_transport_matches_pot_with_identical_projection_matrix(
    optional_reference: Callable[[str], ModuleType],
    order: float,
) -> None:
    """All implementations consume the same directions; RNG seeds are not compared."""
    peer = optional_reference("ot")
    x = np.asarray([[-1.0, 2.0], [2.0, 1.0], [0.5, -1.0]])
    y = np.asarray([[0.0, 0.0], [3.0, -2.0], [-1.0, 1.0], [0.5, 2.0]])
    a, b = np.asarray([2.0, 3.0, 5.0]), np.asarray([1.0, 2.0, 3.0, 4.0])
    design = ProjectionDesign(jax.random.key(91), n_projections=7, p=order)
    directions = np.asarray(design.directions(2, jnp.float64))
    expected_costs = np.asarray(
        [
            _transport_lp(x @ direction, y @ direction, a, b, order)
            for direction in directions
        ]
    )
    reference, evidence = peer.sliced.sliced_wasserstein_distance(
        x,
        y,
        a=a / a.sum(),
        b=b / b.sum(),
        p=order,
        projections=directions.T,
        log=True,
    )

    actual = sliced_wasserstein_under_design(
        design,
        PackedClouds.from_array(
            jnp.asarray(jnp.asarray(x)),
            jnp.asarray(jnp.asarray(a))
            / jnp.sum(jnp.asarray(jnp.asarray(a)), axis=-1, keepdims=True),
        ),
        PackedClouds.from_array(
            jnp.asarray(jnp.asarray(y)),
            jnp.asarray(jnp.asarray(b))
            / jnp.sum(jnp.asarray(jnp.asarray(b)), axis=-1, keepdims=True),
        ),
    )

    np.testing.assert_allclose(np.linalg.norm(directions, axis=1), 1.0, atol=1e-13)
    np.testing.assert_allclose(
        evidence["projected_emds"], expected_costs, rtol=1e-11, atol=1e-11
    )
    assert float(actual) == pytest.approx(float(reference), rel=1e-11, abs=1e-11)
    assert float(actual) == pytest.approx(
        expected_costs.mean() ** (1 / order), rel=1e-11, abs=1e-11
    )


@pytest.mark.unit
@pytest.mark.numerical
@pytest.mark.parametrize("order", [1.0, 2.0, 3.0])
def test_nonunique_exact_cost_assignments_match_closed_form_objective(
    order: float,
) -> None:
    """Several optimal plans share one value; no plan coordinates are promised."""
    costs = jnp.asarray([[0.0, 0.0, 4.0], [0.0, 0.0, 4.0], [4.0, 4.0, 2.0]])
    expected = 2 / 3 ** (1 / order)

    def evaluate_balanced_wasserstein_from_cost_jax(matrix: jax.Array) -> Array:
        return balanced_wasserstein_from_cost_jax(matrix, order)

    actual = jax.jit(evaluate_balanced_wasserstein_from_cost_jax)(costs)

    assert float(actual) == pytest.approx(expected, rel=1e-12, abs=1e-12)
