"""Typed numerical availability preserves actual native solver lane failures."""

import jax.numpy as jnp
import pytest
from statadic.inference._common import (
    ProjectionUnavailableError,
    _require_projection_solution,
)
from statadic.optimize.simplex import NonnegativeQPSolution


@pytest.mark.unit
@pytest.mark.contract
def test_unavailable_projection_retains_typed_lane_counts() -> None:
    """Distinguish completed domain failures from generic runtime failures."""
    solution = NonnegativeQPSolution(
        primal=jnp.zeros((3, 2)),
        objective=jnp.zeros(3),
        active_mask=jnp.ones((3, 2), dtype=bool),
        sweeps=jnp.ones(3, dtype=int),
        input_finite=jnp.array([True, False, True]),
        convex=jnp.array([True, True, False]),
        bounded=jnp.array([True, True, True]),
        converged=jnp.array([True, False, False]),
        projected_gradient_norm=jnp.array([0.0, 0.25, 0.5]),
        flat_mask=jnp.zeros((3, 2), dtype=bool),
    )
    with pytest.raises(ProjectionUnavailableError) as caught:
        _require_projection_solution(solution, context="synthetic native solver")
    error = caught.value
    assert isinstance(error, RuntimeError)
    assert error.lane_counts == {
        "rejected": 2,
        "total": 3,
        "nonfinite": 1,
        "nonconvex": 1,
        "unbounded_like": 0,
        "unconverged": 2,
    }
    assert error.max_projected_kkt == pytest.approx(
        float(solution.projected_gradient_norm[-1])
    )
    assert "rejected 2/3 lanes" in str(error)
