# standalone package-maintenance entry point
"""Independent spatial-operator and sparse transformation checks."""

from __future__ import annotations

from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from hypothesis import example, given
from hypothesis import strategies as st
from jax.experimental.sparse import BCOO
from statadic.core.results import SolverStatus
from statadic.operators.spatial import (
    prepare_spatial_weights,
    spatial_lags,
    spatial_logdet,
    spatial_panel_lag,
    spatial_solve,
)

from .._budgets import examples
from .._strategies import float_arrays

if TYPE_CHECKING:
    from jax import Array

if TYPE_CHECKING:
    from statadic.core.results import SolverResult


@pytest.mark.unit
@pytest.mark.reference
@pytest.mark.parametrize("sparse", [False, True])
@examples(10)
@example(y=jnp.arange(7, dtype=float), rho=0.3)
@given(y=float_arrays(7, bound=5.0), rho=st.floats(-0.5, 0.5))
def test_sparse_and_dense_solve_and_lags(
    y: jax.Array, rho: float, *, sparse: bool
) -> None:
    """Check directed solves against NumPy, including JIT and vmap."""
    w = np.roll(np.eye(7), 1, axis=1)
    values = BCOO.fromdense(jnp.asarray(w)) if sparse else jnp.asarray(w)
    weights = prepare_spatial_weights(values)

    def evaluate_spatial_solve(r: jax.Array) -> SolverResult[Array]:
        return spatial_solve(weights, r, y, rtol=1e-09, atol=1e-10)

    result = jax.jit(evaluate_spatial_solve)(jnp.asarray(rho))
    assert result.converged
    np.testing.assert_allclose(
        result.value, np.linalg.solve(np.eye(7) - rho * w, y), atol=1e-8
    )

    def evaluate_default_tolerance_solve(r: jax.Array) -> SolverResult[Array]:
        return spatial_solve(weights, r, y)

    batched = jax.jit(jax.vmap(evaluate_default_tolerance_solve))(jnp.array([0.1, 0.3]))
    assert np.all(batched.converged)

    def evaluate_spatial_lags(x: jax.Array) -> Array:
        return spatial_lags(weights, x, order=2)

    actual = jax.jit(evaluate_spatial_lags)(y)
    np.testing.assert_allclose(actual, np.stack([w @ y, w @ w @ y]))


@pytest.mark.component
@pytest.mark.numerical
def test_time_varying_weights_are_applied_per_period() -> None:
    """Changing W_t is not equivalent to repeating one spatial operator."""
    w = np.roll(np.eye(5), 1, axis=1)
    permutation = np.array([2, 0, 4, 1, 3])
    second = w[np.ix_(permutation, permutation)]
    assert not np.allclose(w @ second, second @ w)
    weights = jnp.asarray(np.stack([w, second, w @ w]))
    y = jnp.arange(15, dtype=float).reshape(3, 5)
    actual = jax.jit(spatial_panel_lag)(weights, y)
    np.testing.assert_allclose(actual, np.einsum("tij,tj->ti", weights, y))
    sparse = BCOO.fromdense(weights, n_batch=1)
    np.testing.assert_allclose(jax.jit(spatial_panel_lag)(sparse, y), actual)
    assert not np.allclose(actual, spatial_panel_lag(weights[0], y))


@pytest.mark.unit
@pytest.mark.reference
def test_complex_spectrum_logdet_and_explicit_sparse_policy() -> None:
    """Retain complex eigenvalues and reject hidden densification."""
    w = np.roll(np.eye(5), 1, axis=1)
    weights = prepare_spatial_weights(w)
    np.testing.assert_allclose(
        spatial_logdet(weights, jnp.asarray(0.7)),
        np.linalg.slogdet(np.eye(5) - 0.7 * w)[1],
    )
    assert np.isneginf(spatial_logdet(weights, jnp.asarray(1.0)))
    with pytest.raises(TypeError, match="dense"):
        spatial_logdet(
            prepare_spatial_weights(BCOO.fromdense(jnp.asarray(w))), jnp.asarray(0.7)
        )


@pytest.mark.component
@pytest.mark.contract
def test_weight_preparation_rejects_implicit_graph_changes() -> None:
    """Require explicit isolate, normalization and diagonal policies."""
    with pytest.raises(ValueError, match="Isolated"):
        prepare_spatial_weights(np.zeros((3, 3)))
    with pytest.raises(ValueError, match="diagonal"):
        prepare_spatial_weights(np.eye(3))
    weights = prepare_spatial_weights(
        np.array([[0.0, 2.0], [1.0, 0.0]]), normalize_rows=True
    )
    np.testing.assert_allclose(weights.row_sums, 1)


@pytest.mark.unit
@pytest.mark.reference
@pytest.mark.parametrize("sparse", [False, True])
def test_directed_disconnected_graph_and_isolate_preserve_explicit_policy(
    *, sparse: bool
) -> None:
    """Unequal edge weights and both graph axes follow the declared node order."""
    raw = np.array(
        [
            [0.0, 2.0, 1.0, 0.0, 0.0, 0.0],
            [1.0, 0.0, 3.0, 0.0, 0.0, 0.0],
            [4.0, 1.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 2.0, 0.0],
            [0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        ]
    )
    row_sum = raw.sum(axis=1)
    expected = raw / np.where(row_sum > 0, row_sum, 1)[:, None]
    matrix = BCOO.fromdense(jnp.asarray(raw)) if sparse else raw
    weights = prepare_spatial_weights(matrix, normalize_rows=True, allow_isolates=True)
    y = np.array([1.0, -2.0, 3.0, 4.0, -1.0, 5.0])
    np.testing.assert_allclose(
        spatial_lags(weights, jnp.asarray(y), order=2),
        np.stack([expected @ y, expected @ expected @ y]),
        atol=1e-12,
    )
    actual = spatial_solve(
        weights, jnp.asarray(0.35), jnp.asarray(y), rtol=1e-10, atol=1e-11
    )
    assert actual.converged
    assert actual.status == SolverStatus.SUCCESS
    np.testing.assert_allclose(
        actual.value,
        np.linalg.solve(np.eye(len(y)) - 0.35 * expected, y),
        rtol=1e-9,
        atol=1e-10,
    )
    residual = np.linalg.norm((np.eye(len(y)) - 0.35 * expected) @ actual.value - y)
    assert float(actual.residual_norm) == pytest.approx(residual, abs=1e-12)
    order = np.array([5, 2, 4, 0, 3, 1])
    permuted = prepare_spatial_weights(
        expected[np.ix_(order, order)], allow_isolates=True
    )
    transformed = spatial_solve(
        permuted, jnp.asarray(0.35), jnp.asarray(y[order]), rtol=1e-10, atol=1e-11
    )
    assert transformed.converged
    np.testing.assert_allclose(
        transformed.value, actual.value[order], rtol=1e-9, atol=1e-10
    )
    np.testing.assert_allclose(weights.row_sums, expected.sum(axis=1), atol=1e-12)
    assert weights.limit == pytest.approx(
        1 / max(np.linalg.norm(expected, ord=np.inf), 1)
    )


@pytest.mark.component
@pytest.mark.contract
def test_singular_spatial_solve_retains_failure_without_finite_output() -> None:
    """A unit eigenvalue cannot yield an available solution at rho one."""
    w = np.roll(np.eye(5), 1, axis=1)
    weights = prepare_spatial_weights(w)
    result = spatial_solve(weights, jnp.asarray(1.0), jnp.ones(5), max_steps=8)
    assert not result.converged
    assert result.status == SolverStatus.FAILED
    assert np.all(np.isnan(result.value))
    assert np.isneginf(spatial_logdet(weights, jnp.asarray(1.0)))
