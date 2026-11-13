# standalone package-maintenance entry point
"""Fitted least-squares derivatives at interior full-column-rank designs."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from hypothesis import example, given
from hypothesis import strategies as st
from hypothesis.extra import numpy as hnp
from statadic.estimation.linear import (
    LeastSquares,
    LinearProblem,
)

from .._budgets import examples

_OUTCOME = [0.3, -1.0, 1.5, 2.1, 1.7]
_CONTRAST = [0.5, -1.0, 2.0]
_DTYPES = st.sampled_from([np.float32, np.float64])
# Outcomes and contrasts are free; only the conditioned design is fixed, since
# the fitted-solution derivatives are taken at full-column-rank interior designs.
_OUTCOMES = hnp.arrays(np.float64, 5, elements=st.floats(-3.0, 3.0))
_CONTRASTS = hnp.arrays(np.float64, 3, elements=st.floats(-2.0, 2.0))


def _fixture(
    dtype: type,
    outcome: np.ndarray | None = None,
    contrast: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return a conditioned design with the given (or retained) outcome and contrast."""
    design = np.asarray(
        [[1, -2, 0.5], [1, -1, 1], [1, 0, -1], [1, 1, 0.2], [1, 2, 1.5]],
        dtype=dtype,
    )
    y = np.asarray(_OUTCOME if outcome is None else outcome, dtype=dtype)
    a = np.asarray(_CONTRAST if contrast is None else contrast, dtype=dtype)
    return design, y, a


def _analytic_gradients(
    design: np.ndarray, outcome: np.ndarray, contrast: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Differentiate the full-rank normal equation in independent NumPy."""
    x64, y64, a64 = (value.astype(np.float64) for value in (design, outcome, contrast))
    beta = np.linalg.lstsq(x64, y64, rcond=None)[0]
    residual = y64 - x64 @ beta
    dual = np.linalg.solve(x64.T @ x64, a64)
    return residual[:, None] * dual - (x64 @ dual)[:, None] * beta, x64 @ dual


@pytest.mark.component
@pytest.mark.numerical
@examples(20)
@example(dtype=np.float32, outcome=np.asarray(_OUTCOME), contrast=np.asarray(_CONTRAST))
@example(dtype=np.float64, outcome=np.asarray(_OUTCOME), contrast=np.asarray(_CONTRAST))
@given(dtype=_DTYPES, outcome=_OUTCOMES, contrast=_CONTRASTS)
def test_fitted_ols_reverse_derivatives_match_analytic_and_numpy(
    dtype: type, outcome: np.ndarray, contrast: np.ndarray
) -> None:
    """Differentiate the fitted solution, including residual-dependent X terms."""
    design, outcome, contrast = _fixture(dtype, outcome, contrast)
    expected_x, expected_y = _analytic_gradients(design, outcome, contrast)

    def target(x: jax.Array, y: jax.Array) -> jax.Array:
        return jnp.dot(
            LeastSquares().estimate(LinearProblem(x, y)).parameters,
            jnp.asarray(contrast),
        )

    grad_x, grad_y = jax.jit(jax.grad(target, argnums=(0, 1)))(design, outcome)
    # A small conditioned QR solve should preserve directional effects to a
    # few dozen ulps in float32. The budget is relative to the derivative scale.
    tolerance = 100 * np.finfo(dtype).eps
    np.testing.assert_allclose(grad_x, expected_x, rtol=tolerance, atol=tolerance)
    np.testing.assert_allclose(grad_y, expected_y, rtol=tolerance, atol=tolerance)
    assert grad_x.dtype == dtype
    assert grad_y.dtype == dtype

    direction_x = np.cos(np.arange(design.size)).reshape(design.shape)
    direction_y = np.sin(np.arange(len(outcome)) + 0.5)
    step = 1e-5

    def independent(shift: float) -> float:
        return float(
            contrast.astype(np.float64)
            @ np.linalg.lstsq(
                design.astype(np.float64) + shift * direction_x,
                outcome.astype(np.float64) + shift * direction_y,
                rcond=None,
            )[0]
        )

    finite_difference = (independent(step) - independent(-step)) / (2 * step)
    derivative = np.sum(np.asarray(grad_x) * direction_x) + np.dot(grad_y, direction_y)
    np.testing.assert_allclose(
        derivative,
        finite_difference,
        rtol=max(tolerance, 1e-8),
        atol=max(tolerance, 1e-9),
    )


@pytest.mark.component
@pytest.mark.numerical
@examples(20)
@example(dtype=np.float32, outcome=np.asarray(_OUTCOME), contrast=np.asarray(_CONTRAST))
@example(dtype=np.float64, outcome=np.asarray(_OUTCOME), contrast=np.asarray(_CONTRAST))
@given(dtype=_DTYPES, outcome=_OUTCOMES, contrast=_CONTRASTS)
def test_fitted_ols_forward_jacobian_and_batched_reverse_agree(
    dtype: type, outcome: np.ndarray, contrast: np.ndarray
) -> None:
    """Forward and reverse derivatives survive JIT and paired outcome batches."""
    design, outcome, contrast = _fixture(dtype, outcome, contrast)
    expected = np.linalg.pinv(design.astype(np.float64))

    def parameters(y: jax.Array) -> jax.Array:
        return LeastSquares().estimate(LinearProblem(jnp.asarray(design), y)).parameters

    forward = jax.jit(jax.jacfwd(parameters))(outcome)
    reverse = jax.jit(jax.jacrev(parameters))(outcome)
    tolerance = 100 * np.finfo(dtype).eps
    np.testing.assert_allclose(forward, expected, rtol=tolerance, atol=tolerance)
    np.testing.assert_allclose(reverse, expected, rtol=tolerance, atol=tolerance)

    def target(y: jax.Array) -> jax.Array:
        return jnp.asarray(contrast) @ parameters(y)

    outcomes = np.stack([outcome, 2 * outcome + 1])
    gradients = jax.jit(jax.vmap(jax.grad(target)))(outcomes)
    np.testing.assert_allclose(
        gradients,
        np.broadcast_to(contrast @ expected, outcomes.shape),
        rtol=tolerance,
        atol=tolerance,
    )
