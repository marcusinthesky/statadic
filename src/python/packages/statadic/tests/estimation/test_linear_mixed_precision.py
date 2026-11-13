"""Mixed-role WLS working precision and independent numerical witnesses."""

from __future__ import annotations

from itertools import product
from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from hypothesis import example, given
from hypothesis import strategies as st
from statadic.estimation.linear import LeastSquares, LinearProblem
from statadic.inference.linear_covariance import classical_covariance, hc_covariance

from .._budgets import examples
from .._strategies import float_arrays
from .._validation.glm import responses_and_precisions
from .._validation.precision import bfloat16_qr_error

if TYPE_CHECKING:
    from collections.abc import Callable

    from statadic.estimation.linear import LinearFit

# Mixed x64 roles require a caller-enabled x64 process.
_DEFAULT_FLOAT = jnp.float64 if jax.config.x64_enabled else jnp.float32
_DTYPES = [
    dtypes
    for dtypes in product((np.float32, np.float64), repeat=3)
    if jax.config.x64_enabled or np.float64 not in dtypes
]
_Y = [-1.1, 0.2, 0.4, 1.9, 1.8, 3.2]
_W = [0.7, 1.1, 2.3, 0.9, 1.7, 2.1]


def _with_retained_witnesses[F: Callable[..., None]](test: F) -> F:
    """Keep the original data at every retained dtype combination as an example."""
    for dtypes in _DTYPES:
        test = example(values=(_Y, _W), dtypes=dtypes)(test)
    return test


def _responses_and_precisions() -> st.SearchStrategy[tuple[list[float], list[float]]]:
    """Draw shared bounded perturbations around the retained noisy sample."""
    return responses_and_precisions(tuple(_Y))


@pytest.mark.component
@pytest.mark.reference
@examples(20)
@_with_retained_witnesses
@given(values=_responses_and_precisions(), dtypes=st.sampled_from(_DTYPES))
def test_mixed_wls_working_precision_and_reference(  # noqa: PLR0914  # independent fit and covariance witnesses
    values: tuple[list[float], list[float]],
    dtypes: tuple[type, type, type],
) -> None:
    """Compare each represented input to an independent weighted SVD oracle."""
    x = jnp.asarray([[1, -2], [1, -1], [1, 0], [1, 1], [1, 2], [1, 3]], dtype=dtypes[0])
    y = jnp.asarray(values[0], dtype=dtypes[1])
    w = jnp.asarray(values[1], dtype=dtypes[2])
    dtype = jnp.result_type(x, y, w, 1.0)
    nx, ny, nw = (np.asarray(a, dtype=np.float64) for a in (x, y, w))
    root = np.sqrt(nw)
    wx = root[:, None] * nx
    beta = np.linalg.lstsq(wx, root * ny, rcond=None)[0]
    residual = ny - nx @ beta
    mapping = np.linalg.pinv(wx)
    leverage = np.sum(wx * mapping.T, axis=1)
    influence = mapping.T * (root * residual / (1 - leverage))[:, None]
    classical = (nw @ residual**2 / (len(ny) - nx.shape[1])) * (mapping @ mapping.T)
    hc3 = influence.T @ influence
    tol = 2e-5 if dtype == jnp.float32 else 1e-11

    def fit(a: jax.Array, b: jax.Array, c: jax.Array) -> LinearFit:
        return LeastSquares().estimate(LinearProblem(a, b, c))

    eager = fit(x, y, w)
    compiled = jax.jit(fit)(x, y, w)
    batched = jax.jit(jax.vmap(fit))(
        jnp.stack([x, x]), jnp.stack([y, y]), jnp.stack([w, w])
    )

    def evaluate_first_lane(a: jax.Array) -> jax.Array:
        return a[0]

    for result in (eager, compiled, jax.tree.map(evaluate_first_lane, batched)):
        assert bool(result.valid)
        assert result.parameters.dtype == dtype
        assert (
            result.design.dtype,
            result.outcome.dtype,
            result.relative_precision.dtype,
        ) == (x.dtype, y.dtype, w.dtype)
        np.testing.assert_allclose(result.parameters, beta, rtol=tol, atol=tol)
        np.testing.assert_allclose(
            result.scores.values,
            nw[:, None] * nx * residual[:, None],
            rtol=tol,
            atol=tol,
        )
        np.testing.assert_allclose(
            result.weighted_sse, nw @ residual**2, rtol=tol, atol=tol
        )
        np.testing.assert_allclose(
            classical_covariance(result).covariance.values,
            classical,
            rtol=tol,
            atol=tol,
        )
        np.testing.assert_allclose(
            hc_covariance(result, kind="HC3").covariance.values, hc3, rtol=tol, atol=tol
        )
        np.testing.assert_allclose(result.predict(x), nx @ beta, rtol=tol, atol=tol)
    rescaled = fit(x, y, w * 4)
    np.testing.assert_allclose(
        rescaled.parameters, eager.parameters, rtol=tol, atol=tol
    )
    np.testing.assert_allclose(
        classical_covariance(rescaled).covariance.values, classical, rtol=tol, atol=tol
    )


@pytest.mark.component
@pytest.mark.reference
@pytest.mark.parametrize("mixed", [False, True])
def test_integer_inputs_use_jax_default_promotion(mixed: bool) -> None:  # noqa: FBT001  # parametrized case
    """Integer and mixed integer inputs follow the configured JAX promotion."""
    x = jnp.asarray([[1, 0], [1, 1], [1, 2], [1, 3]])
    y = jnp.asarray([1, 2, 4, 5], dtype=jnp.float32 if mixed else None)
    w = jnp.asarray([1, 2, 3, 4])
    fit = LeastSquares().estimate(LinearProblem(x, y, w))
    assert bool(fit.valid)
    assert fit.parameters.dtype == jnp.result_type(x, y, w, 1.0)
    assert fit.design.dtype == x.dtype
    assert fit.outcome.dtype == y.dtype
    assert fit.relative_precision.dtype == w.dtype
    root = np.sqrt(np.asarray(w, dtype=np.float64))
    expected = np.linalg.lstsq(root[:, None] * x, root * y, rcond=None)[0]
    np.testing.assert_allclose(fit.parameters, expected, rtol=2e-5, atol=2e-5)
    residual = np.asarray(y) - np.asarray(x) @ expected
    np.testing.assert_allclose(
        fit.scores.values,
        np.asarray(w)[:, None] * np.asarray(x) * residual[:, None],
        rtol=2e-5,
        atol=2e-5,
    )
    mapping = np.linalg.pinv(root[:, None] * np.asarray(x))
    covariance = (np.asarray(w) @ residual**2 / 2) * (mapping @ mapping.T)
    np.testing.assert_allclose(
        classical_covariance(fit).covariance.values, covariance, rtol=2e-5, atol=2e-5
    )


@pytest.mark.component
@pytest.mark.contract
def test_mixed_wls_invalid_lane_under_jit_vmap() -> None:
    """Invalid weight and deficient rank affect only their own fit lanes."""
    x = jnp.asarray([[1, 0], [1, 1], [1, 2], [1, 3]], dtype=jnp.float32)
    y = jnp.asarray(
        [1, 2, 4, 5], dtype=jnp.float64 if jax.config.x64_enabled else jnp.float32
    )
    w = jnp.asarray([1, 2, 3, 4], dtype=jnp.float32)

    def fit(design: jax.Array, weights: jax.Array) -> LinearFit:
        return LeastSquares().estimate(LinearProblem(design, y, weights))

    designs = jnp.stack([x, x, jnp.ones_like(x)])
    weights = jnp.stack([w, w.at[1].set(-1), w])
    result = jax.jit(jax.vmap(fit))(designs, weights)
    np.testing.assert_array_equal(result.valid, [True, False, False])
    np.testing.assert_array_equal(result.identified, [True, False, False])
    np.testing.assert_array_equal(result.rank, [2, -1, 1])
    assert bool(jnp.all(jnp.isnan(result.parameters[1:])))


@pytest.mark.component
@pytest.mark.contract
def test_bfloat16_wls_does_not_force_float32() -> None:
    """The backend may reject bf16 rather than silently widening it."""
    x = jnp.asarray([[1, 0], [1, 1], [1, 2], [1, 3]], dtype=jnp.bfloat16)
    y = jnp.asarray([1, 2, 4, 5], dtype=jnp.bfloat16)
    w = jnp.asarray([1, 2, 3, 4], dtype=jnp.bfloat16)
    unsupported = bfloat16_qr_error()
    if unsupported is not None:
        with pytest.raises(unsupported, match="bfloat16"):
            LeastSquares().estimate(LinearProblem(x, y, w))
    else:
        fit = LeastSquares().estimate(LinearProblem(x, y, w))
        assert fit.parameters.dtype == jnp.bfloat16
        assert bool(fit.valid)

    promoted = LeastSquares().estimate(LinearProblem(x, y.astype(jnp.float32), w))
    assert bool(promoted.valid)
    assert promoted.parameters.dtype == jnp.result_type(x, y.astype(jnp.float32), w)
    assert promoted.design.dtype == jnp.bfloat16
    assert promoted.relative_precision.dtype == jnp.bfloat16


@pytest.mark.component
@pytest.mark.numerical
@examples(40)
@example(
    y=jnp.asarray([0.2, 0.8, 2.1, 2.6], dtype=_DEFAULT_FLOAT),
    direction=jnp.asarray(
        [[0.2, -0.3], [-0.1, 0.2], [0.1, 0.1], [-0.2, 0.1]], dtype=_DEFAULT_FLOAT
    ),
)
@given(y=float_arrays((4,), bound=3.0), direction=float_arrays((4, 2), bound=0.5))
def test_ols_fitted_parameter_derivatives_match_independent_formula(
    y: jax.Array, direction: jax.Array
) -> None:
    """Factor reuse retains response and design derivatives away from rank changes."""
    dtype = _DEFAULT_FLOAT
    x = jnp.asarray([[1, -1], [1, 0], [1, 1], [1, 2]], dtype=dtype)

    def parameters(a: jax.Array, b: jax.Array) -> jax.Array:
        return LeastSquares().estimate(LinearProblem(a, b)).parameters

    forward = jax.jit(jax.jacfwd(parameters, argnums=1))(x, y)
    reverse = jax.jit(jax.jacrev(parameters, argnums=1))(x, y)

    def evaluate_design_jvp(a: jax.Array, b: jax.Array, d: jax.Array) -> jax.Array:
        def evaluate_design_parameters(z: jax.Array) -> jax.Array:
            return parameters(z, b)

        return jax.jvp(evaluate_design_parameters, (a,), (d,))[1]

    design_derivative = jax.jit(evaluate_design_jvp)(x, y, direction)
    nx, ny, nd = (np.asarray(a, dtype=np.float64) for a in (x, y, direction))
    beta = np.linalg.lstsq(nx, ny, rcond=None)[0]
    expected = np.linalg.pinv(nx) @ (-nd @ beta) + np.linalg.solve(
        nx.T @ nx, nd.T @ (ny - nx @ beta)
    )
    tol = 5e-5 if dtype == jnp.float32 else 1e-11
    np.testing.assert_allclose(forward, np.linalg.pinv(nx), rtol=tol, atol=tol)
    np.testing.assert_allclose(reverse, np.linalg.pinv(nx), rtol=tol, atol=tol)
    np.testing.assert_allclose(design_derivative, expected, rtol=tol, atol=tol)
