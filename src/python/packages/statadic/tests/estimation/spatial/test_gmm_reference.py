# standalone package-maintenance entry point
"""Spatial error/combined GMM against matched spreg moment specifications."""

from __future__ import annotations

from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from hypothesis import example, given
from hypothesis import strategies as st
from statadic.estimation.spatial import SAC, SEM, SpatialGMM, SpatialProblem
from statadic.inference.spatial.covariance import spatial_covariance
from statadic.operators.spatial import prepare_spatial_weights

from ..._budgets import examples

if TYPE_CHECKING:
    from collections.abc import Callable
    from types import ModuleType

    from statadic.estimation.spatial.gmm import SpatialGMMFit


@pytest.mark.component
@pytest.mark.reference
@pytest.mark.external_reference
@pytest.mark.requires_optional("spreg")
@pytest.mark.parametrize("hetero", [False, True])
@pytest.mark.parametrize("lag", [False, True])
@examples(6)
@example(seed=126)
@given(seed=st.integers(0, 2**32 - 1))
def test_spatial_error_gmm_matches_spreg(
    optional_reference: Callable[[str], ModuleType],
    seed: int,
    *,
    hetero: bool,
    lag: bool,
) -> None:
    """Match one-update GS2SLS moments, coefficients and covariance without clipping."""
    spreg = optional_reference("spreg")
    libpysal = optional_reference("libpysal")
    rng = np.random.default_rng(seed)
    n = 60
    w = 0.6 * np.roll(np.eye(n), 1, axis=1) + 0.4 * np.roll(np.eye(n), -1, axis=1)
    x = np.column_stack([np.ones(n), rng.normal(size=(n, 2))])
    shock = rng.normal(size=n) * (0.3 + 0.2 * np.abs(x[:, 1]) if hetero else 0.5)
    y = x @ np.array([0.3, -0.5, 0.8]) + np.linalg.solve(np.eye(n) - 0.3 * w, shock)
    if lag:
        y = np.linalg.solve(np.eye(n) - 0.25 * w, y)
    fit = SpatialGMM(
        variance_model="heteroskedastic" if hetero else "homoskedastic",
        instrument_columns=(1, 2),
        tolerance=1e-09,
    ).estimate(
        SpatialProblem(
            x,
            y,
            prepare_spatial_weights(w),
            process=SAC() if lag else SEM(),
            slx_columns=(),
        )
    )
    if lag:
        cls = spreg.GM_Combo_Het if hetero else spreg.GM_Combo_Hom
        oracle = cls(
            y[:, None],
            x[:, 1:],
            w=libpysal.weights.full2W(w),
            w_lags=1,
            epsilon=1e-9,
            max_iter=1,
        )
    else:
        cls = spreg.GM_Error_Het if hetero else spreg.GM_Error_Hom
        oracle = cls(
            y[:, None], x[:, 1:], w=libpysal.weights.full2W(w), epsilon=1e-9, max_iter=1
        )
    assert fit.valid
    np.testing.assert_allclose(
        fit.moment_variability, fit.moment_variability.T, atol=1e-12
    )
    assert np.all(np.linalg.eigvalsh(fit.moment_variability) > 0)
    np.testing.assert_allclose(
        fit.iteration_change,
        abs(fit.parameters[-1] - fit.coefficient_lambda),
        atol=1e-12,
    )
    np.testing.assert_allclose(
        fit.parameters, np.asarray(oracle.betas).ravel(), rtol=1e-4, atol=2e-5
    )
    np.testing.assert_allclose(
        fit.residuals, np.asarray(oracle.u).ravel(), rtol=1e-4, atol=2e-5
    )
    np.testing.assert_allclose(
        fit.innovations, np.asarray(oracle.e_filtered).ravel(), rtol=1e-4, atol=2e-5
    )
    covariance = spatial_covariance(fit)
    assert covariance.valid
    np.testing.assert_allclose(
        covariance.covariance.values, oracle.vm, rtol=3e-4, atol=2e-5
    )
    np.testing.assert_allclose(fit.solver.value[0], fit.parameters[-1], atol=1e-12)


@pytest.mark.component
@pytest.mark.reference
@examples(8)
@example(seed=71)
@given(seed=st.integers(0, 2**32 - 1))
def test_spatial_gmm_jit_preserves_actual_iteration_state(seed: int) -> None:
    """A numerical update retains the lambda actually used for its beta solve."""
    rng = np.random.default_rng(seed)
    n = 30
    w = 0.7 * np.roll(np.eye(n), 1, axis=1) + 0.3 * np.roll(np.eye(n), -1, axis=1)
    x = np.column_stack([np.ones(n), rng.normal(size=n)])
    y = x @ np.array([0.4, 0.8]) + np.linalg.solve(
        np.eye(n) - 0.3 * w, rng.normal(size=n)
    )
    weights = prepare_spatial_weights(w)

    def evaluate_fit(z: jax.Array) -> SpatialGMMFit:
        return SpatialGMM().estimate(
            SpatialProblem(jnp.asarray(x), z, weights, process=SEM(), slx_columns=())
        )

    evaluate_fit = jax.jit(evaluate_fit)(jnp.asarray(y))
    assert evaluate_fit.valid
    filtered_x = x - float(evaluate_fit.coefficient_lambda) * (w @ x)
    filtered_y = y - float(evaluate_fit.coefficient_lambda) * (w @ y)
    expected = np.linalg.lstsq(filtered_x, filtered_y, rcond=None)[0]
    np.testing.assert_allclose(evaluate_fit.parameters[:-1], expected, atol=1e-9)
