# standalone package-maintenance entry point
"""KKP fixed-W component estimates against spreg's public estimator."""

from __future__ import annotations

from functools import partial
from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from hypothesis import example, given
from hypothesis import strategies as st
from statadic.estimation.spatial import KKP, SEM, SpatialPanelProblem
from statadic.estimation.spatial.kkp import _full_weight_tau
from statadic.inference.linear_covariance import classical_covariance
from statadic.operators.spatial import prepare_spatial_panel_weights
from scipy import optimize

from ..._budgets import examples

if TYPE_CHECKING:
    from collections.abc import Callable
    from types import ModuleType

    from statadic.estimation.spatial.kkp import KKPPanelFit


@pytest.mark.component
@pytest.mark.reference
def test_full_weight_tau_matches_exact_gaussian_quadratic_form_integration() -> None:
    """Degree-four Gaussian moments independently fix both disputed factors."""
    w = np.array([[0.0, 1.0, 0.0], [0.5, 0.0, 0.5], [0.0, 1.0, 0.0]])
    nodes, weights = np.polynomial.hermite.hermgauss(3)
    points = np.stack(np.meshgrid(*([np.sqrt(2) * nodes] * 3), indexing="ij"), axis=-1)
    points = points.reshape(-1, 3)
    products = np.stack(
        np.meshgrid(*([weights / np.sqrt(np.pi)] * 3), indexing="ij"), axis=-1
    )
    probabilities = products.reshape(-1, 3).prod(axis=1)
    forms = np.stack((np.eye(3), w.T @ w, (w + w.T) / 2))
    centered = (
        np.einsum("ni,kij,nj->nk", points, forms, points)
        - np.trace(forms, axis1=1, axis2=2)
    ) / np.sqrt(3)
    expected = centered.T @ (probabilities[:, None] * centered)
    np.testing.assert_allclose(
        expected, [[2, 5 / 3, 0], [5 / 3, 17 / 6, 0], [0, 0, 1.5]], atol=1e-14
    )
    actual = jax.jit(_full_weight_tau)(jnp.asarray(w))
    np.testing.assert_allclose(actual, expected, rtol=2e-6, atol=2e-6)


@pytest.mark.component
@pytest.mark.contract
def test_kkp_rejects_varying_graphs_and_single_period() -> None:
    """The component moments require constant W and at least two periods."""
    w = 0.5 * (np.roll(np.eye(4), 1, axis=1) + np.roll(np.eye(4), -1, axis=1))
    varying = prepare_spatial_panel_weights(np.stack([w, w]), period_ids=np.arange(2))
    with pytest.raises(ValueError, match="fixed W"):
        KKP().estimate(
            SpatialPanelProblem(
                np.ones((2, 4, 1)),
                np.ones((2, 4)),
                varying,
                process=SEM(),
                effects="random",
                slx_columns=(),
            )
        )
    one = prepare_spatial_panel_weights(w, period_ids=np.arange(1))
    with pytest.raises(ValueError, match="at least two periods"):
        KKP().estimate(
            SpatialPanelProblem(
                np.ones((1, 4, 1)),
                np.ones((1, 4)),
                one,
                process=SEM(),
                effects="random",
                slx_columns=(),
            )
        )


@pytest.mark.component
@pytest.mark.reference
@pytest.mark.external_reference
@pytest.mark.requires_optional("spreg")
@examples(6)
@example(seed=126)
@example(seed=1886)
@given(seed=st.integers(0, 2**32 - 1))
def test_kkp_components_and_coefficients_match_spreg(  # noqa: PLR0914
    optional_reference: Callable[[str], ModuleType],
    *,
    seed: int,
) -> None:
    """Match partial weighting; spreg full weighting has different trace factors."""
    spreg = optional_reference("spreg")
    libpysal = optional_reference("libpysal")
    rng = np.random.default_rng(seed)
    t, n, p = 8, 30, 3
    w = 0.6 * np.roll(np.eye(n), 1, axis=1) + 0.4 * np.roll(np.eye(n), -1, axis=1)
    x = np.concatenate([np.ones((t, n, 1)), rng.normal(size=(t, n, 2))], axis=2)
    errors = np.linalg.solve(
        np.eye(n) - 0.3 * w, (rng.normal(size=n) + rng.normal(size=(t, n)) * 0.5).T
    ).T
    y = x @ np.array([0.3, -0.5, 0.8]) + errors
    fit = KKP(full_weights=False, tolerance=1e-9).estimate(
        SpatialPanelProblem(
            x,
            y,
            prepare_spatial_panel_weights(w, period_ids=np.arange(t)),
            process=SEM(),
            effects="random",
            slx_columns=(),
        )
    )
    # GM_KKP exposes no optimizer options. Its default approximate-gradient
    # L-BFGS-B can stop early in the weakly scaled between-variance direction
    # (seed 1886). Tighten only its stopping criteria, preserving spreg's own
    # moments, weights, parameter bounds and transformed coefficient solve.
    with pytest.MonkeyPatch.context() as precision:
        precision.setattr(
            optimize,
            "fmin_l_bfgs_b",
            partial(
                optimize.fmin_l_bfgs_b,
                factr=10.0,
                pgtol=1e-12,
                maxiter=10000,
                maxfun=20000,
            ),
        )
        oracle = spreg.GM_KKP(
            y.reshape(-1, 1),
            x[:, :, 1:].reshape(t * n, p - 1),
            libpysal.weights.full2W(w),
            full_weights=False,
        )
    assert fit.valid
    assert fit.lambda_solver.converged
    actual = np.concatenate(
        [
            np.asarray(fit.parameters),
            np.atleast_1d(fit.spatial_lambda),
            np.asarray(fit.variance_components),
        ]
    )
    np.testing.assert_allclose(
        actual, np.asarray(oracle.betas).ravel(), rtol=2e-4, atol=2e-5
    )
    np.testing.assert_allclose(
        fit.residuals.ravel(), np.asarray(oracle.u).ravel(), rtol=2e-4, atol=2e-5
    )
    # Conditional on the separately estimated components, reconstruct the
    # complete spatial/temporal whitening rather than trusting retained X/y.
    filtered_x = np.einsum("ij,tjp->tip", np.eye(n) - float(fit.spatial_lambda) * w, x)
    filtered_y = y @ (np.eye(n) - float(fit.spatial_lambda) * w).T
    theta = 1 - np.sqrt(float(fit.variance_components[0] / fit.variance_components[1]))
    transformed_x = (filtered_x - theta * filtered_x.mean(axis=0)).reshape(t * n, p)
    transformed_y = (filtered_y - theta * filtered_y.mean(axis=0)).ravel()
    beta = np.linalg.lstsq(transformed_x, transformed_y, rcond=None)[0]
    residual = transformed_y - transformed_x @ beta
    covariance = (
        residual
        @ residual
        / (t * n - p)
        * np.linalg.inv(transformed_x.T @ transformed_x)
    )
    uncertainty = classical_covariance(fit.linear)
    assert uncertainty.valid
    np.testing.assert_allclose(fit.linear.design, transformed_x, atol=1e-10)
    np.testing.assert_allclose(fit.linear.outcome, transformed_y, atol=1e-10)
    np.testing.assert_allclose(
        uncertainty.covariance.values, covariance, rtol=1e-10, atol=1e-12
    )
    np.testing.assert_allclose(
        uncertainty.covariance.values, oracle.vm, rtol=2e-4, atol=2e-5
    )


@pytest.mark.component
@pytest.mark.contract
@examples(6)
@example(seed=126)
@given(seed=st.integers(0, 2**32 - 1))
def test_kkp_batched_parameters_preserve_fit_availability_per_lane(seed: int) -> None:
    """A failed response lane cannot mask coefficient columns in a valid lane."""
    rng = np.random.default_rng(seed)
    periods, units = 5, 12
    w = 0.6 * np.roll(np.eye(units), 1, axis=1) + 0.4 * np.roll(
        np.eye(units), -1, axis=1
    )
    x = np.concatenate(
        [np.ones((periods, units, 1)), rng.normal(size=(periods, units, 2))], axis=-1
    )
    y = (
        x @ np.array([0.3, -0.5, 0.8])
        + rng.normal(size=units)
        + rng.normal(size=(periods, units)) * 0.5
    )
    graph = prepare_spatial_panel_weights(w, period_ids=np.arange(periods))

    def evaluate_fit(z: jax.Array) -> KKPPanelFit:
        return KKP().estimate(
            SpatialPanelProblem(
                jnp.asarray(x),
                z,
                graph,
                process=SEM(),
                effects="random",
                slx_columns=(),
            )
        )

    batch = jax.jit(jax.vmap(evaluate_fit))(
        jnp.asarray(np.stack([y, np.full_like(y, np.nan)]))
    )
    np.testing.assert_array_equal(batch.valid, [True, False])
    assert np.all(np.isfinite(batch.parameters[0]))
    assert np.all(np.isnan(batch.parameters[1]))
