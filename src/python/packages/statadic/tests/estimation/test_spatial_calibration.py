# standalone package-maintenance entry point
"""Bounded Gaussian spatial ML operating-characteristic screens."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from statadic.estimation.spatial import ML, SAR, SpatialProblem
from statadic.inference.spatial import spatial_covariance
from statadic.operators.spatial import prepare_spatial_weights


@pytest.mark.component
@pytest.mark.statistical
@pytest.mark.slow
@pytest.mark.parametrize(("n", "rho"), [(48, 0.0), (96, 0.4)])
def test_gaussian_sar_calibration(  # noqa: PLR0914
    n: int, rho: float
) -> None:  # explicit Monte Carlo state
    """Screen bias, standard errors and coverage in two fixed-design Gaussian cells."""
    replications = 128
    rng = np.random.default_rng(126000 + n)
    w = 0.7 * np.roll(np.eye(n), 1, axis=1) + 0.3 * np.roll(np.eye(n), -2, axis=1)
    x = np.column_stack([np.ones(n), rng.normal(size=(n, 2))])
    truth = np.array([0.4, -0.5, 0.8, rho])
    errors = rng.normal(size=(replications, n)) * 0.5
    ys = np.linalg.solve(np.eye(n) - rho * w, (x @ truth[:3] + errors).T).T
    weights = prepare_spatial_weights(w)

    def one(y: jax.Array) -> tuple[jax.Array, jax.Array, jax.Array]:
        fit = ML().estimate(
            SpatialProblem(x, y, weights, process=SAR(), slx_columns=())
        )
        uncertainty = spatial_covariance(fit)
        return (
            fit.parameters,
            jnp.sqrt(jnp.diag(uncertainty.covariance.values)),
            fit.valid & uncertainty.valid,
        )

    run = jax.jit(jax.vmap(one))
    chunks = [
        jax.device_get(run(jnp.asarray(ys[start : start + 16])))
        for start in range(0, replications, 16)
    ]
    estimates, se, valid = [
        np.concatenate([chunk[i] for chunk in chunks]) for i in range(3)
    ]
    assert np.all(valid), "Unavailable fits must be counted as calibration failures."
    bias = estimates.mean(axis=0) - truth
    empirical = estimates.std(axis=0, ddof=1)
    ratio = np.sqrt(np.mean(se**2, axis=0)) / empirical
    coverage = np.mean(np.abs(estimates - truth) <= 1.959963984540054 * se, axis=0)
    # Predeclared Monte Carlo screens for these two fixed-design Gaussian cells.
    bias_bound = 4.5 * empirical / np.sqrt(replications)
    ratio_bound = 4.5 / np.sqrt(2 * (replications - 1))
    coverage_floor = 0.95 - 4.5 * np.sqrt(0.95 * 0.05 / replications)
    passed = bool(
        np.all(np.abs(bias) <= bias_bound)
        and np.all(np.abs(ratio - 1) <= ratio_bound)
        and np.all(coverage >= coverage_floor)
    )
    assert passed
