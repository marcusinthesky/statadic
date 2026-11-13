"""Dynamic graph likelihood on fixed-distance restrictions against PySAL spreg."""

from __future__ import annotations

from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy.spatial.distance import cdist
from scipy.special import softmax
from statadic.estimation.spatial.graph_likelihood import (
    _graph_mean,
    estimate_graph_ml,
    graph_log_density,
    graph_profile,
)
from statadic.operators.spatial.distance import rbf_weights

if TYPE_CHECKING:
    from collections.abc import Callable
    from types import ModuleType


@pytest.mark.unit
@pytest.mark.reference
@pytest.mark.parametrize("roles", [0, 1])
def test_graph_mean_has_one_pure_transformable_owner(roles: int) -> None:
    """Reduced and structural means follow their distinct Gaussian equations."""
    w = np.roll(np.eye(4), 1, axis=1)[None, ...][:roles]
    rho = np.full(roles, 0.3)
    x = np.column_stack((np.ones(4), np.arange(4)))
    beta = jnp.asarray([0.4, -0.2])
    outcomes = jnp.asarray([1.0, 3.0, 2.0, 4.0])
    a = np.eye(4) - np.einsum("g,gij->ij", rho, w)

    def reduced(coefficients: jax.Array) -> jax.Array:
        return _graph_mean(
            jnp.asarray(w), jnp.asarray(rho), jnp.asarray(x), coefficients
        )

    expected = np.linalg.solve(a, x @ np.asarray(beta))
    np.testing.assert_allclose(jax.jit(reduced)(beta), expected)
    tangent = jnp.asarray([0.2, 0.5])
    _, jvp = jax.jvp(reduced, (beta,), (tangent,))
    np.testing.assert_allclose(jvp, np.linalg.solve(a, x @ np.asarray(tangent)))
    np.testing.assert_allclose(
        jax.vmap(reduced)(jnp.stack((beta, beta))), [expected, expected]
    )
    structural = _graph_mean(
        jnp.asarray(w), jnp.asarray(rho), jnp.asarray(x), beta, outcome=outcomes
    )
    np.testing.assert_allclose(
        structural, x @ np.asarray(beta) + (np.eye(4) - a) @ np.asarray(outcomes)
    )


@pytest.mark.component
@pytest.mark.reference
@pytest.mark.parametrize("partial", [False, True])
def test_no_dependence_profile_is_pooled_ols(*, partial: bool) -> None:
    """A precomputed SLX design needs only its observed independent OLS rows."""
    rng = np.random.default_rng(428)
    designs = tuple(
        np.column_stack((np.ones(n), rng.normal(size=(n, 2)))) for n in (7, 5)
    )
    positions = (np.array([6, 1, 3, 0]), np.array([4, 1, 2])) if partial else None
    outcomes = tuple(
        x @ np.array([0.3, -0.4, 0.7]) + rng.normal(size=len(x)) for x in designs
    )
    selected_x = (
        designs
        if positions is None
        else tuple(x[i] for x, i in zip(designs, positions, strict=True))
    )
    selected_y = (
        outcomes
        if positions is None
        else tuple(y[i] for y, i in zip(outcomes, positions, strict=True))
    )
    x, y = np.concatenate(selected_x), np.concatenate(selected_y)
    beta = np.linalg.lstsq(x, y, rcond=None)[0]
    sigma2 = np.mean((y - x @ beta) ** 2)
    log_likelihood = -len(y) * (np.log(2 * np.pi * sigma2) + 1) / 2
    empty = tuple(jnp.empty((0, len(block), len(block))) for block in designs)
    args = (
        empty,
        empty,
        tuple(map(jnp.asarray, designs)),
        tuple(map(jnp.asarray, selected_y)),
    )
    observed = None if positions is None else tuple(map(jnp.asarray, positions))
    actual = graph_profile(*args, jnp.empty(0), observed=observed)
    np.testing.assert_allclose(actual[0], beta, atol=1e-12)
    np.testing.assert_allclose(actual[1:], [sigma2, log_likelihood], atol=1e-12)
    np.testing.assert_allclose(
        graph_log_density(
            *args,
            jnp.asarray(beta),
            jnp.empty(0),
            jnp.asarray(sigma2),
            observed=observed,
        ),
        log_likelihood,
        atol=1e-12,
    )


@pytest.mark.integration
@pytest.mark.reference
@pytest.mark.external_reference
@pytest.mark.requires_optional("spreg")
@pytest.mark.requires_optional("libpysal")
@pytest.mark.parametrize("process", ["lag", "error"])
@pytest.mark.parametrize("periods", [1, 5], ids=["cross-section", "unit-effects"])
def test_distance_weight_profile_matches_spreg(  # noqa: PLR0914
    optional_reference: Callable[[str], ModuleType], process: str, periods: int
) -> None:
    """Compare beta, spatial coefficient, ML innovation variance and log likelihood."""
    spreg = optional_reference("spreg")
    libpysal = optional_reference("libpysal")
    rng = np.random.default_rng(548721)
    n = 24
    sites = rng.normal(size=(n, 2))
    cost = cdist(sites, sites, metric="sqeuclidean")
    logits = -cost / 0.7
    np.fill_diagonal(logits, -np.inf)
    w = softmax(logits, axis=1)
    graph = libpysal.weights.full2W(w)
    covariate = rng.normal(size=(periods, n))
    # A distinct directed graph supplies the exogenous lag in both estimators.
    x = np.stack((covariate, np.roll(covariate, -1, axis=1)), axis=-1)
    intercept = np.ones(n) * 0.4 if periods == 1 else rng.normal(size=n)
    mean = x @ np.array([-0.5, 0.8]) + intercept
    eps = rng.normal(size=(periods, n)) * 0.4
    a = np.eye(n) - 0.32 * w
    y = (
        np.linalg.solve(a, (mean + eps).T).T
        if process == "lag"
        else mean + np.linalg.solve(a, eps.T).T
    )
    empty = jnp.empty((0, n, n))
    filters = (rbf_weights(jnp.asarray(cost), jnp.asarray(0.7))[None],) * periods
    lag, error = (filters, (empty,) * periods)
    if process == "error":
        lag, error = error, lag
    extras = np.ones((n, 1)) if periods == 1 else np.eye(n)
    design = tuple(jnp.asarray(np.column_stack((xx, extras))) for xx in x)
    outcomes = tuple(map(jnp.asarray, y))

    names = (*(f"beta[{i}]" for i in range(design[0].shape[1])), process)
    fit = estimate_graph_ml(lag, error, design, outcomes, names)
    assert fit.valid
    beta, rho = fit.parameters[:-1], fit.parameters[-1]
    if periods == 1:
        estimator = spreg.ML_Lag if process == "lag" else spreg.ML_Error
        ref = estimator(y.reshape(-1, 1), x[0], graph, method="full", epsilon=1e-9)
        expected = np.asarray(ref.betas[:-1]).ravel()[[1, 2, 0]]
        np.testing.assert_allclose(beta, expected, rtol=2e-6, atol=2e-7)
        offset = 0.0
    else:
        estimator = spreg.Panel_FE_Lag if process == "lag" else spreg.Panel_FE_Error
        ref = estimator(y.reshape(-1, 1), x.reshape(-1, 2), graph, epsilon=1e-9)
        # spreg eliminates unit effects by demeaning and reports just slopes.
        np.testing.assert_allclose(beta[:2], ref.betas[:2, 0], rtol=2e-6, atol=2e-7)
        # Its fitted criterion uses log(SSE) instead of log(SSE / NT).
        offset = (n * periods / 2) * np.log(n * periods)
    np.testing.assert_allclose(rho, ref.betas[-1, 0], atol=2e-6)
    np.testing.assert_allclose(fit.sigma2, np.asarray(ref.sig2).item(), atol=2e-7)
    np.testing.assert_allclose(
        fit.log_likelihood, ref.logll + offset, atol=2e-7, rtol=0
    )
