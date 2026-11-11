"""Independent likelihood references for spatial panels and the SAC model.

Peers and the conventions they fix (spreg source under ``spreg/``):

* Pooled panels: a pooled spatial panel with a common graph ``W`` is exactly a
  cross-section on the stacked rows with ``kron(I_T, W)``, so ``ML_Lag`` and
  ``ML_Error`` are the peers. Their ``k`` counts the design columns, plus rho
  for ``ML_Lag`` only (``ml_lag.py``: ``self.k += 1`` "to include rho in
  count"), and neither the lambda of ``ML_Error`` nor the Gaussian scale. statadic
  counts the design columns, the spatial parameter and the scale, so its ``k``
  exceeds spreg's by one (lag) or two (error).
* Random-effects panels: ``Panel_RE_Lag`` and ``Panel_RE_Error``
  (``panel_re.py``) fit Elhorst's (2003) random-effects Gaussian model, the same
  model as statadic's ``effects="random"`` (``y`` stacked time-major, ``kron(I_T,
  W)``, a constant added by spreg's ``check_constant``). Differences recorded
  and handled explicitly, with no tolerance loosened:

  - ``Panel_RE_Lag.logll`` (``panel_re.py`` ``-res_rho.fun - nt/2 ln 2pi -
    nt/2``) is ``lag_c_loglik_sp``, which omits the ``n/2 ln(phi^2)`` term that
    the same module's ``phi_c_loglik`` optimises. statadic's exact marginal
    likelihood includes it, so ``statadic = spreg.logll + n/2 ln(phi^2)`` with spreg's
    own ``phi``. spreg's ``k`` is the design columns plus rho (``self.k += 1``);
    statadic adds the scale and the variance ratio, an offset of two.
  - ``Panel_RE_Error.logll`` is built from ``err_c_loglik_ord``, whose
    ``nt/2 log(sse)`` uses the sum rather than the mean of squares, so it
    carries the parameter-independent ``nt/2 log(nt)`` offset also seen in
    ``Panel_FE_Lag``. spreg's ``k`` is the design columns only; statadic adds
    lambda, the scale and the variance ratio, an offset of three. The stored
    ``vm`` inversion is singular on many draws (a spreg limitation), so the
    draw used here is one where it succeeds. The error model is compared
    through spreg's own objective ``err_c_loglik_ord`` (with ``phi^2`` equal to
    statadic's variance ratio) at statadic's estimate, and statadic's profile likelihood
    at spreg's estimate; the two functions agree to rounding at both points.
    The optima are also compared: statadic reports convergence only when the
    per-observation score of the search coordinates (lambda and the log
    variance ratio) meets the search tolerance, so it reaches the maximum of
    the shared likelihood that spreg's stopped point also attains.

Fits with no independent peer, recorded here and not compared: learned
graph-parameter counts and censored spatial survival with an estimated ``rho``.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import TYPE_CHECKING, Literal

import jax.numpy as jnp
import numpy as np
import pytest
from statadic.api.data import Observations, PanelLayout
from statadic.api.design import Intercept, Linear
from statadic.api.regression import Regression
from statadic.api.spatial.cross_section import SpatialRegressionResult
from statadic.api.spatial.graphs import (
    PanelData,
    SpatialData,
    SpatialError,
    SpatialLag,
    SpatialLagError,
)
from statadic.api.spatial.panel import SpatialPanelResult
from statadic.estimation.spatial import ML, PanelML, SpatialFit
from statadic.estimation.spatial.panel import spatial_panel_profile
from statadic.operators.spatial import prepare_spatial_weights
from scipy import optimize

if TYPE_CHECKING:
    from collections.abc import Callable
    from types import ModuleType

# k offsets over spreg (see the module docstring) and the sparsity of the SAC graph.
LAG_RE_OFFSET = 2
ERROR_RE_OFFSET = 3
DENSITY = 0.15

pytestmark = [pytest.mark.integration, pytest.mark.reference]


def _metrics(fitted: SpatialPanelResult | SpatialRegressionResult) -> dict[str, float]:
    return {m.name: float(m.value or 0.0) for m in fitted.fit_statistics().metrics}


def _ring(count: int) -> np.ndarray:
    return 0.5 * np.roll(np.eye(count), 1, axis=1) + 0.5 * np.roll(
        np.eye(count), -1, axis=1
    )


def _panel(
    dependence: Literal["lag", "error"], seed: int, periods: int, units: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, PanelData, PanelLayout]:
    """Return weights, features, outcome (time-major), data and layout."""
    rng = np.random.default_rng(seed)
    weights = _ring(units)
    features = rng.normal(size=(periods, units, 2))
    effects = rng.normal(size=units)
    noise = rng.normal(size=(periods, units)) * 0.5
    inverse = np.linalg.inv(np.eye(units) - 0.3 * weights)
    mean = 0.4 + features @ np.array([0.5, -0.4])
    if dependence == "lag":
        outcome = (inverse @ (mean + effects + noise).T).T
    else:
        outcome = mean + (inverse @ (effects + noise).T).T
    rows = Observations(
        {
            "unit": np.tile(np.arange(units), periods),
            "date": np.repeat(np.arange(periods), units),
            "x": features[:, :, 0].ravel(),
            "z": features[:, :, 1].ravel(),
            "y": outcome.ravel(),
        }
    )
    layout = PanelLayout("unit", "date")
    data = PanelData(rows, layout, {"neighbors": prepare_spatial_weights(weights)})
    return weights, features, outcome, data, layout


def _fit_panel(
    dependence: Literal["lag", "error"],
    effects: Literal["pooled", "random"],
    data: PanelData,
    layout: PanelLayout,
) -> SpatialPanelResult:
    fitted = Regression(
        "y",
        Intercept() + Linear("x") + Linear("z"),
        structure=layout,
        dependence=SpatialLag("neighbors")
        if dependence == "lag"
        else SpatialError("neighbors"),
        effects=effects,
        estimator=PanelML(),
    ).estimate(data)
    assert isinstance(fitted, SpatialPanelResult)
    assert fitted.valid
    return fitted


@pytest.mark.external_reference
@pytest.mark.requires_optional("spreg")
@pytest.mark.requires_optional("libpysal")
@pytest.mark.requires_optional("statsmodels")
@pytest.mark.parametrize("dependence", ["lag", "error"])
def test_pooled_spatial_panel_matches_stacked_ml(  # noqa: PLR0914 -- one matched model, many conventions
    optional_reference: Callable[[str], ModuleType],
    dependence: Literal["lag", "error"],
) -> None:
    """Pooled panels equal ML_Lag / ML_Error on stacked rows with kron(I_T, W)."""
    spreg = optional_reference("spreg")
    libpysal = optional_reference("libpysal")
    measures = optional_reference("statsmodels.tools.eval_measures")
    periods, units = 6, 15
    weights, features, outcome, data, layout = _panel(dependence, 11, periods, units)
    fitted = _fit_panel(dependence, "pooled", data, layout)
    total = periods * units
    stacked = np.kron(np.eye(periods), weights)
    peer_type = spreg.ML_Lag if dependence == "lag" else spreg.ML_Error
    peer = peer_type(
        outcome.reshape(-1, 1),
        features.reshape(total, 2),
        libpysal.weights.full2W(stacked),
        epsilon=1e-9,
    )
    np.testing.assert_allclose(
        fitted.parameters, np.asarray(peer.betas).ravel(), atol=2e-6
    )
    np.testing.assert_allclose(fitted.core.log_likelihood, peer.logll, atol=2e-6)
    # peer.k: constant + 2 slopes (+ rho for the lag); statadic adds the spatial
    # parameter (error) and the scale.
    statadic_k = 2 + 1 + 1 + 1
    offset = statadic_k - peer.k
    assert offset == (1 if dependence == "lag" else 2)
    metrics = _metrics(fitted)
    assert metrics["n_parameters"] == statadic_k
    np.testing.assert_allclose(metrics["log_likelihood"], peer.logll, atol=2e-6)
    np.testing.assert_allclose(metrics["aic"], peer.aic + 2 * offset, atol=4e-6)
    np.testing.assert_allclose(
        metrics["bic"], peer.schwarz + offset * np.log(total), atol=4e-6
    )
    np.testing.assert_allclose(
        metrics["hqic"], measures.hqic(peer.logll, total, statadic_k), atol=4e-6
    )


@pytest.mark.external_reference
@pytest.mark.requires_optional("spreg")
@pytest.mark.requires_optional("libpysal")
@pytest.mark.requires_optional("statsmodels")
def test_random_effects_lag_panel_matches_spreg(  # noqa: PLR0914 -- one matched model, many conventions
    optional_reference: Callable[[str], ModuleType],
) -> None:
    """Random-effects SAR matches Panel_RE_Lag after the omitted phi term."""
    spreg = optional_reference("spreg")
    libpysal = optional_reference("libpysal")
    measures = optional_reference("statsmodels.tools.eval_measures")
    periods, units = 8, 20
    weights, features, outcome, data, layout = _panel("lag", 2, periods, units)
    fitted = _fit_panel("lag", "random", data, layout)
    total = periods * units
    peer = spreg.Panel_RE_Lag(
        outcome.reshape(-1, 1),
        features.reshape(total, 2),
        libpysal.weights.full2W(weights),
        epsilon=1e-9,
    )
    # spreg betas: constant, slopes, rho, phi (phi^2 = sig2 / (T sig2_u + sig2)).
    betas = np.asarray(peer.betas).ravel()
    np.testing.assert_allclose(fitted.parameters, betas[:-1], atol=2e-6)
    phi = float(betas[-1])
    # phi^2 = 1 / (1 + T tau) with tau statadic's Var(alpha) / Var(epsilon).
    np.testing.assert_allclose(
        phi**2, 1 / (1 + periods * float(fitted.core.variance_ratio)), rtol=1e-5
    )
    np.testing.assert_allclose(
        fitted.core.sigma2, np.asarray(peer.sig2).item(), rtol=1e-6
    )
    # spreg's logll omits the units/2 * ln(phi^2) term of the exact marginal.
    phi_term = units / 2 * np.log(phi**2)
    np.testing.assert_allclose(
        fitted.core.log_likelihood, peer.logll + phi_term, atol=2e-6
    )
    # peer.k: constant + 2 slopes + rho; statadic adds the scale and variance ratio.
    statadic_k = 2 + 1 + 1 + 1 + 1
    offset = statadic_k - peer.k
    assert offset == LAG_RE_OFFSET
    metrics = _metrics(fitted)
    assert metrics["n_parameters"] == statadic_k
    np.testing.assert_allclose(
        metrics["aic"], peer.aic - 2 * phi_term + 2 * offset, atol=4e-6
    )
    np.testing.assert_allclose(
        metrics["bic"],
        peer.schwarz - 2 * phi_term + offset * np.log(total),
        atol=4e-6,
    )
    np.testing.assert_allclose(
        metrics["hqic"],
        measures.hqic(peer.logll + phi_term, total, statadic_k),
        atol=4e-6,
    )


@pytest.mark.external_reference
@pytest.mark.requires_optional("spreg")
@pytest.mark.requires_optional("libpysal")
@pytest.mark.requires_optional("statsmodels")
def test_random_effects_error_panel_matches_spreg_objective(  # noqa: PLR0914
    optional_reference: Callable[[str], ModuleType],
) -> None:
    """Random-effects SEM equals spreg's own objective at statadic's estimate."""
    spreg = optional_reference("spreg")
    libpysal = optional_reference("libpysal")
    measures = optional_reference("statsmodels.tools.eval_measures")
    spreg_re = optional_reference("spreg.panel_re")
    periods, units = 8, 20
    weights, features, outcome, data, layout = _panel("error", 2, periods, units)
    fitted = _fit_panel("error", "random", data, layout)
    total = periods * units
    graph = libpysal.weights.full2W(weights)
    peer = spreg.Panel_RE_Error(
        outcome.reshape(-1, 1), features.reshape(total, 2), graph, epsilon=1e-9
    )
    names = fitted.parameter_names
    assert names is not None
    lam = float(np.asarray(fitted.parameters)[names.index("lambda[neighbors]")])
    phi = float(np.sqrt(fitted.core.variance_ratio))
    # Rebuild spreg's own inputs: constant first, time-major stack.
    design = np.column_stack([np.ones(total), features.reshape(total, 2)])
    response = outcome.reshape(-1, 1)
    identity = np.identity(units)
    evals, evecs = np.linalg.eigh(weights)  # symmetric ring, as spreg's branch
    one = np.ones((periods, 1))
    q = np.kron(one @ one.T / periods, identity)
    lagged = np.kron(np.eye(periods), weights)

    def spreg_loglik(parameters: tuple[float, float]) -> float:
        value = spreg_re.err_c_loglik_ord(
            parameters,
            evals,
            evecs,
            units,
            periods,
            response,
            design,
            lagged @ response,
            lagged @ design,
            q @ response,
            q @ design,
            identity,
            graph.sparse,
        )
        # Constants and the nt/2 log(nt) offset spreg leaves out of its logll.
        return float(
            -value
            - total / 2 * np.log(2 * np.pi)
            - total / 2
            + total / 2 * np.log(total)
        )

    # spreg's stored logll equals its objective at its own stopped point up to
    # the stated nt/2 log(nt) offset.
    np.testing.assert_allclose(
        spreg_loglik((peer.lam, peer.phi)) - total / 2 * np.log(total),
        peer.logll,
        atol=1e-9,
    )
    np.testing.assert_allclose(
        fitted.core.log_likelihood, spreg_loglik((lam, phi)), atol=2e-6
    )
    assert fitted.core.solver.converged
    np.testing.assert_allclose(lam, peer.lam, atol=2e-6)
    np.testing.assert_allclose(phi**2, peer.phi**2, rtol=1e-5)
    np.testing.assert_allclose(
        fitted.core.log_likelihood, spreg_loglik((peer.lam, peer.phi)), atol=2e-9
    )
    # The profile likelihood is the same function in statadic: evaluated at spreg's
    # stopped point it reproduces spreg's objective there.
    spreg_point = spreg_loglik((peer.lam, peer.phi))
    profile = spatial_panel_profile(
        jnp.array([float(peer.lam)]),
        fitted.core.design,
        fitted.core.outcome,
        fitted.core.weights,
        lag=False,
        error=True,
        effects="random",
        variance_ratio=jnp.asarray(float(peer.phi) ** 2),
    )
    np.testing.assert_allclose(float(profile[3]), spreg_point, atol=2e-6)
    # peer.k: constant + 2 slopes only; statadic adds lambda, scale, variance ratio.
    statadic_k = 2 + 1 + 1 + 1 + 1
    offset = statadic_k - peer.k
    assert offset == ERROR_RE_OFFSET
    ll = spreg_loglik((lam, phi))
    reference = SimpleNamespace(k=peer.k, logll=ll, n=total)
    metrics = _metrics(fitted)
    assert metrics["n_parameters"] == statadic_k
    np.testing.assert_allclose(
        metrics["aic"], spreg.akaike(reference) + 2 * offset, atol=4e-6
    )
    np.testing.assert_allclose(
        metrics["bic"], spreg.schwarz(reference) + offset * np.log(total), atol=4e-6
    )
    np.testing.assert_allclose(
        metrics["hqic"], measures.hqic(ll, total, statadic_k), atol=4e-6
    )


def _sac_log_likelihood(
    parameters: np.ndarray,
    outcome: np.ndarray,
    design: np.ndarray,
    weights: np.ndarray,
) -> float:
    """Dense Gaussian SAC log likelihood at (beta, rho, lambda, sigma2)."""
    count, width = design.shape
    beta, rho, lam, sigma2 = (
        parameters[:width],
        parameters[width],
        parameters[width + 1],
        parameters[width + 2],
    )
    identity = np.eye(count)
    shock = (identity - lam * weights) @ (
        (identity - rho * weights) @ outcome - design @ beta
    )
    logdet = (
        np.linalg.slogdet(identity - rho * weights)[1]
        + np.linalg.slogdet(identity - lam * weights)[1]
    )
    return float(
        -count / 2 * np.log(2 * np.pi * sigma2) - shock @ shock / (2 * sigma2) + logdet
    )


def test_sac_log_likelihood_is_witnessed_and_locally_optimal() -> None:  # noqa: PLR0914 -- one witness, one optimality check
    """A dense numpy likelihood reproduces statadic's SAC value, which no move beats."""
    rng = np.random.default_rng(7)
    count = 40
    raw = rng.uniform(size=(count, count)) * (
        rng.uniform(size=(count, count)) < DENSITY
    )
    np.fill_diagonal(raw, 0.0)
    raw[np.arange(count), (np.arange(count) + 1) % count] += 1.0
    weights = raw / raw.sum(axis=1, keepdims=True)
    features = rng.normal(size=(count, 2))
    design = np.column_stack([np.ones(count), features])
    shock = np.linalg.solve(np.eye(count) - 0.4 * weights, rng.normal(size=count) * 0.6)
    outcome = np.linalg.solve(
        np.eye(count) - 0.3 * weights, design @ np.array([0.4, 0.5, -0.4]) + shock
    )
    rows = Observations({"x": features[:, 0], "z": features[:, 1], "y": outcome})
    data = SpatialData(rows, {"neighbors": prepare_spatial_weights(weights)})
    fitted = Regression(
        "y",
        Intercept() + Linear("x") + Linear("z"),
        dependence=SpatialLagError("neighbors"),
        estimator=ML(),
    ).estimate(data)
    assert isinstance(fitted, SpatialRegressionResult)
    assert fitted.valid
    core = fitted.core
    assert isinstance(core, SpatialFit)
    names = fitted.parameter_names
    assert names[-2:] == ("rho[neighbors]", "lambda[neighbors]")
    estimate = np.concatenate([np.asarray(fitted.parameters), [float(core.sigma2)]])
    witnessed = _sac_log_likelihood(estimate, outcome, design, weights)
    np.testing.assert_allclose(core.log_likelihood, witnessed, atol=1e-8)
    metrics = _metrics(fitted)
    np.testing.assert_allclose(metrics["log_likelihood"], witnessed, atol=1e-8)
    # Design columns, rho, lambda and the scale.
    assert metrics["n_parameters"] == 3 + 2 + 1
    np.testing.assert_allclose(
        metrics["aic"], 2 * metrics["n_parameters"] - 2 * witnessed, atol=1e-8
    )
    np.testing.assert_allclose(
        metrics["bic"],
        metrics["n_parameters"] * np.log(count) - 2 * witnessed,
        atol=1e-8,
    )

    def negative(parameters: np.ndarray) -> float:
        return -_sac_log_likelihood(parameters, outcome, design, weights)

    # Coordinate perturbations: no nearby point improves beyond 1e-8.
    for index in range(estimate.size):
        for step in (-1e-3, 1e-3):
            moved = estimate.copy()
            moved[index] += step
            if moved[-1] <= 0 or abs(moved[3]) >= 1 or abs(moved[4]) >= 1:
                continue
            assert -negative(moved) <= witnessed + 1e-8
    # Local minimisation from statadic's estimate gains no more than 1e-7.
    result = optimize.minimize(
        negative,
        estimate,
        method="L-BFGS-B",
        bounds=[(None, None)] * 3 + [(-0.99, 0.99)] * 2 + [(1e-6, None)],
    )
    assert -result.fun <= witnessed + 1e-7
