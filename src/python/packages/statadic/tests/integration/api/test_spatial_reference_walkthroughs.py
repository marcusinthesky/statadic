"""Public fixed-graph estimation and diagnostics against independent spreg fits."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import numpy as np
import pytest
from statadic.api.data import Observations, PanelLayout
from statadic.api.design import Intercept, Linear
from statadic.api.hypotheses import Normal, TestOutcome
from statadic.api.regression import Regression
from statadic.api.spatial.cross_section import (
    SpatialInformation,
    SpatialRegressionResult,
)
from statadic.api.spatial.diagnostics import AnselinKelejian, ResidualMoran, SpatialLM
from statadic.api.spatial.graph_regression import GraphRegressionResult
from statadic.api.spatial.graphs import PanelData, SpatialData, SpatialError, SpatialLag
from statadic.api.spatial.learning import GraphML
from statadic.api.spatial.observed import ObservedSpatialPanelResult
from statadic.api.spatial.panel import SpatialPanelResult
from statadic.estimation.spatial import (
    ML,
    ObservedPanelML,
    PanelML,
    SpatialFit,
    SpatialIV,
)
from statadic.operators.spatial import prepare_spatial_weights

if TYPE_CHECKING:
    from collections.abc import Callable
    from types import ModuleType

pytestmark = [pytest.mark.integration, pytest.mark.reference]


@pytest.mark.external_reference
@pytest.mark.requires_optional("spreg")
@pytest.mark.requires_optional("statsmodels")
@pytest.mark.parametrize("dependence", ["lag", "error"])
def test_cross_section_ml_public_fit_matches_spreg(  # noqa: PLR0914
    optional_reference: Callable[[str], ModuleType],
    dependence: Literal["lag", "error"],
) -> None:
    """The actual Regression ML route matches estimates, objective and predictions."""
    spreg = optional_reference("spreg")
    libpysal = optional_reference("libpysal")
    measures = optional_reference("statsmodels.tools.eval_measures")
    rng = np.random.default_rng(126)
    count = 40
    weights = 0.8 * np.roll(np.eye(count), 1, axis=1) + 0.2 * np.roll(
        np.eye(count), -1, axis=1
    )
    features = rng.normal(size=(count, 2))
    outcome = np.linalg.solve(
        np.eye(count) - 0.35 * weights,
        0.4
        - 0.5 * features[:, 0]
        + 0.8 * features[:, 1]
        + rng.normal(size=count) * 0.4,
    )
    rows = Observations({"x": features[:, 0], "z": features[:, 1], "y": outcome})
    data = SpatialData(rows, {"neighbors": prepare_spatial_weights(weights)})
    fitted = Regression(
        "y",
        Intercept() + Linear("x") + Linear("z"),
        dependence=SpatialLag("neighbors")
        if dependence == "lag"
        else SpatialError("neighbors"),
        estimator=ML(),
    ).estimate(data)
    assert isinstance(fitted, SpatialRegressionResult)
    assert isinstance(fitted.core, SpatialFit)
    assert fitted.valid
    peer_type = spreg.ML_Lag if dependence == "lag" else spreg.ML_Error
    peer = peer_type(
        outcome[:, None], features, libpysal.weights.full2W(weights), epsilon=1e-9
    )
    np.testing.assert_allclose(
        fitted.parameters, np.asarray(peer.betas).ravel(), atol=2e-6
    )
    np.testing.assert_allclose(fitted.core.log_likelihood, peer.logll, atol=2e-7)
    # spreg counts k = design columns, plus rho for ML_Lag only (its source adds
    # one "to include rho in count"), and never the lambda of ML_Error or the
    # Gaussian scale. statadic counts the design columns, the one spatial parameter
    # and the scale, so its k exceeds spreg's by one (lag) or two (error); AIC
    # and BIC are compared after that offset in k.
    statadic_k = features.shape[1] + 1 + 1 + 1
    offset = statadic_k - peer.k
    assert offset == (1 if dependence == "lag" else 2)
    metrics = {m.name: float(m.value or 0.0) for m in fitted.fit_statistics().metrics}
    assert metrics["n_parameters"] == statadic_k
    np.testing.assert_allclose(metrics["log_likelihood"], peer.logll, atol=2e-7)
    np.testing.assert_allclose(metrics["aic"], peer.aic + 2 * offset, atol=4e-7)
    np.testing.assert_allclose(
        metrics["bic"], peer.schwarz + offset * np.log(count), atol=4e-7
    )
    # statsmodels: -2 llf + 2 ln(ln(nobs)) df, with statadic's parameter count.
    np.testing.assert_allclose(
        metrics["hqic"], measures.hqic(peer.logll, count, statadic_k), atol=4e-7
    )
    np.testing.assert_allclose(
        fitted.core.sigma2, np.asarray(peer.sig2).item(), atol=2e-7
    )
    np.testing.assert_allclose(
        fitted.predict(data, kind="structural"),
        np.asarray(peer.predy).ravel(),
        atol=2e-6,
    )
    uncertainty = fitted.infer(covariance=SpatialInformation()).uncertainty
    np.testing.assert_allclose(
        uncertainty.covariance.values, peer.vm, rtol=1e-5, atol=1e-7
    )


@pytest.mark.external_reference
@pytest.mark.requires_optional("spreg")
@pytest.mark.requires_optional("statsmodels")
@pytest.mark.parametrize("backend", ["observed", "complete", "graph"])
def test_panel_ml_public_routes_match_spreg(  # noqa: PLR0914 -- one matched statistical model through three lowering paths
    optional_reference: Callable[[str], ModuleType],
    backend: Literal["observed", "complete", "graph"],
) -> None:
    """The three real panel routes match the same constant-W unit-effects SAR.

    This exercises full balanced observations only. It grants no changing-roster,
    random-effects, multi-operator or learned-graph reference coverage.
    """
    spreg = optional_reference("spreg")
    libpysal = optional_reference("libpysal")
    measures = optional_reference("statsmodels.tools.eval_measures")
    rng = np.random.default_rng(0)
    periods, units, width = 8, 12, 2
    weights = 0.5 * np.roll(np.eye(units), 1, axis=1) + 0.5 * np.roll(
        np.eye(units), -1, axis=1
    )
    features = rng.normal(size=(periods, units, width))
    effects = rng.normal(size=units)
    noise = rng.normal(size=(periods, units)) * 0.4
    outcome = np.linalg.solve(
        np.eye(units) - 0.3 * weights,
        (features @ np.array([0.5, -0.4]) + effects + noise).T,
    ).T
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
    estimator = (
        GraphML()
        if backend == "graph"
        else PanelML()
        if backend == "complete"
        else ObservedPanelML()
    )
    fitted = Regression(
        "y",
        Linear("x") + Linear("z"),
        structure=layout,
        dependence=SpatialLag("neighbors"),
        effects="unit",
        estimator=estimator,
    ).estimate(data)
    assert isinstance(
        fitted, (ObservedSpatialPanelResult, SpatialPanelResult, GraphRegressionResult)
    )
    assert isinstance(
        fitted,
        {
            "observed": ObservedSpatialPanelResult,
            "complete": SpatialPanelResult,
            "graph": GraphRegressionResult,
        }[backend],
    )
    assert fitted.valid
    names = fitted.parameter_names
    assert names is not None
    selected = np.asarray(fitted.parameters)[
        [names.index(name) for name in ("x", "z", "rho[neighbors]")]
    ]
    peer = spreg.Panel_FE_Lag(
        outcome.reshape(-1, 1),
        features.reshape(periods * units, width),
        libpysal.weights.full2W(weights),
        epsilon=1e-9,
    )
    np.testing.assert_allclose(selected, np.asarray(peer.betas).ravel(), atol=3e-6)
    # spreg's panel logll retains the parameter-independent NT/2*log(NT) offset.
    np.testing.assert_allclose(
        fitted.core.log_likelihood,
        peer.logll + outcome.size / 2 * np.log(outcome.size),
        atol=2e-6,
    )
    # spreg counts the coefficients and rho (its source adds rho to k) but
    # neither the Gaussian scale nor the unit effects. statadic counts fixed effects
    # whether explicit dummies or profiled out by the within transform, and the
    # scale, so on every route its k exceeds spreg's by the unit effects plus
    # one. The log likelihood is the same on all three routes; criteria are
    # compared after the logll offset above and that difference in k.
    ll_offset = outcome.size / 2 * np.log(outcome.size)
    extra_k = 1 + units
    metrics = {m.name: float(m.value or 0.0) for m in fitted.fit_statistics().metrics}
    assert metrics["n_parameters"] == peer.k + extra_k
    np.testing.assert_allclose(
        metrics["aic"], peer.aic - 2 * ll_offset + 2 * extra_k, atol=2e-5
    )
    np.testing.assert_allclose(
        metrics["bic"],
        peer.schwarz - 2 * ll_offset + extra_k * np.log(outcome.size),
        atol=2e-5,
    )
    np.testing.assert_allclose(
        metrics["hqic"],
        measures.hqic(peer.logll + ll_offset, outcome.size, peer.k + extra_k),
        atol=2e-5,
    )
    prediction = fitted.predict(data, kind="structural")
    np.testing.assert_allclose(
        np.concatenate([np.asarray(value) for value in prediction.values]),
        outcome.ravel() - np.asarray(peer.u).ravel(),
        atol=3e-6,
    )
    if backend == "complete":
        # spreg's Panel_FE_Lag.vm is the inverse *expected* (Anselin) Gaussian
        # information, and so is statadic's PanelML covariance (SpatialInformation,
        # debiased=False, scale nuisance eliminated after inversion). Both are
        # on the (beta, rho) block without the unit effects; spreg has no
        # degrees-of-freedom option. The observed and graph routes report
        # *observed* information instead (2e-3 to 7e-2 relative standard-error
        # differences here), a different estimator of the same covariance, so
        # they have no spreg peer and are not compared.
        covariance = np.asarray(
            fitted.infer(covariance=SpatialInformation()).uncertainty.covariance.values
        )
        order = [names.index(name) for name in ("x", "z", "rho[neighbors]")]
        np.testing.assert_allclose(
            covariance[np.ix_(order, order)], np.asarray(peer.vm), rtol=1e-6, atol=1e-9
        )


@pytest.mark.external_reference
@pytest.mark.requires_optional("spreg")
def test_ols_moran_lm_public_journey_matches_spreg(  # noqa: PLR0914 -- matched host/reference evidence
    optional_reference: Callable[[str], ModuleType],
) -> None:
    """Named rows, OLS fitting and requests retain the same fixed-graph tests."""
    spreg = optional_reference("spreg")
    libpysal = optional_reference("libpysal")
    rng = np.random.default_rng(126)
    count = 30
    weights = 0.8 * np.roll(np.eye(count), 1, axis=1) + 0.2 * np.roll(
        np.eye(count), -1, axis=1
    )
    x = rng.normal(size=(count, 2))
    y = rng.normal(size=count)
    observations = Observations({"x": x[:, 0], "z": x[:, 1], "y": y})
    fit = Regression.ols("y", Intercept() + Linear("x") + Linear("z")).estimate(
        observations
    )
    core = fit.core
    graph = prepare_spatial_weights(weights)
    outcomes = fit.diagnose(
        ResidualMoran(graph),
        *(
            SpatialLM(graph, kind=kind)
            for kind in ("lag", "error", "robust-lag", "robust-error", "joint")
        ),
    )
    oracle_graph = libpysal.weights.full2W(weights)
    oracle = spreg.OLS(y[:, None], x, w=oracle_graph, spat_diag=True, moran=True)
    for name, expected in (
        ("lm_lag", oracle.lm_lag),
        ("lm_error", oracle.lm_error),
        ("lm_robust_lag", oracle.rlm_lag),
        ("lm_robust_error", oracle.rlm_error),
        ("lm_joint", oracle.lm_sarma),
    ):
        outcome = outcomes[name]
        assert isinstance(outcome, TestOutcome)
        assert outcome.available
        assert outcome.statistic is not None
        assert outcome.pvalue is not None
        np.testing.assert_allclose(
            [outcome.statistic, outcome.pvalue], expected, rtol=1e-8, atol=1e-10
        )
        assert outcome.evidence is not None
    moran = outcomes["moran_residual"]
    assert isinstance(moran, TestOutcome)
    assert moran.statistic is not None
    assert moran.pvalue is not None
    expected_moran = spreg.MoranRes(oracle, oracle_graph, z=True)
    np.testing.assert_allclose(
        [moran.statistic, moran.pvalue],
        [expected_moran.I, expected_moran.p_norm],
        atol=1e-10,
    )
    assert isinstance(moran.reference, Normal)
    assert moran.reference.location == pytest.approx(expected_moran.eI)
    assert moran.reference.scale**2 == pytest.approx(expected_moran.vI)
    assert fit.core is core


@pytest.mark.external_reference
@pytest.mark.requires_optional("spreg")
def test_spatial_iv_ak_public_journey_matches_spreg(
    optional_reference: Callable[[str], ModuleType],
) -> None:
    """The host uses the actual fitted spatial-IV instruments and residuals."""
    spreg = optional_reference("spreg")
    libpysal = optional_reference("libpysal")
    rng = np.random.default_rng(126)
    count = 40
    weights = 0.6 * np.roll(np.eye(count), 1, axis=1) + 0.4 * np.roll(
        np.eye(count), -1, axis=1
    )
    x = rng.normal(size=(count, 2))
    y = np.linalg.solve(
        np.eye(count) - 0.3 * weights,
        0.3 - 0.5 * x[:, 0] + 0.8 * x[:, 1] + rng.normal(size=count) * 0.5,
    )
    rows = Observations({"x": x[:, 0], "z": x[:, 1], "y": y})
    fit = Regression(
        "y",
        Intercept() + Linear("x") + Linear("z"),
        dependence=SpatialLag("neighbors"),
        estimator=SpatialIV(instrument_columns=(1, 2)),
    ).estimate(SpatialData(rows, {"neighbors": prepare_spatial_weights(weights)}))
    assert isinstance(fit, SpatialRegressionResult)
    retained = fit.core
    outcome = fit.diagnose(AnselinKelejian())["anselin_kelejian"]
    assert isinstance(outcome, TestOutcome)
    oracle_graph = libpysal.weights.full2W(weights)
    oracle_fit = spreg.GM_Lag(y[:, None], x, w=oracle_graph, w_lags=1)
    expected = spreg.AKtest(oracle_fit, oracle_graph, case="gen")
    assert outcome.available
    assert outcome.statistic is not None
    assert outcome.pvalue is not None
    np.testing.assert_allclose(
        [outcome.statistic, outcome.pvalue],
        [expected.ak, expected.p],
        rtol=1e-8,
        atol=1e-10,
    )
    assert fit.core is retained


def test_fixed_effects_panel_criteria_agree_across_routes() -> None:
    """One unit-effects SAR panel has one k, log likelihood and criteria per route."""
    rng = np.random.default_rng(0)
    periods, units = 8, 12
    weights = 0.5 * np.roll(np.eye(units), 1, axis=1) + 0.5 * np.roll(
        np.eye(units), -1, axis=1
    )
    features = rng.normal(size=(periods, units, 2))
    effects = rng.normal(size=units)
    noise = rng.normal(size=(periods, units)) * 0.4
    outcome = np.linalg.solve(
        np.eye(units) - 0.3 * weights,
        (features @ np.array([0.5, -0.4]) + effects + noise).T,
    ).T
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
    results = {}
    for name, estimator in (
        ("observed", ObservedPanelML()),
        ("complete", PanelML()),
        ("graph", GraphML()),
    ):
        fitted = Regression(
            "y",
            Linear("x") + Linear("z"),
            structure=layout,
            dependence=SpatialLag("neighbors"),
            effects="unit",
            estimator=estimator,
        ).estimate(data)
        assert fitted.valid
        results[name] = {
            m.name: float(m.value or 0.0) for m in fitted.fit_statistics().metrics
        }
    # Two slopes, rho, the scale and one effect per unit, on every route.
    k = 2 + 1 + 1 + units
    for metrics in results.values():
        assert metrics["n_parameters"] == k
        for key in ("log_likelihood", "aic", "bic", "hqic"):
            np.testing.assert_allclose(
                metrics[key], results["complete"][key], rtol=1e-8, atol=2e-6
            )
