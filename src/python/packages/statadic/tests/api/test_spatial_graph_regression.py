"""Graph-qualified fits retain full roster identities and Gaussian predictions."""

import json
import math
from dataclasses import replace
from pathlib import Path  # noqa: TC003
from typing import TYPE_CHECKING

import jax.numpy as jnp
import numpy as np
import pytest
from statadic.api.artifacts import ArtifactError, load_graph_predictor, save_graph_predictor
from statadic.api.data import Observations, PanelLayout
from statadic.api.design import Intercept, Linear
from statadic.api.regression import Regression
from statadic.api.spatial.cross_section import SpatialInformation
from statadic.api.spatial.graph_regression import (
    GraphRegressionResult,
    PreparedGraphRegression,
)
from statadic.api.spatial.graphs import (
    PanelData,
    SpatialData,
    SpatialError,
    SpatialLag,
    SpatialX,
)
from statadic.api.spatial.prediction import SpatialPanelPrediction
from statadic.core.typing import Array
from statadic.operators.spatial import prepare_spatial_weights
from scipy.stats import multivariate_normal

if TYPE_CHECKING:
    from statadic.api.capabilities import SummarizesLikelihood


@pytest.mark.component
@pytest.mark.reference
def test_mixed_graph_marginal_score_and_prediction(tmp_path: Path) -> None:  # noqa: PLR0914, PLR0915
    """Compare three graph roles and shuffled, partially observed outcomes."""
    rng = np.random.default_rng(2718)
    n, t = 8, 6
    layout = PanelLayout("id", "time")
    a: list[np.ndarray] = []
    b: list[np.ndarray] = []
    s: list[np.ndarray] = []
    for graphs in (a, b, s):
        for _ in range(t):
            matrix = rng.uniform(size=(n, n)) ** 3
            np.fill_diagonal(matrix, 0)
            graphs.append(matrix / matrix.sum(axis=1, keepdims=True))
    x = rng.normal(size=(t, n))
    y = np.array(
        [
            np.linalg.solve(
                np.eye(n) - 0.3 * a[i],
                1
                + x[i]
                + 0.8 * s[i] @ x[i]
                + np.linalg.solve(np.eye(n) - 0.25 * b[i], rng.normal(size=n)),
            )
            for i in range(t)
        ]
    )
    obs = Observations(
        {
            "id": np.tile(np.arange(n), t),
            "time": np.repeat(np.arange(t), n),
            "x": x.ravel(),
            "y": y.ravel(),
        }
    )
    graphs = {
        name: tuple((i, prepare_spatial_weights(w)) for i, w in enumerate(mats))
        for name, mats in zip(("a", "b", "s"), (a, b, s), strict=True)
    }
    graphs["b"] = tuple(reversed(graphs["b"]))
    data = PanelData(obs.take(rng.permutation(t * n).tolist()), layout, graphs)
    model = Regression(
        "y",
        Intercept() + Linear("x") + SpatialX("x", "s"),
        dependence=SpatialLag("a") + SpatialError("b"),
    )
    complete = model.prepare(data)
    assert isinstance(complete, PreparedGraphRegression)
    assert complete.observed is None
    assert model.inspect(complete).likelihood == "complete"
    forced = replace(model, panel_likelihood="observed")
    state = forced.prepare(data)
    assert isinstance(state, PreparedGraphRegression)
    assert state.observed is not None
    for positions in state.observed:
        np.testing.assert_array_equal(positions, np.arange(n))
    assert forced.inspect(state).likelihood == "observed"
    result = model.estimate_prepared(complete)
    assert isinstance(result, GraphRegressionResult)
    assert result.valid
    assert result.parameter_names == ("intercept", "x", "W[s]:x", "rho[a]", "lambda[b]")
    np.testing.assert_allclose(
        result.log_score(data), result.core.log_likelihood, atol=1e-8
    )
    prediction = result.predict(data)
    save_graph_predictor(tmp_path, result)
    frozen = load_graph_predictor(tmp_path)
    restored = frozen.predict(data.nodes)
    assert isinstance(restored, SpatialPanelPrediction)
    assert restored.period_ids == prediction.period_ids
    for actual, expected_values in zip(restored.values, prediction.values, strict=True):
        np.testing.assert_allclose(actual, expected_values, atol=1e-9)
    with pytest.raises(ValueError, match="keys must match"):
        frozen.predict(data.nodes.take([0, 1]))
    keep = np.array([1, 3, 4, 7])
    idx = (np.arange(t)[:, None] * n + keep).ravel()
    sample = replace(data, outcomes=obs.take(rng.permutation(idx).tolist()))
    xx, theta, wx, rho, lam = np.asarray(result.parameters)
    expected = 0.0
    for i in range(t):
        filter_y, filter_e = np.eye(n) - rho * a[i], np.eye(n) - lam * b[i]
        mean = np.linalg.solve(filter_y, xx + theta * x[i] + wx * s[i] @ x[i])
        root = np.linalg.inv(filter_e @ filter_y)
        np.testing.assert_allclose(prediction.values[i], mean, atol=1e-8)
        cov = float(result.core.sigma2) * root @ root.T
        expected += multivariate_normal.logpdf(
            y[i, keep], mean=mean[keep], cov=cov[np.ix_(keep, keep)]
        )
    np.testing.assert_allclose(result.log_score(sample), expected, atol=1e-8)
    with pytest.raises(ValueError, match="neighbor outcome"):
        result.predict(sample, kind="structural")
    assert (
        result.infer(covariance=SpatialInformation())
        .intervals(reference="normal")
        .valid
    )
    assert result.report().metadata["outcome_lag_graphs"] == ("a",)
    # Replacing only the error field leaves the mean unchanged, but both
    # evaluation modes still require an admissible fitted Gaussian process.
    new_error = tuple(
        (i, prepare_spatial_weights(w * (2 / abs(lam)) if i == 1 else w))
        for i, w in enumerate(b)
    )
    invalid = replace(data, graphs={**graphs, "b": new_error})
    for evaluate in (result.predict, result.log_score):
        with pytest.raises(
            ValueError, match="admissible range for prediction period 1"
        ):
            evaluate(invalid)


@pytest.mark.component
@pytest.mark.reference
def test_graph_specific_wx_matches_direct_design(tmp_path: Path) -> None:
    """Both WX columns and report identifiers survive a pure SLX fit."""
    rng = np.random.default_rng(73)
    n = 14
    x, y = rng.normal(size=(2, n))
    w = np.roll(np.eye(n), 1, axis=1)
    v = w.T
    data = SpatialData(
        Observations({"x": x, "y": y}),
        {"a": prepare_spatial_weights(w), "b": prepare_spatial_weights(v)},
    )
    result = Regression(
        "y", Linear("x") + SpatialX("x", "a") + SpatialX("x", "b")
    ).estimate(data)
    assert isinstance(result, GraphRegressionResult)
    assert result.valid
    assert result.parameter_names == ("x", "W[a]:x", "W[b]:x")
    save_graph_predictor(tmp_path, result)
    frozen = load_graph_predictor(tmp_path)
    restored = frozen.predict(data.observations)
    assert isinstance(restored, Array)
    np.testing.assert_allclose(restored, result.predict(data))
    manifest = tmp_path / "manifest.json"
    state = json.loads(manifest.read_text())
    state["capabilities"].append("refit")
    manifest.write_text(json.dumps(state))
    with pytest.raises(ArtifactError):
        load_graph_predictor(tmp_path)
    design = np.column_stack((x, w @ x, v @ x))
    beta = np.linalg.lstsq(design, y, rcond=None)[0]
    np.testing.assert_allclose(result.parameters, beta, atol=1e-9)
    np.testing.assert_allclose(result.predict(data), design @ beta, atol=1e-9)


@pytest.mark.component
@pytest.mark.contract
def test_graph_effects_retain_intercept_provenance(tmp_path: Path) -> None:
    """A covariate named intercept remains laggable and keeps every unit effect."""
    rng = np.random.default_rng(38)
    n, t = 5, 4
    x, y = rng.normal(size=(2, n * t))
    rows = Observations(
        {
            "intercept": x,
            "y": y,
            "unit": np.tile(np.arange(n), t),
            "period": np.repeat(np.arange(t), n),
        }
    )
    w = np.roll(np.eye(n), 1, axis=1)
    data = PanelData(
        rows,
        PanelLayout("unit", "period"),
        {"a": prepare_spatial_weights(w), "b": prepare_spatial_weights(w.T)},
    )
    terms = (
        Linear("intercept") + SpatialX("intercept", "a") + SpatialX("intercept", "b")
    )
    fit = Regression("y", terms, effects="unit").estimate(data)
    assert isinstance(fit, GraphRegressionResult)
    assert fit.valid
    assert fit.parameter_names[:3] == ("intercept", "W[a]:intercept", "W[b]:intercept")
    assert len(fit.parameter_names) == n + 3
    save_graph_predictor(tmp_path, fit)
    frozen = load_graph_predictor(tmp_path).predict(rows)
    assert isinstance(frozen, SpatialPanelPrediction)
    np.testing.assert_allclose(frozen.values, fit.predict(data).values)


@pytest.mark.unit
@pytest.mark.contract
def test_graph_roles_reject_duplicates_and_mismatched_rosters() -> None:
    """Reject duplicate coefficients and inconsistent node order."""
    with pytest.raises(ValueError, match="once per process"):
        SpatialLag("a") + SpatialLag("a")
    w = np.roll(np.eye(4), 1, axis=1)
    sources = {
        "a": prepare_spatial_weights(w),
        "b": prepare_spatial_weights(w, unit_ids=np.arange(4)[::-1]),
    }
    data = SpatialData(Observations({"x": np.arange(4), "y": np.arange(4)}), sources)
    with pytest.raises(ValueError, match="ordered numeric unit IDs"):
        Regression(
            "y", Linear("x"), dependence=SpatialLag("a") + SpatialError("b")
        ).prepare(data)


@pytest.mark.component
@pytest.mark.reference
def test_graph_regression_fit_statistics_match_the_gaussian_likelihood() -> None:  # noqa: PLR0914
    """A pure SLX graph fit is Gaussian OLS: k = coefficients + scale, n = rows."""
    rng = np.random.default_rng(73)
    n = 30
    x, y = rng.normal(size=(2, n))
    w = np.roll(np.eye(n), 1, axis=1)
    data = SpatialData(
        Observations({"x": x, "y": y}),
        {"a": prepare_spatial_weights(w), "b": prepare_spatial_weights(w.T)},
    )
    result = Regression(
        "y", Linear("x") + SpatialX("x", "a") + SpatialX("x", "b")
    ).estimate(data)
    assert isinstance(result, GraphRegressionResult)
    design = np.column_stack((x, w @ x, w.T @ x))
    beta = np.linalg.lstsq(design, y, rcond=None)[0]
    sse = float(np.sum((y - design @ beta) ** 2))
    expected = -n / 2 * (np.log(2 * np.pi * sse / n) + 1)
    statistics = result.fit_statistics()
    summary: SummarizesLikelihood = result
    fit_summary = summary.likelihood_summary()
    assert fit_summary is not None
    assert fit_summary.log_likelihood == pytest.approx(expected)
    assert (fit_summary.n_parameters, fit_summary.n_observations) == (4, n)
    assert (
        fit_summary.unavailable_reason
        == "Spatial Gaussian log likelihood needs a valid fit."
    )
    metrics = {m.name: m.value for m in statistics.metrics}
    assert metrics["n_parameters"] == 4  # noqa: PLR2004
    assert metrics["log_likelihood"] == pytest.approx(expected)
    assert metrics["aic"] == pytest.approx(-2 * expected + 8)
    assert metrics["bic"] == pytest.approx(-2 * expected + 4 * np.log(n))
    assert metrics["hqic"] == pytest.approx(-2 * expected + 8 * np.log(np.log(n)))
    report = result.report()
    assert [m.name for m in report.metrics].count("log_likelihood") == 1
    assert statistics.notes[0] in report.notes
    inference = result.infer(covariance=SpatialInformation())
    assert inference.fit_statistics == statistics
    broken = replace(result.core, valid=jnp.asarray(False))  # noqa: FBT003
    invalid = replace(result, core=broken)
    assert invalid.fit_statistics().metrics[0].value == 4  # noqa: PLR2004
    assert not invalid.fit_statistics().metrics[1].available
    invalid_summary = invalid.likelihood_summary()
    assert invalid_summary is not None
    assert math.isnan(invalid_summary.log_likelihood)
    assert invalid_summary.n_parameters == 4  # noqa: PLR2004
