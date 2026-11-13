"""Whole-date graph refits preserve contemporaneous alignment and declarations."""

from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING

import jax.numpy as jnp
import numpy as np
import pytest
from statadic.api.base import FitContext
from statadic.api.data import MeasureColumn, Observations, PanelLayout
from statadic.api.design import Intercept, Linear
from statadic.api.regression import Regression, RegressionContext
from statadic.api.representation import Energy
from statadic.api.spatial.graph_regression import GraphRegressionResult
from statadic.api.spatial.graphs import (
    BandwidthRef,
    DistanceGraph,
    PanelData,
    ProjectionRef,
    RBFWeights,
    SpatialX,
)
from statadic.api.spatial.learning import GraphML, LearnBandwidth, LearnLinear
from statadic.api.spatial.resampling import TimeBlockPairs, bootstrap_graph_regression
from statadic.operators.longrun import circular_block_bootstrap_indices
from statadic.operators.spatial import prepare_spatial_weights
from statadic.sample.packed import PackedClouds

if TYPE_CHECKING:
    from statadic.api.spatial.graph_regression import (
        PreparedGraphRegression,
    )


@pytest.mark.component
@pytest.mark.reference
def test_whole_dates_match_independent_effects_regression(  # noqa: PLR0914
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Paired observations and changing W travel together across block boundaries."""
    rng = np.random.default_rng(328)
    periods, n = 5, 4
    x = rng.normal(size=(periods, n))
    ws = [np.roll(np.eye(n), 1 + t % 3, axis=1) for t in range(periods)]
    wx = np.einsum("tij,tj->ti", ws, x)
    y = 0.3 * x + 0.5 * wx + rng.normal(size=(periods, n))
    keys = {
        "unit": np.tile(np.arange(n), periods),
        "date": np.repeat(np.arange(periods) + 80, n),
    }
    nodes = Observations({"x": x.ravel(), **keys})
    outcomes = Observations({"y": y.ravel(), **keys}).take(
        rng.permutation(y.size).tolist()
    )
    data = PanelData(
        nodes,
        PanelLayout("unit", "date"),
        {"w": tuple((t + 80, prepare_spatial_weights(w)) for t, w in enumerate(ws))},
        outcomes,
    )
    model = Regression(
        "y", Linear("x") + SpatialX("x", "w"), effects="unit", estimator=GraphML()
    )
    original = model.estimate(data)
    assert isinstance(original, GraphRegressionResult)
    estimate = GraphML.estimate
    refits: list[PreparedGraphRegression] = []

    def record(
        self: GraphML, prepared: PreparedGraphRegression
    ) -> GraphRegressionResult:
        refits.append(prepared)
        return estimate(self, prepared)

    monkeypatch.setattr(GraphML, "estimate", record)
    attempts = 5
    request = TimeBlockPairs(n_boot=attempts, block_length=2, seed=171)
    res = original.infer(resampling=request)
    assert res.original is original
    assert res.original.prepared is original.prepared
    assert len(refits) == attempts
    np.testing.assert_array_equal(
        res.indices, circular_block_bootstrap_indices(periods, 2, attempts, seed=171)
    )
    assert res.n_eff == attempts
    assert res.periods == tuple(range(80, 85))
    expected = []
    for idx in np.asarray(res.indices):
        np.testing.assert_array_equal(idx[1::2], (idx[:-1:2] + 1) % periods)
        design = np.column_stack(
            (x[idx].ravel(), wx[idx].ravel(), np.tile(np.eye(n), (periods, 1)))
        )
        expected.append(np.linalg.lstsq(design, y[idx].ravel(), rcond=None)[0])
    np.testing.assert_allclose(np.asarray(res.draws, dtype=float), expected, atol=1e-6)
    report = res.report(confidence=0.5)
    np.testing.assert_allclose(
        np.asarray([r.std_error for r in report.parameters], dtype=float),
        np.std(expected, axis=0, ddof=1),
        atol=1e-6,
    )
    np.testing.assert_allclose(
        np.asarray([r.ci_lower for r in report.parameters], dtype=float),
        np.quantile(expected, 0.25, axis=0),
        atol=1e-6,
    )
    assert all(r.pvalue is None for r in report.parameters)
    assert (
        res.report().parameters[0].interval_unavailable_reason
        == "too-few-refits-for-tail"
    )
    with pytest.raises(ValueError, match="balanced"):
        bootstrap_graph_regression(
            model,
            replace(data, outcomes=outcomes.take([0, 1])),
            n_boot=2,
            block_length=1,
            seed=1,
        )
    with pytest.raises(ValueError, match="pooled or unit"):
        bootstrap_graph_regression(
            replace(model, effects="time"), data, n_boot=2, block_length=1, seed=1
        )
    context = original.context
    with pytest.raises(TypeError, match="canonical Regression"):
        replace(
            original,
            context=FitContext(
                context.observations, context.model_name, context.response
            ),
        ).infer(resampling=request)
    with pytest.raises(ValueError, match="another model"):
        replace(
            original,
            context=RegressionContext(
                context.observations,
                context.model_name,
                context.response,
                replace(model),
            ),
        ).infer(resampling=request)


@pytest.mark.component
def test_failed_draws_remain_in_denominator() -> None:
    """Resampling a single original date aliases its indicator with the intercept."""
    n = 5
    rng = np.random.default_rng(987)
    rows = Observations(
        {
            "x": rng.normal(size=2 * n),
            "d": np.repeat([0, 1], n),
            "unit": np.tile(np.arange(n), 2),
            "date": np.repeat([81, 82], n),
            "y": rng.normal(size=2 * n),
        }
    )
    data = PanelData(
        rows,
        PanelLayout("unit", "date"),
        {"w": prepare_spatial_weights(np.roll(np.eye(n), 1, axis=1))},
    )
    model = Regression(
        "y",
        Intercept() + Linear("d") + Linear("x") + SpatialX("x", "w"),
        estimator=GraphML(),
    )
    attempts = 8
    original = model.estimate(data)
    assert isinstance(original, GraphRegressionResult)
    result = original.infer(
        resampling=TimeBlockPairs(n_boot=attempts, block_length=1, seed=73)
    )
    assert result.original is original
    successes = sum(len(set(pair)) == len(pair) for pair in result.indices.tolist())
    assert 0 < result.n_eff == successes < attempts
    assert result.failures.count("invalid-fit") == attempts - successes
    report = result.report(confidence=0.5)
    assert report.metadata["n_eff"] == successes
    assert all(
        row.ci_lower is None and row.std_error is None for row in report.parameters
    )
    assert all(
        row.interval_unavailable_reason == "failed-refits" for row in report.parameters
    )


@pytest.mark.integration
def test_refits_projection_and_bandwidth(monkeypatch: pytest.MonkeyPatch) -> None:  # noqa: PLR0914
    """Record actual jointly optimized maps and bandwidths from each refit."""
    n, t = 7, 5
    rng = np.random.default_rng(37)
    ids = tuple(range(n))
    clouds = PackedClouds.from_clouds([jnp.array(rng.normal(size=(2, 2))) for _ in ids])
    p, h = ProjectionRef("map"), BandwidthRef("h")
    graph = DistanceGraph.from_measures(
        MeasureColumn(ids, clouds),
        geometry=Energy(),
        projection=p,
        weighting=RBFWeights(h),
    )
    direction = jnp.array([[0.8], [0.6]])
    w = np.asarray(graph.weights({"map": direction, "h": jnp.asarray(0.7)}))
    x = rng.normal(size=(t, n))
    y = 0.5 * x + 2 * x @ w.T + 0.025 * rng.normal(size=x.shape)
    rows = Observations(
        {
            "y": y.ravel(),
            "x": x.ravel(),
            "unit": np.tile(ids, t),
            "date": np.repeat(np.arange(t), n),
        }
    )
    data = PanelData(rows, PanelLayout("unit", "date"), {"w": graph})
    model = Regression(
        "y",
        Linear("x") + SpatialX("x", "w"),
        estimator=GraphML(
            projections={p: LearnLinear(1, initial=direction, norm=1)},
            bandwidths={h: LearnBandwidth(0.7)},
        ),
    )
    maps = []
    fit = GraphML.estimate

    def record(
        self: GraphML, problem: PreparedGraphRegression
    ) -> GraphRegressionResult:
        result = fit(self, problem)
        fitted = np.asarray(result.graph_parameters["map"])
        maps.append(fitted @ fitted.T)
        return result

    monkeypatch.setattr(GraphML, "estimate", record)
    attempts = 2
    result = bootstrap_graph_regression(
        model, data, n_boot=attempts, block_length=2, seed=302
    )
    assert result.n_eff == attempts
    assert len(maps) == attempts + 1
    assert not np.allclose(maps[1], maps[2])
    report = result.report()
    assert report.parameters[-1].name == "bandwidth[h]"
    se = report.parameters[-1].std_error
    assert se is not None
    assert se > 0
    refs = report.metadata["graph_refits"]
    assert isinstance(refs, tuple)
    assert set(refs) == {"h", "map"}
