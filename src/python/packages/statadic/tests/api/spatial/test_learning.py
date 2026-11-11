"""Recovery and out-of-period scoring with parameterized spatial graph sources."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace
from typing import TYPE_CHECKING

import jax.numpy as jnp
import numpy as np
import pytest
from statadic.api.artifacts import load_graph_predictor, save_graph_predictor
from statadic.api.base import InvalidFitError
from statadic.api.data import MeasureColumn, Observations, PanelLayout
from statadic.api.design import Intercept, Linear
from statadic.api.regression import Regression
from statadic.api.representation import Energy
from statadic.api.spatial.graph_regression import (
    GraphRegressionResult,
    PreparedGraphRegression,
)
from statadic.api.spatial.graphs import (
    BandwidthRef,
    DistanceGraph,
    PanelData,
    ProjectionRef,
    RBFWeights,
    SpatialX,
)
from statadic.api.spatial.learning import GraphML, LearnBandwidth, LearnLinear
from statadic.api.spatial.prediction import SpatialPanelPrediction
from statadic.api.spatial.selection import select_bandwidth
from statadic.sample.packed import PackedClouds

if TYPE_CHECKING:
    from pathlib import Path

_IDS = tuple(range(7))
_LAYOUT = PanelLayout("unit", "date")
_H = BandwidthRef("smooth")


def _sample(graph: DistanceGraph, *, seed: int, start: int, count: int) -> PanelData:
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(count, len(_IDS)))
    state = {"smooth": jnp.asarray(1.3), "map": jnp.array([[0.7], [0.3]])}
    w = np.asarray(graph.weights(state))
    y = 0.3 + 0.7 * x + 2.1 * x @ w.T + 0.04 * rng.normal(size=x.shape)
    rows = Observations(
        {
            "x": x.ravel(),
            "y": y.ravel(),
            "date": np.repeat(np.arange(start, start + count), len(_IDS)),
            "unit": np.tile(_IDS, count),
        }
    )
    return PanelData(rows, _LAYOUT, {"information": graph})


@pytest.mark.integration
def test_bandwidth_recovery_and_frozen_validation(tmp_path: Path) -> None:
    """A jointly learned bandwidth and a held-out grid recover a planted field."""
    d = np.abs(np.subtract.outer(np.arange(7), np.arange(7)))
    graph = DistanceGraph.from_distances(d, unit_ids=_IDS, weighting=RBFWeights(_H))
    train = _sample(graph, seed=8, start=0, count=5)
    validation = _sample(graph, seed=12, start=5, count=4)
    terms = Intercept() + Linear("x") + SpatialX("x", "information")
    result = Regression(
        "y", terms, estimator=GraphML(bandwidths={_H: LearnBandwidth(1.0)})
    ).estimate(train)
    assert isinstance(result, GraphRegressionResult)
    assert result.valid
    assert float(result.graph_parameters["smooth"]) == pytest.approx(1.3, abs=0.2)
    parameters = result.report().metadata["graph_parameters"]
    assert isinstance(parameters, Mapping)
    assert parameters["smooth"] == {
        "shape": (),
        "value": float(result.graph_parameters["smooth"]),
    }
    save_graph_predictor(tmp_path, result)
    frozen = load_graph_predictor(tmp_path)
    realized = frozen.predict(train.nodes)
    expected = result.predict(train)
    assert isinstance(realized, SpatialPanelPrediction)
    assert isinstance(expected, SpatialPanelPrediction)
    np.testing.assert_allclose(realized.values, expected.values, atol=1e-9)
    with pytest.raises(ValueError, match="keys must match"):
        frozen.predict(validation.nodes)
    sweep = select_bandwidth(
        Regression("y", terms),
        train,
        validation,
        parameter=_H,
        candidates=(0.2, 1.3, 4.0),
    )
    assert sweep.bandwidth == pytest.approx(1.3)
    assert tuple(h for h, _ in sweep.scores) == (0.2, 1.3, 4.0)
    assert sweep.refit_periods == ()
    with pytest.raises(ValueError, match="precede"):
        select_bandwidth(
            Regression("y", terms), validation, train, parameter=_H, candidates=(1.0,)
        )


@pytest.mark.integration
@pytest.mark.regression
def test_selected_bandwidth_replaces_a_learning_binding_and_refits() -> None:
    """The candidate fixes h while retaining caller solver options unchanged."""
    d = np.abs(np.subtract.outer(np.arange(7), np.arange(7)))
    graph = DistanceGraph.from_distances(d, unit_ids=_IDS, weighting=RBFWeights(_H))
    train = _sample(graph, seed=8, start=0, count=3)
    validation = _sample(graph, seed=12, start=3, count=2)
    refit = _sample(graph, seed=14, start=0, count=5)
    estimator = GraphML(bandwidths={_H: LearnBandwidth(0.8)}, max_steps=71)
    result = select_bandwidth(
        Regression(
            "y",
            Intercept() + Linear("x") + SpatialX("x", "information"),
            estimator=estimator,
        ),
        train,
        validation,
        parameter=_H,
        candidates=(1.3,),
        refit=refit,
    )
    assert result.bandwidth == pytest.approx(1.3)
    assert result.refit_periods == (0, 1, 2, 3, 4)
    assert result.result.valid
    assert result.result.graph_parameters == {}
    assert estimator.bandwidths == {_H: LearnBandwidth(0.8)}
    fixed = Regression(
        "y", Intercept() + Linear("x") + SpatialX("x", "information")
    ).estimate(
        replace(
            refit, graphs={"information": replace(graph, weighting=RBFWeights(1.3))}
        )
    )
    np.testing.assert_allclose(result.result.parameters, fixed.parameters)


@pytest.mark.integration
def test_joint_projection_improves_spatial_mean() -> None:
    """The result retains a trained atom-map rather than freezing the initial graph."""
    cloud = PackedClouds.from_clouds(
        [jnp.array([[float(i), (-1.0) ** i], [float(i) + 0.3, 0.5]]) for i in _IDS]
    )
    p = ProjectionRef("map")
    graph = DistanceGraph.from_measures(
        MeasureColumn(_IDS, cloud),
        geometry=Energy(),
        projection=p,
        weighting=RBFWeights(1.3),
    )
    train = _sample(graph, seed=16, start=0, count=5)
    spec = LearnLinear(1, initial=jnp.array([[0.8], [0.15]]))
    result = Regression(
        "y",
        Intercept() + Linear("x") + SpatialX("x", "information"),
        estimator=GraphML(projections={p: spec}),
    ).estimate(train)
    assert isinstance(result, GraphRegressionResult)
    assert result.valid
    metadata = result.report().metadata
    params = metadata["graph_parameters"]
    assert isinstance(params, Mapping)
    matrix = params["map"]
    assert isinstance(matrix, Mapping)
    assert matrix["shape"] == (2, 1)
    assert matrix["rank"] == 1
    assert "value" not in matrix
    assert matrix["singular_values"] == pytest.approx(
        (float(jnp.linalg.norm(result.graph_parameters["map"])),)
    )
    graphs = metadata["graph_provenance"]
    assert isinstance(graphs, Mapping)
    information = graphs["information"]
    assert isinstance(information, Mapping)
    assert information["binding"] == "fixed"
    assert information["recipes"] == (
        {
            "source": "measures",
            "geometry": "Energy",
            "weighting": "RBFWeights",
            "bandwidth": 1.3,
            "bandwidth_ref": None,
            "projection": "map",
            "period_count": 5,
        },
    )
    fitted = result.graph_parameters["map"]
    truth = np.array([[0.7], [0.3]])
    assert np.linalg.norm(np.asarray(fitted) - truth) < np.linalg.norm(
        np.asarray(spec.initial) - truth
    )


@pytest.mark.integration
def test_unlearnable_bandwidth_with_equal_neighbor_costs() -> None:
    """An exactly invariant graph cannot support a learned bandwidth."""
    d = np.abs(np.subtract.outer(np.arange(7), np.arange(7)))
    active = DistanceGraph.from_distances(d, unit_ids=_IDS, weighting=RBFWeights(_H))
    ref = BandwidthRef("flat")
    invariant = DistanceGraph.from_distances(
        np.ones_like(d) - np.eye(len(_IDS)), unit_ids=_IDS, weighting=RBFWeights(ref)
    )
    data = _sample(active, seed=8, start=0, count=5)
    data = replace(data, graphs={**data.graphs, "uniform": invariant})
    model = Regression(
        "y",
        Intercept()
        + Linear("x")
        + SpatialX("x", "information")
        + SpatialX("x", "uniform"),
        estimator=GraphML(
            bandwidths={_H: LearnBandwidth(1.0), ref: LearnBandwidth(1.0)}
        ),
    )
    result = model.estimate(data)
    assert isinstance(result, GraphRegressionResult)
    assert result.core.converged
    assert result.core.identified
    assert not result.valid
    assert result.report().metadata["graph_profile_perturbation_detected"] == {
        "smooth": True,
        "flat": False,
    }
    assert np.isnan(result.parameters).all()
    # NaN coefficients must report the invalid fit, not a graph-domain failure.
    with pytest.raises(InvalidFitError, match="spatial regression"):
        result.predict(data)
    with pytest.raises(InvalidFitError):
        result.log_score(data)


@pytest.mark.unit
def test_graph_ml_requires_observed_degrees_of_freedom() -> None:
    """Count finite outcome records rather than complete graph nodes."""
    d = np.abs(np.subtract.outer(np.arange(7), np.arange(7)))
    graph = DistanceGraph.from_distances(d, unit_ids=_IDS, weighting=RBFWeights(_H))
    train = _sample(graph, seed=14, start=0, count=2)
    train = replace(train, outcomes=train.nodes.take([0, 1, 2]))
    model = Regression(
        "y",
        Intercept() + Linear("x") + SpatialX("x", "information"),
        estimator=GraphML(bandwidths={_H: LearnBandwidth(1.0)}),
    )
    with pytest.raises(ValueError, match="more observed outcomes"):
        model.estimate(train)


@pytest.mark.component
@pytest.mark.contract
def test_graph_bindings_precede_joint_scale_ambiguity() -> None:
    """Missing bindings are diagnosed before a fully bound scale convention."""
    projection = ProjectionRef("map")
    graph = DistanceGraph.from_measures(
        MeasureColumn(_IDS, PackedClouds.from_array(jnp.arange(14.0).reshape(7, 1, 2))),
        geometry=Energy(),
        projection=projection,
        weighting=RBFWeights(_H),
    )
    data = _sample(graph, seed=3, start=0, count=2)
    estimator = GraphML(
        projections={projection: LearnLinear(1, initial=jnp.array([[0.7], [0.3]]))}
    )
    model = Regression(
        "y",
        Intercept() + Linear("x") + SpatialX("x", "information"),
        estimator=estimator,
    )
    prepared = model.prepare(data)
    assert isinstance(prepared, PreparedGraphRegression)
    with pytest.raises(ValueError, match="bindings must match"):
        estimator.estimate(prepared)
    complete = replace(estimator, bandwidths={_H: LearnBandwidth(1.0)})
    with pytest.raises(ValueError, match="Fix the projection norm"):
        complete.estimate(prepared)


@pytest.mark.component
def test_learned_bandwidth_is_counted_in_the_information_criteria() -> None:
    """K counts mean coefficients, the learned bandwidth coordinate and the scale."""
    d = np.abs(np.subtract.outer(np.arange(7), np.arange(7)))
    graph = DistanceGraph.from_distances(d, unit_ids=_IDS, weighting=RBFWeights(_H))
    train = _sample(graph, seed=8, start=0, count=5)
    terms = Intercept() + Linear("x") + SpatialX("x", "information")
    result = Regression(
        "y", terms, estimator=GraphML(bandwidths={_H: LearnBandwidth(1.0)})
    ).estimate(train)
    assert isinstance(result, GraphRegressionResult)
    metrics = {m.name: m.value for m in result.report().metrics}
    # Three mean coefficients, one bandwidth and the Gaussian scale; n = 35 rows.
    assert metrics["n_parameters"] == 5  # noqa: PLR2004
    ll = float(result.core.log_likelihood)
    assert metrics["log_likelihood"] == pytest.approx(ll)
    assert metrics["aic"] == pytest.approx(-2 * ll + 10)
    assert metrics["bic"] == pytest.approx(-2 * ll + 5 * np.log(35))
    assert metrics["hqic"] == pytest.approx(-2 * ll + 10 * np.log(np.log(35)))


@pytest.mark.component
@pytest.mark.parametrize(("norm", "free"), [(None, 2), (0.76, 1)])
def test_learned_projection_counts_its_free_dimension(
    norm: float | None, free: int
) -> None:
    """A fixed Frobenius norm removes one dimension from the projection count."""
    cloud = PackedClouds.from_clouds(
        [jnp.array([[float(i), (-1.0) ** i], [float(i) + 0.3, 0.5]]) for i in _IDS]
    )
    p = ProjectionRef("map")
    graph = DistanceGraph.from_measures(
        MeasureColumn(_IDS, cloud),
        geometry=Energy(),
        projection=p,
        weighting=RBFWeights(1.3),
    )
    train = _sample(graph, seed=16, start=0, count=5)
    spec = LearnLinear(1, initial=jnp.array([[0.8], [0.15]]), norm=norm)
    result = Regression(
        "y",
        Intercept() + Linear("x") + SpatialX("x", "information"),
        estimator=GraphML(projections={p: spec}),
    ).estimate(train)
    assert isinstance(result, GraphRegressionResult)
    assert result.graph_parameters["map"].size == 2  # noqa: PLR2004
    # Three mean coefficients, the projection's free dimension and the scale.
    metrics = {m.name: m.value for m in result.report().metrics}
    assert metrics["n_parameters"] == 3 + free + 1
    assert metrics["n_observations"] == 35  # noqa: PLR2004
